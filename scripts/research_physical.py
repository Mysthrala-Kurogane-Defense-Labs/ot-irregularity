"""Development-only physical/regime ablation with complete interval denominators."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import polars as pl
import torch
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.research_detection import cache_windows, metrics, write_json
from scripts.validate_contextual import ui_state
from ot_irregularity.contextual import run_bucket, select_columns, tail_errors
from ot_irregularity.metrics import evaluate_event_intervals
from ot_irregularity.models import ae_errors, train_ae
from ot_irregularity.pipeline import _scores, cdf_calibrate

PHYSICAL = {'cavitation', 'bearing_degradation', 'cooling_degradation', 'mechanical_overload'}
VALUE_SUFFIXES = ('_mean', '_median', '_min', '_max', '_range', '_std', '_mad', '_last', '_delta', '_slope')


def center_regimes(fit, frames, columns):
    """Fit per-regime medians on normal fit runs; unseen regimes use class median."""
    fallback = np.median(fit.select(columns).to_numpy(), axis=0)
    centers = {}
    counts = {}
    for regime in sorted(fit['operating_regime'].unique()):
        part = fit.filter(pl.col('operating_regime') == regime)
        counts[regime] = len(part)
        if len(part) >= 20:
            centers[regime] = np.median(part.select(columns).to_numpy(), axis=0)
    arrays = []
    unknown = []
    for frame in frames:
        offsets = np.array([centers.get(r, fallback) for r in frame['operating_regime']])
        arrays.append(frame.select(columns).to_numpy().astype(float) - offsets)
        unknown.append(sum(r not in centers for r in frame['operating_regime']))
    return arrays, {'centers': centers, 'fallback': fallback, 'fit_counts': counts, 'fallback_windows': unknown}


def load_events(raw_root, dev):
    """Namespace original validation truth exactly as the historical assembled dataset."""
    runs = set(dev['run_id'].to_list())
    events, hashes, matched = [], {}, set()
    for truth in sorted(raw_root.glob('raw-*/validation/*/ground_truth.json')):
        doc = json.loads(truth.read_text())
        batch = truth.parents[2].name.replace('raw-', 'dev-')
        run = batch + '::' + doc['run_id']
        if run not in runs:
            continue
        if run in matched:
            raise ValueError('Duplicated development run ground truth')
        matched.add(run)
        hashes[truth.relative_to(raw_root).as_posix()] = hashlib.sha256(truth.read_bytes()).hexdigest()
        for event in doc['events']:
            def micros(value):
                return int(dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() * 1e6)
            events.append({'run_id': run, 'asset_id': event['asset_id'],
                           'event_id': run + '/' + event['event_id'], 'family': event['type'],
                           'start_us': micros(event['start']), 'end_us': micros(event['end'])})
    if not events or matched != runs:
        raise ValueError('Incomplete matching development ground truth')
    return events, hashes


def summarize(frame, scores, threshold, events):
    result = metrics(frame, scores, threshold)
    intervals = evaluate_event_intervals(events, scores, threshold, frame['run_id'].to_numpy(),
        frame['asset_id'].to_numpy(), frame['window_start'].to_numpy(), frame['window_end'].to_numpy())
    families = {e['event_id']: e['family'] for e in events}
    result.update({k: v for k, v in intervals.items() if k != 'events'})
    result['mean_time_to_first_detection_seconds'] = intervals['mean_detection_latency_seconds']
    by_family = {}
    for family in sorted(set(families.values())):
        selected = [e for e in intervals['events'] if families[e['event_id']] == family]
        by_family[family] = {'total': len(selected), 'detected': sum(e['detected'] for e in selected)}
    result['by_family'] = by_family
    physical = [e for e in intervals['events'] if families[e['event_id']] in PHYSICAL]
    result['physical_events'] = {'total': len(physical), 'detected': sum(e['detected'] for e in physical)}
    return result


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    schema = json.loads((args.baseline / 'feature_schema.json').read_text())
    cfg = yaml.safe_load((args.baseline / 'training_config.yaml').read_text())
    frames = cache_windows(args.dataset, schema, cfg, out / 'cache')
    tr, va = frames['train'], frames['validation']
    if 'is_anomaly' not in tr.columns or tr['is_anomaly'].null_count() or tr['is_anomaly'].any():
        raise ValueError('Training requires explicitly normal-only windows')
    if 'is_anomaly' not in va.columns or va['is_anomaly'].null_count():
        raise ValueError('Validation requires explicit normal/anomaly labels')
    if set(tr['run_id']) & set(va['run_id']):
        raise ValueError('Train/validation run leakage')
    fit = tr.filter(pl.Series([run_bucket(r, 5) != 0 for r in tr['run_id']]))
    stop = tr.filter(pl.Series([run_bucket(r, 5) == 0 for r in tr['run_id']]))
    cal = va.filter(pl.Series([run_bucket(r, 2) == 0 for r in va['run_id']])).filter(~pl.col('is_anomaly'))
    dev = va.filter(pl.Series([run_bucket(r, 2) == 1 for r in va['run_id']]))
    events, hashes = load_events(args.raw_root, dev)
    write_json(out / 'protocol.json', {'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'seed': args.seed, 'test_used': False, 'physical_families': sorted(PHYSICAL),
        'variants': ['physical', 'regime-all', 'regime-physical', 'max-physical', 'max-regime-physical'],
        'selection': 'Require more physical events, no loss of total detected events, precision >=0.5 and no increase in false windows versus v0.4 on development. Otherwise retain v0.4 without a fresh test.',
        'normal_quantile': .99, 'steps_per_class': 3000, 'truth_hashes': hashes,
        'baseline_manifest_sha256': hashlib.sha256((args.baseline / 'contextual_models.json').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'counts': {k: len(v) for k, v in [('fit', fit), ('stop', stop), ('cal', cal), ('dev', dev)]}})
    _, _, _, _, base_cal, threshold = _scores(args.baseline, cal)
    _, _, _, _, base_dev, _ = _scores(args.baseline, dev)
    rows = [{'id': 'v0.4', 'metrics': summarize(dev, base_dev, threshold, events)}]
    prediction = dev.select(['run_id', 'asset_id', 'window_start', 'window_end', 'is_anomaly']).with_columns(pl.Series('v0.4', base_dev))
    score_sets = {}
    for index, variant in enumerate(('physical', 'regime-all', 'regime-physical')):
        ui_state(out, 'screening', index / 3, rows, variant)
        cs, ds = np.zeros(len(cal)), np.zeros(len(dev))
        for group in sorted(fit['asset_class'].unique()):
            subsets = [f.filter(pl.col('asset_class') == group) for f in (fit, stop, cal, dev)]
            cols = select_columns(fit, schema, group)
            if variant != 'regime-all':
                cols = [c for c in cols if c.endswith(VALUE_SUFFIXES)]
            regime = None
            arrays = [f.select(cols).to_numpy().astype(float) for f in subsets]
            if variant.startswith('regime-'):
                arrays, regime = center_regimes(subsets[0], subsets, cols)
            scaler = RobustScaler().fit(arrays[0])
            x, xe, xc, xd = [scaler.transform(a) for a in arrays]
            root = out / variant / group
            root.mkdir(parents=True)
            ae = train_ae(x, xe, {**cfg['autoencoder'], 'device': args.device}, args.seed, root / 'autoencoder.pt')
            iso = IsolationForest(n_estimators=200, random_state=args.seed, n_jobs=2).fit(x)
            scale = np.maximum(np.quantile(ae_errors(ae, xe), .95, axis=0), 1e-8)
            rc, _ = tail_errors(ae_errors(ae, xc), scale, 4)
            rd, _ = tail_errors(ae_errors(ae, xd), scale, 4)
            cs[cal['asset_class'].to_numpy() == group] = cdf_calibrate(rc, rc)
            ds[dev['asset_class'].to_numpy() == group] = cdf_calibrate(rc, rd)
            joblib.dump({'features': cols, 'scaler': scaler, 'regime': regime, 'isolation': iso,
                'scale': scale, 'reference': rc}, root / 'detector.joblib')
            print(variant, group, 'complete', flush=True)
        score_sets[variant] = (cs, ds)
        th = float(np.nextafter(np.quantile(cs, .99, method='higher'), np.inf))
        rows.append({'id': variant, 'metrics': summarize(dev, ds, th, events)})
        prediction = prediction.with_columns(pl.Series(variant, ds))
        print(json.dumps(rows[-1]), flush=True)
    for name in ('physical', 'regime-physical'):
        cs, ds = score_sets[name]
        cs, ds = np.maximum(cs, base_cal), np.maximum(ds, base_dev)
        th = float(np.nextafter(np.quantile(cs, .99, method='higher'), np.inf))
        rows.append({'id': 'max-' + name, 'metrics': summarize(dev, ds, th, events)})
        prediction = prediction.with_columns(pl.Series('max-' + name, ds))
    base = rows[0]['metrics']
    eligible = [r['id'] for r in rows[1:] if
        r['metrics']['physical_events']['detected'] > base['physical_events']['detected'] and
        r['metrics']['event_detection_rate'] >= base['event_detection_rate'] and
        r['metrics']['precision'] >= .5 and
        r['metrics']['false_positives_per_asset_day'] <= base['false_positives_per_asset_day']]
    write_json(out / 'results.json', {'rows': rows, 'eligible': eligible, 'test_used': False,
        'event_count': len(events), 'selection_after_test': False})
    prediction.write_parquet(out / 'development_predictions.parquet')
    ui_state(out, 'completed', 1., rows)
    print('ELIGIBLE', eligible, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'baseline', 'raw-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261005)
    parser.add_argument('--device', default='cuda:0')
    run(parser.parse_args())
