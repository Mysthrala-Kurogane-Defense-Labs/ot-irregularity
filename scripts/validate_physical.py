"""Freeze a physical AE supplement, then compare once on new test batches."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_otlab import prepare
from scripts.research_detection import write_json
from scripts.research_physical import PHYSICAL, summarize
from scripts.validate_contextual import bootstrap, ui_state
from ot_irregularity.contextual import tail_errors
from ot_irregularity.metrics import evaluate_event_intervals
from ot_irregularity.models import ae_errors, load_ae
from ot_irregularity.pipeline import _scores, _windows, cdf_calibrate


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def freeze(args):
    args.output.mkdir(parents=True, exist_ok=False)
    results = json.loads((args.research / 'results.json').read_text())
    chosen = next(r for r in results['rows'] if r['id'] == 'max-physical')
    if chosen['id'] not in results['eligible']:
        raise ValueError('Candidate does not pass the declared development gate')
    write_json(args.output / 'frozen.json', {'frozen_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'baseline_files': hashes(args.baseline), 'physical_files': hashes(args.research / 'physical'),
        'decision_threshold': chosen['metrics']['threshold'], 'development': chosen,
        'holdout_seeds': [910511, 910512, 910513], 'seed': 20261005,
        'promotion_rule': 'More physical events, no loss of overall events, precision >=0.5, no increase in false windows. Report paired uncertainty; a small uncertain gain is not sufficient evidence for promotion.',
        'test_used_for_selection': False})
    ui_state(args.output, 'control', 0., [], 'Frozen; waiting for new holdouts')


def supplement(research, windows, device='auto'):
    score = np.zeros(len(windows))
    for group in sorted(windows['asset_class'].unique()):
        root = research / 'physical' / group
        model = joblib.load(root / 'detector.joblib')
        mask = windows['asset_class'].to_numpy() == group
        x = model['scaler'].transform(windows.select(model['features']).to_numpy()[mask])
        ae = load_ae(root / 'autoencoder.pt', device)
        raw, _ = tail_errors(ae_errors(ae, x), model['scale'], 4)
        score[mask] = cdf_calibrate(model['reference'], raw)
    return score


def evaluate(args):
    out = args.output
    frozen = json.loads((out / 'frozen.json').read_text())
    if (out / 'results.json').exists():
        raise FileExistsError('Holdout already evaluated')
    if hashes(args.baseline) != frozen['baseline_files'] or hashes(args.research / 'physical') != frozen['physical_files']:
        raise ValueError('Artifacts changed after freeze')
    schema = json.loads((args.baseline / 'feature_schema.json').read_text())
    cfg = yaml.safe_load((args.baseline / 'training_config.yaml').read_text())
    parts, events, sources = [], [], []
    for source in args.sources:
        prepared = out / ('prepared-' + source.name)
        manifest = prepare(source, prepared, max_train_runs=None, max_validation_runs=None, max_test_runs=None)
        seed = manifest['source_master_seed']
        if seed not in frozen['holdout_seeds'] or seed in [s['seed'] for s in sources]:
            raise ValueError('Unexpected or duplicated holdout seed')
        w = _windows(prepared / 'test', cfg, schema)
        w = w.with_columns((pl.lit(source.name + '::') + pl.col('run_id')).alias('run_id'),
            pl.when(pl.col('event_id').is_not_null()).then(pl.lit(source.name + '::') + pl.col('event_id')).otherwise(None).alias('event_id'))
        parts.append(w)
        truth_hashes = {}
        for truth in sorted((source / 'test').glob('*/ground_truth.json')):
            doc = json.loads(truth.read_text())
            truth_hashes[truth.relative_to(source).as_posix()] = hashlib.sha256(truth.read_bytes()).hexdigest()
            for event in doc['events']:
                def micros(s): return int(dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp() * 1e6)
                run = source.name + '::' + doc['run_id']
                events.append({'run_id': run, 'asset_id': event['asset_id'], 'event_id': run + '/' + event['event_id'],
                    'family': event['type'], 'start_us': micros(event['start']), 'end_us': micros(event['end'])})
        sources.append({'seed': seed, 'manifest_sha256': hashlib.sha256((source / 'dataset_manifest.json').read_bytes()).hexdigest(),
            'truth_hashes': truth_hashes, 'license': manifest['source_license'], 'attribution': manifest['source_attribution'],
            'source_url': manifest['source_url'], 'simulator_version': manifest['source_simulator_version'], 'test_runs': manifest['partitions']['test']['runs']})
    if {s['seed'] for s in sources} != set(frozen['holdout_seeds']):
        raise ValueError('Incomplete holdout set')
    w = pl.concat(parts)
    _, _, _, _, baseline, threshold = _scores(args.baseline, w)
    physical = supplement(args.research, w, args.device)
    candidate = np.maximum(baseline, physical)
    thresholds = [threshold, frozen['decision_threshold']]
    rows = [{'id': name, 'metrics': summarize(w, scores, th, events)}
            for name, scores, th in zip(('v0.4', 'max-physical'), (baseline, candidate), thresholds)]
    predictions = w.select(['run_id', 'asset_id', 'asset_class', 'window_start', 'window_end', 'is_anomaly', 'event_id', 'event_start_us'])
    predictions.with_columns(pl.Series('v0.4', baseline), pl.Series('physical', physical),
        pl.Series('max-physical', candidate)).write_parquet(out / 'predictions.parquet')
    write_json(out / 'events.json', events)
    details = [evaluate_event_intervals(events, scores, th, w['run_id'].to_numpy(), w['asset_id'].to_numpy(),
        w['window_start'].to_numpy(), w['window_end'].to_numpy())['events'] for scores, th in zip((baseline, candidate), thresholds)]
    families = {e['event_id']: e['family'] for e in events}
    runs = sorted(set(w['run_id']))
    counts = []
    for table in details:
        counts.append(np.array([[sum(e['run_id'] == run for e in table),
            sum(e['run_id'] == run and e['detected'] for e in table),
            sum(e['run_id'] == run and families[e['event_id']] in PHYSICAL for e in table),
            sum(e['run_id'] == run and families[e['event_id']] in PHYSICAL and e['detected'] for e in table)] for run in runs]))
    rng = np.random.default_rng(20261005)
    deltas = {'event_detection_rate': [], 'physical_event_detection_rate': []}
    for _ in range(2000):
        pick = rng.integers(0, len(runs), len(runs))
        a, b = [c[pick].sum(axis=0) for c in counts]
        for key, n, d in [('event_detection_rate', 1, 0), ('physical_event_detection_rate', 3, 2)]:
            if a[d] and b[d]: deltas[key].append(b[n] / b[d] - a[n] / a[d])
    interval_ci = {k: np.quantile(v, [.025, .975]).tolist() for k, v in deltas.items()}
    window_ci = bootstrap(w, [baseline, candidate], thresholds)
    window_ci.pop('event_detection_rate', None)  # Legacy single-ID event bootstrap is not authoritative.
    a, b = [r['metrics'] for r in rows]
    gate = (b['physical_events']['detected'] > a['physical_events']['detected'] and
            b['detected_events'] >= a['detected_events'] and b['precision'] >= .5 and
            b['false_positive_windows'] <= a['false_positive_windows'])
    write_json(out / 'results.json', {'rows': rows, 'sources': sources, 'test_windows': len(w),
        'exposure_asset_hours': len(w) / 60, 'event_count': len(events),
        'paired_window_ci95': window_ci, 'paired_interval_ci95': interval_ci,
        'interval_bootstrap_replicates': 2000, 'selection_after_test': False,
        'no_regression_gate_passed': bool(gate),
        'decision': 'Review paired uncertainty before promotion' if gate else 'Retain v0.4; candidate failed the frozen gate'})
    ui_state(out, 'completed', 1., rows)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'evaluate'])
    for key in ('baseline', 'research', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs='+')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.command == 'evaluate' and not args.sources:
        parser.error('evaluate requires --sources')
    (freeze if args.command == 'freeze' else evaluate)(args)
