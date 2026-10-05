"""Controlled normal-coverage ablation; historical and new development only."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import polars as pl
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.normal_exposure_metrics import alarm_counts, summarize as normal_summary
from scripts.prepare_otlab import _sha256
from scripts.research_detection import cache_windows, write_json
from scripts.research_physical import PHYSICAL, load_events, summarize
from scripts.research_relationships import closed
from scripts.validate_physical import hashes
from ot_irregularity.contextual import run_bucket, train_contextual_windows
from ot_irregularity.metrics import evaluate_event_intervals
from ot_irregularity.pipeline import _git_commit, _git_working_tree_dirty, _scores, cdf_calibrate


def integrity_losses(baseline, candidate, families):
    key = lambda e: (e['run_id'], e['asset_id'], e['event_id'])
    detected = {key(e) for e in candidate if e['detected']}
    return sorted(key(e) for e in baseline if e['detected'] and families[key(e)] not in PHYSICAL and key(e) not in detected)


def eligible_component(mixed, normal, lost, baseline_mixed, baseline_normal):
    return bool(not lost and mixed['detected_events'] >= baseline_mixed['detected_events']
        and mixed['physical_events']['detected'] >= baseline_mixed['physical_events']['detected']
        and mixed['false_positive_windows'] <= baseline_mixed['false_positive_windows']
        and mixed['precision'] >= .5
        and normal['overall']['false_windows'] <= baseline_normal['overall']['false_windows']
        and all(m['false_windows_per_asset_day'] <= 10 for m in normal['by_class'].values()))


def normal_metrics(frame, score, threshold, attributes):
    records = []
    data = frame.with_columns(pl.Series('_alert', score >= threshold))
    for (run_id, group), windows in data.group_by(['run_id', 'asset_class']):
        records.append({'run_id': run_id, 'asset_class': group, **attributes[run_id],
                        'models': {'candidate': alarm_counts(windows, windows['_alert'].to_numpy())}})
    return {'overall': normal_summary(records, ['candidate'])['candidate'],
            'by_class': {g: normal_summary([r for r in records if r['asset_class'] == g], ['candidate'])['candidate']
                         for g in sorted(frame['asset_class'].unique())},
            'strata': {key: {str(value): normal_summary([r for r in records if r[key] == value], ['candidate'])['candidate']
                             for value in sorted({r[key] for r in records})}
                       for key in ('profile', 'sampling_interval_ms', 'expose_operating_regime')}}


def event_table(events, frame, score, threshold):
    return evaluate_event_intervals(events, score, threshold, frame['run_id'].to_numpy(),
        frame['asset_id'].to_numpy(), frame['window_start'].to_numpy(), frame['window_end'].to_numpy())['events']


def run(args):
    out = args.output
    result_path = out / 'coverage-results.json'
    if result_path.exists() or (out / 'models').exists():
        raise FileExistsError('Existing ablation; inspect completed work before repeating')
    protocol_path = Path(__file__).resolve().parents[1] / 'docs/NORMAL_COVERAGE_DEVELOPMENT.md'
    declared = json.loads((out / 'data_protocol.json').read_text())
    declared_files = {name.replace('\\', '/'): sha for name, sha in declared['files'].items()}
    repo = Path(__file__).resolve().parents[1]
    if any(_sha256(repo / name) != sha for name, sha in declared_files.items()):
        raise ValueError('Development protocol or suite changed')
    cache = out / 'normal-cache'; provenance = json.loads((cache / 'manifest.json').read_text())
    if provenance['source_seed'] != 920611 or provenance['test_used']:
        raise ValueError('Only declared new development seed is allowed')
    if provenance['feature_schema_sha256'] != _sha256(args.baseline / 'feature_schema.json'):
        raise ValueError('Feature schema differs from cache')
    for name, sha in provenance['files'].items():
        if _sha256(cache / name) != sha:
            raise ValueError('Normal feature cache changed')
    source_manifest = args.normal_source / 'dataset_manifest.json'
    if _sha256(source_manifest) != provenance['source_manifest_sha256']:
        raise ValueError('Normal source manifest changed')
    source = json.loads(source_manifest.read_text())
    profiles = yaml.safe_load((repo / 'configs/normal-coverage-development.yaml').read_text())['generation']['regime_profiles']
    profile_by_run = {'normal-dev-920611::' + e['run_id']: next(p['profile_id'] for p in profiles
                       if p['shift_pattern'] == e['configured_shift_pattern']) for e in source['runs']}
    attributes = {r['run_id']: {'profile': profile_by_run[r['run_id']],
                   'sampling_interval_ms': r['sampling_interval_ms'],
                   'expose_operating_regime': r['expose_operating_regime']} for r in provenance['runs']}
    torch.set_num_threads(2)
    cfg = yaml.safe_load((args.baseline / 'training_config.yaml').read_text())
    if cfg['ensemble'] != {'autoencoder_weight': 1.0, 'isolation_weight': 0.0}:
        raise ValueError('Coverage ablation requires the declared AE-only ensemble; IF is retained separately')
    schema = json.loads((args.baseline / 'feature_schema.json').read_text())
    old = cache_windows(args.dataset, schema, cfg, args.cache)
    old = {p: closed(w, pl.read_parquet(args.dataset / p / 'telemetry.parquet')) for p, w in old.items()}
    new = {p: pl.read_parquet(cache / (p + '.parquet')) for p in ('train', 'validation')}
    for p in new:
        if new[p]['is_anomaly'].null_count() or new[p]['is_anomaly'].any():
            raise ValueError('New normal source has missing/anomalous labels')
        if any(not r.startswith('normal-dev-920611::') for r in new[p]['run_id']):
            raise ValueError('Unexpected development namespace')
    sets = [set(f['run_id']) for f in [old['train'], old['validation'], new['train'], new['validation']]]
    if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]):
        raise ValueError('Run leakage between sources/partitions')
    tr = pl.concat([old['train'], new['train']], how='diagonal_relaxed')
    va = pl.concat([old['validation'], new['validation']], how='diagonal_relaxed')
    if tr['is_anomaly'].null_count() or tr['is_anomaly'].any():
        raise ValueError('Training must be explicitly normal')
    historical = old['validation'].filter(pl.Series([run_bucket(r, 2) == 1 for r in old['validation']['run_id']]))
    normal_dev = new['validation'].filter(pl.Series([run_bucket(r, 2) == 1 for r in new['validation']['run_id']]))
    if not len(normal_dev):
        raise ValueError('No new normal development runs')
    events, truth_hashes = load_events(args.raw_root, historical)
    families = {(e['run_id'], e['asset_id'], e['event_id']): e['family'] for e in events}
    _, _, _, _, base, bth = _scores(args.baseline, historical)
    _, _, _, _, base_normal, _ = _scores(args.baseline, normal_dev)
    bm = summarize(historical, base, bth, events); bn = normal_metrics(normal_dev, base_normal, bth, attributes)
    be = event_table(events, historical, base, bth)
    identity = {'protocol_sha256': _sha256(protocol_path), 'baseline_files': hashes(args.baseline),
                'normal_cache_manifest_sha256': _sha256(cache / 'manifest.json'),
                'historical_cache_manifest': json.loads((args.cache / 'cache_manifest.json').read_text()),
                'truth_hashes': truth_hashes, 'test_used': False}
    write_json(out / 'ablation-inputs.json', identity)
    dataset_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    rows = []; selected = None; confirmation = []
    for seed in (20261005, 20261006, 20261007):
        if seed != 20261005 and selected is None:
            break
        random.seed(seed); np.random.seed(seed)
        model = out / 'models' / str(seed); model.mkdir(parents=True)
        local_cfg = deepcopy(cfg)
        local_cfg.update(seed=seed, device=args.device, model_version='0.6.0-coverage-development')
        local_cfg['window']['tail_policy'] = 'complete'
        metadata = {'model_version': local_cfg['model_version'], 'training_date': datetime.now(timezone.utc).isoformat(),
                    'git_commit': _git_commit(), 'git_working_tree_dirty': _git_working_tree_dirty(),
                    'dataset_hash': dataset_hash, 'dataset_schema_version': '1', 'stride_seconds': 60,
                    'source_windows': {kind: {p: len(w) for p, w in frames.items()} for kind, frames in [('historical', old), ('new_normal', new)]}}
        write_json(out / 'progress.json', {'phase': 'training', 'seed': seed, 'models_completed': len(confirmation)})
        train_contextual_windows(tr, va, local_cfg, model, deepcopy(schema), metadata)
        _, _, _, _, values, _ = _scores(model, historical)
        _, _, _, _, nv, _ = _scores(model, normal_dev)
        manifest = json.loads((model / 'contextual_models.json').read_text())
        refs = np.concatenate([cdf_calibrate(m['autoencoder_reference'], m['autoencoder_reference']) for m in manifest['groups'].values()])
        quantiles = [.99, .995, .9975] if seed == 20261005 else [selected['quantile']]
        for q in quantiles:
            th = float(np.nextafter(np.quantile(refs, q, method='higher'), np.inf))
            if th > 1:
                rows.append({'seed': seed, 'quantile': q, 'eligible': False, 'reason': 'Unresolvable normal quantile'})
                continue
            mixed = summarize(historical, values, th, events)
            normal = normal_metrics(normal_dev, nv, th, attributes)
            lost = integrity_losses(be, event_table(events, historical, values, th), families)
            row = {'seed': seed, 'quantile': q, 'threshold': th, 'historical': mixed, 'new_normal': normal,
                   'lost_baseline_nonphysical_events': lost, 'eligible': eligible_component(mixed, normal, lost, bm, bn)}
            rows.append(row)
            print(json.dumps({'seed': seed, 'quantile': q, 'events': mixed['detected_events'],
                  'false_historical': mixed['false_positive_windows'], 'false_new_normal': normal['overall']['false_windows'],
                  'lost_integrity': len(lost), 'eligible': row['eligible']}), flush=True)
        historical.select(['run_id', 'asset_id', 'window_start', 'window_end']).with_columns(pl.Series('score', values)).write_parquet(model / 'historical_scores.parquet')
        normal_dev.select(['run_id', 'asset_id', 'asset_class', 'window_start', 'window_end']).with_columns(pl.Series('score', nv)).write_parquet(model / 'normal_scores.parquet')
        if seed == 20261005:
            eligible = [r for r in rows if r['eligible']]
            selected = max(eligible, key=lambda r: (r['historical']['detected_events'], r['historical']['physical_events']['detected'],
                -r['new_normal']['overall']['false_windows'], -r['quantile'])) if eligible else None
        confirmation.append({'seed': seed, 'files': hashes(model)})
    result = {'protocol_sha256': identity['protocol_sha256'], 'baseline_historical': bm, 'baseline_new_normal': bn,
              'rows': rows, 'selected': selected, 'models': confirmation, 'test_used': False,
              'secondary_seeds_skipped': selected is None,
              'confirmed_development': selected is not None and len(rows) == 5 and all(r['eligible'] for r in rows if r['seed'] != 20261005),
              'historical_windows': len(historical), 'new_normal_windows': len(normal_dev),
              'new_normal_development_runs': normal_dev['run_id'].n_unique(),
              'artifact_threshold_note': 'Artifacts retain default training q=.99; selected operating point is explicit in this report and requires separate freeze before confirmation.'}
    write_json(result_path, result)
    write_json(out / 'progress.json', {'phase': 'completed', 'selected': selected, 'confirmed_development': result['confirmed_development']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'baseline', 'dataset', 'cache', 'raw-root', 'normal-source'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    run(p.parse_args())
