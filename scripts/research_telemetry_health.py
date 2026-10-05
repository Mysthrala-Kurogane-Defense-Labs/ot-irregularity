"""Predeclared telemetry-health supplement; development only, no network retraining."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_normal_exposure import verify_run
from scripts.prepare_otlab import _sha256, normalize_observations
from scripts.relationship_operating_points import residual_scores
from scripts.research_detection import cache_windows, write_json
from scripts.research_normal_coverage import event_table, integrity_losses, normal_metrics
from scripts.research_physical import PHYSICAL, load_events, summarize
from scripts.research_relationships import closed
from scripts.validate_physical import hashes
from ot_irregularity.contextual import run_bucket
from ot_irregularity.pipeline import _git_commit, _scores
from ot_irregularity.telemetry_health import KEYS, TelemetryHealthReference, extract_health


def align_scores(windows, scores):
    if scores.select(KEYS).is_duplicated().any():
        raise ValueError('Duplicate health score identity')
    if not windows.select(KEYS).sort(KEYS).equals(scores.select(KEYS).sort(KEYS)):
        raise ValueError('Health score identities differ from windows')
    return windows.select(KEYS).join(scores, on=KEYS, maintain_order='left')


def candidate_gate(metrics, normal, lost, physical_lost, baseline, relational, relational_normal):
    return bool(not lost and not physical_lost
        and metrics['detected_events'] >= relational['detected_events']
        and metrics['false_positive_windows'] <= baseline['false_positive_windows']
        and metrics['precision'] >= .5
        and normal['overall']['false_windows'] <= relational_normal['overall']['false_windows']
        and all(m['false_windows_per_asset_day'] <= 10 for m in normal['by_class'].values()))


def run(args):
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists() or (out / 'health-reference.json').exists():
        raise FileExistsError('Health study already started/completed; inspect before repeating')
    cfg = yaml.safe_load((args.baseline / 'training_config.yaml').read_text())
    schema = json.loads((args.baseline / 'feature_schema.json').read_text())
    relational = json.loads((args.relationships / 'frozen.json').read_text())
    if [m['seed'] for m in relational['models']] != [20261005, 20261006, 20261007]:
        raise ValueError('Declared primary and confirmation seeds must be ordered and complete')
    if hashes(args.baseline) != relational['baseline_files']:
        raise ValueError('Baseline differs from frozen control')
    for model in relational['models']:
        if hashes(args.relationships / model['directory']) != model['files']:
            raise ValueError('Relational model changed')
    provenance = json.loads((args.normal_cache / 'manifest.json').read_text())
    if provenance['source_seed'] != 920611 or provenance['test_used']:
        raise ValueError('Only declared new normal development is allowed')
    if _sha256(args.normal_source / 'dataset_manifest.json') != provenance['source_manifest_sha256']:
        raise ValueError('New normal source changed')
    for name, sha in provenance['files'].items():
        if _sha256(args.normal_cache / name) != sha:
            raise ValueError('Normal window cache changed')
    if provenance['feature_schema_sha256'] != _sha256(args.baseline / 'feature_schema.json'):
        raise ValueError('Feature schema mismatch')
    old = cache_windows(args.dataset, schema, cfg, args.cache)
    raw = {p: pl.read_parquet(args.dataset / p / 'telemetry.parquet') for p in old}
    old = {p: closed(frame, raw[p]) for p, frame in old.items()}
    new = {p: pl.read_parquet(args.normal_cache / (p + '.parquet')) for p in ('train', 'validation')}
    populations = [set(f['run_id']) for f in [*old.values(), *new.values()]]
    if any(a & b for i, a in enumerate(populations) for b in populations[i+1:]):
        raise ValueError('Run leakage between partitions/sources')
    source = json.loads((args.normal_source / 'dataset_manifest.json').read_text())
    profiles = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/normal-coverage-development.yaml').read_text())['generation']['regime_profiles']
    profile_by_run = {'normal-dev-920611::' + e['run_id']: next(p['profile_id'] for p in profiles if p['shift_pattern'] == e['configured_shift_pattern']) for e in source['runs']}
    attributes = {r['run_id']: {'profile': profile_by_run[r['run_id']], 'sampling_interval_ms': r['sampling_interval_ms'],
                  'expose_operating_regime': r['expose_operating_regime']} for r in provenance['runs']}
    repo = Path(__file__).resolve().parents[1]
    identity = {'protocol_sha256': _sha256(repo / 'docs/TELEMETRY_HEALTH_PROTOCOL.md'), 'git_commit': _git_commit(),
                'implementation_sha256': _sha256(repo / 'src/ot_irregularity/telemetry_health.py'),
                'baseline_files': relational['baseline_files'], 'relational_models': relational['models'],
                'historical_cache_manifest': json.loads((args.cache / 'cache_manifest.json').read_text()),
                'normal_cache_manifest_sha256': _sha256(args.normal_cache / 'manifest.json'), 'test_used': False}
    write_json(out / 'inputs.json', identity)
    cache = out / 'health-cache'; cache.mkdir(exist_ok=True)
    cache_manifest = cache / 'manifest.json'
    if cache_manifest.exists():
        saved = json.loads(cache_manifest.read_text())
        if saved['inputs'] != identity or any(_sha256(cache / name) != sha for name, sha in saved['files'].items()):
            raise ValueError('Health cache provenance changed')
    else:
        if any(cache.glob('*.parquet')):
            raise ValueError('Incomplete health cache; preserve and inspect it')
        for p in old:
            extract_health(raw[p], old[p], schema['signals_by_asset_class']).write_parquet(cache / ('old-' + p + '.parquet'))
            write_json(out / 'progress.json', {'phase': 'extract_historical', 'partition': p})
        parts = {'train': [], 'validation': []}
        for n, entry in enumerate(source['runs'], 1):
            if entry['partition'] not in parts:
                raise ValueError('Do not read normal source test/challenge')
            folder, _, _, _ = verify_run(args.normal_source, entry, entry['partition'])
            observations = normalize_observations(pl.read_parquet(folder / 'telemetry.parquet')).with_columns(
                (pl.lit('normal-dev-920611::') + pl.col('run_id')).alias('run_id'))
            windows = new[entry['partition']].filter(pl.col('run_id') == 'normal-dev-920611::' + entry['run_id'])
            parts[entry['partition']].append(extract_health(observations, windows, schema['signals_by_asset_class']))
            write_json(out / 'progress.json', {'phase': 'extract_new_normal', 'completed_runs': n, 'total_runs': len(source['runs'])})
        for p, frames in parts.items():
            pl.concat(frames, how='diagonal_relaxed').write_parquet(cache / ('new-' + p + '.parquet'))
        write_json(cache_manifest, {'inputs': identity, 'files': {p.name: _sha256(p) for p in cache.glob('*.parquet')}})
    del raw
    h = {kind: {p: pl.read_parquet(cache / (kind + '-' + p + '.parquet')) for p in old} for kind in ('old', 'new')}
    tr = pl.concat([h['old']['train'], h['new']['train']], how='diagonal_relaxed')
    va = pl.concat([h['old']['validation'], h['new']['validation']], how='diagonal_relaxed')
    fit = tr.filter(pl.Series([run_bucket(r, 5) != 0 for r in tr['run_id']]))
    cal = va.filter(pl.Series([run_bucket(r, 2) == 0 for r in va['run_id']]) & ~va['is_anomaly'])
    model = TelemetryHealthReference().fit(fit, cal); model.save(out / 'health-reference.json')
    historical = old['validation'].filter(pl.Series([run_bucket(r, 2) == 1 for r in old['validation']['run_id']]))
    normal = new['validation'].filter(pl.Series([run_bucket(r, 2) == 1 for r in new['validation']['run_id']]))
    events, truth_hashes = load_events(args.raw_root, historical)
    families = {(e['run_id'], e['asset_id'], e['event_id']): e['family'] for e in events}
    _, _, _, _, base, bth = _scores(args.baseline, historical)
    base_normal = _scores(args.baseline, normal)[4]
    baseline = summarize(historical, base, bth, events); baseline_events = event_table(events, historical, base, bth)
    health = {}; availability = {}
    for enabled in (False, True):
        for kind, windows in [('historical', historical), ('normal', normal)]:
            subset = va.join(windows.select(KEYS), on=KEYS, how='semi')
            scored, details = model.score(subset, repetition=enabled)
            scored = align_scores(windows, scored)
            scored.write_parquet(out / f'health-{kind}-repetition-{enabled}.parquet')
            if enabled:
                details.write_parquet(out / f'health-details-{kind}.parquet')
            health[enabled, kind] = scored['health_score'].fill_null(0).to_numpy()
            if enabled:
                availability[kind] = {'windows': len(scored), 'available': {
                    c: len(scored) - scored[c].null_count()
                    for c in ('health_score', 'quality_score', 'sampling_score', 'repetition_score')}}
    rows = []; controls = []; selected = None
    for spec in relational['models']:
        seed = spec['seed']
        if seed != 20261005 and selected is None:
            break
        directory = args.relationships / spec['directory']; threshold = spec['threshold']
        original = np.maximum(base, residual_scores(directory, historical, args.device)['ae'])
        original_normal = np.maximum(base_normal, residual_scores(directory, normal, args.device)['ae'])
        for kind, windows, values in [('historical', historical, original), ('normal', normal, original_normal)]:
            windows.select(KEYS).with_columns(pl.Series('original_score', values),
                pl.lit(threshold).alias('original_threshold')).write_parquet(out / f'original-{seed}-{kind}.parquet')
        control = summarize(historical, original, threshold, events)
        control_normal = normal_metrics(normal, original_normal, threshold, attributes)
        controls.append({'seed': seed, 'historical': control, 'new_normal': control_normal})
        original_events = event_table(events, historical, original, threshold)
        physical_ids = {(e['run_id'], e['asset_id'], e['event_id']) for e in original_events if e['detected'] and families[(e['run_id'], e['asset_id'], e['event_id'])] in PHYSICAL}
        points = [(enabled, th) for enabled in (False, True) for th in (.05, .10, .20)] if seed == 20261005 else [(selected['repetition'], selected['health_threshold'])]
        for enabled, th in points:
            margin = np.maximum(original / threshold, health[enabled, 'historical'] / th)
            ranking = margin / (1 + margin)
            normal_alert = (original_normal >= threshold) | (health[enabled, 'normal'] >= th)
            if not np.array_equal(ranking >= .5, (original >= threshold) | (health[enabled, 'historical'] >= th)):
                raise ValueError('Ranking/union decision mismatch')
            m = summarize(historical, ranking, .5, events)
            nm = normal_metrics(normal, normal_alert.astype(float), .5, attributes)
            event_rows = event_table(events, historical, ranking, .5)
            lost = integrity_losses(baseline_events, event_rows, families)
            detected = {(e['run_id'], e['asset_id'], e['event_id']) for e in event_rows if e['detected']}
            physical_lost = sorted(physical_ids - detected)
            row = {'seed': seed, 'repetition': enabled, 'health_threshold': th, 'historical': m, 'new_normal': nm,
                   'lost_baseline_nonphysical_events': lost, 'lost_relational_physical_events': physical_lost,
                   'eligible': candidate_gate(m, nm, lost, physical_lost, baseline, control, control_normal)}
            rows.append(row)
            print(json.dumps({'seed': seed, 'repetition': enabled, 'threshold': th, 'events': m['detected_events'],
                  'historical_false': m['false_positive_windows'], 'normal_false': nm['overall']['false_windows'],
                  'lost_integrity': len(lost), 'eligible': row['eligible']}), flush=True)
        if seed == 20261005:
            eligible = [r for r in rows if r['eligible']]
            selected = max(eligible, key=lambda r: (r['historical']['detected_events'], -r['new_normal']['overall']['false_windows'],
                -int(r['repetition']), r['health_threshold'])) if eligible else None
    write_json(out / 'results.json', {'baseline': baseline, 'controls': controls, 'rows': rows, 'selected': selected,
        'truth_hashes': truth_hashes, 'health_reference_sha256': _sha256(out / 'health-reference.json'),
        'confirmed_development': selected is not None and len(controls) == 3 and all(r['eligible'] for r in rows if r['seed'] != 20261005),
        'reference_windows': {'fit_role_windows': len(fit), 'calibration_role_windows': len(cal)},
        'component_availability': availability,
        'test_used': False, 'network_retraining': False,
        'ranking': 'm=max(original_relational/original_threshold,health/health_threshold); rank=m/(1+m); not probability'})
    write_json(out / 'progress.json', {'phase': 'completed', 'selected': selected})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'baseline', 'relationships', 'dataset', 'cache', 'raw-root', 'normal-source', 'normal-cache'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0'); run(p.parse_args())
