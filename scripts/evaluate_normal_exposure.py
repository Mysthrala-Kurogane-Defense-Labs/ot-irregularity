"""Evaluate frozen controls on normal-only runs, with verified resumable run outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.normal_exposure_metrics import alarm_counts, summarize
from scripts.prepare_otlab import _annotate, _epoch_us, _sha256, normalize_observations
from scripts.relationship_operating_points import residual_scores
from scripts.research_detection import write_json
from scripts.research_joint_decisions import relational_magnitude_scores
from scripts.validate_physical import hashes
from ot_irregularity.pipeline import _model_config, _scores, _windows
from ot_irregularity.sample_thermal import aggregate_residuals, bounded_tail_score

MODELS = ['v04', 'v05-relational', 'thermal-union', 'joint-rejected']
CLASSES = {'CNC', 'PUMP', 'COMPRESSOR', 'CONVEYOR'}


def verify_source(source, frozen):
    manifest = json.loads((source / 'dataset_manifest.json').read_text())
    if (manifest.get('master_seed') != frozen['master_seed']
            or manifest.get('suite_sha256') != frozen['suite_sha256']
            or manifest.get('simulator_version') != '0.6.0'
            or manifest.get('data_license') != 'CC-BY-4.0'
            or manifest.get('synthetic') is not True
            or manifest.get('generated') is not True
            or manifest.get('customer_data') is not False):
        raise ValueError('Source identity, license or generation provenance differs from freeze')
    entries = manifest['runs']
    if (len(entries) != frozen['runs'] or len({e['run_id'] for e in entries}) != len(entries)
            or any(e['partition'] != 'test' for e in entries)):
        raise ValueError('Incomplete or duplicated normal-only test population')
    return manifest


def verify_run(source, entry, partition='test'):
    run_id = entry['run_id']
    if Path(run_id).name != run_id or run_id in ('.', '..'):
        raise ValueError('Invalid run identity')
    if partition not in ('train', 'validation', 'test') or entry.get('partition', partition) != partition:
        raise ValueError('Unexpected source partition')
    folder = source / partition / run_id
    source_hashes = {}
    for name, field in [('telemetry.parquet', 'telemetry_sha256'), ('ground_truth.json', 'ground_truth_sha256'),
                        ('run_metadata.json', 'metadata_sha256')]:
        actual = _sha256(folder / name)
        if actual != entry[field]:
            raise ValueError('Source file hash mismatch: ' + run_id + '/' + name)
        source_hashes[name] = actual
    truth = json.loads((folder / 'ground_truth.json').read_text())
    scenario = yaml.safe_load((folder / 'scenario.yaml').read_text())
    # Lab hashes canonical scenario JSON, not serialized YAML bytes.
    canonical = hashlib.sha256(json.dumps(scenario, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if canonical != entry['scenario_sha256']:
        raise ValueError('Scenario semantic hash mismatch: ' + run_id)
    source_hashes['scenario.yaml'] = _sha256(folder / 'scenario.yaml')
    if truth['run_id'] != run_id or truth.get('events') != [] or scenario.get('anomalies'):
        raise ValueError('Normal exposure requires explicitly empty anomaly truth')
    if scenario['duration_s'] != 3600 or {a['asset_class'].upper() for a in scenario['assets']} != CLASSES:
        raise ValueError('Unexpected scenario duration or classes')
    return folder, truth, scenario, source_hashes


def truth_regimes(windows, truth):
    # Evaluation metadata only, computed after all model scores.
    intervals = [(x['asset_id'], _epoch_us(x['start']), _epoch_us(x['end']), x['regime'])
                 for x in truth['operating_regimes']]
    result = []
    for row in windows.iter_rows(named=True):
        regimes = {regime for asset, start, end, regime in intervals
                   if asset == row['asset_id'] and start < row['window_end'] and end > row['window_start']}
        result.append(next(iter(regimes)) if len(regimes) == 1 else ('mixed' if regimes else 'uncovered'))
    return result


def run(args):
    repo = Path(__file__).resolve().parents[1]
    out = args.output
    frozen = json.loads((out / 'frozen.json').read_text())
    for root, key in [(args.baseline, 'baseline_files'), (args.sample, 'sample_files'),
                      (args.relational, 'relational_control_files')]:
        if hashes(root) != frozen[key]:
            raise ValueError('Frozen model changed: ' + key)
    joint_root = args.relationships / frozen['joint_relational_model']['directory']
    if hashes(joint_root) != frozen['joint_relational_model']['files']:
        raise ValueError('Joint model changed')
    immutable = ['src', 'scripts/research_joint_decisions.py', 'scripts/relationship_operating_points.py']
    if subprocess.check_output(['git', '-C', str(repo), 'diff', frozen['git_commit'], '--', *immutable], text=True):
        raise ValueError('Scoring implementation changed since freeze')
    if _sha256(repo / 'configs/normal-exposure-v05.yaml') != frozen['suite_sha256']:
        raise ValueError('Generation suite changed')
    if _sha256(repo / 'docs/NORMAL_EXPOSURE_PROTOCOL.md') != frozen['protocol_sha256']:
        raise ValueError('Evaluation protocol changed')
    manifest = verify_source(args.source, frozen)
    implementation = {p.relative_to(repo).as_posix(): _sha256(p) for p in
        [*sorted((repo / 'src').rglob('*.py')), *sorted((repo / 'scripts').glob('*.py'))]}
    execution = {'frozen_sha256': _sha256(out / 'frozen.json'),
                 'manifest_sha256': _sha256(args.source / 'dataset_manifest.json'),
                 'implementation': implementation, 'device': args.device}
    execution_path = out / 'execution.json'
    if execution_path.exists():
        if json.loads(execution_path.read_text()) != execution:
            raise ValueError('Resume requires identical source, freeze and evaluation implementation')
    else:
        write_json(execution_path, execution)
    execution_hash = _sha256(execution_path)
    cfg = _model_config(args.baseline, tail_policy='complete')
    schema = json.loads((args.baseline / 'feature_schema.json').read_text())
    selected = frozen['thermal_selected']
    stat = ['abs_time_weighted_mean_rate', 'q95_abs_rate'].index(selected['score'])
    thermal_models = {}
    for group, roles in schema['signals_by_asset_class'].items():
        for target in (r for r in roles if 'temperature' in r):
            key = f"initial-{selected['initial_context']}-{group}-{target}"
            thermal_models[group, key] = joblib.load(args.sample / (key + '.joblib'))['dynamics']
    availability_models = {group: joblib.load(joint_root / group / 'detector.joblib')['relationship'] for group in CLASSES}
    profiles = yaml.safe_load((repo / 'configs/normal-exposure-v05.yaml').read_text())['generation']['regime_profiles']
    records = []
    for n, entry in enumerate(manifest['runs'], 1):
        folder, truth, scenario, source_hashes = verify_run(args.source, entry)
        dest = out / 'runs' / entry['run_id']
        dest.mkdir(parents=True, exist_ok=True)
        marker = dest / 'record.json'
        predictions = dest / 'predictions.parquet'
        if marker.exists():
            record = json.loads(marker.read_text())
            if (record['execution_sha256'] != execution_hash or record['source_hashes'] != source_hashes
                    or record['predictions_sha256'] != _sha256(predictions)):
                raise ValueError('Completed run provenance differs: ' + entry['run_id'])
        else:
            raw = _annotate(normalize_observations(pl.read_parquet(folder / 'telemetry.parquet')), [])
            if set(raw['run_id'].unique()) != {entry['run_id']} or set(raw['asset_class'].unique()) != CLASSES:
                raise ValueError('Unexpected telemetry population')
            with tempfile.TemporaryDirectory(prefix='normal-exposure-') as temporary:
                path = Path(temporary) / 'telemetry.parquet'
                raw.write_parquet(path)
                windows = _windows(path, cfg, schema)
            if set(windows['asset_class'].unique()) != CLASSES or windows['is_anomaly'].any():
                raise ValueError('Missing class windows or unexpected anomaly labels')
            _, _, _, _, base, bth = _scores(args.baseline, windows)
            if bth != frozen['baseline_threshold']:
                raise ValueError('Baseline threshold changed')
            original = np.maximum(base, residual_scores(args.relational / 'relationships', windows, args.device)['ae'])
            relational, anchors = relational_magnitude_scores(joint_root, windows, args.device)
            if anchors != frozen['joint_relational_anchors']:
                raise ValueError('Relational normal-reference anchors changed')
            thermal = np.zeros(len(windows))
            available_t = np.zeros(len(windows), dtype=bool)
            available_r = np.zeros(len(windows), dtype=bool)
            for group in sorted(CLASSES):
                mask = windows['asset_class'].to_numpy() == group
                gw = windows.filter(pl.Series(mask))
                gr = raw.filter(pl.col('asset_class') == group)
                _, valid_r, _ = availability_models[group].transform(gw)
                available_r[mask] = valid_r
                local = np.zeros(len(gw)); valid_any = np.zeros(len(gw), dtype=bool)
                for (owner, key), model in thermal_models.items():
                    if owner != group:
                        continue
                    values, valid = aggregate_residuals(model.transform(gr), gw)
                    score = bounded_tail_score(values[:, stat], frozen['thermal_anchors'][key][stat])
                    score[~valid] = 0
                    local = np.maximum(local, score); valid_any |= valid
                thermal[mask] = local; available_t[mask] = valid_any
            decisions = {'v04': base >= bth, 'v05-relational': original >= frozen['relational_control_threshold'],
                         'thermal-union': (base >= bth) | (thermal >= selected['threshold'])}
            decisions['joint-rejected'] = decisions['thermal-union'] | (relational >= frozen['joint_point']['threshold'])
            scores = [base, original, relational, thermal]
            if any(not np.isfinite(s).all() or np.any((s < 0) | (s > 1)) for s in scores):
                raise ValueError('Invalid score')
            pred = windows.select(['run_id', 'asset_id', 'asset_class', 'window_start', 'window_end', 'operating_regime'])
            pred = pred.with_columns(*[pl.Series(name, values) for name, values in
                {**decisions, 'baseline_score': base, 'original_relational_score': original,
                 'relational_magnitude': relational, 'thermal_score': thermal,
                 'thermal_available': available_t, 'relational_available': available_r,
                 'truth_regime_diagnostic': truth_regimes(windows, truth)}.items()])
            profile = next(p['profile_id'] for p in profiles if p['shift_pattern'] == scenario['shift_pattern'])
            rows = []
            for group in sorted(CLASSES):
                mask = windows['asset_class'].to_numpy() == group
                gw = windows.filter(pl.Series(mask))
                rows.append({'run_id': entry['run_id'], 'asset_class': group,
                    'sampling_interval_ms': scenario['sampling_interval_ms'],
                    'expose_operating_regime': scenario['expose_operating_regime'], 'profile': profile,
                    'models': {name: alarm_counts(gw, decision[mask]) for name, decision in decisions.items()},
                    'thermal_available_windows': int(available_t[mask].sum()),
                    'relational_available_windows': int(available_r[mask].sum())})
            pred.write_parquet(predictions)
            record = {'run_id': entry['run_id'], 'execution_sha256': execution_hash,
                      'source_hashes': source_hashes, 'predictions_sha256': _sha256(predictions), 'rows': rows}
            write_json(marker, record)
        records.extend(record['rows'])
        write_json(out / 'progress.json', {'phase': 'evaluating', 'completed_runs': n, 'total_runs': len(manifest['runs']),
                   'asset_hours': sum(r['models']['v04']['asset_hours'] for r in records),
                   'false_windows': {name: sum(r['models'][name]['false_windows'] for r in records) for name in MODELS}})
        print(json.dumps({'completed_runs': n, 'total_runs': len(manifest['runs'])}), flush=True)
    kwargs = {'repetitions': frozen['bootstrap_repetitions'], 'seed': frozen['bootstrap_seed']}
    by_class = {g: summarize([r for r in records if r['asset_class'] == g], MODELS, **kwargs) for g in sorted(CLASSES)}
    strata = {key: {str(v): summarize([r for r in records if r[key] == v], MODELS, **kwargs)
                   for v in sorted({r[key] for r in records})}
              for key in ['sampling_interval_ms', 'expose_operating_regime', 'profile']}
    result = {'runs': len(manifest['runs']), 'overall': summarize(records, MODELS, **kwargs), 'by_class': by_class,
              'strata': strata, 'availability': {g: {key: sum(r[key] for r in records if r['asset_class'] == g)
                  for key in ['thermal_available_windows', 'relational_available_windows']} for g in sorted(CLASSES)},
              'exposure_target_met': all(m['v04']['asset_hours'] >= frozen['minimum_asset_hours_per_class'] for m in by_class.values()),
              'no_selection_or_retraining': True, 'execution_sha256': execution_hash,
              'source_attribution': 'OT Irregularity Lab 0.6.0, Mysthrala Kurogane Defense Labs; CC BY 4.0',
              'source_url': 'https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab',
              'limitations': ['Normal-only synthetic exposure cannot establish recall or industrial generalization.',
                             'One-hour runs aggregated by class do not establish continuous week-long stability.',
                             'Run-cluster bootstrap conditions on this simulator population; zero-count intervals do not prove zero risk.']}
    write_json(out / 'results.json', result)
    write_json(out / 'progress.json', {'phase': 'completed', 'completed_runs': len(manifest['runs']),
                                     'total_runs': len(manifest['runs']), 'overall': result['overall']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'baseline', 'sample', 'relational', 'relationships'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    run(p.parse_args())
