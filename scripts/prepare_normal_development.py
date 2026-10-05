"""Create verified per-run feature caches from the newly declared normal development source."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_normal_exposure import verify_run
from scripts.prepare_otlab import _annotate, _sha256, normalize_observations
from scripts.research_detection import write_json
from ot_irregularity.pipeline import _model_config, _windows


def prepare(args):
    protocol = json.loads((args.output / 'data_protocol.json').read_text())
    if protocol['seed'] != 920611:
        raise ValueError('Unexpected development protocol seed')
    repo = Path(__file__).resolve().parents[1]
    for name, expected in protocol['files'].items():
        if _sha256(repo / name.replace('\\', '/')) != expected:
            raise ValueError('Declared protocol changed')
    source = json.loads((args.source / 'dataset_manifest.json').read_text())
    if (source.get('master_seed') != protocol['seed'] or source.get('run_count') != 180
            or source.get('partition_counts') != {'train': 120, 'validation': 60, 'test': 0}
            or source.get('suite_sha256') != _sha256(repo / 'configs/normal-coverage-development.yaml')
            or source.get('data_license') != 'CC-BY-4.0' or source.get('simulator_version') != '0.6.0'
            or source.get('synthetic') is not True or source.get('generated') is not True
            or source.get('customer_data') is not False):
        raise ValueError('Source differs from the new normal-development protocol')
    entries = source['runs']
    if (len(entries) != 180 or len({e['run_id'] for e in entries}) != 180
            or any(e['partition'] not in ('train', 'validation') for e in entries)):
        raise ValueError('Invalid development run population')
    cache = args.output / 'normal-cache'
    if cache.exists():
        raise FileExistsError('Cache exists; inspect its provenance before any repeat')
    cache.mkdir()
    cfg = _model_config(args.baseline, tail_policy='complete')
    schema = json.loads((args.baseline / 'feature_schema.json').read_text())
    frames = {'train': [], 'validation': []}; records = []
    for n, entry in enumerate(entries, 1):
        folder, truth, scenario, source_hashes = verify_run(args.source, entry, entry['partition'])
        raw = _annotate(normalize_observations(pl.read_parquet(folder / 'telemetry.parquet')), [])
        if set(raw['run_id'].unique()) != {entry['run_id']}:
            raise ValueError('Telemetry run identity differs')
        raw = raw.with_columns((pl.lit('normal-dev-920611::') + pl.col('run_id')).alias('run_id'))
        with tempfile.TemporaryDirectory(prefix='normal-development-') as temporary:
            path = Path(temporary) / 'telemetry.parquet'
            raw.write_parquet(path)
            windows = _windows(path, cfg, schema)
        frames[entry['partition']].append(windows)
        records.append({'run_id': 'normal-dev-920611::' + entry['run_id'], 'partition': entry['partition'],
                        'windows': len(windows), 'source_hashes': source_hashes,
                        'sampling_interval_ms': scenario['sampling_interval_ms'],
                        'expose_operating_regime': scenario['expose_operating_regime']})
        write_json(args.output / 'progress.json', {'phase': 'prepare_normal_development', 'completed_runs': n, 'total_runs': 180})
    for partition, parts in frames.items():
        pl.concat(parts, how='diagonal_relaxed').write_parquet(cache / (partition + '.parquet'))
    metadata = {'source_manifest_sha256': _sha256(args.source / 'dataset_manifest.json'),
                'data_protocol_sha256': _sha256(args.output / 'data_protocol.json'),
                'feature_schema_sha256': _sha256(args.baseline / 'feature_schema.json'),
                'files': {p.name: _sha256(p) for p in cache.glob('*.parquet')}, 'runs': records,
                'source_seed': 920611, 'source_license': 'CC-BY-4.0',
                'source_attribution': 'OT Irregularity Lab 0.6.0; Mysthrala Kurogane Defense Labs',
                'source_url': 'https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab',
                'feature_config': cfg, 'test_used': False}
    write_json(cache / 'manifest.json', metadata)
    print(json.dumps({'runs': len(records), 'windows': {k: sum(len(w) for w in v) for k, v in frames.items()}}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'baseline'):
        p.add_argument('--' + name, type=Path, required=True)
    prepare(p.parse_args())
