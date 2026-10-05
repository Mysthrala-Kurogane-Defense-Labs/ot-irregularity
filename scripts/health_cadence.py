"""Verify Lab provenance before attaching configured cadence to health inputs."""
import hashlib
import json
import math
from pathlib import Path
import re

import polars as pl
import yaml

from scripts.prepare_otlab import _sha256, normalize_observations


def declared_cadence(source, entry, *, allow_test=False):
    source = Path(source).resolve()
    partition, run = entry['partition'], entry['run_id']
    allowed = ('train', 'validation', 'test') if allow_test else ('train', 'validation')
    if partition not in allowed or not re.fullmatch(partition + r'-[0-9]+', run):
        raise ValueError('Only explicit development run identities are allowed')
    folder = (source / partition / run).resolve()
    if not folder.is_relative_to(source):
        raise ValueError('Run path escapes source')
    hashes = {}
    for name, key in [('run_metadata.json', 'metadata_sha256'), ('telemetry.parquet', 'telemetry_sha256')]:
        hashes[name] = _sha256(folder / name)
        if hashes[name] != entry.get(key):
            raise ValueError('Cadence source hash mismatch: ' + name)
    meta = json.loads((folder / 'run_metadata.json').read_text())
    scenario = yaml.safe_load((folder / 'scenario.yaml').read_text())
    canonical = hashlib.sha256(json.dumps(scenario, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if canonical != entry['scenario_sha256'] or meta['scenario_sha256'] != canonical or meta['run_id'] != run:
        raise ValueError('Cadence scenario or run identity mismatch')
    period = meta.get('sampling_interval_ms')
    if isinstance(period, bool) or not isinstance(period, (int, float)) or not math.isfinite(period) or period <= 0:
        raise ValueError('Positive finite configured cadence required')
    if period != scenario.get('sampling_interval_ms'):
        raise ValueError('Metadata/scenario cadence disagreement')
    hashes['scenario.yaml'] = _sha256(folder / 'scenario.yaml')
    return folder, float(period), hashes


def attach_declared_cadence(frame, period):
    if not math.isfinite(period) or period <= 0:
        raise ValueError('Positive finite configured cadence required')
    if 'declared_sampling_interval_ms' in frame.columns or 'observed_sampling_interval_ms' in frame.columns:
        raise ValueError('Cadence semantics already assigned; do not overwrite')
    expressions = [pl.lit(float(period)).alias('declared_sampling_interval_ms')]
    if 'sampling_interval_ms' in frame.columns:
        expressions.append(pl.col('sampling_interval_ms').alias('observed_sampling_interval_ms'))
    return frame.with_columns(expressions)


def historical_cadence(frame, raw_root, partition):
    if partition not in ('train', 'validation'):
        raise ValueError('Development partition required')
    prepared, evidence = [], {}
    manifests = {}
    compare = ['asset_id', 'timestamp', 'measurement_role', 'value', 'unit', 'quality', 'sampling_interval_ms']
    for (run,), part in frame.partition_by('run_id', as_dict=True).items():
        match = re.fullmatch(r'dev-([0-9]+)::((train|validation)-[0-9]+)', run)
        if not match or match[3] != partition:
            raise ValueError('Historical run namespace or partition mismatch')
        batch, original = 'raw-' + match[1], match[2]
        source = Path(raw_root) / batch
        if batch not in manifests:
            manifest = json.loads((source / 'dataset_manifest.json').read_text())
            manifests[batch] = manifest
        matches = [r for r in manifests[batch]['runs'] if r['partition'] == partition and r['run_id'] == original]
        if len(matches) != 1:
            raise ValueError('Historical cadence requires one manifest entry')
        folder, period, hashes = declared_cadence(source, matches[0])
        original_frame = normalize_observations(pl.read_parquet(folder / 'telemetry.parquet'))
        keys = ['asset_id', 'timestamp', 'measurement_role']
        if not part.select(compare).sort(keys).equals(original_frame.select(compare).sort(keys)):
            raise ValueError('Historical observations do not match verified raw cadence source')
        prepared.append(attach_declared_cadence(part, period))
        evidence[run] = {'declared_sampling_interval_ms': period, 'files': hashes,
                         'manifest_sha256': _sha256(source / 'dataset_manifest.json')}
    return pl.concat(prepared, how='diagonal_relaxed'), evidence
