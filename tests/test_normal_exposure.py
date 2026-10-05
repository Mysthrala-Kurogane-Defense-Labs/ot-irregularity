import numpy as np
import polars as pl
import pytest

from scripts.normal_exposure_metrics import alarm_counts, summarize
from scripts.prepare_otlab import normalize_observations


def test_semantic_normalization_removes_labels_and_preserves_values():
    raw = pl.DataFrame({'tag_id': ['rpm', 'state'], 'unit': ['rpm', 'CODE'],
                        'quality': ['GOOD', 'BAD'], 'value': [100., 2.],
                        'is_anomaly': [True, False], 'event_id': ['secret', None]})
    out = normalize_observations(raw)
    assert out['measurement_role'].to_list() == ['rpm', 'state']
    assert out['value_kind'].to_list() == ['continuous', 'categorical']
    assert out['value'].equals(raw['value'])
    assert out['quality'].to_list() == ['good', 'bad']
    assert not {'is_anomaly', 'event_id'} & set(out.columns)


def test_episode_boundaries_exposure_and_no_overlap():
    w = pl.DataFrame({'run_id': ['r']*6+['s'], 'asset_id': ['a']*5+['b','a'],
                      'window_start': [0,60,120,180,300,0,0],
                      'window_end': [60,120,180,240,360,60,60]}).with_columns(
                          pl.col('window_start')*1_000_000, pl.col('window_end')*1_000_000)
    m = alarm_counts(w, np.array([True, True, False, True, True, True, True]))
    assert m['false_windows'] == 6 and m['alarm_episodes'] == 5
    assert m['alarm_seconds'] == 360 and m['asset_hours'] == pytest.approx(7/60)
    with pytest.raises(ValueError, match='Overlapping'):
        alarm_counts(pl.concat([w, w.head(1)]), np.ones(8, dtype=bool))
    with pytest.raises(ValueError, match='boolean'):
        alarm_counts(w, np.ones(7))


def test_clustered_metrics_and_paired_identical_controls():
    records = []
    for run, count in [('a', 0), ('b', 2)]:
        for group in ['PUMP', 'CNC']:
            m = {'asset_hours': 1., 'false_windows': count, 'alarm_episodes': int(count>0),
                 'alarm_seconds': count*60., 'windows': 60}
            records.append({'run_id': run, 'asset_class': group, 'models': {'base': m, 'same': m}})
    result = summarize(records, ['base', 'same'])
    assert result['base']['asset_hours'] == 4
    assert result['base']['runs'] == 2
    assert result['base']['fraction_runs_with_alarm'] == .5
    assert result['base']['false_windows_per_asset_day'] == 24
    assert result['same']['paired_false_windows_per_asset_day_delta_ci95'] == [0, 0]
    assert result == summarize(records, ['base', 'same'])
    zero = summarize([records[0]], ['base', 'same'])
    assert zero['base']['zero_count_caution']


def test_run_source_verification_uses_lab_canonical_scenario_hash(tmp_path):
    import hashlib
    import json
    import yaml
    from scripts.evaluate_normal_exposure import verify_run
    folder = tmp_path/'test'/'r'
    folder.mkdir(parents=True)
    scenario = {'duration_s': 3600, 'anomalies': [], 'assets': [
        {'asset_class': name} for name in ('cnc', 'pump', 'compressor', 'conveyor')]}
    (folder/'scenario.yaml').write_text(yaml.safe_dump(scenario))
    (folder/'ground_truth.json').write_text(json.dumps({'run_id': 'r', 'events': []}))
    (folder/'run_metadata.json').write_text('{}')
    (folder/'telemetry.parquet').write_bytes(b'fixture-not-read-by-provenance-check')
    entry = {'run_id': 'r', 'scenario_sha256': hashlib.sha256(
        json.dumps(scenario, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    for name, key in [('telemetry.parquet', 'telemetry_sha256'), ('ground_truth.json', 'ground_truth_sha256'),
                      ('run_metadata.json', 'metadata_sha256')]:
        entry[key] = hashlib.sha256((folder/name).read_bytes()).hexdigest()
    assert verify_run(tmp_path, entry)[2] == scenario
    scenario['duration_s'] = 1
    (folder/'scenario.yaml').write_text(yaml.safe_dump(scenario))
    with pytest.raises(ValueError, match='semantic hash'):
        verify_run(tmp_path, entry)
