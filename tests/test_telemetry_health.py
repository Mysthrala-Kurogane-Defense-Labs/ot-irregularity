from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from ot_irregularity.telemetry_health import TelemetryHealthReference, extract_health


def raw_samples(start=0, count=60, interval=1000):
    return pl.DataFrame({'run_id': ['r']*count, 'asset_id': ['a']*count,
        'measurement_role': ['flow']*count, 'unit': ['l/min']*count,
        'timestamp': [datetime(2025,1,1)+timedelta(milliseconds=start+i*interval) for i in range(count)],
        'value': np.arange(count, dtype=float), 'quality': ['good']*count,
        'sampling_interval_ms': [interval]*count, 'declared_sampling_interval_ms': [interval]*count})


def window(start=0):
    # Use the timestamp's own epoch representation, independent of local timezone.
    epoch = pl.Series([datetime(2025,1,1)]).dt.timestamp('us')[0]
    return pl.DataFrame({'run_id': ['r'], 'asset_id': ['a'], 'asset_class': ['PUMP'],
                        'window_start': [epoch+start*1000], 'window_end': [epoch+(start+60000)*1000],
                        'is_anomaly': [False]})


def test_health_counts_quality_and_missing_signal_are_separate():
    raw = raw_samples().with_columns(pl.Series('quality', ['bad']*12+['good']*48))
    result = extract_health(raw, window(), {'PUMP': ['flow']}).row(0, named=True)
    assert result['bad_ratio'] == .2 and result['coverage_ratio'] == 1
    assert result['repetition_ratio'] == 0
    missing = extract_health(raw, window(60000), {'PUMP': ['flow']}).row(0, named=True)
    assert missing['coverage_ratio'] == 0 and missing['bad_ratio'] is None
    assert missing['finite_value_count'] == 0 and missing['expected_sample_count'] == 60


def test_unknown_quality_is_unavailable_and_future_cadence_does_not_rewrite_past():
    raw = raw_samples().drop('quality')
    first = extract_health(raw, window(), {'PUMP': ['flow']})
    assert first['bad_ratio'][0] is None and first['repetition_ratio'][0] is None
    future = raw_samples(60000, 120, 500).drop('quality')
    combined = pl.concat([raw, future])
    assert extract_health(combined, window(), {'PUMP': ['flow']}).equals(first)
    changed = extract_health(combined, window(60000), {'PUMP': ['flow']})
    assert changed['coverage_ratio'][0] is None
    assert changed['cadence_reason'][0] == 'declared_cadence_changed'
    before = extract_health(future, window(), {'PUMP': ['flow']})
    assert before['coverage_ratio'][0] is None and before['unit'][0] is None


def test_health_rejects_duplicates_and_does_not_bridge_bad_pairs():
    raw = raw_samples().with_columns(pl.lit(5.).alias('value'))
    with pytest.raises(ValueError, match='Duplicate observation'):
        extract_health(pl.concat([raw, raw.head(1)]), window(), {'PUMP': ['flow']})
    raw = raw.with_columns(pl.Series('quality', ['good']*10+['bad']*40+['good']*10))
    result = extract_health(raw, window(), {'PUMP': ['flow']})
    assert result['usable_pairs'][0] == 18 and result['repetition_ratio'][0] is None


def references():
    base = extract_health(raw_samples(), window(), {'PUMP': ['flow']}).row(0, named=True)
    def frame(prefix):
        return pl.DataFrame([{**base, 'run_id': f'{prefix}-{i}', 'mean_value': float(i%3),
            'repetition_ratio': 1. if i%3 == 0 else .1} for i in range(30)])
    return frame('fit'), frame('cal')


def test_health_reference_scores_explanations_and_json_roundtrip(tmp_path):
    fit, cal = references()
    model = TelemetryHealthReference(minimum_windows=2).fit(fit, cal)
    observed = cal.head(1).with_columns(pl.lit('new').alias('run_id'), pl.lit(1.).alias('mean_value'),
                                      pl.lit(.8).alias('repetition_ratio'), pl.lit(.2).alias('bad_ratio'))
    score, details = model.score(observed)
    assert score['quality_score'][0] == .2
    assert score['repetition_score'][0] == pytest.approx(7/9)
    assert score['health_score'][0] == pytest.approx(7/9)
    assert details['normal_repetition_upper'][0] == .1
    assert score['observations'][0].to_list() == ['quality_deviation', 'signal_repetition']
    without, _ = model.score(observed, repetition=False)
    assert without['health_score'][0] == .2
    # A normal zero plateau belongs to a different learned value context.
    plateau, _ = model.score(observed.with_columns(pl.lit(0.).alias('mean_value'), pl.lit(0.).alias('bad_ratio')))
    assert plateau['repetition_score'][0] == 0
    model.save(tmp_path/'health.json')
    loaded = TelemetryHealthReference.load(tmp_path/'health.json')
    assert loaded.score(observed)[0].equals(score)
    with pytest.raises(ValueError, match='unit mismatch'):
        loaded.score(observed.with_columns(pl.lit('kg/s').alias('unit')))


def test_health_requires_normal_disjoint_references_and_preserves_unavailability():
    fit, cal = references()
    with pytest.raises(ValueError, match='normal'):
        TelemetryHealthReference().fit(fit.with_columns(pl.lit(True).alias('is_anomaly')), cal)
    with pytest.raises(ValueError, match='leakage'):
        TelemetryHealthReference().fit(fit, fit)
    model = TelemetryHealthReference(minimum_windows=2).fit(fit, cal)
    features = extract_health(raw_samples().drop('quality', 'declared_sampling_interval_ms'), window(), {'PUMP': ['flow']})
    scores, details = model.score(features)
    assert scores['health_score'][0] is None
    assert scores['observations'][0].to_list() == []
    assert details['quality_score_reason'][0] == 'quality_unobserved'
    assert details['sampling_score_reason'][0] == 'no_prior_declared_cadence'


def test_health_signal_loss_and_partial_sampling_have_distinct_observations():
    fit, cal = references()
    model = TelemetryHealthReference(minimum_windows=2).fit(fit, cal)
    empty = extract_health(raw_samples(), window(60000), {'PUMP': ['flow']})
    scored, _ = model.score(empty)
    assert scored['sampling_score'][0] == 1
    assert scored['observations'][0].to_list() == ['signal_loss']
    partial = extract_health(raw_samples(count=30), window(), {'PUMP': ['flow']})
    scored, _ = model.score(partial)
    assert scored['sampling_score'][0] == .5
    assert scored['observations'][0].to_list() == ['sampling_degradation']


def test_health_study_alignment_and_gate_reject_substitutions():
    from scripts.research_telemetry_health import align_scores, candidate_gate
    windows = pl.concat([window(), window(60000)])
    scores = windows.drop('is_anomaly').with_columns(pl.Series('health_score', [.1, .2]))
    assert align_scores(windows, scores.reverse())['health_score'].to_list() == [.1, .2]
    with pytest.raises(ValueError, match='identities differ'):
        align_scores(windows, scores.head(1))
    with pytest.raises(ValueError, match='Duplicate'):
        align_scores(windows, pl.concat([scores, scores.head(1)]))
    mixed = {'detected_events': 80, 'false_positive_windows': 4, 'precision': .8}
    normal = {'overall': {'false_windows': 3}, 'by_class': {'PUMP': {'false_windows_per_asset_day': 5}}}
    args = [mixed, normal, [], [], mixed, mixed, normal]
    assert candidate_gate(*args)
    for index, value in [(2, [('run', 'asset', 'event')]), (3, [('run', 'asset', 'event')]),
                         (0, {**mixed, 'false_positive_windows': 5}),
                         (0, {**mixed, 'precision': .49}),
                         (0, {**mixed, 'detected_events': 79}),
                         (1, {**normal, 'overall': {'false_windows': 4}}),
                         (1, {**normal, 'by_class': {'PUMP': {'false_windows_per_asset_day': 11}}})]:
        changed = args.copy(); changed[index] = value
        assert not candidate_gate(*changed)


def test_declared_cadence_survives_observed_jitter_and_dropout():
    raw = raw_samples().with_columns(pl.Series('sampling_interval_ms', [1000, 999, 1001]*20))
    score = extract_health(raw, window(), {'PUMP': ['flow']})
    assert score['coverage_ratio'][0] == 1
    assert score['cadence_reason'][0] is None
    missing = raw.filter(pl.int_range(pl.len()) % 3 != 1)
    feature = extract_health(missing, window(), {'PUMP': ['flow']})
    assert feature['coverage_ratio'][0] == pytest.approx(2/3)
    # Observed gaps must not become an implicit configured expectation.
    unknown = extract_health(raw.drop('declared_sampling_interval_ms'), window(), {'PUMP': ['flow']})
    assert unknown['coverage_ratio'][0] is None
    assert unknown['repetition_ratio'][0] is None


def test_lab_cadence_provenance_and_assignment(tmp_path):
    import hashlib
    import json
    import yaml
    from scripts.health_cadence import attach_declared_cadence, declared_cadence
    from scripts.prepare_otlab import _sha256
    folder = tmp_path/'train'/'train-00001'; folder.mkdir(parents=True)
    scenario = {'sampling_interval_ms': 1000}
    scenario_hash = hashlib.sha256(json.dumps(scenario, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (folder/'scenario.yaml').write_text(yaml.safe_dump(scenario))
    meta = {'run_id': 'train-00001', 'sampling_interval_ms': 1000, 'scenario_sha256': scenario_hash}
    (folder/'run_metadata.json').write_text(json.dumps(meta))
    raw_samples().drop('declared_sampling_interval_ms').write_parquet(folder/'telemetry.parquet')
    entry = {'partition':'train', 'run_id':'train-00001', 'scenario_sha256':scenario_hash,
             'metadata_sha256':_sha256(folder/'run_metadata.json'), 'telemetry_sha256':_sha256(folder/'telemetry.parquet')}
    _, period, hashes = declared_cadence(tmp_path, entry)
    assigned = attach_declared_cadence(pl.read_parquet(folder/'telemetry.parquet'), period)
    assert period == 1000 and len(hashes) == 3
    assert assigned['observed_sampling_interval_ms'].equals(assigned['sampling_interval_ms'])
    with pytest.raises(ValueError, match='already assigned'):
        attach_declared_cadence(assigned, period)
    with pytest.raises(ValueError, match='development'):
        declared_cadence(tmp_path, {**entry, 'partition':'test'})
    with pytest.raises(ValueError, match='hash mismatch'):
        declared_cadence(tmp_path, {**entry, 'metadata_sha256':'changed'})
    meta['sampling_interval_ms'] = 500
    (folder/'run_metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='disagreement'):
        declared_cadence(tmp_path, {**entry, 'metadata_sha256':_sha256(folder/'run_metadata.json')})
