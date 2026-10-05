import json

import numpy as np
import polars as pl

from scripts.research_physical import center_regimes, load_events, summarize


def test_regime_centers_use_only_fit_and_fallback_is_explicit():
    fit = pl.DataFrame({'operating_regime': ['low'] * 20 + ['high'] * 20,
                        'signal_mean': [2.] * 20 + [10.] * 20})
    later = pl.DataFrame({'operating_regime': ['low', 'high', 'unseen'],
                          'signal_mean': [100., 11., 9.]})
    arrays, metadata = center_regimes(fit, [fit, later], ['signal_mean'])
    np.testing.assert_array_equal(arrays[0], np.zeros((40, 1)))
    np.testing.assert_array_equal(arrays[1].ravel(), [98., 1., 3.])
    assert metadata['fallback_windows'] == [0, 1]
    assert metadata['centers']['low'][0] == 2.


def test_original_events_are_namespaced_and_filtered_to_development(tmp_path):
    folder = tmp_path / 'raw-01' / 'validation' / 'r'
    folder.mkdir(parents=True)
    truth = {'run_id': 'r', 'events': [{'event_id': 'e', 'asset_id': 'a',
        'type': 'cavitation', 'start': '2025-01-01T00:00:00Z', 'end': '2025-01-01T00:01:00Z'}]}
    (folder / 'ground_truth.json').write_text(json.dumps(truth))
    events, hashes = load_events(tmp_path, pl.DataFrame({'run_id': ['dev-01::r']}))
    assert events[0]['event_id'] == 'dev-01::r/e'
    assert events[0]['family'] == 'cavitation'
    assert len(hashes) == 1


def test_physical_metrics_include_unobserved_intervals():
    frame = pl.DataFrame({'run_id': ['r'], 'asset_id': ['a'], 'is_anomaly': [True],
        'event_id': ['r/e1'], 'event_start_us': [0], 'window_start': [0], 'window_end': [60_000_000]})
    events = [{'run_id': 'r', 'asset_id': 'a', 'event_id': 'r/e1', 'family': 'cavitation',
               'start_us': 0, 'end_us': 1_000_000},
              {'run_id': 'r', 'asset_id': 'a', 'event_id': 'r/e2', 'family': 'cavitation',
               'start_us': 120_000_000, 'end_us': 180_000_000}]
    result = summarize(frame, np.array([.999]), .99, events)
    assert result['physical_events'] == {'total': 2, 'detected': 1}
    assert result['event_count'] == 2
    assert result['events_without_windows'] == 1
