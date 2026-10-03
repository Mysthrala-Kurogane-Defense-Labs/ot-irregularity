# Feature definitions v1

One row represents an asset/run window. Windows are aligned to epoch at configured stride, use fixed duration, and are emitted only when the full window is observable. Signal value statistics and slopes use all records except `quality=bad`; `uncertain` and null quality are retained because uncertainty does not mean the measurement is unusable. Quality and sampling counts include all records.

| Feature suffix | Formula | Inputs | Reason |
|---|---|---|---|
| `mean`, `median`, `min`, `max` | corresponding statistic | non-bad values in window | level and excursion |
| `range` | max - min | non-bad values | excursion size |
| `std` | population standard deviation | non-bad values | variability |
| `mad` | median(abs(x - median(x))) | non-bad values | robust variability |
| `last` | last non-bad value by timestamp | non-bad values | latest observed state |
| `delta` | last - first non-bad value | non-bad values | net within-window change |
| `slope` | least-squares slope against seconds relative to the window start | non-bad values | temporal trend |
| `sample_count` | count of records | all quality states | observed sampling |
| `expected_sample_count` | window duration / declared interval | sampling metadata | expected sampling |
| `coverage_ratio` | min(sample_count / expected, 1) | counts above | sampling degradation |
| `good_ratio`, `uncertain_ratio`, `bad_ratio` | quality count / total count | quality | data quality |
| `missing` | 1 if no record for signal in window, else 0 | signal presence | explicit signal loss |
| `context_asset_class_*` | train-vocabulary one-hot | asset_class | asset class context |
| `context_operating_regime_*` | train-vocabulary one-hot | operating_regime | regime context |
| `context_regime_unknown` | regime absent from train vocabulary | operating_regime | explicit regime mismatch |

Signal prefixes come from normalized `signal_class` values. Cross-signal correlations, lag features and physical ratios are deferred; v0.1 does not generate them automatically.
