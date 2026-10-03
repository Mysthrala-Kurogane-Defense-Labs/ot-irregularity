# Feature definitions v1

One row represents an asset/run window. Windows are aligned to epoch at configured stride, use fixed duration, and are emitted only when the full window is observable. Signal value statistics use only `quality=good` records; quality and sampling counts include all records.

| Feature suffix | Formula | Inputs | Reason |
|---|---|---|---|
| `mean`, `median`, `min`, `max` | corresponding statistic | good values in window | level and excursion |
| `range` | max - min | good values | excursion size |
| `std` | population standard deviation | good values | variability |
| `mad` | median(abs(x - median(x))) | good values | robust variability |
| `last` | last good value by timestamp | good values | latest observed state |
| `delta` | last - first good value | good values | net within-window change |
| `slope` | least-squares slope against epoch microseconds | all values | temporal trend |
| `sample_count` | count of records | all quality states | observed sampling |
| `expected_sample_count` | window duration / declared interval | sampling metadata | expected sampling |
| `coverage_ratio` | min(sample_count / expected, 1) | counts above | sampling degradation |
| `good_ratio`, `uncertain_ratio`, `bad_ratio` | quality count / total count | quality | data quality |
| `missing` | 1 if no record for signal in window, else 0 | signal presence | explicit signal loss |
| `context_asset_class_*` | train-vocabulary one-hot | asset_class | asset class context |
| `context_operating_regime_*` | train-vocabulary one-hot | operating_regime | regime context |
| `context_regime_unknown` | regime absent from train vocabulary | operating_regime | explicit regime mismatch |

Signal prefixes come from normalized `signal_class` values. Cross-signal correlations, lag features and physical ratios are deferred; v0.1 does not generate them automatically.
