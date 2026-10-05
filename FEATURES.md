# Feature definitions v1

One row represents an asset/run window. Windows are aligned to epoch at configured stride, use fixed duration, and are emitted only when the full window is observable. Signal value statistics and slopes use all records except `quality=bad`; `uncertain` and null quality are retained because uncertainty does not mean the measurement is unusable. Quality and sampling counts include all records.

Window aggregation now keys channels by optional `measurement_role`; when absent, it falls back to `signal_class`. This preserves distinct measurement points that share a broad class. The OT Lab adapter maps its canonical `tag_id` values to `measurement_role` for this pseudo-synthetic study; generic datasets should supply stable semantic roles rather than product-specific identifiers. See the [semantic feature audit](docs/FEATURE_SEMANTICS_AUDIT_OTLAB.md).

Training stores which signal classes occur for each `asset_class`. Inference uses that map before describing `signal_loss` or `sampling_degradation`; legacy artifacts without it suppress those observations because structural absence and real loss cannot be distinguished. The underlying feature columns for structurally absent channels remain in the current fixed-width model input, so this does not yet correct model training semantics.

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

Rows with `value_kind` other than `continuous` are currently excluded from this numeric feature pipeline. This prevents state codes from being treated as continuous measurements; category occupancy and transition features remain future work. Datasets without `value_kind` retain the legacy numeric behavior.

Signal prefixes come from normalized `measurement_role` values when supplied, otherwise `signal_class`. Cross-signal correlations, lag features and physical ratios are deferred; v0.1 does not generate them automatically.
