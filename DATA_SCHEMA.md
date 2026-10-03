# Data schema v1

Required long-format columns: `run_id`, `asset_id`, `timestamp`, `tag_id`, `signal_class`, `value`. Recommended columns: `asset_class`, `unit`, `quality` (`good`, `uncertain`, `bad` or null), `sampling_interval_ms`, `operating_regime`, `engineering_min`, `engineering_max`. Additional fields are allowed. `is_anomaly` (Boolean/0/1) and `event_id` are optional evaluation labels. Product-specific identifiers are not required.

CSV and Parquet are supported; Parquet is recommended for large data. Timestamp must parse, values must be finite numeric, required fields cannot be null, and `(run_id, asset_id, timestamp, tag_id)` must be unique. `signal_class` is the generic semantic feature name used by the pipeline. Normalize vendor tags into this shared field before training.

Semantic normalization (`tag_id` to `signal_class`, engineering units, and source-quality interpretation) is separate from ML scaling. `normalization.canonical_units` converts supported compatible units before window generation and fails on missing, unknown, or dimensionally incompatible units. `RobustScaler` or `StandardScaler` is fit afterward and saved with the model. Signal statistics include `good`, `uncertain`, and ungraded observations, and exclude `bad`; this lets data without quality grades remain usable while retaining quality ratios as separate features.

Dataset directories contain `train/`, `validation/`, optional `test/`, and optional `challenge/`. Train and validation split groups must be disjoint. Training reads only those first two partitions.
