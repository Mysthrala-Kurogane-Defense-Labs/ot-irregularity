# Training

Training sequence: read only `train/` and `validation/`; validate schema and duplicate keys; reject overlap in configured split group; window both partitions; reject anomalous train windows; fit RobustScaler (default) or StandardScaler using train windows only; train autoencoder on train-normal with normal-validation early stopping/checkpoint; fit Isolation Forest on train-normal; calibrate each score from normal validation windows only; compute validation metrics when labels exist; persist artifacts.

`is_anomaly` and `event_id` labels are optional. If validation labels exist, windows labeled anomalous are excluded from score calibration but remain in supervised metrics. Without labels, validation is assumed normal and that assumption appears in metrics. `test/` and `challenge/` are not opened. `evaluate --challenge` is the explicit challenge access path.

The configuration includes seed, window/stride, scaler, autoencoder, Isolation Forest, ensemble weights and decision threshold. Python, NumPy and Torch seeds are set; sklearn receives the configured random state. Dataset SHA-256 covers sorted file names and bytes in train and validation only. Git commit is recorded when training runs inside a Git checkout, otherwise null.

Artifacts: autoencoder state/architecture, Isolation Forest, scaler, feature/context/signal schema, resolved YAML, metadata, normal calibration references and thresholds, metrics, and model card. ONNX is deferred pending parity evidence.
