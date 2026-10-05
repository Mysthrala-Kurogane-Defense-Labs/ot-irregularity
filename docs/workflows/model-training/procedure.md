# Procedure: train a model from feature-search learnings

- Use only the standalone OT Irregularity repo and simulator-generated or otherwise licensed normalized data.
- Preserve the test partition untouched until the model, scaler, feature set, training budget, and operating threshold are fixed.

## Steps

1. Revalidate dataset hashes, source license, row schema and measurement semantics. Continuous physical measurements use their normalized role; categorical states/codes must not be treated as numeric magnitude.
2. Join only the intended fixed partitions; namespace run and event identifiers by source batch. Confirm train contains normal-only rows and run IDs are disjoint across train, validation and test.
3. Preserve signals unless paired evidence supports removing one. Use paired holdouts and report detection/false-alarm trade-offs; an inconclusive rank is not a feature-selection verdict.
4. Fit RobustScaler on training windows only. Train the Autoencoder only on normal windows; fit Isolation Forest on the same scaled normal windows. Calibrate score distributions on validation normal windows only. Keep individual detector scores and fixed configurable ensemble weights.
5. Persist model version, config, dataset hashes, seed, device, scaler, feature schema, thresholds, metrics and model card. Stream training progress to JSONL.
6. Once the candidate is fixed, evaluate the reserved test once. Do not tune on test scores or labels. Verify inference repeatability and summarize model-specific as well as ensemble metrics.

## Cautions

Report PR-AUC, precision, recall, event detection, false alarms per asset-day and latency together. Do not present accuracy as primary or infer faults/cause from scores. A synthetic test only characterizes that simulator suite.

For contextual models, preserve whole-run fit/early-stop/calibration/development separation and freeze candidates before generating test conclusions. Check cache provenance and hashes before reuse. Compare event counts to original ground-truth intervals with `evaluate --events` or the holdout audit; a single event identifier per window loses overlaps. Preserve the old metric when reconciling definitions. Report the failed acceptance target explicitly even when the candidate improves its baseline.
