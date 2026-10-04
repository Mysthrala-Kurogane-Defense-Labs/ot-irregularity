# Evaluation

Scores are empirical normal-reference ranks, not probabilities of failure. The default alert threshold `0.95` is a configurable starting point and has not been optimized. `evaluate` reports precision, recall, F1, PR-AUC, ROC-AUC, false-positive windows, false positives per asset-hour/day, event detection rate and detection latency when the corresponding labels exist. Accuracy is not a primary metric. Without `is_anomaly`, it reports window count and explains why supervised metrics are absent. Without `event_id`, event metrics are omitted.

False-positive exposure uses scored window count times stride as monitored exposure; overlapping samples are not treated as independent asset hours beyond the configured stride. Event latency is the end time of the first alerting window minus the first labeled event timestamp, floored at zero.

Challenge data is evaluated only with `uv run ot-irregularity evaluate --model ... --dataset .../challenge --challenge`. It is never loaded by `train`. The synthetic fixture is deliberately simple and cannot support claims about industrial generalization.
