# Evaluation

Scores are empirical normal-reference ranks, not probabilities of failure. The default alert threshold `0.95` is a configurable starting point and has not been optimized. `evaluate` reports precision, recall, F1, PR-AUC, ROC-AUC, false-positive windows, false positives per asset-hour/day, event detection rate and detection latency when the corresponding labels exist. Accuracy is not a primary metric. Without `is_anomaly`, it reports window count and explains why supervised metrics are absent. Without `event_id`, event metrics are omitted.

False-positive exposure uses scored window count times stride as monitored exposure; overlapping samples are not treated as independent asset hours beyond the configured stride. Event latency is the end time of the first alerting window minus the first labeled event timestamp, floored at zero.

Challenge data is evaluated only with `uv run ot-irregularity evaluate --model ... --dataset .../challenge --challenge`. It is never loaded by `train`. The synthetic fixture is deliberately simple and cannot support claims about industrial generalization.
## Eventos solapados y cobertura completa

`evaluate --events path/to/events.json` añade `event_intervals` para cada detector. El JSON contiene `{"events": [{"run_id": "r", "asset_id": "a", "event_id": "e", "start_us": 0, "end_us": 60000000}]}`. Cuenta todos los intervalos originales, incluso varios eventos por ventana o eventos sin ventanas observables; estos últimos cuentan como no detectados. La latencia usa el primer cierre de ventana que solapa el evento y supera el umbral. El adaptador de Lab genera este sidecar por partición; sus campos nunca se incluyen en features.

Si hay etiquetas por fila, las métricas de ventanas las conservan; si solo se aporta el JSON de eventos, se derivan por solapamiento. Las tasas existentes `false_positives_per_asset_day` cuentan **ventanas falsas**, divididas por la exposición sumada en ventanas/stride; no son incidentes o episodios de alarma deduplicados. La evaluación heredada por un solo `event_id` no garantiza cobertura de eventos solapados. El [informe v0.4](docs/MODEL_V0.4_CONTEXTUAL.md) reconcilia ambas cuentas.
