# OT Irregularity

Proyecto open source independiente para aprender comportamiento operacional normal y puntuar desviaciones temporales y multivariables. No diagnostica averías ni depende de Kurogane Hub. Python >=3.12, uv; licencia Apache-2.0.

## Inicio rápido

```bash
uv sync --extra test
uv run ot-irregularity train --dataset examples/synthetic --config configs/baseline.yaml --output artifacts/model-v0.1
uv run ot-irregularity infer --model artifacts/model-v0.1 --dataset examples/synthetic/test --output predictions.jsonl
uv run ot-irregularity evaluate --model artifacts/model-v0.1 --dataset examples/synthetic/test
uv run ot-irregularity evaluate --model artifacts/model-v0.1 --dataset examples/synthetic/challenge --challenge
uv run ot-irregularity inspect --model artifacts/model-v0.1
uv run pytest
```

`train/` y `validation/` son obligatorios y se separan por `run_id` (configurable). Train debe ser normal. Las etiquetas opcionales `is_anomaly` y `event_id` permiten medir resultados; los eventos anómalos de validation quedan fuera de la calibración. `test/` y `challenge/` nunca se cargan por train. Challenge se evalúa explícitamente con `--challenge`.

El formato de entrada es long-format CSV o Parquet, descrito en [DATA_SCHEMA.md](DATA_SCHEMA.md). La salida de inferencia es JSONL o Parquet; guarda ambos scores de detector, ensamble, contribuciones de reconstrucción y observaciones de señal/regímenes.

Lee [ARCHITECTURE.md](ARCHITECTURE.md), [FEATURES.md](FEATURES.md), [TRAINING.md](TRAINING.md), [EVALUATION.md](EVALUATION.md) y [MODEL_CARD_TEMPLATE.md](MODEL_CARD_TEMPLATE.md) para revisar supuestos y límites. El dataset de ejemplo es sintético y solo prueba el flujo técnico.
