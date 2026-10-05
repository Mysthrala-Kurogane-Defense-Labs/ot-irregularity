# OT Irregularity

Proyecto open source independiente para aprender comportamiento operacional normal y puntuar desviaciones temporales y multivariables. No diagnostica averías ni depende de Kurogane Hub. Python >=3.12, uv; licencia Apache-2.0.

El [candidato contextual v0.4](docs/MODEL_V0.4_CONTEXTUAL.md) mejora la detección en una evaluación pseudo-sintética independiente: 55/119 eventos frente a 8/119 de v0.3, con menos ventanas falsas. Sigue perdiendo familias físicas completas y no alcanza aún el objetivo exploratorio de alarmas. Es una opción de investigación reproducible, no una validación industrial.

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

Las fuentes de datasets consideradas y sus licencias están en [DATASET_SOURCES.md](docs/DATASET_SOURCES.md). Los procedimientos y resultados de Petrobras 3W 1.1.1 y 2.0.0 están en [BENCHMARK_3W.md](docs/BENCHMARK_3W.md) y [BENCHMARK_3W_OOD.md](docs/BENCHMARK_3W_OOD.md); los datos originales no se incluyen. El benchmark pseudo-sintético independiente de OT Irregularity Lab y su adaptador reproducible están en [BENCHMARK_OT_IRREGULARITY_LAB.md](docs/BENCHMARK_OT_IRREGULARITY_LAB.md); los datos y modelos generados permanecen fuera de Git. La comparación de configuraciones, regímenes expuestos/ocultos y stream nuevo está en [MODEL_ARENA.md](docs/MODEL_ARENA.md). La corrida CUDA de 100.000 actualizaciones y el monitor de estado están documentados en [CUDA_TRAINING.md](docs/CUDA_TRAINING.md). La búsqueda reproducible de grupos de features y su tablero local están en [FEATURE_SEARCH.md](docs/FEATURE_SEARCH.md). Lee [ARCHITECTURE.md](ARCHITECTURE.md), [FEATURES.md](FEATURES.md), [TRAINING.md](TRAINING.md), [EVALUATION.md](EVALUATION.md) y [MODEL_CARD_TEMPLATE.md](MODEL_CARD_TEMPLATE.md) para revisar supuestos y límites. El dataset de ejemplo es sintético y solo prueba el flujo técnico. El candidato entrenado v0.3 con los aprendizajes de la búsqueda está resumido en [MODEL_V0.3_CANDIDATE.md](docs/MODEL_V0.3_CANDIDATE.md); modelo y datos permanecen fuera de Git.
