# Training

Training sequence: read only `train/` and `validation/`; validate schema and duplicate keys; reject overlap in configured split group; window both partitions; reject anomalous train windows; fit RobustScaler (default) or StandardScaler using train windows only; train autoencoder on train-normal with normal-validation early stopping/checkpoint; fit Isolation Forest on train-normal; calibrate each score from normal validation windows only; compute validation metrics when labels exist; persist artifacts.

`is_anomaly` and `event_id` labels are optional. If validation labels exist, windows labeled anomalous are excluded from score calibration but remain in supervised metrics. Without labels, validation is assumed normal and that assumption appears in metrics. `test/` and `challenge/` are not opened. `evaluate --challenge` is the explicit challenge access path.

The configuration includes seed, window/stride, scaler, autoencoder, Isolation Forest, ensemble weights and decision threshold. Python, NumPy and Torch seeds are set; sklearn receives the configured random state. Dataset SHA-256 covers sorted file names and bytes in train and validation only. Git commit is recorded when training runs inside a Git checkout, otherwise null.

Artifacts: autoencoder state/architecture, Isolation Forest, scaler, feature/context/signal schema, resolved YAML, metadata, normal calibration references and thresholds, metrics, and model card. ONNX is deferred pending parity evidence.

## CUDA y entrenamientos extensos

El autoencoder acepta `device: auto`, `cpu` o `cuda:0` en YAML. `auto` selecciona CUDA cuando PyTorch la ve; un `cuda:N` explícito falla claramente si ese dispositivo no está disponible. La inferencia usa la misma selección y el Isolation Forest de scikit-learn sigue ejecutándose en CPU. La configuración `configs/benchmark-otlab-1m-cuda-100k.yaml` pide 100.000 actualizaciones del optimizador (unas 3.334 épocas con lotes de 128), hasta 100.000 épocas como máximo y guarda el checkpoint con menor pérdida de validación. `training_metadata.json` registra dispositivo, épocas completadas, mejor época y duración.

En Windows con GPU NVIDIA, instala la variante fijada en `requirements-cuda-cu130.txt` con `uv pip install --python .venv/Scripts/python.exe --reinstall -r requirements-cuda-cu130.txt`, después verifica `torch.cuda.is_available()` antes de entrenar. El selector oficial de [PyTorch Start Locally](https://pytorch.org/get-started/locally/) sirve para actualizar la rueda si cambian los requisitos. El lock multiplataforma no fuerza CUDA; usa `uv run --no-sync` para no restaurar la rueda CPU.

```powershell
uv run --no-sync ot-irregularity train `
  --dataset $OT_IRREGULARITY_LOCAL\otlab-v0.3 `
  --config configs\benchmark-otlab-1m-cuda-100k.yaml `
  --output $OT_IRREGULARITY_LOCAL\model-otlab-v0.3-1m-cuda-100k
```


Para visualizar una corrida en curso, abre otra terminal y ejecuta:

```powershell
uv run --no-sync python scripts/watch_training.py --run $OT_IRREGULARITY_LOCAL\model-otlab-v0.3-1m-cuda-100k --refresh 1
```

El monitor lee `training_progress.jsonl` (fases, actualizaciones, época, pérdida y VRAM asignada/reservada por PyTorch) y consulta `nvidia-smi` para uso/utilización de GPU. Mantén la misma ruta de salida que el comando de entrenamiento.
# Entrenamiento contextual opcional

`configs/model-v0.4-contextual.yaml` activa modelos AE/IF por `asset_class`. Requiere suficientes runs normales por clase para ajuste, early stopping y calibración separados. Los artefactos por clase se guardan en `groups/` y la inferencia rechaza clases desconocidas. Ver [protocolo y resultados](docs/MODEL_V0.4_CONTEXTUAL.md). El conjunto de ejemplo mínimo está destinado al baseline global; no tiene la cobertura necesaria para calibrar un cuantil 0,99 por clase.
