# Entrenamiento CUDA 100k: ejecución y lectura

Ejecución local del 2026-10-03 en Windows 11 con una NVIDIA GeForce RTX 5070 Ti de 16 GB. El primer intento detectó el driver, pero el entorno tenía la rueda CPU (`torch 2.14.1+cpu`). Se instaló PyTorch CUDA 13.0 en el `.venv` del proyecto. La instalación se fija ahora en [requirements-cuda-cu130.txt](../requirements-cuda-cu130.txt); PyTorch recomienda seleccionar Windows/Pip/CUDA en su [selector oficial de instalación](https://pytorch.org/get-started/locally/). El `uv.lock` general sigue multiplataforma y no fuerza ruedas CUDA.

## Corrida

- Modelo anterior comparable: `$OT_IRREGULARITY_LOCAL\model-otlab-v0.3-1m`.
- Nuevo artefacto: `$OT_IRREGULARITY_LOCAL\model-otlab-v0.3-1m-cuda-100k` (fuera de Git).
- Configuración: `configs/benchmark-otlab-1m-cuda-100k.yaml`; seed 42, 1m, RobustScaler, batch 128, autoencoder CUDA, Isolation Forest CPU.
- Train/validation: mismo hash `8400310416d2f737b172b20e366094fe5a07d018041e05db34be4ea361f93e79` que el candidato 1m, con 3,727 ventanas normales de train y 917 de validación.
- Trabajo completado: exactamente 100,000 actualizaciones de gradiente en 3,334 épocas; 213.0 s de tiempo del autoencoder. Mejor checkpoint en época 732 con pérdida normal de validación 2.6871159. Se entrena todo el presupuesto pedido y se conserva el checkpoint de validación más bajo.
- La prueba CUDA ejecutó una multiplicación matricial en `cuda:0` y verificó `torch 2.14.1+cu130`, CUDA 13.0 y `NVIDIA GeForce RTX 5070 Ti`.
- En consultas puntuales durante la corrida, `nvidia-smi` mostró entre 13–15% de GPU y 1.7–1.8 GiB usados de 16,303 MiB. El modelo es pequeño: 3,727 × 218 features en train; el tensor float32 es de unos pocos MiB. La VRAM libre no constituye trabajo paralelo automáticamente. Un lote mayor cambia el número de ejemplos por actualización y, por tanto, la dinámica/semántica del entrenamiento; no se aumentó a ciegas.
- La inferencia se repitió y generó bytes idénticos: SHA-256 `33058150ABB6B75F3EAE32E6F2977533918CA331B94F525949CA098B3C3D2D9B`.

## Comparación al umbral 0.95

| Conjunto | Modelo | Ventanas (positivas) | PR-AUC | ROC-AUC | Precisión | Recall | FP/activo-día | Eventos detectados | Latencia media |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lab v0.3 test | candidato 1m, entrenamiento previo | 2463 (299) | 0.138 | 0.516 | 0.191 | 0.057 | 42.10 | 16/144 | — |
| Lab v0.3 test | CUDA 100k | 2463 (299) | 0.153 | 0.544 | 0.239 | 0.054 | 29.82 | 13/144 | 54.3 s |
| Transiciones normales | candidato 1m, entrenamiento previo | 604 (0) | — | — | — | — | 23.84 | — | — |
| Transiciones normales | CUDA 100k | 604 (0) | — | — | — | — | 38.15 | — | — |
| Stream nuevo, régimen visible | candidato 1m, entrenamiento previo | 395 (45) | 0.208 | 0.651 | 0.111 | 0.022 | 29.16 | 1/15 | 65.0 s |
| Stream nuevo, régimen visible | CUDA 100k | 395 (45) | 0.255 | 0.686 | 0.429 | 0.133 | 29.16 | 3/15 | 74.7 s |
| Stream nuevo, régimen oculto | candidato 1m, entrenamiento previo | 395 (45) | 0.201 | 0.639 | 0.111 | 0.022 | 29.16 | 1/15 | 65.0 s |
| Stream nuevo, régimen oculto | CUDA 100k | 395 (45) | 0.239 | 0.677 | 0.375 | 0.133 | 36.46 | 3/15 | 74.7 s |

## Interpretación

La nueva corrida mejora el ranking PR-AUC en Lab y detecta más eventos en el stream nuevo; sin embargo, empeora el stress de operación normal (38.15 frente a 23.84 falsas alarmas por activo-día) y cae de 16 a 13 eventos detectados en el test Lab. Más actualizaciones no equivalen a una mejora general. La validación normal dejó de mejorar después de la época 732; el resto del presupuesto cumplió la solicitud de 100,000 actualizaciones, pero no mejoró el checkpoint. El artefacto no está validado para alarmas industriales.

## Monitor de corridas

Se añadió [watch_training.py](../scripts/watch_training.py), un monitor de consola que muestra barra de avance, actualizaciones, épocas, mejor pérdida, tiempo/ETA, VRAM y telemetría `nvidia-smi`. El entrenamiento ahora crea `training_progress.jsonl` desde carga de datos hasta el cierre y añade snapshots durante el autoencoder. Para futuras corridas, instálalo en una terminal con:

```powershell
uv pip install --python .venv/Scripts/python.exe --reinstall -r requirements-cuda-cu130.txt
uv run --no-sync ot-irregularity train --dataset $OT_IRREGULARITY_LOCAL\otlab-v0.3 --config configs\benchmark-otlab-1m-cuda-100k.yaml --output $OT_IRREGULARITY_LOCAL\model-otlab-v0.3-1m-cuda-100k
```

En otra terminal:

```powershell
uv run --no-sync python scripts/watch_training.py --run $OT_IRREGULARITY_LOCAL\model-otlab-v0.3-1m-cuda-100k --refresh 1
```

En esta ejecución el monitor se añadió después de empezar el entrenamiento; pudo leer el metadata final y mostrar el estado completado, mientras que el progreso en curso se comunicó con las líneas de estado del comando. Corridas posteriores dejarán el registro JSONL completo.

## Artefactos y controles

`training_metadata.json`, `metrics.json`, `model_card.md`, config copiada, scaler, Isolation Forest y Autoencoder están bajo `$OT_IRREGULARITY_LOCAL\model-otlab-v0.3-1m-cuda-100k`. SHA-256 de `autoencoder.pt`: `B9CA4ECE57C8CADD8994D28D3421D91782C3C4BAFC90B8A877E85F6CF68D3B32`. Hash de config: `0DBD30D019058C5191A5EF44B1C02C1E92C6B90663570E6B1FF3B59776A134DF`. El metadata conserva el commit base `e9bbaf91ce3998f3aee26de1eab7ac9872a438c6` y marca correctamente el árbol de trabajo como dirty.

Verificación local: 23 tests pasan bajo PyTorch CUDA; inference repetible byte por byte; sin commit, push, publicación ni despliegue.
