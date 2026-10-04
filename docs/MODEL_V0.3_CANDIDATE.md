# OT Irregularity model v0.3 candidate

## Result

Se entrenó un candidato Autoencoder + Isolation Forest con la semántica de tags corregida y las 15 particiones normales de entrenamiento del experimento pareado. Se conservaron todas las señales continuas; la evidencia de ablations no justificó excluir `rotational_speed`, y quitar vibración redujo detección de eventos. Los códigos categóricos no entran en las features numéricas.

- Train: 421 runs normales, 3.484.797 observaciones, 6.749 ventanas y 289 features.
- Validation/calibración: 270 runs mixtos, 4.412 ventanas; 3.904 ventanas normales para checkpoint y calibración.
- Autoencoder: 30.000 actualizaciones en `cuda:0` (RTX 5070 Ti), 72,3 s; mejor checkpoint época 297 de 567, loss 9,8244. RobustScaler. Isolation Forest: 300 estimadores. Ensamble 0,5/0,5.
- Test reservado: 270 runs, 4.267 ventanas, 573 positivas y 276 eventos; evaluado tras fijar el modelo, sin ajuste de threshold.

| Detector | PR-AUC | Precisión @ 0,95 | Recall @ 0,95 | Eventos detectados | FP / activo-día | Latencia media |
|---|---:|---:|---:|---:|---:|---:|
| Autoencoder | 0,1577 | 0,1749 | 0,0803 | 43 / 276 (15,6%) | 73,23 | 66,2 s |
| Isolation Forest | 0,1527 | 0,1464 | 0,0716 | 38 / 276 (13,8%) | 80,66 | 63,9 s |
| Ensamble | 0,1599 | 0,1702 | 0,0419 | 23 / 276 (8,3%) | 39,48 | 60,9 s |

El ensamble mejora PR-AUC y reduce falsas alarmas frente a cada detector por separado en este test, a cambio de menor recall y menos eventos detectados. Incluso así, 39,48 FP por activo-día es alto y la detección de eventos es baja. Este resultado no acredita utilidad en una planta real ni debe activar alarmas operativas.

## Artefactos y reproducción

Modelo y datos (fuera de Git): `$OT_IRREGULARITY_LOCAL\model-training\`.

```powershell
.venv\Scripts\python.exe scripts\assemble_feature_search_dataset.py `
  --search-run $OT_IRREGULARITY_LOCAL\feature-search\feature-search-paired-20261004 `
  --output $OT_IRREGULARITY_LOCAL\model-training\dataset-v0.3-all-learnings

.venv\Scripts\ot-irregularity.exe train `
  --dataset $OT_IRREGULARITY_LOCAL\model-training\dataset-v0.3-all-learnings `
  --config configs\model-v0.3-candidate.yaml `
  --output $OT_IRREGULARITY_LOCAL\model-training\model-v0.3-full-learnings

.venv\Scripts\ot-irregularity.exe evaluate `
  --model $OT_IRREGULARITY_LOCAL\model-training\model-v0.3-full-learnings `
  --dataset $OT_IRREGULARITY_LOCAL\model-training\dataset-v0.3-all-learnings\test
```

Inferencia repetida sobre test generó el mismo SHA-256 de JSONL: `0F5FACB1F12BA288A49496B8263122ACCE547F99A50F410EBE2614F7A18D36D2`. Los hashes de artefactos y manifests están en el directorio externo. Procedencia: OT Irregularity Lab 0.3.1, suite `training-v0.2.yaml`, licencia CC BY 4.0, 15 semillas y manifests registrados en el dataset combinado. El artefacto registra commit de modelo `e9bbaf91ce3998f3aee26de1eab7ac9872a438c6`; el checkout estaba dirty.

## Límites

Pseudodatos de un único simulador y sus procesos simplificados; 1m window/stride; threshold operativo 0,95 heredado de la configuración; sin calibración específica por activo ni validación en datasets industriales abiertos. El modelo describe irregularidad, no causa. `test` se evaluó una vez tras el fit y no se utilizó para elegir arquitectura, features, scaler, pasos ni threshold. Versión candidata local, no publicada.

Registro de ejecución: [workflow y evidencia](workflows/model-training/runs/2026-10-04.md).
