# Candidato v0.4: detección por clase de activo

Fecha: 2026-10-05. Investigación reproducible con datos pseudo-sintéticos. El candidato mejora claramente v0.3 en esta evaluación, pero todavía no cumple el objetivo exploratorio de detectar al menos el 50 % de eventos con precisión >=50 % y <=10 ventanas falsas por activo/día. No está validado como alarma industrial.

## Resultado independiente

Se congelaron arquitectura, pesos, umbral y tres semillas de entrenamiento antes de evaluar tres lotes nuevos. Son 108 runs de test, 1.695 ventanas de un minuto, 239 ventanas positivas y **119 eventos originales**. La exposición evaluada equivale a 28,25 horas-activo; las tasas diarias son extrapolaciones de esta muestra corta.

| Métrica | v0.3 publicado | v0.4, semilla principal 20261005 |
|---|---:|---:|
| PR-AUC (average precision) | 0,1532 | **0,5658** |
| Precisión de ventanas | 16,7 % | **83,3 %** |
| Recall de ventanas | 4,2 % | **37,7 %** |
| Eventos detectados, intervalos originales | 8/119 (6,7 %) | **55/119 (46,2 %)** |
| Ventanas falsas | 50 | **18** |
| Ventanas falsas / activo / día | 42,48 | **15,29** |
| Latencia media de eventos detectados | 40,63 s | 50,18 s |

La latencia se calcula al cierre de la ventana y sobre conjuntos distintos de eventos detectados; no demuestra una mejora o regresión de velocidad por sí sola. Una alerta en una ventana que solapa dos eventos puede detectar ambos; las cuentas por familia no prueban diagnóstico causal.

Las semillas adicionales 20261006 y 20261007 alcanzaron PR-AUC 0,5753 y 0,5653; detección 54/119 y 56/119; precisión 84,6 % y 83,7 %; falsas ventanas 13,59 y 14,44/activo-día. Se conserva la semilla principal elegida antes del test, sin seleccionar la mejor después.

Bootstrap pareado por **run completo**: diferencia de PR-AUC +0,4126, IC95 % [0,3228; 0,4918] (500 remuestreos); diferencia de detección de eventos +39,5 puntos, IC95 % [28,3; 49,6] (2.000 remuestreos con todos los intervalos); diferencia de ventanas falsas/día -27,19, IC95 % [-43,92; -12,80]. Estos intervalos describen variación entre runs de este simulador, no entre plantas reales.

## Qué cambió

1. Autoencoder e Isolation Forest independientes por `asset_class`, usando únicamente los roles de medida que aplican a esa clase. Así se evita que la unión de canales de activos distintos domine el escalado y el error.
2. `RobustScaler` ajustado solo con runs normales de ajuste, conservando unidades/normalización semántica como fase previa. Se comparó también `StandardScaler`.
3. Error cuadrático por feature dividido por su percentil 95 en runs normales de early stopping, con suelo numérico `1e-8`. El score bruto del AE es la media de los **cuatro errores normalizados más altos**; no promedia una anomalía local entre todas las columnas.
4. Calibración CDF empírica independiente por clase. Umbral global: cuantil 0,99 del score combinado en calibración normal, con corte estricto mediante `nextafter`. Es una elección de desarrollo, no una garantía de falsas alarmas futuras ni una probabilidad de avería.
5. Pesos elegidos en desarrollo: AE=1, IF=0. Isolation Forest sigue entrenado, persistido y disponible en inferencia. La mezcla 50/50 y las mezclas 90/10 y 75/25 redujeron la separación y detección en esta muestra. No hay razón experimental para imponerles un peso positivo.

La arquitectura sigue siendo pequeña: AE denso con latent 8, hasta 3.000 actualizaciones por clase y early stopping. Más actualizaciones no eran el requisito que faltaba. Este candidato condiciona por **clase**, no aprende todavía una dinámica física específica por régimen; `operating_regime` sigue disponible para contexto y observaciones de vocabulario desconocido.

## Separación de datos

La selección utilizó solo `train/` y `validation/` de los 15 lotes históricos. El test histórico no se volvió a abrir para seleccionar configuraciones.

- Ajuste: `SHA256(run_id)[:8] % 5 != 0`, 5.155 ventanas normales.
- Early stopping y escala de residuos: resto de runs de train, 1.594 ventanas normales.
- Calibración: runs de validation con `SHA256(run_id)[:8] % 2 == 0`, solo ventanas normales, 1.927 ventanas.
- Selección: los demás runs completos de validation, 2.234 ventanas y 124 IDs de evento representados.

La exploración comparó cinco representaciones (global/por clase, robust/standard, completa/compacta), cinco scores por representación y 36 combinaciones de pesos/cuantiles en desarrollo. Son múltiples comparaciones: la evidencia de mejora principal es la evaluación posterior congelada, no el ranking de desarrollo.

El protocolo requiere `asset_class` y etiquetas explícitas de normalidad en validation. Rechaza clases desconocidas, particiones insuficientes y cuantiles cuya resolución empírica desactivaría todas las alertas. Los directorios de salida no se sobrescriben. Los modelos globales v0.1–v0.3 mantienen su formato y ruta de inferencia.

## Auditoría de eventos y límites por familia

La evaluación antigua guardaba un solo `event_id` por ventana: representaba 117 de los 119 eventos y contaba 52 detecciones para el candidato. La nueva evaluación usa todos los intervalos de ground truth por `(run_id, asset_id, event_id)`: cuenta también eventos solapados o sin ventanas observables. En este test los 119 tenían ventanas; el resultado correcto por ese contrato es 55/119. Se conservaron las métricas antiguas y la reconciliación; no se reentrenó ni ajustó el umbral después del test.

| Familia del simulador | v0.3 | v0.4 principal |
|---|---:|---:|
| asset_communication_loss | 1/8 | 7/8 |
| single_signal_loss | 0/13 | 12/13 |
| missing_telemetry | 0/8 | 5/8 |
| quality_degradation | 0/7 | 7/7 |
| multivariate_novelty | 2/6 | 6/6 |
| sensor_stuck | 0/5 | 5/5 |
| sensor_bias | 1/7 | 4/7 |
| sensor_drift | 1/17 | 5/17 |
| sudden_spike | 0/5 | 1/5 |
| regime_mismatch | 2/11 | 2/11 |
| maintenance_activity | 0/7 | 1/7 |
| mechanical_overload | 1/7 | 0/7 |
| bearing_degradation | 0/8 | 0/8 |
| cooling_degradation | 0/6 | 0/6 |
| cavitation | 0/4 | 0/4 |

El avance se concentra en integridad de señal y algunas desviaciones multivariables. Persisten fallos graves en cambios físicos sutiles. No se deben presentar los nombres del ground truth como diagnósticos producidos por el modelo.

## Entrenar y usar

```bash
uv run ot-irregularity train --dataset datasets/development --config configs/model-v0.4-contextual.yaml --output artifacts/model-v0.4
uv run ot-irregularity infer --model artifacts/model-v0.4 --dataset datasets/test --output artifacts/predictions.jsonl
uv run ot-irregularity evaluate --model artifacts/model-v0.4 --dataset datasets/test --events datasets/test/events.json
uv run ot-irregularity inspect --model artifacts/model-v0.4
```

`device: auto` utiliza CUDA cuando está disponible; puede fijarse a `cuda:0` o `cpu`. Para este entrenamiento se usó RTX 5070 Ti, PyTorch 2.14.1+cu130, Python 3.12.10. No se garantiza identidad binaria entre hardware/versiones.

Artefactos: `contextual_models.json`, `feature_schema.json`, config, metadata, thresholds, metrics y model card en la raíz; `groups/<hash-de-clase>/autoencoder.pt`, `scaler.joblib`, `isolation_forest.joblib` por clase. `feature_contributions` explica la fracción del error top-k normalizado, indicada en `contribution_basis`; no representa causalidad ni atribuciones de Isolation Forest.

Los cuatro Autoencoders reentrenados con la misma semilla durante cribado y congelación tuvieron tensores `state_dict` bit-idénticos en este entorno CUDA. Esto no garantiza igualdad entre plataformas.

Dos inferencias del artefacto principal sobre 561 ventanas fueron byte-idénticas. SHA-256 JSONL: `64ffe8160c2037464f1ac51ffa56e29c5888b3e3b9a27f1d79b8df457fb2457d`. Las correcciones posteriores de explicabilidad y conteo de eventos cambiaron **0,0** los scores comparados con las predicciones congeladas.

## Fuentes y reproducción experimental

Fuente: [OT Irregularity Lab](https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab), suite `training-v0.2.yaml`, licencia de datos **CC BY 4.0**. Atribución: Mysthrala Kurogane Defense Labs, OT Irregularity Lab generated pseudo-synthetic telemetry. Desarrollo generado con Lab 0.3.1; nuevos lotes con Lab 0.6.0, checkout limpio `718babb7772c3a21f0b87c403f628540cbce58db`. La suite tiene el mismo SHA-256 `a768415b78245f671ac41edb75b249e9c8ab0523c1088aa9848b9c62f613e091`; no se asume identidad del generador entre versiones.

Semillas nuevas: 910501, 910502, 910503, 240 runs por lote; solo sus 36 runs de test/lote se evaluaron. Los manifiestos conservan hashes, licencia y atribución. No se han probado plantas reales, nuevas clases de activos ni todas las familias de fallo posibles.

Herramientas: `scripts/research_detection.py` (cribado y `--operating-points-only`), `scripts/validate_contextual.py fit` (tres semillas, congelación de hashes), `scripts/validate_contextual.py evaluate-holdout` (acceso explícito a test nuevo), `scripts/audit_holdout_events.py` (reconciliación de intervalos). El estado JSON puede servirse mediante el tablero existente `scripts/feature_search.py serve`.

Véanse [registro de ejecución](workflows/model-training/runs/2026-10-05.md) y [evidencia numérica](results/v0.4-contextual.json). Datos, pesos y predicciones completos permanecen fuera de Git.

## Próxima investigación

Separar el rendimiento de integridad de telemetría del de desviaciones físicas; evaluar contexto de régimen y relaciones temporales con selección en desarrollo, una nueva reserva independiente y largas trazas normales. Un mejor umbral elegido mirando este test no sería una validación nueva. El objetivo de 50 %/50 %/10 ventanas falsas por día sigue pendiente, y tampoco equivale por sí solo a aceptación operacional.
