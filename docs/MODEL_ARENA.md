# Model arena: 3W, Lab y stream nuevo

Fecha: 2026-10-03. Ejecución local; sin commit, publicación ni despliegue.

El análisis por evento/familia, reproducibilidad y límites del candidato CUDA 100k está en [diagnóstico de detección](DIAGNOSIS_EVENT_DETECTION.md).

La auditoría posterior confirmó problemas semánticos en columnas/mediciones de Lab y un falso `signal_loss`/`sampling_degradation` constante, ya corregido en inferencia: [auditoría de semántica de features](FEATURE_SEMANTICS_AUDIT_OTLAB.md).

## Pregunta y comparación

Se compararon el modelo 5m `model-otlab-v0.3` y el candidato 1m `model-otlab-v0.3-1m`. Ambos se ajustaron sobre las mismas ventanas normales de entrenamiento y validación del dataset Lab v0.3.0; comparten el hash de datos de entrenamiento `8400310416d2f737b172b20e366094fe5a07d018041e05db34be4ea361f93e79`, seed 42, RobustScaler y configuración 50/50. Cada uno conserva scores de Autoencoder e Isolation Forest y score combinado. Las métricas de puntos/eventos usan umbral 0.95 persistido por el modelo.

El modelo anterior `model-unseen-wells-v0.1` se evaluó también en su arena nativa 3W. No se mezclan sus métricas con las de Lab: los esquemas semánticos y espacios de features son incompatibles (7 señales 3W y 122 features frente a 12 clases de señal Lab y 212/218 features). No existe una comparación directa justa entre esos artefactos sin volver a entrenar bajo un esquema común.

## Stream generado nuevo

Se creó un par de datasets desde la suite `challenge-v0.1` de OT Irregularity Lab, con CC BY 4.0 declarado explícitamente, seed `20261008`, simulador 0.3.1 y exactamente un evento por run. Ambos tienen 120 runs (84 train, 18 validation, 18 test); la evaluación usa 18 runs, 140,276 filas y 15 eventos. Familias: missing telemetry (4), asset communication loss (2), sensor drift (1), multivariate novelty (5), bearing degradation (2) y cavitation (1). El modelo se entrenó solo con la partición normal de Lab v0.3.0; estos patrones son nuevos respecto a su ajuste, aunque los conozca el simulador.

Suite visible: `$OT_LAB_LOCAL\challenge-state-visible-cc-by-v0.1.yaml`, hash `aa6d16bbb482a3d8450cdad37fecc91b1373b9e82a0b9ea97741f6e2155be3ed`; manifiesto de salida `2855b75bf71abee3938733fd9385654fbe1e3e69269d59b1f4eba8aa364285c0`.

Suite oculta: `$OT_LAB_LOCAL\challenge-state-hidden-cc-by-v0.1.yaml`, hash `4a0aaa83f846f50f24f37418aed46fc248871294fb8d1b03311d0a5d1cb3138a`; manifiesto `d2890e0f0598d615657fb7dfbaa12f9cc8f41869979f770cf4343ddb1b8639a9`.

Se comprobó que ambas particiones de test tienen las mismas 140,276 observaciones y los mismos 17 campos físicos exactamente; solo `operating_regime` difiere (140,276 valores presentes frente a todos nulos). El estado esperado se expuso al modelo solo en la variante visible. Los eventos etiquetados se mantienen como metadatos de evaluación, fuera de las features. Los streams, predicciones y modelos permanecen fuera de Git.

## Resultados del stream nuevo

PR-AUC mide ranking sobre las ventanas etiquetadas; `eventos` indica eventos con al menos una alerta al umbral 0.95. El denominador de FP/día son ventanas negativas normalizadas por tiempo observado y activos. Conjunto pequeño e intencionadamente cargado de eventos: no estima prevalencia de campo.

| Regímenes | Ventana | Detector | Ventanas / positivas | PR-AUC | ROC-AUC | Precisión | Recall | FP/activo-día | Eventos detectados | Latencia media |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| visibles | 5m | Autoencoder | 255 / 76 | 0.3620 | 0.5880 | 0.324 | 0.145 | — | — | — |
| visibles | 5m | Isolation Forest | 255 / 76 | 0.3451 | 0.5244 | 0.258 | 0.211 | — | — | — |
| visibles | 5m | conjunto | 255 / 76 | 0.4129 | 0.5974 | 0.400 | 0.158 | 101.65 | 2/15 | 59.5 s |
| visibles | 1m | Autoencoder | 395 / 45 | 0.1946 | 0.6330 | 0.154 | 0.044 | — | — | — |
| visibles | 1m | Isolation Forest | 395 / 45 | 0.2057 | 0.6431 | 0.273 | 0.133 | — | — | — |
| visibles | 1m | conjunto | 395 / 45 | 0.2076 | 0.6505 | 0.111 | 0.022 | 29.16 | 1/15 | 65.0 s |
| ocultos | 5m | Autoencoder | 255 / 76 | 0.3626 | 0.5885 | — | — | — | — | — |
| ocultos | 5m | Isolation Forest | 255 / 76 | 0.3441 | 0.5250 | — | — | — | — | — |
| ocultos | 5m | conjunto | 255 / 76 | 0.4134 | 0.5999 | 0.387 | 0.158 | 107.29 | 2/15 | 59.5 s |
| ocultos | 1m | Autoencoder | 395 / 45 | 0.1947 | 0.6337 | — | — | — | — | — |
| ocultos | 1m | Isolation Forest | 395 / 45 | 0.1999 | 0.6254 | — | — | — | — | — |
| ocultos | 1m | conjunto | 395 / 45 | 0.2014 | 0.6393 | 0.111 | 0.022 | 29.16 | 1/15 | 65.0 s |

Las métricas de umbral por detector individual en la variante oculta no se incluyen aquí; los resultados completos están en el registro de ejecución. Los agregados visibles/ocultos son cercanos: exponer estos regímenes aporta poco a esta prueba, no demuestra que el contexto sea inútil. `regime_mismatch` aparece en 35/255 ventanas visibles para 5m y 17/395 para 1m porque los nombres de régimen de esta suite no están en el vocabulario del modelo; aparece en 0 ventanas ocultas. Es una observación de contexto desconocido, no una causa de fallo. Una corrección de inferencia impide etiquetar `regime_mismatch` cuando el régimen falta.

## Arena 3W separada

Modelo `model-unseen-wells-v0.1`, evaluado contra los conjuntos que corresponden a su esquema. En `test`, 72,570 ventanas (58,959 positivas): Autoencoder PR-AUC 0.8523 / ROC-AUC 0.5522; Isolation Forest 0.7978 / 0.5118; conjunto 0.8641 / 0.5656, recall 0.1754, 1/13 eventos y 0 FP/activo-día al umbral configurado. En `challenge`, 1,786 ventanas (614 positivas): conjunto PR-AUC 0.2642 / ROC-AUC 0.3253, 0/6 eventos; Isolation Forest detecta 6/6 con 55.63 FP/activo-día. Ver detalles y límites en [benchmark 3W](BENCHMARK_3W_OOD.md). Alta prevalencia en 3W test condiciona PR-AUC; no compararla directamente con Lab.

## Lectura para el candidato 1m

No hay ganador universal. Frente a 5m, el candidato 1m fue mucho más silencioso en los dos conjuntos normales previos (23.84 frente a 142.11 FP/activo-día en el stress de transiciones), pero conserva una tasa inaceptable. En el benchmark Lab v0.3.0 la PR-AUC conjunta apenas cambia (0.138 frente a 0.328; las prevalencias de ventana son distintas por tamaño de ventana), recall al umbral cae (0.057 frente a 0.060) y detecta 16/144 eventos frente a 11/134 evaluables a 5m. En el stream nuevo el 5m logra mayor PR-AUC y 2/15 eventos frente a 1/15 con 1m; el 1m marca menos falsas alarmas, con la misma prevalencia de test por run, pero sigue detectando muy pocos eventos.

Conclusión: 1m es una variante experimental con menor volumen de falsas alarmas y menor detección, no una mejora general demostrada para declarar superior a v0.1 ni para adoptar como configuración por defecto. Puede publicarse como una iteración experimental si se presenta junto con las limitaciones, sin afirmar rendimiento industrial. La decisión de release queda fuera de esta arena.

## Reproducción

Requiere el checkout local del Lab en `$OT_LAB_CHECKOUT` y artefactos/datos bajo `$LOCAL_EXPERIMENTS\`; los archivos no se versionan.

```powershell
uv run python scripts/prepare_otlab.py --source $OT_LAB_LOCAL\arena-state-visible-v0.1 --output $OT_IRREGULARITY_LOCAL\arena-state-visible-v0.1
uv run python scripts/prepare_otlab.py --source $OT_LAB_LOCAL\arena-state-hidden-v0.1 --output $OT_IRREGULARITY_LOCAL\arena-state-hidden-v0.1
uv run ot-irregularity evaluate --model $OT_IRREGULARITY_LOCAL\model-otlab-v0.3 --dataset $OT_IRREGULARITY_LOCAL\arena-state-visible-v0.1\test
uv run ot-irregularity evaluate --model $OT_IRREGULARITY_LOCAL\model-otlab-v0.3-1m --dataset $OT_IRREGULARITY_LOCAL\arena-state-visible-v0.1\test
uv run ot-irregularity evaluate --model $OT_IRREGULARITY_LOCAL\model-otlab-v0.3 --dataset $OT_IRREGULARITY_LOCAL\arena-state-hidden-v0.1\test
uv run ot-irregularity evaluate --model $OT_IRREGULARITY_LOCAL\model-otlab-v0.3-1m --dataset $OT_IRREGULARITY_LOCAL\arena-state-hidden-v0.1\test
uv run ot-irregularity infer --model $OT_IRREGULARITY_LOCAL\model-otlab-v0.3-1m --dataset $OT_IRREGULARITY_LOCAL\arena-state-hidden-v0.1\test --output $OT_IRREGULARITY_LOCAL\arena-model-otlab-v0.3-1m-hidden.jsonl
uv run pytest
```

> Las cuatro evaluaciones y predicciones usadas en esta ejecución ya se generaron; las predicciones están en `$OT_IRREGULARITY_LOCAL\arena-*.jsonl`. El adaptador puede volver a ejecutarse en una ruta externa vacía. Para regenerar los streams hacen falta las dos suites YAML externas archivadas y el CLI del simulador; conservar sus hashes y seed indicados arriba.

## Procedencia y límites

El stream nuevo es generado por el simulador independiente OT Irregularity Lab, no es tráfico de red/protocolo ni telemetría física. La suite se declaró CC BY 4.0 localmente; debe revisarse que esa atribución sea válida para cualquier redistribución según los términos/origen de la suite. Los resultados provienen de 15 eventos y 18 runs de test, por lo que tienen alta incertidumbre y no sirven como estimación de alarmas en industria. Los datasets/modelos/predicciones siguen bajo `$LOCAL_EXPERIMENTS\`; el repositorio no contiene estos datos. El estado cambiante del checkout Lab y las limitaciones de procedencia se conservan en el [registro de ejecución](workflows/model-arena/runs/2026-10-03.md).
