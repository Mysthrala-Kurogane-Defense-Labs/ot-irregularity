# Diagnóstico de detección de eventos esperados (v0.3, 1m, CUDA 100k)

Fecha: 2026-10-03. Análisis local reproducible; no hubo publicación, entrenamiento nuevo ni cambios en los artefactos/modelos.

## Respuesta corta

Sí, hay un valle serio en aciertos. En el test Lab el candidato de 100.000 actualizaciones alerta 13 de 144 eventos (9,0%) con el umbral configurado. El PR-AUC del ensamble es 0,153 frente a 0,121 de prevalencia positiva en ventanas: apenas ordena las ventanas anómalas por encima de azar. ROC-AUC 0,544 confirma separación débil. No parece simplemente una mala elección de umbral.

## Qué comprobé

- El parquet contiene 1.380.342 observaciones, 150 runs y 4 clases de activo; 86.455 filas están marcadas dentro de intervalos de evento.
- El pipeline genera 2.463 ventanas de un minuto; 299 (12,14%) positivas. Hay 144 IDs de evento y los 144 aparecen en al menos una ventana. No encontré intervalos de ground truth solapados en el mismo run/activo. Por tanto, el bajo conteo no se debe a eventos perdidos por falta de cobertura en este test.
- Cada ventana se convierte a 218 features. El modelo se entrena solo con ventanas normales (3.727 entrenamiento; 917 para calibración normal). `is_anomaly` y `event_id` quedan fuera de las features.
- El umbral `.95` se aplica a un promedio 50/50 de dos percentiles empíricos CDF, calibrados por separado en ventanas normales de validación. Es un ranking de rareza relativa, no una probabilidad de avería ni un umbral elegido para maximizar detección de eventos.

## Evidencia de dónde falla

Detección por familia al umbral actual:

| Familia | Detectados / evaluables | Score máximo mediano entre eventos |
|---|---:|---:|
| mechanical_overload | 0/15 | 0,520 |
| maintenance_activity | 0/7 | 0,620 |
| regime_mismatch | 0/12 | 0,540 |
| quality_degradation | 0/7 | 0,778 |
| sensor_stuck | 0/10 | 0,686 |
| sensor_bias | 0/8 | 0,560 |
| asset_communication_loss | 2/5 | 0,712 |
| multivariate_novelty | 2/7 | 0,925 |
| otras nueve familias | 13/85 | variable |

Los eventos muy cortos son difíciles con stride de 1 minuto (p.ej. sudden_spike dura 12 s de mediana y se detecta 1/10), pero no es una explicación completa: mechanical_overload dura 55 s de mediana y no se detecta ninguno; otras familias con mayor duración también quedan fuera. Hay patrones que no levantan suficiente error/anomalía en las features actuales, aunque no puede inferirse una causa física solo desde los scores.

En este mismo test y al corte `.95`, Autoencoder por sí solo alcanza PR-AUC 0,168, recall de ventanas 0,117 y 25/144 eventos; Isolation Forest 0,139, recall 0,110 y 27/144. El ensamble 50/50 cae a PR-AUC 0,153, recall 0,054 y 13/144. En esta ejecución, la combinación diluye señal útil de ambos detectores. Los pesos 50/50 son iniciales, no óptimos.

Barrido descriptivo del score de ensamble (no es una selección validada):

| Umbral | Ventanas alertadas | Falsas positivas | Recall de ventanas | Eventos detectados |
|---:|---:|---:|---:|---:|
| 0,95 | 67 | 51 | 5,4% | 13/144 |
| 0,90 | 165 | 131 | 11,4% | 28/144 |
| 0,85 | 225 | 186 | 13,0% | 33/144 |
| 0,80 | 349 | 290 | 19,7% | 42/144 |

Bajar el umbral recupera eventos a costa de una subida rápida de alarmas; no resuelve la separabilidad.

## ¿Es determinista?

- **Inferencia del mismo artefacto en esta máquina:** sí, repetible en la evidencia disponible. Dos ejecuciones guardadas tienen el mismo SHA-256 (`33058150ABB6B75F3EAE32E6F2977533918CA331B94F525949CA098B3C3D2D9B`); el notebook también recalcula y obtiene diferencia máxima 0.0 en todos los scores.
- **Reentrenamiento bit a bit:** no está demostrado. El entrenamiento registra seed 42 para Python/NumPy/PyTorch y scikit-learn; PyTorch pide algoritmos deterministas en modo `warn_only=True`, y el entrenamiento ocurre en CUDA. No repetimos dos entrenamientos desde cero comparando hashes/errores. No afirmar reproducibilidad bit a bit entre hardware, versiones o entrenamientos.

## Causas probables (evidencia, no causalidad física)

1. **Features y representación no separan muchas familias.** Scores máximos medianos bajos en varias familias; más tiempo de optimizador no inventa señal discriminante ausente.
2. **El ensamble actual perjudica en este corte.** Ambos detectores solos superan al ensamble en PR-AUC, recall y eventos al mismo umbral.
3. **Granularidad temporal inadecuada para eventos rápidos.** Ventana/stride de 60 s difumina transitorios breves. El efecto existe para spikes, aunque no explica todos los fallos.
4. **Calibración/umbral no alineados con el coste operativo.** El score es CDF de normalidad, pero el `.95` no fue seleccionado con un objetivo de detección por evento y FP tolerables.
5. **Evidencia de entrenamiento acotada.** Son 3.727 ventanas normales para 218 features. Se pidieron 100.000 actualizaciones, pero el mejor checkpoint de validación se produjo en época 732 de 3.334 completadas. Esto muestra que más actualizaciones no mejoraron el mejor checkpoint; por sí solo no prueba sobreajuste.
6. **Benchmark de un simulador y escenarios diseñados.** Los resultados explican este benchmark sintético; no estiman calidad ni prevalencia en planta real.

El notebook incluye visualización por familia, distribución de score frente a duración, métricas de cada detector, barrido del umbral, auditoría de cobertura de eventos y verificación de repetibilidad.

## Reproducir

Desde la raíz del repo, con el checkout de OT Irregularity Lab y artefactos locales disponibles:

```powershell
uv pip install --python .venv/Scripts/python.exe nbformat nbclient ipykernel pandas
@'
import nbformat
from nbclient import NotebookClient
from pathlib import Path
p=Path('notebooks/diagnostico_deteccion_eventos.ipynb')
nb=nbformat.read(p, as_version=4)
NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(Path.cwd())}}).execute()
nbformat.write(nb, p)
'@ | .venv\Scripts\python.exe -
```

Rutas absolutas de los datos/modelo se declaran en la primera celda de configuración del notebook. El contenido de datos, predicciones y modelos no se incorpora a este repositorio.

## Próxima iteración recomendada

Hacer la siguiente comparación sobre split de desarrollo separado: (a) umbral/ensemble con coste por evento y límite de FP explícitos; (b) stride 10–15 s y agregación temporal que preserve transitorios; (c) pruebas de ablación por grupos de features y por detector; (d) métricas por familia/severidad sin elegir en el test final. Incluir más runs normales y anómalos del simulador y, cuando haya permiso/procedencia, datos industriales externos antes de hacer claims de generalización.
