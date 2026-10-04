# Auditoría semántica de features: OT Lab v0.3

Fecha: 2026-10-04. Solo se inspeccionaron contratos, código y datasets locales; no se modificaron el dataset ni los artefactos del modelo.

## Hallazgos confirmados

### 1. `signal_class` no identifica siempre una serie de medida única

El pipeline de ventanas agrupa los registros de un activo únicamente por `signal_class`; `tag_id` queda fuera. El contrato del Lab define `signal_class` como categoría de ingeniería y `tag_id` como nombre del punto.

En el compresor de este dataset hay dos temperaturas simultáneas bajo `signal_class=temperature`: `oil_temperature_c` y `discharge_temperature_c`. En entrenamiento suman 122.796 filas para esas dos series. `make_windows` combina sus observaciones en un solo grupo y calcula `mean`, `range`, `std`, `last`, `slope`, `sample_count` y cobertura sobre la mezcla. Los valores de `last` y `slope` no tienen significado temporal inequívoco cuando dos puntos distintos comparten timestamp. Un cambio anómalo en uno de ellos puede diluirse con el otro.

No se debe usar el `tag_id` de un fabricante como feature categórica genérica. La capa semántica necesita una identidad canónica opcional del punto, independiente del identificador fuente: por ejemplo `oil_temperature` y `discharge_temperature`.

### 2. `state` es código discreto, no medida continua

El simulador emite `cycle_state`, unidad `code`, con valores observados 0 y 1. Sin embargo declara límites de ingeniería 0–4, y el pipeline calcula sobre el código las mismas 17 estadísticas continuas que sobre temperatura, presión o vibración. Para un canal binario, media/proporciones y transiciones sí pueden ser informativas; pendiente, rango, MAD y reconstrucción continua no tienen la misma interpretación. El tipo debe ser explícito y el contrato del simulador debe explicar la discrepancia 0–4 frente a valores 0/1.

### 3. Se confunde ausencia estructural con pérdida de señal

El modelo genera features para la unión global de 12 `signal_class`; cada clase de activo emite solo una parte de ellas. Cuando una señal no existe para un activo, `_signal_features` devuelve `missing=1`, `coverage_ratio=0` y `bad_ratio=1`, igual que si una señal aplicable hubiese desaparecido.

La explicación de inferencia comprobaba cualquiera de esos valores sin conocer aplicabilidad por clase de activo. En las predicciones guardadas del test Lab, las **2.463/2.463 ventanas** recibieron `signal_loss` y `sampling_degradation`, incluidas las normales. Esas observaciones son inválidas para este dataset. Los flags de ausencia estructural también se incorporan como features del modelo; el one-hot de `asset_class` puede ayudarlo a aprender el patrón, pero no arregla la semántica ni la explicación.

Se corrigió la emisión de estas observaciones: modelos recién entrenados guardan `signals_by_asset_class` en `feature_schema.json` y la inferencia solo evalúa pérdida/cobertura de canales aplicables. Para modelos antiguos sin ese mapa, se suprimen ambos diagnósticos porque no se puede distinguir de forma fiable ausencia estructural de pérdida real. Una inferencia en memoria del artefacto CUDA existente confirmó 0 flags en sus 2.463 ventanas tras la corrección. Esto **no cambia los features ni scores numéricos**, y no reescribe sus JSONL históricos.

### 4. Dimensión alta y peso implícito por señal

Las 218 features son 12 señales ×17 resúmenes más 14 columnas de contexto. Las 17 columnas de cada señal no son 17 pruebas independientes: varias describen el mismo nivel, variabilidad o forma. Además, la pérdida del Autoencoder promedia el error cuadrático por feature; un desvío concentrado en un único canal queda promediado con el resto. Este diseño puede rebajar anomalías locales y dar peso proporcional al número de resúmenes de cada canal. Debe compararse el scoring por feature actual con agregación primero por canal semántico, sin asumir que usar `max` será mejor (puede elevar falsas alarmas).

### 5. Qué no parece mal en este dataset

- Las unidades de medida físicas son consistentes dentro de cada `signal_class` en las particiones inspeccionadas. La configuración de este entrenamiento deja `canonical_units` vacío; no se observa por ahora una mezcla de °C/°F o bar/Pa que explique este resultado concreto.
- El transformador de unidades sí existe y debe usarse cuando el dataset de entrada mezcle unidades compatibles.
- `engineering_min`/`engineering_max` del simulador describen límites declarados, no operación normal. Ignorarlos en las features no es por sí solo un defecto. Se podrían comparar como escalado relativo o flags fuera de rango, manteniendo los valores físicos y sin tratarlos como límites de alarma.
- Calidad y muestreo se deben conservar, pero sus métricas deben evaluarse solo en señales aplicables.

## Diseño semántico recomendado

Ampliar de forma compatible el formato normalizado con metadatos opcionales:

- `measurement_role`: identidad semántica canónica del punto (`motor_current`, `oil_temperature`, etc.). No contiene el identificador del producto/historian.
- `value_kind`: `continuous`, `binary`, `categorical` o `counter`.
- Un mapa de aplicabilidad por `asset_class`/tipo de activo para distinguir “no aplica” de “perdió muestras”.
- Para `categorical`, registrar el vocabulario/etiquetas y mapearlo solo desde entrenamiento; extraer proporción por categoría, último estado y transiciones/duración en estado. No promediar códigos con orden arbitrario.
- Si `measurement_role` no está disponible, mantener fallback explícito al `signal_class`, emitir una advertencia de posible mezcla de puntos y no presentar pérdida de señal para clases estructuralmente ausentes.
- Agregar errores de reconstrucción por `measurement_role` antes de combinarlos entre roles; comparar media, percentil alto y top-k contra la reducción actual, calibrando cada alternativa en validación normal.

La semántica canónica forma parte del preprocesamiento industrial. `RobustScaler`/`StandardScaler` siguen siendo una fase posterior de escala ML y deben persistirse junto al modelo.

## ¿Conviene probar todas las combinaciones?

No como primera búsqueda. Hay 12 familias globales: todos los subconjuntos no vacíos son `2^12 - 1 = 4.095`; a 10–15 repeticiones son **40.950–61.425 entrenamientos**. El entrenamiento de 100k pasos medido tardó 213 s, lo que da aproximadamente **101–151 días en serie**. Enumerar las 218 columnas crudas implica `2^218` subconjuntos, fuera de escala. El paralelismo CUDA ayudaría, pero no reduce el espacio ni escala linealmente.

La comparación además sería engañosa antes de arreglar semántica: una combinación podría parecer “mejor” simplemente porque elimina el canal agregado de forma ambigua o los falsos flags de señales no aplicables.

### Búsqueda viable por etapas

1. **Corregir representación y congelar contrato:** roles de medida, tipos de valor y aplicabilidad. Crear un nuevo dataset de desarrollo etiquetado con el simulador; mantener el test Lab actual sin volver a seleccionar features sobre él.
2. **Screening agrupado:** baseline corregido, ablación de cada familia de señal (12 variantes), y unas pocas variantes justificadas de calidad/muestreo/contexto y scoring por canal. Unos 20 candidatos ×3 repeticiones =60 ajustes, ~3,6 h al coste actual de 100k cada uno. Se puede pilotar con límite de pasos y early stopping medido por validación para acortar, no asumir de entrada que 20k pasos bastan.
3. **Finalistas:** elegir hasta cinco configuraciones con mejora consistente en splits por `run_id`/escenario y repetir 10–15 veces en folds/semillas pareados (50–75 ajustes; ~3,0–4,4 h a 100k pasos cada uno).
4. **Criterios:** PR-AUC comparado con prevalencia, detección por evento/familia, falsos positivos por activo-hora, latencia y dispersión entre repeticiones. Reportar media, dispersión e intervalo por bootstrap agrupado por run; no usar accuracy ni ajustar con el test final.

Las repeticiones deben variar inicialización y partición de runs/escenarios. Diez seeds sobre una única partición solo miden variación del optimizador, no estabilidad entre datos. Una feature/grupo “estable” es aquella cuya utilidad aparece consistentemente en splits independientes; no es lo mismo que inferencia determinista del mismo artefacto.

## Evidencia de entrada

- Fuente: `ot-irregularity-training-v0.3-release` del checkout `ot-irregularity-lab`, archivos Parquet y `ground_truth.json` por run.
- Código observado: `src/ot_irregularity/features.py`, `src/ot_irregularity/pipeline.py`, `scripts/prepare_otlab.py` del repo del modelo; `src/ot_lab/process.py` y `TELEMETRY_SCHEMA.md` del simulador.
- Métricas actuales por familia: [diagnóstico de detección](DIAGNOSIS_EVENT_DETECTION.md); comparación original: [arena](MODEL_ARENA.md).
- La corrección de aplicabilidad evita explicaciones constantes erróneas; aún queda rediseñar el eje de puntos de medida y el tratamiento categórico antes de volver a entrenar el modelo.
