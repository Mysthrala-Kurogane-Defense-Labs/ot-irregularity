# Registro del estudio de detección OT

Este registro conserva avances, resultados negativos y cambios de medición. Los datos son pseudo-sintéticos de [OT Irregularity Lab](https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab), CC BY 4.0; atribución: Mysthrala Kurogane Defense Labs. Los nombres de eventos pertenecen al ground truth y no son diagnósticos del detector.

## Evolución hasta el 5 de octubre de 2026

[Cadencia corregida](workflows/model-training/runs/2026-10-05-telemetry-health-r2.md): el canal sin repetición recupera todas las detecciones de integridad requeridas: 71→86/124 eventos, manteniendo cuatro falsas ventanas históricas y 25 normales. 90 tests y scores reproducibles. La confirmación completa falla en las otras semillas por límites ya incumplidos por sus controles relacionales: compresores 10,25 y 22,71 falsas ventanas/día. El canal de calidad/disponibilidad añade cero falsas ventanas en las tres semillas; el siguiente problema es estabilidad relacional. Sin nuevo test ni promoción.

[Canal de telemetría](workflows/model-training/runs/2026-10-05-telemetry-health.md): 71→79/124 eventos con cuatro falsas ventanas históricas y 25 en normalidad nueva, pero aún pierde seis eventos baseline. Ningún punto se acepta. Se detecta un error semántico del adaptador: el intervalo por fila del simulador representa el gap observado, no la cadencia configurada. El control de cambios desactiva cobertura/repetición en casi todas las ventanas. Corregir esa separación y repetir desarrollo; el ensayo actual no permite juzgar la eficacia del canal de disponibilidad. 88 tests locales y repetibilidad exacta de las cuatro tablas de scores tras cargar JSON.

[Ampliación de normalidad](workflows/model-training/runs/2026-10-05-normal-coverage.md): 180 runs nuevos de desarrollo, misma arquitectura y tres umbrales. Ninguno supera el criterio: q=.99 gana eventos (53→60/124), pero sube falsas ventanas históricas (4→10) y pierde un evento de calidad pese a bad_ratio=19,17 %. La calidad está representada, pero no cruza el umbral global. Siguiente hipótesis: canal explícito de calidad/disponibilidad, calibrado por separado; sin eliminar señales físicas ni reajustar sobre test.

[Exposición normal prolongada](workflows/model-training/runs/2026-10-05-normal-exposure.md): 180 runs y 178,53 horas-activo por clase. La unión conjunta aumenta falsas ventanas de 5,24 a 11,63/activo-día; diferencia pareada IC95% [4,57, 8,33]. Térmica: 7,60 global, pero 11,56 en compresores. El baseline también falla en perfiles de mantenimiento. No se promueve candidato: mejorar cobertura y representación de normalidad con nuevos datos de desarrollo.

[Decisión conjunta](workflows/model-training/runs/2026-10-05-joint-decisions.md): los 24 puntos se rechazan en desarrollo por falsas ventanas. El punto principal q=.9975 conserva todas las alertas baseline y alcanza 87/124 eventos, 18/35 físicos y 5/10 de refrigeración, pero pasa de 4 a 6 falsas ventanas. No se consume otro test ni se promueve candidato. La siguiente medición sigue el [protocolo normal prolongado](NORMAL_EXPOSURE_PROTOCOL.md).

[Dinámica térmica antes de agregar](workflows/model-training/runs/2026-10-05-sample-thermal.md): componente seleccionado en desarrollo y contrastado en tres lotes nuevos. Test: 62/105 eventos frente a 60/105 baseline, refrigeración 2/7 frente a 1/7, falsas ventanas 8 frente a 7, sin perder eventos baseline. Cumple el gate exploratorio, pero los intervalos incluyen ganancia cero; v0.5 relacional detecta 67/105 y sigue siendo mejor en conjunto. No se sustituye el modelo ni se reajusta sobre test. Contexto inicial y normalización de magnitud siguen siendo supuestos explícitos.

[Ensayo térmico y pares simulados](workflows/model-training/runs/2026-10-05-thermal.md): el modelo de dinámica por minuto no mejora refrigeración y se rechaza. Diez replays exactos confirman desviaciones térmicas de 0,59–4,01 °C; sus medias por minuto reducen el pico a 0,31–2,61 °C. La versión declarada 0.3.1 no reproduce estos datos; el checkout congelado 0.6.0 sí, con hashes registrados. La procedencia histórica requiere ese matiz.

Última iteración: [presupuestos separados de alerta](workflows/model-training/runs/2026-10-05-alert-budgets.md) rechazados en desarrollo. La unión conserva todas las alertas baseline y gana eventos físicos, pero pasa de 4 a entre 8 y 17 falsas ventanas. Ninguna de tres semillas cumple el gate; no se consume un nuevo test.

| Etapa | Pregunta y evidencia | Decisión |
|---|---|---|
| v0.3, búsqueda de features | 120 entrenamientos; test histórico: PR-AUC 0,1599, 23/276 eventos, 39,48 falsas ventanas/activo-día | Conservar el resultado débil como referencia; no afirmar calidad operacional |
| v0.4, selección en desarrollo | Cinco representaciones, cinco scores y 36 combinaciones de pesos/cuantiles. Separación por runs de ajuste, early stopping, calibración y selección | Modelos por clase; RobustScaler; media de los cuatro residuos normalizados mayores; AE=1 e IF=0 en la combinación |
| v0.4, evaluación congelada | Tres semillas de entrenamiento y tres lotes nuevos: 108 runs, 1.695 ventanas, 119 intervalos de evento | Mejora reproducida en este simulador; objetivo exploratorio aún incumplido |
| Auditoría de la métrica de eventos | Un solo ID por ventana perdía solapamientos: 117 eventos representados frente a 119 originales | Usar todos los intervalos; corregir 52/117 a 55/119 sin cambiar scores, pesos ni umbral |
| Siguiente iteración | La mejora se concentra en integridad de telemetría; cuatro familias físicas suman 0/25 detecciones | Medir detección física y falsas ventanas por separado; estudiar representación y contexto de régimen |

Las cifras del test histórico de v0.3 y las de v0.4 proceden de conjuntos distintos. La comparación pareada válida es la siguiente, que vuelve a ejecutar ambos modelos sobre los mismos datos nuevos.

## Comparación pareada independiente: v0.3 frente a v0.4

| Indicador | v0.3 | v0.4 principal | Objetivo exploratorio |
|---|---:|---:|---:|
| PR-AUC | 0,1532 | 0,5658 | Maximizar con control de falsas alarmas |
| Precisión | 16,7 % | 83,3 % | >=50 %: cumplido en esta muestra |
| Eventos detectados | 8/119 (6,7 %) | 55/119 (46,2 %) | >=50 %: pendiente |
| Falsas ventanas/activo-día | 42,48 | 15,29 | <=10: pendiente |
| Cavitation / bearing / cooling / overload | 1/25 | 0/25 | Mejorar sin ocultarlo en el agregado |

Solo se han observado 28,25 horas-activo. Las tasas diarias extrapolan esa exposición; no cuentan episodios de alerta. Las otras dos semillas de v0.4 detectaron 54/119 y 56/119 eventos. No se eligió la mejor semilla mirando test. La latencia media se calcula solo entre eventos detectados y no permite comparar velocidad entre cohortes distintas sin un análisis pareado adicional.

Detalles, intervalos de confianza, hashes, fuentes y resultados por familia: [informe v0.4](MODEL_V0.4_CONTEXTUAL.md), [evidencia numérica](results/v0.4-contextual.json).

## Protocolo de la siguiente iteración

1. Tratar los lotes 910501–910503 como evaluación ya consumida: sirven para formular hipótesis, nunca como nueva prueba independiente.
2. Reutilizar exclusivamente las particiones históricas de ajuste, early stopping, calibración y desarrollo para seleccionar. Mantener normal-only en ajuste y calibración.
3. Contrastar un número acotado de representaciones: separar features físicas de calidad y centrar features respecto a la mediana normal de su régimen. El régimen debe estar disponible en inferencia; no usar etiquetas de anomalía como contexto.
4. Registrar PR-AUC, precisión, recall, falsas ventanas/día y detección por intervalos completos y familias. Informar también cobertura y tamaño de cada régimen.
5. Congelar candidato y criterio antes de un nuevo holdout. Si desarrollo no respalda una mejora, conservar v0.4 y publicar el resultado negativo sin consumir otro test.

## Segunda iteración: resultado negativo conservado

Se ejecutaron tres representaciones nuevas (12 AE en CUDA y 12 Isolation Forest) y dos combinaciones con v0.4. El presupuesto máximo fue de 3.000 actualizaciones por AE con early stopping. El criterio de selección se escribió antes del cribado: más eventos físicos, ninguna pérdida en detecciones totales, precisión >=50 % y ninguna subida de falsas ventanas frente a v0.4.

Desarrollo: 2.234 ventanas, **124 intervalos originales**, de ellos 35 físicos. El conteo completo corrige también las detecciones de desarrollo de v0.4 de 51 a 53, sin alterar scores. Las cifras siguientes usan el mismo contrato de intervalos para todas las variantes.

| Variante | PR-AUC | Eventos/124 | Físicos/35 | Falsas ventanas |
|---|---:|---:|---:|---:|
| v0.4 | 0,5001 | 53 | 4 | 15 |
| Solo features físicas | 0,3650 | 25 | 3 | 11 |
| Todas, centradas por régimen | 0,4848 | 51 | 4 | 17 |
| Físicas, centradas por régimen | 0,3479 | 24 | 3 | 11 |
| Máximo de v0.4 y AE físico | 0,5246 | 53 | 5 | 14 |
| Máximo de v0.4 y AE físico centrado | 0,5151 | 52 | 5 | 13 |

La combinación seleccionada toma el máximo de dos percentiles, **recalibrando su umbral con normales** para no sumar presupuestos de falsas alarmas sin control. No cambia el umbral mirando eventos de test. El centramiento resta la mediana por clase/régimen aprendida exclusivamente en ajuste normal; categorías con menos de 20 ventanas usan la mediana normal de clase. En desarrollo, 1.151 ventanas tienen régimen `unknown` y 680 `mixed`: 1.831/2.234 (82,0 %). Esto limita la información del contexto; no demuestra por sí solo la causa de los errores.

### Nuevo holdout después de congelar el candidato

Semillas Lab **910511, 910512, 910513**, suite 0.2.0 y checkout limpio `718babb7772c3a21f0b87c403f628540cbce58db`. Se generaron 240 runs por lote y se evaluaron únicamente sus 36 runs de test: 108 runs, 1.905 ventanas, 114 eventos, 31,75 horas-activo. Los artefactos y el umbral se congelaron antes de generar estos lotes. Una semilla de entrenamiento para este ensayo; el resultado negativo no justifica promocionar ni elegir retrospectivamente otra semilla.

| Indicador | v0.4 congelado | Combinación física |
|---|---:|---:|
| PR-AUC | 0,4377 | 0,4435 |
| Precisión | 79,5 % | 78,8 % |
| Eventos | **53/114 (46,5 %)** | 50/114 (43,9 %) |
| Eventos físicos | 4/29 | 4/29 |
| Falsas ventanas | 18 | 18 |
| Falsas ventanas/activo-día | 13,61 | 13,61 |

Bootstrap pareado por run: ΔPR-AUC +0,0058, IC95 % [-0,0038; 0,0165] (500 réplicas); Δdetección total -2,63 puntos, IC95 % [-8,04; 1,96]; Δdetección física 0 puntos, IC95 % [-10,00; 10,53] (2.000 réplicas). Se gana una cavitación y se pierde una sobrecarga, además de otras detecciones de señal. No existe evidencia suficiente de mejora física ni global.

**Decisión: conservar v0.4.** El suplemento permanece como experimento reproducible y no se incorpora al modelo principal. Estos nuevos lotes ya están consumidos para evaluación; cualquier selección posterior necesita otro holdout. La variación de PR-AUC de v0.4 entre lotes no es un cambio del modelo: los mismos pesos producen resultados distintos ante otras muestras.

Evidencia completa: [segunda iteración](results/physical-ablation-20261005.json). Reproducción, controles y límites: [registro de ejecución](workflows/model-training/runs/2026-10-05-physical.md).

## Indicadores siguientes y estado

### Cuarta iteración: candidato relacional v0.5

Diagnóstico en desarrollo: 22/35 eventos físicos contienen algún valor bruto extremo respecto a la referencia normal marginal, pero ninguno supera el percentil 99 del indicador agregado de distancia de features. Esto no prueba pérdida de información: falta una ejecución normal equivalente y cambian las oportunidades de observar extremos.

Se prueban seis residuos por clase: cada media de señal se predice a partir de las otras cinco, con Ridge ajustado solo sobre normales. Una variante añade el minuto anterior sin cruzar runs, activos ni huecos. Después se comparan residuos directos y AE sobre residuos, solos y como suplemento de v0.4. Todas las variantes usan la misma política de cierre y ventanas. La variante temporal no resulta elegida.

El primer umbral q99 produce demasiadas falsas alertas. Una **segunda fase adaptativa de desarrollo**, explícitamente registrada, compara cuatro cuantiles normales: 24 puntos resolubles, cinco elegibles. Se selecciona el máximo de v0.4 y el AE de residuos estáticos, q99,75: 71/124 eventos, 15/35 físicos y cuatro falsos minutos frente a 53/124, 4/35 y cuatro de v0.4. La selección no utiliza test.

Tres semillas AE y umbrales quedan congelados antes de generar los lotes **910531–910533**. Nuevo test: 108 runs, 1.418 minutos completos, 96 intervalos originales, 33 físicos, misma cobertura para todos los modelos.

| Indicador | v0.4 con cierre completo | v0.5 principal |
|---|---:|---:|
| PR-AUC | 0,5423 | **0,7036** |
| Precisión | 88,1 % | **97,6 %** |
| Eventos detectados | 40/96 (41,7 %) | **61/96 (63,5 %)** |
| Eventos físicos | 1/33 | **23/33** |
| Falsos minutos | 7 | **2** |
| Falsos minutos/activo-día | 7,11 | **2,03** |

Las otras dos semillas detectan 59 y 58 eventos, 21 físicos cada una, con tres falsos minutos. Las tres pasan el objetivo exploratorio en esta muestra. Bootstrap pareado: ΔPR-AUC IC95% [0,0919; 0,2213]; Δdetección total [8,08; 35,17] puntos; Δdetección física [48,48; 83,79] puntos. La reducción de falsas alertas todavía tiene incertidumbre: su diferencia de tasa tiene IC95% [-11,39; +1,03], que incluye cero.

**Regresiones que no se ocultan:** comunicaciones 8/8 -> 6/8; telemetría ausente 6/7 -> 4/7; calidad 7/8 -> 3/8; señal congelada 8/9 -> 5/9. Refrigeración sigue **0/7**. El avance físico se concentra en bearing (1/11 -> 9/11), cavitación (0/8 -> 8/8) y sobrecarga (0/7 -> 6/7). Son familias de ground truth, no diagnósticos emitidos.

Se conserva v0.4 y se empaqueta v0.5 como **candidato experimental**, con inferencia JSONL/Parquet reproducible, scores separados, disponibilidad y contribuciones de residuos. No sustituye los defaults ni se publica un release de pesos. Los controles de cuantil equivalente y residuo directo también se reportan: el avance no se explica únicamente por recalibrar el umbral.

[Protocolo, familias, controles y uso](workflows/model-training/runs/2026-10-05-relationships.md) · [Evidencia numérica](results/relationships-20261005.json).

### Tercera iteración: resolución temporal y cierre de archivos

Las ventanas reales de 30s y 15s ya están implementadas, con compatibilidad explícita para los artefactos antiguos. Ambas alternativas **fallan** el filtro de desarrollo. Sobre 2.057 minutos comunes, v0.4 detecta 53/124 eventos con 5 minutos falsos; 30s detecta 55 con 27 falsos; 15s detecta 43 con 12 falsos. No se promocionan ni se prueban contra nuevos holdouts.

La auditoría de cobertura encuentra otra causa de falsas alertas: colas incompletas de los archivos. Se formula y congela una hipótesis separada: emitir solo ventanas cuyo final haya alcanzado el último timestamp observado del activo, conservando pesos y umbral de v0.4.

| Nuevo test, semillas 910521–910523 | Cierre histórico | Solo ventanas cerradas |
|---|---:|---:|
| Eventos detectados | 33/84 | **33/84** |
| Eventos físicos | 2/22 | 2/22 |
| Minutos con falsa alerta | 23 | **8** |
| Precisión | 69,3 % | **86,4 %** |
| Falsos minutos/día observado, misma exposición | 19,20 | **6,68** |
| Ventanas emitidas | 1.856 | 1.631 |

Se excluyen **225 minutos incompletos**, cuatro de ellos positivos. Ningún evento queda sin ventanas observables y no se pierde ninguna detección de evento en esta muestra. Son 108 runs, 28,75 horas-activo observadas y 27,18 horas cubiertas por ventanas completas. El IC95% pareado del cambio de falsos minutos es [-25; -6]. El denominador de eventos incluye todos los intervalos originales.

**Avance aceptado: opción de cierre correcta para archivos finitos**, disponible con `infer/evaluate --tail-policy complete`. El modo predeterminado conserva compatibilidad y los pesos siguen siendo v0.4. No mejora la sensibilidad física ni convierte 39,3% de detección en un ratio suficiente. En monitorización continua hace falta un reloj externo para cerrar ventanas durante ausencia de muestras; esa función sigue pendiente. El recorte de cobertura se publica y no se presenta como mejor ranking del modelo.

Detalles y reproducción: [ejecución temporal](workflows/model-training/runs/2026-10-05-temporal.md); [evidencia numérica](results/temporal-and-tail-20261005.json).

Prioridad tras v0.5: recuperar las detecciones de integridad mediante presupuestos separados para integridad y desviación física; investigar contexto térmico para refrigeración; y medir falsas alertas con trazas normales mucho más largas. Son hipótesis pendientes, que requieren selección en desarrollo y otro holdout. Los doce lotes de test evaluados hasta aquí están consumidos.

El PR permanece draft. El candidato v0.5 pasa el objetivo exploratorio conjunto en el último test pseudo-sintético, con mejoras físicas y regresiones de integridad explícitas. No hay validación de planta real ni CI remoto configurado; el resultado no establece aceptación operacional.
