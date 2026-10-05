# Registro del estudio de detección OT

Este registro conserva avances, resultados negativos y cambios de medición. Los datos son pseudo-sintéticos de [OT Irregularity Lab](https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab), CC BY 4.0; atribución: Mysthrala Kurogane Defense Labs. Los nombres de eventos pertenecen al ground truth y no son diagnósticos del detector.

## Evolución hasta el 5 de octubre de 2026

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

Prioridad: sensibilidad física por familia/severidad, relaciones temporales justificadas y falsas alertas durante transiciones normales. Acortar ventanas sin más ya se ha probado y no basta. Las relaciones entre señales siguen **sin probarse**; deben evaluarse con selección en desarrollo y un nuevo holdout, manteniendo explícita la política de cierre. No se recomienda aumentar épocas ni bajar el umbral como sustituto de esa investigación.

El PR permanece draft. Hay una corrección de procesamiento de archivos validada en datos sintéticos; no hay un nuevo modelo físico que promocionar, validación de planta real ni CI remoto configurado. El objetivo exploratorio conjunto de >=50 % de eventos y <=10 falsas ventanas/día sigue pendiente.
