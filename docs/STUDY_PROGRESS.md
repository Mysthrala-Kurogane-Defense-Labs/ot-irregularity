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

Estado: investigación en curso; ninguna nueva mejora validada todavía. El PR permanece draft. No hay validación de planta real ni CI remoto configurado; los controles locales se detallan en cada ejecución.
