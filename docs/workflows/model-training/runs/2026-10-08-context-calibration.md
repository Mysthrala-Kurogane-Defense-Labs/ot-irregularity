# Calibración contextual — 2026-10-08

Protocolo 495940b declarado antes de resultados; implementación a7dbc36, repetición con registro completo 556a248. Solo calibración normal y selección de desarrollo; sin test consumido, redes nuevas ni cambio de canales de integridad. Fuente: OT Irregularity Lab (Mysthrala Kurogane Defense Labs), CC BY 4.0, mismas fuentes históricas y normal-development 920611 verificadas en estudios anteriores.

## Resultado

Cumple el gate de desarrollo declarado, pero no se prioriza para nuevo holdout: no añade detecciones frente al modelo por clase y aumenta falsas ventanas normales. No se cambia retroactivamente el criterio ni se presenta como rechazo del gate.

- Primario con salud: 86/124 eventos, 16/35 físicos, 4 falsas ventanas históricas, 25 normales, PR-AUC .717892.
- Candidato por clase: 92/124 eventos, 20/35 físicos, 3 falsas históricas, 14 normales, PR-AUC .708312.
- Contextual: 92/124 eventos, 20/35 físicos, 3 falsas históricas, 20 normales, PR-AUC .709857. Conserva todas las detecciones del primario, incluidas integridad baseline y físicos requeridos. Latencia media 50.71s frente a 50.71s del candidato por clase, condicionada a detecciones.

Solo cuatro grupos cumplen >=400 ventanas normales: mixed en CNC (507), COMPRESSOR (458), CONVEYOR (512), PUMP (506). Ningún régimen estable tiene suficiente cobertura. No se reduce el mínimo. Contexto utilizado en 579/1965 ventanas históricas y 2312/7864 normales; resto usa referencia de clase o mantiene indisponibilidad relacional (47 históricas). Un régimen desconocido no se trata como evidencia de anomalía ni como régimen entrenado.

## Evidencia

136 tests locales pasan. Rutas de fallback, referencias insuficientes, contextos desconocidos, entradas inválidas y serialización cubiertos. Repetición: predicciones de ambas poblaciones y referencia JSON idénticas byte a byte; métricas idénticas. Cache/truth/input hashes y particiones de runs verificados. [Resultados completos](../../../results/context-calibration-20261008.json), [referencia y cobertura](../../../results/context-reference-20261008.json).

## Decisión siguiente

La limitación medida es cobertura normal por régimen y falta de mejora incremental en esta regla. Preparar desarrollo normal equilibrado por régimen estable y transiciones, con runs independientes de calibración y selección, sin reutilizar test. No asumir que más datos garantizan mejorar: comparar mismo presupuesto de alarmas y eventos físicos con controles congelados. Mantener por separado el especialista térmico y sus requisitos causales. No promoción ni destilación todavía.
