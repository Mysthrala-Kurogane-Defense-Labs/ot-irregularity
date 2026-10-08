# Confirmación independiente del ensemble por clase — 2026-10-08

## Decisión

**Rechazado para promoción.** Evaluación congelada en 65a0ed4 completada sin reajustes sobre los 288 runs reservados. Fuentes mixtas 910561/2/3 y normal 920631 de OT Irregularity Lab, commit 718babb7772c3a21f0b87c403f628540cbce58db, CC BY 4.0, atribución Mysthrala Kurogane Defense Labs. Estos datasets pasan a ser test consumido, excluido de ajuste/calibración/selección futuros.

## Resultados pareados

Control primario con salud frente al candidato por clase:

- Eventos: 71→73/96; físicos 17→16/30.
- Falsas ventanas mixtas: 4→5; incumple el límite congelado.
- PR-AUC: .698865→.710544; precisión .962963→.954955; recall .525253→.535354.
- Pierde un evento físico requerido: seed-910563::test-00002/evt-1, CNC-01, etiqueta mechanical_overload. Es una etiqueta del simulador, no diagnóstico del modelo.
- Ninguna detección de integridad requerida del baseline se pierde.

Normalidad: 180 runs, 42,852 ventanas, 714.2 horas-activo en total. Falsas ventanas/episodios 144→85; tasa diaria 4.839→2.856. Tasas candidato por clase: CNC 3.092, compresor 4.973, transportador 1.344, bomba 2.016. Cumple el criterio normal y la exposición mínima por clase; falla el criterio mixto, por tanto falla la confirmación completa.

Bootstrap pareado por run, 2,000 réplicas: delta de eventos IC95% [-2.128, 8.001] puntos porcentuales; delta PR-AUC [-.011596, .033048], ambos inconclusos. Delta físico [-11.111, 0] puntos. Delta de falsas ventanas normales/activo-día [-2.823, -1.210]: reducción respaldada en esta población del simulador. No ocultar la pérdida física con la mejora agregada.

## Verificación y continuidad

Comando evaluate_consensus.py terminó con exit 0. Se auditaron los 288 record.json, sus referencias a la congelación y los hashes de todos los predictions.parquet; coinciden. No se repitió todavía la ejecución para comprobar reanudación en esta confirmación. No se cambiaron modelos, scalers, referencias ni umbrales. [Resultados completos](../../../results/class-confirmation-20261008.json).

Siguiente trabajo: integrar salidas explícitas de especialistas y analizar, solo en desarrollo, retención física y falsas alarmas conjuntas. El resultado no autoriza elegir retrospectivamente otro control ni tocar umbrales mirando este test. La rama de especialistas aporta contrato/adaptador con paridad; aún no demuestra un modelo mejor. Destilación condicionada a un teacher confirmado. No release ni v1.0.
