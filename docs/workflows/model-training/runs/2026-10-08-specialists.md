# Especialistas: primera implementación — 2026-10-08

Autorización: continuar mejoras y estudiar especialización por problemas con posible destilación posterior. Aplicados engineering-review, code-quality/quality-core y política MKDL. Rama research/specialists-v06 desde 296a3a1 en worktree independiente para no alterar el runtime congelado de la confirmación en ejecución.

Implementado contrato SpecialistResult con ventanas, versiones de modelo/calibración, disponibilidad explícita, scores, umbral, observaciones y contribuciones. Combiner de máximo margen relativo con inventario obligatorio y política explícita para operación parcial; rechaza identidades mezcladas, duplicados y versiones incorrectas. Rank no es probabilidad ni garantía de tasa conjunta. Adaptador de TelemetryHealthReference separa calidad y muestreo preservando scores. No se han cambiado defaults ni la CLI existente.

Validación: 123 tests pasan en Windows/Python 3.12, CUDA deshabilitada para suite, usando el entorno ya fijado del worktree anterior. No se reinstalaron dependencias. Paridad adicional ejecutada con la referencia health-r2 sobre 3,875 ventanas históricas de validación: scores individuales y decisiones combinadas exactamente iguales. No se utilizó test para esa comprobación.

Revisión local: corregido antes de entrega un caso de desbordamiento de score/threshold con umbral subnormal, que podía producir rank NaN. La fórmula algebraicamente equivalente score/(score+threshold) evita ese desbordamiento y tiene regression test. Contratos inválidos, falta de componentes, indisponibilidad total, operación parcial, serialización y límites incluidos en tests. Sin hallazgos pendientes identificados en este cambio acotado; no equivale a una auditoría exhaustiva de todo el repositorio.

Pendiente: completar confirmación congelada; adaptar especialistas relacional/térmico, empaquetar CLI/bundle y demostrar utilidad incremental. La destilación permanece condicionada a un teacher confirmado, y no sustituye automáticamente controles deterministas de cadencia/calidad. No se declara un nuevo modelo aceptado, release ni validación industrial.

Ver [decisión de arquitectura](../../../SPECIALIST_ARCHITECTURE.md). La confirmación usa el worktree research/relational-v05 y sus hashes congelados, no esta rama.

## Relational adapter and fixed union diagnostic

Added relational margin adapter with explicit unavailable state and monotonic rank margin/(1+margin), threshold .5. No fitting or causal attribution. Unit suite: 132 tests pass. Saved development score adapter parity: 9,829 windows, exact ranks and decisions; no consumed test read.

Protocol committed at 109f41d before the single diagnostic. Historical primary 86 events/4 false windows; class median 92/3; union 92/6. New-normal false windows primary 25, class median 14, union 29. Union fails both overall false-window limits (4 historical, 25 normal). It is rejected without generating another test. Retaining all votes increases false alarms without additional development event detections here.

Input cache checksums, exact identity joins, development run partitions and truth hashes verified. Saved input scores are preserved and hashed; this diagnostic did not re-run all original networks. Script uncommitted at execution, recorded as such. [Full diagnostic](../../../results/specialist-union-20261008.json).

Next hypothesis: context-conditioned specialist calibration/routing selected in development, preserving quality/sampling and exact physical event requirements. No routing rule has yet been selected or validated. Do not use rejected independent confirmation sources for that selection. Thermal adapter and teacher distillation remain pending.
