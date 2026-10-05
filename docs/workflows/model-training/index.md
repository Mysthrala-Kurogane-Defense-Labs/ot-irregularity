# Model training workflow

| Date | Status | Scope | Evidence |
| --- | --- | --- | --- |
| 2026-10-05 | Corrected health primary passes; secondary full-model gates fail | Verified configured cadence, 90 tests, integrity restored for three seeds; relational false-alarm stability next | [run record](runs/2026-10-05-telemetry-health-r2.md) |
| 2026-10-05 | Health points rejected; observed/declared cadence mismatch identified | Separate quality/availability/repetition, 88 tests, exact reference reload; source adapter correction next | [run record](runs/2026-10-05-telemetry-health.md) |
| 2026-10-05 | Normal coverage alone rejected at all three thresholds | 180 new development runs, unchanged AE/IF architecture, exact integrity-event gate and serialized score repeatability | [run record](runs/2026-10-05-normal-coverage.md) |
| 2026-10-05 | Normal-exposure target met; joint alarm target failed | 180 normal runs, 178.53 asset-hours/class; cluster intervals, episode counts and profile diagnostics | [run record](runs/2026-10-05-normal-exposure.md) |
| 2026-10-05 | All 24 joint operating points rejected in development | Baseline alert preservation, relational magnitude and thermal decisions; normal-exposure protocol declared | [run record](runs/2026-10-05-joint-decisions.md) |
| 2026-10-05 | Thermal component passes exploratory fresh-test gate; gain small/uncertain, no model replacement | Corrected per-run cadence, initial-context ablation, magnitude calibration, three new frozen holdouts | [run record](runs/2026-10-05-sample-thermal.md) |
| 2026-10-05 | Minute thermal model rejected; paired signal effect verified | Normal-only dynamics and ten exact original replay/intervention pairs; version discrepancy recorded | [run record](runs/2026-10-05-thermal.md) |
| 2026-10-05 | Separate alert decisions rejected in development | Three frozen seeds, no baseline alerts lost but false windows increase; no new test | [run record](runs/2026-10-05-alert-budgets.md) |
| 2026-10-05 | Relational v0.5 candidate passes exploratory targets; family regressions remain | Diagnostics, eight representations, adaptive operating points, three AE seeds and fresh holdout | [run record](runs/2026-10-05-relationships.md); [study progress](../../STUDY_PROGRESS.md) |
| 2026-10-05 | Short-window candidates rejected; batch endpoint policy validated | Temporal screen; fresh frozen tail-policy holdout, coverage and original event denominators | [run record](runs/2026-10-05-temporal.md); [study progress](../../STUDY_PROGRESS.md) |
| 2026-10-05 | Experiment complete; supplement rejected | Physical/regime ablation, development selection and fresh frozen holdout; v0.4 retained | [run record](runs/2026-10-05-physical.md); [study progress](../../STUDY_PROGRESS.md) |
| 2026-10-05 | Experiment complete; operational target unmet | Contextual AE residual tails, three seeds, new independent holdout, complete event intervals | [run record](runs/2026-10-05.md); [candidate report](../../MODEL_V0.4_CONTEXTUAL.md) |
| 2026-10-04 | Complete locally | Fit v0.3 candidate from 15 semantically corrected simulator batches; held-out test and deterministic inference | [run record](runs/2026-10-04.md); [candidate report](../../MODEL_V0.3_CANDIDATE.md) |

## Entrega y continuación

v0.4 integrada mediante PR #1 (79b32ce). v0.5 continúa en su rama independiente: [mantenimiento y protocolo siguiente](../../V0.5_MAINTENANCE.md). Validación de separación: árbol v0.4 idéntico a f3dc360 y 60 tests; candidato restaurado en v0.5 con 65 tests. CI Windows verificado en ambos PR; la dependencia tzdata y el workflow ya están en main. [Registro de entrega](runs/2026-10-05-delivery.md).
