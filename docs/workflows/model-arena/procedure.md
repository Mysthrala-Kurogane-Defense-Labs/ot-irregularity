# Procedure: model arena

- Purpose: compare saved model generations/configurations on known-source tests, normal-only variation and independent seeded streams with explicit/hidden state context.
- Do not refit, recalibrate or select hyperparameters using arena test labels. Preserve each detector score as well as ensemble score.
- Keep incompatible semantic schemas in separate lanes; compare models directly only when inputs, split, labels and preprocessing are comparable.
- Preserve seeds, exact suite/dataset/config hashes, model metadata, per-run partitions and a paired-input check when one field (for example operating regime exposure) is the experimental variable.
- Record event-level counts and latencies alongside point metrics and normal false positives. Treat thresholds as operating points, not universally calibrated policy.
- Store generated data, model weights and inference files outside Git; record licenses and source attribution before redistribution.
- Read the [arena report](../../MODEL_ARENA.md) and [run record](runs/2026-10-03.md) for the current execution and caveats.
