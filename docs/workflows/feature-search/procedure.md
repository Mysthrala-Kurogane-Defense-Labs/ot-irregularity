# Procedure: OT irregularity feature search

- Purpose: identify which continuous signal classes and justified feature families contribute to the generic normal-behavior detector under repeated simulator variation.
- Keep pseudo-synthetic data, logs, models, checkpoints and raw telemetry outside Git. Do not load the reserved `test/` partition during selection or control.
- Use the local loopback dashboard to observe phase, datasets, worker progress, CUDA allocation, candidates, and aggregate ranking. Runner state is atomically written to `state.json`; append-only `events.jsonl` is the progress history.

## Steps

1. Record the model and simulator commits/worktree state, suite and configuration hashes, Python environment, CUDA device, and run identifier.
2. Generate independently seeded datasets and hash each prepared dataset manifest. Prepare event-free training, mixed labeled validation and separate reserved test sets.
3. Screen all continuous signal groups and the predefined statistical, slope, quality and sampling family ablations with the screening seeds.
4. Select a small finalist set by validation PR-AUC, event detection and false positives per asset-day. Train each finalist on the same new holdout datasets and per-dataset paired training seeds.
5. Train a full-feature baseline on those exact holdout datasets and seeds, even if the screening baseline was not a finalist.
6. Compute mean/sample standard deviation per model and paired per-dataset differences with two-sided 95% Student t intervals. Preserve per-seed differences; intervals describe generated seeds only.
7. Write report, full machine-readable state and paired analysis; verify zero failed runs and `reserved_test_used=false`.
8. Run unit/integration checks appropriate to code changes. Inspect Git changes; do not commit or publish without explicit authorization.

## Caveats

Ablations identify sensitivity, not causal feature value. Selecting finalists on the screening sample induces selection bias; paired new seeds reduce direct comparison noise but do not remove that bias. Synthetic generator performance does not establish behavior on physical plants. Low event counts can make precision, event rate and latency unstable.
