# Normal coverage development protocol

## Question and fixed scope

The consumed normal-exposure test showed false alarms in maintenance, mixed-regime windows and longer thermal operation. First isolate the effect of **normal training/calibration coverage** with the existing contextual Autoencoder + Isolation Forest architecture. Do not initially add new features, larger networks or rule-based suppression. This stage can improve the baseline component without constituting a full v0.5 replacement.

## New data

- Pinned OT Irregularity Lab `718babb7772c3a21f0b87c403f628540cbce58db`, CC BY 4.0, attribution Mysthrala Kurogane Defense Labs.
- `configs/normal-coverage-development.yaml`, 180 one-hour normal runs, **new master seed 920611**: 120 train, 60 validation, zero test. Every run contains all four classes. Same declared sampling/ambient/profile distributions as the previous normal exposure; independent seeds and observations.
- Namespaced IDs `normal-dev-920611::RUN`. Verify source telemetry, truth, scenario and metadata; explicitly reject nonempty anomalies. Existing historical train/validation remain available. No reading historical test/challenge or normal-exposure seed 920601 as training/calibration input.
- Combine historical complete one-minute windows with new complete one-minute windows. Preserve the historical semantic roles, unit handling, feature schema, sampling reference and scaler policy. This tests data coverage only; changed cadence semantics or feature choices require separate ablations.
- Whole-run allocation unchanged: train SHA-256 bucket modulo 5 separates fitting from early stop; validation modulo 2 separates normal calibration from development. Namespace before hashing new IDs. No row-random split. Preserve actual run lists and source/cache hashes.
- Longer runs contribute more windows; row weights remain uniform, as in the existing trainer. Record source-specific window counts and do not claim balanced runs or equal historical/new weighting.

## Fits and thresholds

Primary seed 20261005; existing v0.4 architecture, features and hyperparameters, latent 8, at most 3,000 updates/class, patience 40, RobustScaler fitted only on fit runs, q95 early-stop reconstruction normalization, top-four residual aggregation. Train/persist AE and IF individually; retain the existing selected AE=1/IF=0 ensemble for the controlled comparison. CUDA; deterministic seeds. Report actual optimization steps and early-stop epochs.

Only three normal-calibrated threshold quantiles: .99, .995, .9975. Compute from the persisted calibration references; no extra fit per threshold. Do not choose thresholds on test. Frozen v0.4 is scored on exactly the same development populations.

This first stage evaluates the augmented baseline alone. Relational and thermal supplementation are separate subsequent stages; no claim of improved physical detection follows from a better baseline false-alarm rate.

## Development gate and explicit change of invariant

The earlier union preserved **every baseline alert**, including normal false positives. This stage instead requires preservation of every **labeled nonphysical event detected by baseline** on historical development, evaluated by exact run/asset/event identity against original intervals. Nonphysical here means every ground-truth family except bearing degradation, cavitation, cooling degradation and mechanical overload; it is an evaluation grouping, not a diagnostic taxonomy.

Requirements for an eligible component:

1. No baseline-detected nonphysical event lost; total event detections >= baseline; physical detections >= baseline.
2. Historical mixed-development false windows <= baseline and precision >=.5.
3. On the new normal-development subset: false windows <= baseline overall, <=10/asset-day in every class. Report profile/cadence strata and episodes even if the gate passes.
4. Among eligible primary-seed thresholds choose most total events, then physical events, then fewer new-normal false windows, then the lower quantile. Confirm the selected quantile using seeds 20261006 and 20261007; do not switch seeds to obtain a pass.

Failure is retained as evidence; no holdout is generated for an ineligible component. A component passing development still needs a new frozen mixed-event and normal-exposure confirmation, followed by separate relational/thermal integration. No v1.0 promotion follows from this ablation alone. The original rejections remain unchanged.
