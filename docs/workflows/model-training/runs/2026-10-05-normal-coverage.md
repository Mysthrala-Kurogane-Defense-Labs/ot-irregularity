# Normal coverage ablation

## Declared experiment

The [protocol](../../../NORMAL_COVERAGE_DEVELOPMENT.md) and suite were committed as `6f071e8` before generating new seed 920611. Pinned Lab `718babb` generated 180 normal one-hour runs: 120 train, 60 validation, zero test. Normal-development and consumed normal-exposure run seeds were checked disjoint using manifest metadata only. Source: OT Irregularity Lab 0.6.0, Mysthrala Kurogane Defense Labs, CC BY 4.0.

This isolates normal coverage using the unchanged v0.4 contextual AE/IF architecture, feature schema and training budget. Historical train/validation are retained. Longer runs contribute more windows under the existing uniform-window sampling; no claim of balanced source weighting. Whole-run hashing separates fit, early stop, normal calibration and development after namespacing new IDs.

The gate explicitly preserves baseline-detected nonphysical event **identities**, rather than all baseline alert decisions. It also constrains historical detections/false windows and new normal-development false alarms. This is a new predeclared component ablation; it does not retrospectively change the rejected union results or imply full-model promotion.

## Reproduce

```powershell
python scripts/prepare_normal_development.py --source NEW_NORMAL_SOURCE --output STUDY_OUTPUT --baseline BASELINE
python scripts/research_normal_coverage.py --output STUDY_OUTPUT --baseline BASELINE --dataset HISTORICAL_TRAIN_VALIDATION --cache VERIFIED_HISTORICAL_CACHE --raw-root HISTORICAL_RAW_BATCHES --normal-source NEW_NORMAL_SOURCE --device cuda:0
```

`STUDY_OUTPUT/data_protocol.json` records the pre-generation source commit, time, seed and suite/protocol hashes. Preparation verifies normal source provenance and writes reusable window caches with source hashes. Training verifies cache bytes and run separation, persists all models/scalers/IF/reference distributions, and reports exact original event intervals as well as normal-only exposure and profile/cadence strata.

Three primary-seed operating points are declared (.99/.995/.9975 normal quantiles). Seeds 20261006 and 20261007 are only run if the primary seed yields an eligible point; their quantile cannot be reselected. Artifacts retain their training q=.99 threshold, while any selected alternative is explicit in `coverage-results.json` and must be frozen separately before independent confirmation.

## Verification and status

81 local tests pass. New tests reject equal-count substitutions that lose a specific integrity event, each declared gate regression and an explicit source-partition mismatch. Existing tests remain green. These checks establish software behavior, not model quality.

Generation and preparation completed: 28,564 new train windows and 14,280 new validation windows. Historical complete windows contribute 5,902 train and 3,875 validation. After the declared whole-run split: 26,168 fit windows, 8,298 early-stop windows, 8,084 normal calibration windows and 9,829 development windows (1,965 historical mixed + 7,864 new normal, from 33 new development runs).

Primary models trained at clean code commit `bb4446c`, on CUDA, 3,000 updates per class. Both AE and IF, scalers, references, metadata and predictions were persisted. Reloaded scores match saved arrays exactly on all 1,965 historical and 7,864 normal development windows.

## Results: coverage alone rejected

| Operating point | Historical events /124 | Physical /35 | Historical false windows | Lost baseline nonphysical events | New normal false windows |
|---|---:|---:|---:|---:|---:|
| Frozen baseline | 53 | 4 | 4 | 0 | 31 |
| Augmented q=.99 | 60 | 7 | 10 | 1 | 82 |
| Augmented q=.995 | 52 | 3 | 5 | 3 | 49 |
| Augmented q=.9975 | 49 | 3 | 2 | 5 | 34 |

New normal-development exposure is 131.067 asset-hours overall. The q=.99 rates also exceed 10 false windows/day in every class. Higher quantiles lower false alarms but lose integrity events and physical detections. **None of the three points passes the predeclared component gate.** Secondary seeds and new holdouts were therefore skipped as declared, not omitted after an unfavorable test. No model promotion or release.

The result refutes the practical sufficiency of this particular coverage augmentation under the fixed features, uniform-window weighting, architecture and optimization budget. It does not prove that additional normal data are generally harmful or that longer training would help. The new long-run windows dominate the training mix; alternative weighting is an untested hypothesis. [Numeric evidence](../../../results/normal-coverage-20261005.json) includes all operating points, strata, event identities, hashes and metadata.

## Why the q=.99 integrity event was lost

The lost event is `dev-13::validation-00010/evt-1`, CNC quality degradation. In the complete 00:08–00:09 window, each of six applicable continuous roles has **bad_ratio=0.191667**. Quality-ratio features dominate the top-four contributions in both models:

- Baseline score 0.992410 exceeds its 0.990566 threshold.
- Augmented score 0.986893 is below its 0.990291 threshold.

Thus the quality observation survives normalization/feature extraction and contributes to the score; the final global reconstruction-percentile decision misses it. The added data change fitted weights, scaling and calibration together, so this is not a causal attribution to one of those mechanisms. Scores are reference percentiles, not probabilities of valid telemetry or faults.

Next hypothesis: an explicit, separately calibrated channel for observed telemetry quality/availability, preserving its measured ratios and reference context instead of requiring the global physical reconstruction score to cross its threshold. Keep physical signals and reconstruction contributions. Normal reference quality must be learned per applicable role/source policy; an unknown quality grade must not be silently relabeled good or bad. Declare that experiment before implementation/evaluation; do not apply it retrospectively to pass this ablation.

No test/challenge telemetry was used. Consumed normal-exposure seed 920601 remains excluded from fitting, calibration and selection.
