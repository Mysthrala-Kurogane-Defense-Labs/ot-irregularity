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

Generation complete; preparation/training results pending. No test/challenge used, no release or model promotion.
