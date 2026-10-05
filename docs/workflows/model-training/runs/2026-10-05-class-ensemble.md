# Class-calibrated reconstruction ensemble — 2026-10-05

## Decision and provenance

**Selected for new independent confirmation; not promoted.** The six-point [protocol](../../../CLASS_CALIBRATED_ENSEMBLE_PROTOCOL.md) was committed at `5901f0c`; execution used clean `7a02daf` on CUDA. [Machine-readable results](../../../results/class-ensemble-20261005.json) preserve every candidate, exact event losses, source/model hashes and class/stratum results.

Sources: historical train/validation plus normal-development seed 920611 from [OT Irregularity Lab](https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab), Mysthrala Kurogane Defense Labs, CC BY 4.0. Historical source provenance is verified against raw telemetry and configured cadence. No test/challenge source is read. Previously consumed confirmation seeds 910551/2/3 and 920621 remain excluded.

## Method

Keep baseline and three relational AE/Ridge/scaler members frozen. Normalize each member's raw top-two reconstruction error by class-normal q99; compare mean and median, with class thresholds at .99/.995/.9975. Only explicitly normal whole-run calibration partitions contribute: 8,084 windows in 164 runs. Long new normal runs have greater window weight; no source-balanced claim. Require all three relational members available; otherwise retain unavailable state while baseline and health continue.

Combine class margin with baseline/original-primary threshold and corrected health/.05; repetition off. Scores are rankings, not probabilities. No network retraining. This is one three-member ensemble, not three independent ensemble replications.

## Complete development screen

| Aggregate | Quantile | Events /124 | Historical false windows | New normal false windows | Eligible |
|---|---:|---:|---:|---:|---|
| mean | 0.99 | 94 | 15 | 75 | No |
| mean | 0.995 | 92 | 6 | 36 | No |
| mean | 0.9975 | 91 | 5 | 25 | No |
| median | 0.99 | 94 | 13 | 59 | No |
| median | 0.995 | 92 | 5 | 37 | No |
| median | 0.9975 | 92 | 3 | 14 | Yes |

Selected **median, q=.9975** preserves every primary-corrected detected event, including all required physical and baseline integrity events. Physical detection improves 16→20/35, total 86→92/124; historical false windows 4→3 and new normal 25→14.

**Ranking tradeoff:** PR-AUC falls from primary-corrected .717892 to .708312 (fixed majority previously .729560). Improved threshold decisions do not establish an overall ranking improvement. The declared gate does not require PR-AUC non-regression. Precision .979021, recall .557769. Mean detection latency 50.71 seconds is conditional on the detected cohort and is not a paired speed comparison.

Normal development: 33 runs, 7,864 windows, 131.067 asset-hours; 14 false windows/episodes, 2.564 per asset-day, run-bootstrap 95% interval [.734, 4.582]. Class rates: CNC 1.465, COMPRESSOR 3.662, CONVEYOR 1.465, PUMP 3.662. These are development observations used in selection, not independent effectiveness estimates.

## Validation and next step

108 local tests pass without warnings, including class-scale invariance, explicit-normal guards, unavailable values and JSON round trips. All six reference reloads reproduce score arrays exactly. Fresh original member/control and health scores exactly match frozen development controls. Relational availability: 1,918/1,965 historical and 7,864/7,864 normal development windows. Other channels remain active outside relational availability.

Freeze the selected reference, all component artifacts and code before generating new independent mixed and prolonged-normal sources. Preserve the previous failed majority confirmation and report event retention, paired uncertainty and PR-AUC tradeoffs. No further point selection on new test. Industrial generalization, thermal coverage, streaming terminal silence and generic CLI packaging remain incomplete toward v1.0.

## Confirmation preparation checkpoint

CI for development evidence commit `7fe727d` passed on 2026-10-05. The [new confirmation protocol](../../../CLASS_ENSEMBLE_CONFIRMATION_PROTOCOL.md) declares mixed seeds 910561/2/3 and normal 920631 before generation. Acceptance and paired-bootstrap helpers now accept an explicit candidate name, preserving majority defaults and identical criteria; 110 local tests pass. The executable freeze and inference extension are still pending. No new source has been generated or evaluated at this checkpoint; no confirmation result is claimed.
