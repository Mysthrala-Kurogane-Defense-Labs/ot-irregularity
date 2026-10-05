# Independent class-magnitude confirmation protocol

Declared after the class median q=.9975 passed development, before new source generation. Prior majority confirmation failed and remains unchanged. The selected class-normal JSON anchors and thresholds are frozen alongside the components below. Freeze the exact three original relational members and thresholds, baseline artifacts, corrected health JSON, health threshold .05, repetition disabled, complete one-minute windows, inference/evaluation code and simulator/suite hashes. No selection, retraining or recalibration on this confirmation.

## New sources

Pinned OT Irregularity Lab commit `718babb7772c3a21f0b87c403f628540cbce58db`, simulator 0.6.0. Data CC BY 4.0, attribution Mysthrala Kurogane Defense Labs, [source](https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab).

- Mixed: suite `suites/training-v0.2.yaml`, 240 generated runs per master seed **910561, 910562, 910563**. Evaluate every manifest `test` run, expected 36 per batch / 108 total. Generation retains the suite's fixed partitions; evaluation never reads train/validation telemetry or uses it for fitting.
- Normal: existing `configs/normal-exposure-v05.yaml`, **180** one-hour runs, master seed **920631**, test only. Require >=168 evaluated asset-hours per class and empty anomaly truth. Preserve the existing six weighted transition profiles, cadence/jitter, ambient/drift and regime exposure distributions. This is many independent one-hour runs, not continuous week-long operation.
- Source directories must not exist when freezing. Check actual per-run seed overlap with source manifests of historical development, normal development and consumed prior evaluations before reading new test telemetry. Any overlap aborts. Existing source manifests may be inspected only for identity/seed metadata.

## Controls and outputs

1. Original v0.4, original threshold.
2. Original primary v0.5 relational, original shared threshold.
3. Primary v0.5 plus corrected health .05, repetition off.
4. Fixed majority of all three original member margins plus the same health channel, retained as a comparator.
5. Selected class-magnitude median q=.9975 plus baseline floor at the original primary shared threshold and health .05. Require all three relational members; preserve unavailable margins rather than inventing values. This fifth model is the sole candidate under test.

Preserve raw member reconstruction magnitudes, class-normal margins and availability, original member scores, AE/IF component scores, health components and availability, decision rank and exact window identity. Normalize acquisition semantics independently from ML scaling. Verify per-run metadata/scenario nominal cadence; retain observed gaps separately. Never derive expectations from labels or future gaps. Include every original event interval, including ones with no surviving observation/windows. Persist verified per-run outputs so interrupted evaluation can resume only under identical code, source and model hashes.

## Metrics and acceptance

Mixed confirmation: report PR-AUC, precision, recall, F1, ROC-AUC, false windows/asset-day, event/family detection, exact lost-event identities, time to first detection/latency and unavailable windows. Candidate must retain every v0.4-detected nonphysical event and every primary-corrected physical event, total detections >= primary corrected, false windows <= primary corrected, precision >=.5 and event detection >=.5.

Normal confirmation: report per-class and overall exposure, false windows and alarm episodes/day, fraction of runs alerted, profile/cadence/regime-exposure strata and component availability. Candidate must have no more overall false windows than primary corrected and <=10/asset-day in every class, with the minimum exposure reached in all classes. Source/inference failures are explicit failures, never silently removed from denominators.

Pair comparisons by entire run with 2,000 bootstrap resamples, seed 20261005. Report event/physical detection deltas, PR-AUC and false-window-rate intervals against primary corrected. A confidence interval including zero is inconclusive evidence of improvement, even if the point-estimate acceptance gate passes. Zero observed false alarms do not establish zero population rate.

Both populations must pass for exploratory candidate confirmation. Do not relax the gate or choose another member/threshold after seeing test. Preserve any failure and use a new development iteration/new test for a later candidate. Passing remains limited to this simulator population; public packaging, streaming, thermal coverage and licensed real-industrial validation are still separate v1.0 work. No automatic release is authorized by a green CI or this study alone.

## Selection and interpretation boundaries

All prior mixed sources through 910551/2/3 and normal 920621 are consumed test; exclude their run seeds in addition to the existing inventory. Historical development and normal-development 920611 remain development only. No new test results may change the selected median, quantile, anchors, thresholds or members.

Development PR-AUC regressed .717892 to .708312 despite improved threshold decisions. Report this tradeoff alongside the new paired PR-AUC interval. The unchanged exploratory gate concerns thresholded detection and false alerts; passing it does not establish a ranking improvement or general industrial validity. Record all lost primary-corrected event identities, including families outside the required integrity/physical subsets.
