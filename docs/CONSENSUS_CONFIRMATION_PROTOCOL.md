# Independent consensus confirmation protocol

Declared after majority passed development, before new source generation. Freeze the exact three original relational members and thresholds, baseline artifacts, corrected health JSON, health threshold .05, repetition disabled, complete one-minute windows, inference/evaluation code and simulator/suite hashes. No selection, retraining or recalibration on this confirmation.

## New sources

Pinned OT Irregularity Lab commit `718babb7772c3a21f0b87c403f628540cbce58db`, simulator 0.6.0. Data CC BY 4.0, attribution Mysthrala Kurogane Defense Labs, [source](https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab).

- Mixed: suite `suites/training-v0.2.yaml`, 240 generated runs per master seed **910551, 910552, 910553**. Evaluate every manifest `test` run, expected 36 per batch / 108 total. Generation retains the suite's fixed partitions; evaluation never reads train/validation telemetry or uses it for fitting.
- Normal: existing `configs/normal-exposure-v05.yaml`, **180** one-hour runs, master seed **920621**, test only. Require >=168 evaluated asset-hours per class and empty anomaly truth. Preserve the existing six weighted transition profiles, cadence/jitter, ambient/drift and regime exposure distributions. This is many independent one-hour runs, not continuous week-long operation.
- Source directories must not exist when freezing. Check actual per-run seed overlap with source manifests of historical development, normal development and consumed prior evaluations before reading new test telemetry. Any overlap aborts. Existing source manifests may be inspected only for identity/seed metadata.

## Controls and outputs

1. Original v0.4, original threshold.
2. Original primary v0.5 relational, original shared threshold.
3. Primary v0.5 plus corrected health .05, repetition off.
4. Fixed majority of all three original member margins plus the same health channel.

Preserve member scores, AE/IF component scores, health components and availability, decision rank and exact window identity. Normalize acquisition semantics independently from ML scaling. Verify per-run metadata/scenario nominal cadence; retain observed gaps separately. Never derive expectations from labels or future gaps. Include every original event interval, including ones with no surviving observation/windows. Persist verified per-run outputs so interrupted evaluation can resume only under identical code, source and model hashes.

## Metrics and acceptance

Mixed confirmation: report PR-AUC, precision, recall, F1, ROC-AUC, false windows/asset-day, event/family detection, exact lost-event identities, time to first detection/latency and unavailable windows. Candidate must retain every v0.4-detected nonphysical event and every primary-corrected physical event, total detections >= primary corrected, false windows <= primary corrected, precision >=.5 and event detection >=.5.

Normal confirmation: report per-class and overall exposure, false windows and alarm episodes/day, fraction of runs alerted, profile/cadence/regime-exposure strata and component availability. Candidate must have no more overall false windows than primary corrected and <=10/asset-day in every class, with the minimum exposure reached in all classes. Source/inference failures are explicit failures, never silently removed from denominators.

Pair comparisons by entire run with 2,000 bootstrap resamples, seed 20261005. Report event/physical detection deltas, PR-AUC and false-window-rate intervals against primary corrected. A confidence interval including zero is inconclusive evidence of improvement, even if the point-estimate acceptance gate passes. Zero observed false alarms do not establish zero population rate.

Both populations must pass for exploratory candidate confirmation. Do not relax the gate or choose another member/threshold after seeing test. Preserve any failure and use a new development iteration/new test for a later candidate. Passing remains limited to this simulator population; public packaging, streaming, thermal coverage and licensed real-industrial validation are still separate v1.0 work. No automatic release is authorized by a green CI or this study alone.
