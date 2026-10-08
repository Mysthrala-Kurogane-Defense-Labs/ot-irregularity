# Fixed-model normal exposure protocol

Declared before generation. This measures false alarms and availability; it cannot measure recall, PR-AUC or event latency without anomalous events.

## Source and population

- OT Irregularity Lab, commit `718babb7772c3a21f0b87c403f628540cbce58db`, simulator 0.6.0. Generated data: CC BY 4.0; attribution: Mysthrala Kurogane Defense Labs, [source](https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab).
- `configs/normal-exposure-v05.yaml`: 180 independent one-hour runs, master seed **920601**, all four classes in every run, no injected anomaly, all runs assigned to test. Reject nonempty event truth.
- Sampling 500/1,000 ms, jitter 0–30 ms, ambient 15–32 °C, drift ±4 °C, exposed/hidden regime. Generic process parameters unchanged.
- Six existing production profiles retain their weights; each shift pattern repeats eight times. This avoids stretching four/five operating phases across an hour. Lab still assigns 8% warmup and 7% cooldown; interior phases are about 77–128 seconds. This is a new duration/transition distribution, not an exact population match to the shorter development runs.
- Target: >=168 evaluated asset-hours per class after complete-tail exclusions. Approximately 177 hours/class are expected. Report actual exposure. Aggregated one-hour runs do not demonstrate continuous week-long stability.

## Frozen controls

1. v0.4 baseline, original thresholds, complete-tail windows.
2. Original primary v0.5 shared-threshold relational model.
3. Baseline OR selected sample-first thermal component.
4. Baseline OR relational magnitude q=.9975 (seed 20261005) OR selected thermal component. This is the **rejected development diagnostic point**, chosen for its primary-seed detection gain with six false windows. It is not a promoted candidate.

Before generation persist code/suite hashes, model hashes, every threshold and normal-reference anchor, source commit, master seed and timestamp. Recheck hashes when evaluating. No refit, recalibration or operating-point selection on this exposure dataset. Preserve per-run predictions and source hashes; process runs individually to bound memory. Resume only verified completed runs under the identical frozen protocol.

## Metrics and interpretation

- Per class and overall: evaluated asset-hours, false windows per asset-day, alarm episodes per asset-day, fraction of runs with an alarm, component availability.
- Episode: consecutive alerting minute windows for the same run/asset; end/start must touch. A normal window or missing window ends an episode. Never join runs or different assets. Report count and duration separately from false windows.
- Cluster bootstrap by run, 2,000 resamples, seed 20261005; ratios use resampled counts divided by resampled exposure. Report paired rate differences between controls. If all observed counts are zero, label the degenerate bootstrap interval as insufficient evidence for a zero population rate.
- Report strata for cadence, regime exposure and production profile; use truth regimes only in diagnostics, never as hidden inference inputs.
- The existing <=10 false windows/asset-day target is exploratory. This benchmark can reveal failure of that target; passing it alone cannot promote the joint model, prove industrial suitability, or repair the failed development gate.

Record absent/invalid runs and failed inference explicitly. Do not reduce the denominator population by silently dropping failures. A later improvement informed by this benchmark requires a new independent dataset for confirmation.
