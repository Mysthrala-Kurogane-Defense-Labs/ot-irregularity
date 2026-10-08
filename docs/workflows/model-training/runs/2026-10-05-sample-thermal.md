# Sample-first thermal development and tail normalization

## Cadence correction

The first implementation used the baseline schema's pooled interval (about 507 ms). Historical validation has both 500 ms (261 runs/assets) and 1,000 ms (276) declared cadences. Only 916/1,965 development windows remained available. This was an input-contract error, not a meaningful test of the representation. The attempt is preserved at code commit `dc87395` and in the numeric evidence.

The corrected experiment uses the first declared interval of each run/asset for causal gap and coverage checks; later rows cannot retroactively change that cadence. Exact timestamps align roles, without interpolation or future data. Residuals require good finite current/previous measurements, gap <=1.5 initial declared intervals, and minute windows with >=90% expected valid pairs and at least 20 pairs. Availability becomes **1,917/1,965 windows**; baseline covers all 1,965. This remains a full-run batch experiment, not a stateful streaming API. Asynchronous roles and cadence changes need explicit future handling.

## Representation and first screen

Per class/thermal role, normal-only Ridge(alpha=1) predicts measured temperature rate `(T[t]-T[t-1])/dt` from previous temperature and current operating drivers. Current temperature never enters its own predictors. RobustScaler fits on normal fit runs only. Two variants exclude/include the first fully usable temperature in the run, retained causally as initial context. That value is not a measured ambient temperature and may depend on cold/warm startup.

Two minute summaries: absolute time-weighted mean residual rate (persistent deviation) and q95 absolute residual rate (excursion). Empirical normal-CDF calibration, quantiles .99/.995/.9975, and union with the unchanged baseline decision were predeclared. Every resolvable candidate preserved all baseline alert windows but failed the false-window gate. Best cooling count was 9/10 with initial context and q95, versus baseline 2/10; false windows rose from 4 to 9 at q=.995. Without initial context the maximum was 3/10. Thus initial context materially affects development results.

## Adaptive tail stage, disclosed before new test

CDF saturation maps every residual above a calibration maximum to 1; a strict cutoff above that value disables alerts. A separate adaptive development stage retained magnitude using `s = r / (r + a)`, where `a` is that thermal role/statistic's normal calibration q99 (floor 1e-9 degC/s). Maximum score across available thermal roles; pooled normal thresholds .99/.995/.9975/.999. This score is a bounded relative magnitude, **not a percentile or probability of a fault**. No new model fit or test access.

One of 16 operating points passes the unchanged development gate:

| Metric on identical development windows | v0.4 baseline | Selected thermal union |
|---|---:|---:|
| Detected events | 53/124 | 61/124 |
| Cooling events | 2/10 | 5/10 |
| False windows | 4 | 4 |
| Lost baseline alert windows | — | 0 |

Selected: initial context enabled, absolute time-weighted mean residual, normal threshold quantile .999. Other candidates remain in [numeric evidence](../../../results/sample-thermal-20261005.json), including stronger cooling counts with unacceptable false alarms. This is adaptive development evidence and may overfit repeated selection; it needs a frozen new holdout. It is a thermal component, not an automatic replacement for the relational v0.5 candidate, whose physical-family strengths must remain visible in comparison.

## Freeze and evaluation protocol

`freeze_thermal_candidate.py` requires clean committed code, checks baseline and sample-model hashes against development evidence, rejects pre-existing holdout directories, and freezes the selected point, source hashes, anchors and declared new seeds before generation. Controls: unchanged v0.4 with complete-tail policy and original shared-threshold relational v0.5 primary. New seeds reserved: 910541/42/43, 240 runs per batch using frozen Lab `718babb`, suite `training-v0.2.yaml`; evaluate test partitions only, no challenge. These are new seeds of the same simulator suite, not industrial field validation.

Gate: no baseline events lost, cooling improvement, precision >=50%, all-event detection >=50%, false windows/asset-day <=10. Report every family, coverage, latency and paired run confidence intervals, including failures. Do not promote this component as the full v0.5 model merely because it improves cooling. A joint relational/thermal model requires its own development selection and independent validation.

## Validation and attribution

74 local tests passed after correcting an Int32/Int64 mismatch in the new mixed-cadence test fixture. Tests cover causal context, target exclusion, missing/bad sample history, duplicate identities, serialization, time-weighted aggregation, mixed cadences and monotonic tail scores. Earlier empirical-CDF implementation: 72 tests; its pooled-cadence results are explicitly superseded.

Data: OT Irregularity Lab, Mysthrala Kurogane Defense Labs, CC BY 4.0. Historical source version caveat remains as documented in the [paired thermal audit](2026-10-05-thermal.md). Data/models stay outside Git; results and provenance are public.

## Independent holdout: small component gain, no overall model promotion

The candidate was frozen at clean code commit `838be89`, before generation of seeds 910541/42/43. All three batches completed (240 runs each); only their 108 test runs were evaluated. Same 1,512 complete minute windows (25.2 asset-hours), 105 original events, 34 physical events including seven cooling events. The thermal supplement is available on 1,461 windows; the baseline remains active on all windows. No baseline event detection was lost.

| Metric on the same fresh test | v0.4 complete | Thermal union | Original v0.5 relational control |
|---|---:|---:|---:|
| Events detected | 60/105 | 62/105 | 67/105 |
| Physical events | 6/34 | 8/34 | 18/34 |
| Cooling | 1/7 | 2/7 | 1/7 |
| False windows | 7 | 8 | 7 |
| False windows/asset-day | 6.67 | 7.62 | 6.67 |
| Precision | 91.57% | 90.80% | 92.71% |
| PR-AUC | .6216 | .6102 | .6949 |
| Mean detection latency, seconds | 46.48 | 46.77 | 49.18 |

Thermal union adds one cooling and one overload event. It passes the previously frozen exploratory gate, but the improvement is small and uncertain: paired run-bootstrap IC95% for event-rate change [0, .0516], cooling-rate change [0, .5], PR-AUC change [-.0364, .0148], and false-window/day change [0, 3.2146]. All-event/cooling intervals use 2,000 resamples; window metrics use 500. Precision and ranking point estimates fall slightly, and the original relational model remains better overall on this sample. **Do not replace v0.5 with this thermal component or claim statistically established thermal improvement.**

For the union's PR-AUC, the continuous ranking is the maximum relative decision margin `m=max(baseline/baseline_threshold, thermal/thermal_threshold)`, mapped to `m/(1+m)` with decision threshold .5. An assertion verifies identical decisions to the frozen Boolean union. This ranking is not a calibrated probability. Event denominators use every original interval, not one event ID per window.

The relational control still loses integrity detections compared with baseline: communication 7 vs 8/9, missing telemetry 5 vs 10/11, quality 1 vs 3/4, single-signal loss 4 vs 5/5. Its physical gains do not erase those losses. The thermal union preserves baseline detections but has not yet been combined with the relational supplement. The next joint decision must be selected on development, then independently tested; do not construct and promote it from these test outcomes.

[Frozen manifest, source hashes and complete results](../../../results/thermal-holdout-20261005.json). `validate_thermal_candidate.py` checks artifact hashes and rejects runtime source changes since freeze, evaluates the selected component plus the two frozen controls, and refuses a second evaluation once results exist. No threshold or model was changed after viewing this test. Fifteen test seed batches are now consumed. Long normal exposure and real-industrial generalization remain unvalidated.
