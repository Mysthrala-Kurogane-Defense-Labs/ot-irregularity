# Telemetry health development protocol

## Motivation and scope

Development diagnosis before this experiment finds eleven baseline-detected nonphysical events missed by the original primary relational candidate: six sample/signal/asset-loss intervals, two quality-degradation intervals and three sensor-stuck intervals. The coverage ablation also missed observed 19.17% bad quality. Do not require every kind of telemetry evidence to exceed a global reconstruction-percentile threshold.

This experiment adds a separately explained **observation-health channel** to the unchanged original relational v0.5 decision. Preserve baseline AE, IF and relational scores. Do not infer fault causes. Compare quality/availability alone with an explicit repetition ablation. No network retraining, bigger architecture or thermal model change.

## Inputs and causal extraction

Use normalized raw observations plus complete nonoverlapping minute windows and the trained applicable-role schema. Keep roles distinct and units consistent. Extract each role's observed bad/uncertain fractions, finite-value sample count, expected count, coverage ratio, mean value and exact repetition fraction. Missing/unknown quality is unavailable, never fabricated good/bad. Repetition uses consecutive finite good-quality pairs with a positive gap <=1.5 declared sample intervals, with at least 20 usable pairs.

The expected interval is the first positive declared sampling interval already observed for that run/asset/role. Allow the first observation within 1.5 intervals of window start; otherwise a window before sampling begins has unavailable coverage. Do not use future observations to establish earlier cadence. A later declared cadence change makes affected/subsequent availability and repetition unavailable until a new run/segment is supplied. Emit that reason; do not silently borrow pooled training cadence. Finite-value absence is not proof of communication failure.

Quality ratios describe received grades; no samples means unavailable quality. Coverage counts finite measurements regardless of grade and remains separate from quality. Retain unclipped coverage for explanation, though the score below only detects shortfall. Complete-file windows cannot detect silence after the final observation; streaming watermark support remains separate work.

## Normal references and scores

Historical normal fit/calibration windows and the new seed-920611 normal-development source retain their declared whole-run splits. No consumed test/challenge is read for fitting/calibration/selection. Fit repetition context boundaries only on fit windows: unique quartiles of the same role's finite mean value. Calibrate repeat upper bounds within each mean-value bin using normal calibration windows; fewer than 20 available calibration windows in a bin falls back to the role's global bound and records that fallback. This conservative conditioning separates common zero plateaus from nonzero operation where data support it. A signal stuck at a legitimate normal level can still be missed; other-signal or temporal context is not claimed.

Per class/role, normal-calibration q=.995 gives upper bad/uncertain/repetition bounds; q=.005 gives lower coverage. Require >=20 available calibration windows for a reference. Unit mismatch is rejected. Persist reference counts, bin cuts, bounds and fallback flags as JSON.

- Upper-tail excess: `max(0, observed - upper)/(1-upper)`; bound 1 has no possible upward excess.
- Coverage shortfall: `max(0, lower-coverage)/lower`; lower=0 is uninformative/unavailable.
- Clamp scores to [0,1]. Max across roles for each component; quality is max of bad/uncertain excess. Final health is max of available components. Entirely unavailable components remain null with availability reasons, not evidence of normality.

These are bounded relative deviations, not percentiles or probabilities. Observations: `quality_deviation`, `sampling_degradation`, `signal_loss` for zero finite samples with a supported reference, and `signal_repetition` for excess identical pairs. Repeated values do not diagnose a failed sensor.

## Fixed comparison and gate

Six primary operating points: repetition disabled/enabled × health thresholds .05/.10/.20. Decision: frozen original v0.5 alert OR health >= threshold. Primary AE seed 20261005; other already-trained seeds 20261006/7 confirm the selected health point without reselection. No new AE fit. Selection uses historical development and seed-920611 normal development only; repeated development use is adaptive research and requires new confirmation.

Eligibility:

1. Preserve every baseline-detected nonphysical event by exact identity and every original v0.5 physical event; total detections >= original v0.5.
2. Historical false windows <= v0.4 baseline, precision >=.5.
3. New normal-development false windows <= original v0.5 overall and <=10/asset-day per class. Report all profiles/cadences, component availability and episode counts.

Among eligible primary points: most events, then fewer new normal false windows, prefer repetition disabled, then the higher threshold. Keep all rejected points. If none passes, record remaining event identities/false-alarm strata and generate no new holdout. An eligible choice must be frozen and evaluated on new mixed-event and long-normal data before promotion. This changes no earlier rejection and does not establish v1.0 readiness.

## Declared correction r2: configured versus observed intervals

The first execution at `896f383` exposed an adapter error: Lab row intervals contain observed elapsed time, including jitter and dropouts, rather than configured cadence. Preserve its rejected results at `a4f913b`. Before the corrected evaluation, require explicit `declared_sampling_interval_ms` for this channel; absent declarations make sampling/repetition unavailable. Preserve the source's observed interval separately as `observed_sampling_interval_ms` and leave the existing feature pipeline unchanged.

For Lab runs, verified run metadata and canonical scenario must agree on positive nominal cadence. Match metadata identity and hash, scenario hash and telemetry hash to the source manifest. Match normalized historical observations to those exact raw runs before joining cadence by namespaced run ID. Read only the train/validation runs already selected; no test/challenge telemetry or anomaly labels determine cadence. Retain source hashes. The cadence is configuration known before acquisition, but extraction retains its conservative first-observation availability rule. Generic sources must supply their own verified declared-cadence contract; do not guess from gaps.

Repeat the same six primary points, reference quantiles, split, frozen original models and exact-event/false-alarm gates in a new output directory. No new hyperparameters or threshold selection are introduced. Add jitter, dropout, unknown declaration, true declared change and provenance-mismatch tests before evaluation. Confirm any selected point with the two original secondary seeds, without reselection. Only then freeze a candidate for new independent mixed and normal data.
