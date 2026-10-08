# Class-calibrated reconstruction ensemble development protocol

Declared after the fixed-vote confirmation failed event retention. All confirmation sources through mixed seeds 910551/2/3 and normal 920621 are consumed test and excluded here. Historical train/validation and normal-development seed 920611 are the only permitted development sources.

## Hypothesis and fixed screen

Voting discards reconstruction magnitude, and a shared threshold can distribute false alarms unevenly across classes. Keep all three existing relational models and baseline weights frozen. Extract each member's raw top-two normalized squared reconstruction error before its empirical-CDF transform. Require valid relational inputs from all three members; unavailable relational scores remain null with an availability flag, and other channels still operate.

Use explicitly normal, whole-run calibration windows only: validation run hash modulo 2 ==0 from historical and new normal-development sources. No model/scaler/Ridge fitting changes. Per class and member, divide raw error by its normal-calibration q99, floored at 1e-9. Compare **mean** and **median** across those three magnitudes. For each aggregate, class thresholds are normal-calibration quantiles **.99/.995/.9975**, `method=higher`, moved one floating-point step upward. Require >=400 available normal calibration windows/class, positive finite thresholds and all four trained classes. New long runs contribute more windows; no equal-source/run weighting claim.

Six points total; no further search. Decision margin is `max(baseline_score/original_primary_shared_threshold, class_aggregate/class_threshold, corrected_health/.05)`. Repetition remains off. Rank `margin/(1+margin)` at .5 is a decision ranking, not probability. Preserve raw member errors, anchors, thresholds, availability and component scores. The baseline floor deliberately matches the original primary shared threshold, not the looser standalone v0.4 threshold. Existing frozen controls are reported alongside every candidate.

## Gate and selection

Keep the previous development floor: every baseline-detected nonphysical event and every primary-corrected physical event retained by exact identity; total events >=86/124, false historical windows <=4, precision >=.5. New normal-development false windows <=25 overall and <=10/asset-day in every class. Report all event losses, per-class/stratum false rates, family detection, PR-AUC, recall and latency. Among eligible points choose most events, then fewer normal false windows, then fewer historical false windows, prefer median and higher quantile. This is one ensemble of three seeds, not independent ensemble replication.

If none passes, retain results and generate no holdout. If one passes, freeze the full selected reference/model/code bundle and use new independent confirmation data before promotion. Do not reuse the prior confirmation for fitting, threshold choice or reevaluation. Earlier failed gates are unchanged. Industrial generalization, thermal coverage and operational readiness are not established by this development screen.
