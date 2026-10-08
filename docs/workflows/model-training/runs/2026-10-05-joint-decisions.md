# Joint integrity, relational and thermal decisions

## Protocol and result

Development only, no new fitting or test access. The baseline decision remains fixed. Relational AE top-two normalized reconstruction error uses `r/(r+a)`, with `a` its existing normal-reference q99, floor 1e-9. This retains magnitude beyond the empirical CDF maximum; it is not a probability. Four normal calibration quantiles (.99, .995, .9975, .999), thermal enabled/disabled, and three previously trained AE seeds give 24 operating points. Thermal models, anchors and the previously selected development threshold remain fixed.

Gate declared before execution: preserve all baseline alerts, match or exceed original relational total/physical detections and thermal-union cooling detections, precision >=.5, and no more false windows than baseline. Only seed 20261005 is eligible for selection; the other seeds assess sensitivity.

**All 24 points fail the false-window gate. No candidate is selected or promoted.**

| Development, same 1,965 windows | Baseline | Original relational | Joint primary q=.9975 |
|---|---:|---:|---:|
| Events detected /124 | 53 | 71 | 87 |
| Physical events /35 | 4 | 15 | 18 |
| Cooling /10 | 2 | 2 | 5 |
| False windows | 4 | 4 | 6 |
| Baseline alert windows lost | 0 | Not constrained | 0 |

The joint column is a diagnostic rejected point, not an independently validated model. Other seeds at q=.999 reach 86/88 events and 18 physical events with five false windows, still above the unchanged budget. Do not choose a secondary seed after observing its scores. Adaptive development comparisons do not establish generalization.

The two additional false windows in the primary q=.9975 point belong to PUMP-01 in `dev-03::validation-00001` and `dev-10::validation-00001`, both 00:05–00:06 UTC, with regime hidden from inference. Original truth contains no pump anomaly; the second run's sensor drift belongs to a conveyor. A transient operating-context mismatch is a hypothesis to investigate, not a reason to relabel these false positives.

## Reproduction and evidence

`scripts/research_joint_decisions.py --baseline BASELINE --relationships RELATIONAL_FREEZE --thermal THERMAL_DEVELOPMENT --dataset DEVELOPMENT --cache VERIFIED_CACHE --raw-root ORIGINAL_BATCHES --output NEW_OUTPUT`

The experiment verifies baseline/relational artifact hashes, source-aware window cache and thermal score row identities. All inputs are the existing normal training/calibration/development partitions, never the fifteen consumed test batches. [Numeric evidence](../../../results/joint-decisions-20261005.json) preserves every point and truth/model hashes. Raw datasets and weights remain outside Git.

Next: the [normal exposure protocol](../../../NORMAL_EXPOSURE_PROTOCOL.md) measures fixed-model false alarms across longer normal runs. It does not relax this rejected development gate or retune thresholds.

## Verification

75 local tests pass, including a gate regression test for every declared detection floor and the unchanged false-window limit. Repeating the complete development evaluation after extracting the gate function produces exactly equal results JSON on CUDA. No weights or thresholds changed. Remote CI is reported separately per commit.
