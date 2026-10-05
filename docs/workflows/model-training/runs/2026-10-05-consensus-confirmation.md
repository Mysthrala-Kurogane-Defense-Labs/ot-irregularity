# Frozen independent consensus confirmation

Protocol committed at `2cddaf0`; executable freeze at clean `025b553`, before any new generation. 101 local tests pass, including source population/seed leakage, exact event-retention gates, minimum exposure and run bootstrap with an event lacking windows. Tests establish these software contracts; full evaluator behavior and independent performance require the execution below.

The [protocol](../../../CONSENSUS_CONFIRMATION_PROTOCOL.md) fixes four controls, all original model thresholds, corrected health .05 with repetition off and the majority rule. Code, models, references, suites and pinned Lab commit are hashed. 32 prior source manifests provide 5,760 excluded run seeds. New master seeds: 910551/2/3 (240 generated runs each, 36 reserved test runs each) and 920621 (180 normal one-hour runs, test only). Generated data: OT Irregularity Lab, Mysthrala Kurogane Defense Labs, CC BY 4.0.

[Public freeze record](../../../results/consensus-confirmation-freeze-20261005.json) retains the full manifest hash and a hash/count of the excluded seed inventory. Raw telemetry and model weights remain outside Git. Evaluation checks every new run seed against the full local inventory before reading test telemetry, verifies source files and persists run predictions with resumable provenance checks. No data/threshold reselection or refit is permitted.

## Reproduction

Use the frozen checkout **025b553** for exact evaluation/resume. Later progress-writer maintenance intentionally changes the runtime hash and must not be silently substituted into this frozen experiment.

```powershell
python scripts/freeze_consensus.py --output NEW_FREEZE --lab PINNED_LAB --source-root NEW_SOURCE_ROOT --study CONSENSUS_DEVELOPMENT --baseline BASELINE --relationships FROZEN_RELATIONSHIPS --health-reference CORRECTED_HEALTH_JSON --prior-manifests PRIOR_RAW_MANIFESTS
# In pinned Lab checkout, repeat for 910552 and 910553:
python -m ot_lab.cli dataset create --suite suites/training-v0.2.yaml --runs 240 --seed 910551 --workers 4 --output NEW_SOURCE_ROOT/holdout-910551
python -m ot_lab.cli dataset create --suite DETECTOR_REPO/configs/normal-exposure-v05.yaml --runs 180 --seed 920621 --workers 4 --output NEW_SOURCE_ROOT/normal-exposure-920621
# In detector checkout, after all four generation commands complete:
python scripts/evaluate_consensus.py --output NEW_FREEZE --baseline BASELINE --relationships FROZEN_RELATIONSHIPS --health-reference CORRECTED_HEALTH_JSON --sources MIXED_910551 MIXED_910552 MIXED_910553 NORMAL_920621 --device cuda:0
```

## Execution status

All four generation commands completed successfully: 720 mixed generated runs, 108 reserved mixed test runs, plus 180 normal one-hour test runs. All **288** reserved runs were evaluated, with no source failures or exclusions. Mixed population: **1,780 complete windows, 101 original events, 17 physical-label events**. Normal population: **42,880 windows, 714.667 asset-hours overall, 178.667 hours in each class**. Health, quality and sampling scores are available at window level throughout both populations; this means at least one supported role, not universal role availability.

## Independent results

| Control | PR-AUC | Precision | Recall | Events /101 | Physical /17 | Mixed false windows |
|---|---:|---:|---:|---:|---:|---:|
| v0.4 | .577178 | .932432 | .353846 | 47 | 2 | 5 |
| Original v0.5 primary | .634761 | .936842 | .456410 | 59 | 8 | 6 |
| Primary plus corrected health | .673145 | .948276 | .564103 | 73 | 9 | 6 |
| Fixed majority plus health | .678701 | .981651 | .548718 | 70 | 9 | 2 |

Majority preserves every baseline-detected nonphysical event and every primary-corrected physical event. It nevertheless loses three other primary-corrected detections and gains no replacement detections in this sample:

- `seed-910551::test-00020/evt-2`, PUMP, `maintenance_activity`.
- `seed-910551::test-00023/evt-2`, PUMP, `regime_mismatch`.
- `seed-910553::test-00027/evt-1`, PUMP, `missing_telemetry`.

These are simulator labels, not diagnoses or grounds for retroactively removing events from the denominator. Mixed acceptance **fails** on total event retention, 70 < 73. Mean detection latency is 48.929 seconds for majority and 49.658 for primary-health, each conditional on its detected events; different detected sets prevent interpreting this as an unqualified latency improvement.

| Normal control | False windows | False windows/asset-day |
|---|---:|---:|
| v0.4 | 129 | 4.332 |
| Original v0.5 / primary-health | 153 | 5.138 |
| Majority-health | 95 | 3.190 |

Every observed normal false window is a separate episode here. Majority rates by class: CNC 1.075, COMPRESSOR 8.328, CONVEYOR 1.881, PUMP 1.478. The declared normal gate **passes** (all point rates <=10/day and required exposure met). However compressors worsen versus primary-health **4.970→8.328/day**, while other classes improve. Do not hide that tradeoff behind the overall reduction or claim that the upper confidence bound is below the target.

Paired run bootstrap, 2,000 resamples:

- Event detection delta CI95% **[-6.668, 0] percentage points**.
- Physical detection delta **[0, 0]**, same detected physical identities in this sample.
- PR-AUC delta **[-.006330, .020832]**: improvement inconclusive.
- Mixed false-window-rate delta **[-6.801, -.756]/asset-day**.
- Normal false-window-rate delta **[-2.819, -1.142]/asset-day**.

Thus false-alarm reduction is supported in this simulator sample, with a real event-retention cost. **Overall confirmation fails; no promotion or release.** Both new populations are now consumed test data and excluded from future fitting/calibration/selection.

## Verification and continuation

All 288 prediction hashes and their freeze references were audited. A resume attempt stopped with Windows `PermissionError` replacing `progress.json`; the exact frozen-code retry succeeded. Results were **byte identical**, and all 288 record hashes remained unchanged. The frozen execution is retained at 025b553. Subsequent writer maintenance adds bounded `PermissionError` retries and unique temporary files, tested for transient locks, persistent locks preserving previous state, unrelated errors and a real Windows reader handle; the final suite passes 105 tests without warnings; it does not change these results.

The health control's 59→73 event gain without added false windows is useful component evidence, but this protocol did not authorize choosing a different full candidate after test. Next development should investigate disagreement/event retention and class calibration, retaining these negative and positive findings. Thermal performance, real industrial generalization, streaming and public bundle/CLI integration remain open. [Full metrics, exact losses, strata, intervals and verification](../../../results/consensus-confirmation-20261005.json).
