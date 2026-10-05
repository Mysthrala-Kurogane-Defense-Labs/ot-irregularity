# Configured cadence correction and health confirmation

The correction was declared at `0ede90a`, then executed at clean `1384155`. The [first failed study](2026-10-05-telemetry-health.md) remains unchanged. The corrected adapter verifies run metadata, canonical scenario and telemetry hashes; metadata/scenario nominal periods must agree. It matches all selected historical observations to their raw sources (421 train and 270 validation runs) before attaching `declared_sampling_interval_ms`. Observed row gaps remain separate. No consumed test/challenge data were read.

Source remains [OT Irregularity Lab](https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab), Mysthrala Kurogane Defense Labs, CC BY 4.0. The new normal-development source is seed 920611. Old version metadata caveats remain recorded in the thermal study; this correction verifies the actual source bytes and does not rewrite their provenance.

## Verification

90 local tests pass without warnings. Added cases cover observed jitter, missing samples, unknown declarations, observed interval preservation, attempted overwrite, metadata hash mismatch, metadata/scenario disagreement and forbidden test partition. Existing true-cadence-change and future-observation tests still pass. Reloaded JSON reference scores exactly reproduce all four corrected health score tables. No AE/IF retraining or changed frozen model features.

Sampling is now available in **1,965/1,965 historical** and **7,864/7,864 new normal** windows. Repetition is available in 1,964 historical and all normal windows; availability at window level means at least one role has a reference, not that every role is available. Per-role reasons remain in the persisted detail tables. Quality is available in all windows.

## Fixed primary screen

| Repetition | Threshold | Events /124 | Historical false windows | Lost baseline integrity events | Normal false windows |
|---|---:|---:|---:|---:|---:|
| Original v0.5 | — | 71 | 4 | 11 | 25 |
| Off | .05 | 86 | 4 | 0 | 25 |
| Off | .10 | 84 | 4 | 0 | 25 |
| Off | .20 | 81 | 4 | 2 | 25 |
| On | .05 | 86 | 6 | 0 | 38 |
| On | .10 | 84 | 5 | 0 | 29 |
| On | .20 | 81 | 4 | 2 | 27 |

The declared primary selection is repetition **off**, threshold **.05**. Historical PR-AUC rises from .684395 to .717892, precision .971014 and recall .533865. It detects 86/124 intervals and retains every baseline-detected nonphysical interval and every primary relational physical interval. Physical-label detections are 16/35 versus 15/35; this does not establish a physical fault diagnosis or causal benefit because grades/availability can co-occur with physical labels.

New normal exposure is 131.067 asset-hours across 33 development runs. False windows remain 25 (4.578/asset-day). Per-class rates: CNC 2.197, COMPRESSOR 2.930, CONVEYOR 6.592, PUMP 6.592. Repetition adds no event detections at the tested thresholds and adds false alarms; it remains disabled in the selected component.

## Secondary confirmation: full candidate not accepted

| Original AE seed | Events before → after | Historical false before → after | Normal false before → after | Compressor false windows/day | Gate |
|---|---|---|---|---:|---|
| 20261005 | 71 → 86 | 4 → 4 | 25 → 25 | 2.930 | Primary passes |
| 20261006 | 75 → 90 | 4 → 4 | 24 → 24 | 10.254 | Fails per-class limit |
| 20261007 | 74 → 89 | 5 → 5 | 36 → 36 | 22.706 | Fails historical and per-class limits |

All three retain the required baseline integrity and original physical detections after the supplement. The health channel adds **zero false windows** in both evaluated development populations for all three seeds. But the full decision still violates the predeclared limits on secondary seeds, already violated by those frozen relational controls. Therefore `confirmed_development=false`; no new holdout, promotion or release. Do not select a favorable seed after seeing these results or loosen the gate retrospectively.

The next issue is relational decision stability between seeds, especially normal compressor windows. Investigate their score/reference distributions and false-window overlap before choosing another representation or ensemble. Keep the validated semantic correction and useful health component, preserve individual AE/IF/relational/health scores, and declare the next selection rule before evaluating it. Industrial generalization and streaming remain unproved.

Descriptive overlap on the same normal development windows: compressors have 4/14/31 false windows by seed, 36 distinct windows overall; **26 occur in only one seed and three in all three**. This supports investigating a consensus hypothesis but says nothing yet about lost real-event detections. Do not interpret the overlap as an evaluated ensemble or independent validation. [Overlap counts and score-file hashes](../../../results/relational-seed-overlap-20261005.json).

Reproduce with the command in the first study, a **new** output directory and corrected source code. [All points, controls, strata, availability and verification hashes](../../../results/telemetry-health-20261005-r2.json).
