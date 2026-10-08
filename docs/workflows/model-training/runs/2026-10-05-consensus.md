# Fixed majority plus telemetry health

Protocol declared at `df849ed`; executed at clean `30ff484`. Three original seeds, original thresholds, fixed majority of their normalized decision margins, plus corrected health >=.05 with repetition disabled. No fit, recalibration or threshold search. Unanimity/union are diagnostic comparators, not selectable alternatives. See [protocol](../../../RELATIONAL_CONSENSUS_PROTOCOL.md).

Before scoring, the script verifies model/reference/cache/source hashes and exact window identities. Fresh CUDA inference exactly matches all six saved original member score tables; JSON health reference reload matches both corrected development score tables. 98 local tests pass. Existing component scores remain separately available; the ensemble rank is not a probability.

Data: historical mixed development (1,965 windows) and new normal seed 920611 development (7,864 windows, 131.067 asset-hours). Source: [OT Irregularity Lab](https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab), Mysthrala Kurogane Defense Labs, CC BY 4.0. No consumed test/challenge data used.

## Results

| Decision | Events /124 | Historical false windows | Normal false windows | Lost baseline integrity | Lost primary corrected physical | Eligible |
|---|---:|---:|---:|---:|---:|---|
| Primary corrected control | 86 | 4 | 25 | 0 | 0 | Control |
| Fixed majority | 89 | 2 | 14 | 0 | 0 | Yes, development only |
| Unanimity | 85 | 1 | 6 | 0 | 0 | No |
| Union | 91 | 10 | 65 | 0 | 0 | No |

Majority: PR-AUC **.729560**, precision **.985507**, recall **.541833**, 19/35 physical-label intervals, event detection 71.774%, mean detection latency 49.427 seconds. These are different denominators: high precision does not imply high recall. Normal false windows/day **2.564**, bootstrap CI95% [1.098, 4.218]. Class rates: CNC .732, COMPRESSOR 7.325, CONVEYOR .732, PUMP 1.465. The bootstrap describes resampled development runs and does not remove adaptive selection bias.

Exact losses remain visible: majority loses one `sensor_bias` event detected by primary corrected control (`dev-15::validation-00014/evt-1`, PUMP), and one `bearing_degradation` event detected by secondary seed 20261006 (`dev-07::validation-00008/evt-1`, CNC). It loses no corrected seed-20261007 events. These losses do not violate the declared primary physical/baseline integrity gate, but disallow a claim of preserving every member detection. The ensemble's three members are not three independent replications of ensemble performance.

## Decision and next gate

The fixed majority passes development and is the only candidate selected. It is **not yet independently confirmed or promoted**. Next freeze all three models, original thresholds, health JSON and cadence adapter plus the independent evaluation protocol before generating new mixed and long-normal data. Measure event identities, family performance, false windows/episodes, latency, paired run uncertainty and normal strata. Do not tune on those new test results. Thermal weaknesses, generic CLI packaging, streaming and real industrial generalization remain open toward v1.0.

```powershell
python scripts/research_consensus.py --output NEW_OUTPUT --health-study CORRECTED_HEALTH_STUDY --baseline BASELINE --relationships FROZEN_RELATIONSHIPS --dataset HISTORICAL_TRAIN_VALIDATION --cache VERIFIED_HISTORICAL_CACHE --normal-source NEW_NORMAL_DEVELOPMENT --normal-cache VERIFIED_NORMAL_CACHE --raw-root HISTORICAL_RAW_BATCHES --device cuda:0
```

[All points, exact lost identities, strata and hashes](../../../results/consensus-20261005.json).
