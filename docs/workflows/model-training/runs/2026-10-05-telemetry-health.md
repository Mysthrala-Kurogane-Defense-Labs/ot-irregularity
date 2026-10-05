# Separate telemetry health development study

## Scope and declaration

Protocol committed at `71860f1`; implementation at clean `896f383`. Keep the three original relational models frozen, including their thresholds. Supplement their decision with separately explained quality, sample availability and optionally exact value repetition. No neural training and no thermal change.

Use historical normal fit/calibration partitions and seed 920611 normal development (180 one-hour runs; 120 train, 60 validation). Historical anomaly development is used only for selection/evaluation. Consumed test and challenge data are excluded. Source: [OT Irregularity Lab](https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab), Mysthrala Kurogane Defense Labs, CC BY 4.0. Synthetic results do not establish industrial generalization.

Original development diagnosis identified eleven baseline nonphysical detections lost by original v0.5: six loss intervals, two quality intervals and three sensor-stuck ground-truth intervals. These are evaluation labels, not asserted sensor diagnoses.

## Reproduce

```powershell
python scripts/research_telemetry_health.py --output STUDY_OUTPUT --baseline BASELINE --relationships FROZEN_RELATIONSHIPS --dataset HISTORICAL_TRAIN_VALIDATION --cache VERIFIED_HISTORICAL_CACHE --raw-root HISTORICAL_RAW_BATCHES --normal-source NEW_NORMAL_SOURCE --normal-cache VERIFIED_NORMAL_CACHE --device cuda:0
```

The script verifies original model hashes, dataset/cache provenance and whole-run separation. It persists raw-derived health caches, JSON reference bounds/counts, nullable component scores, observation labels and original control scores. `progress.json` records extraction progress. Repeated invocation refuses an existing reference/results artifact; inspect an interrupted run before reuse. Six primary points use repetition off/on and thresholds .05/.10/.20. Secondary seeds only confirm an eligible primary point without reselection.

## Validation

88 local tests pass without warnings. New tests cover quality versus missing samples, future cadence, unavailable quality/cadence, duplicate observations, contiguous good pairs, legitimate zero plateaus, JSON roundtrip, unit mismatch, normal-only disjoint calibration, identity alignment and all gate constraints. Review checked causal extraction, persisted availability, existing model compatibility and preservation of exact event identities. Existing generic inference remains unchanged; this is a research component.

The [protocol](../../../TELEMETRY_HEALTH_PROTOCOL.md) fixes the acceptance criteria before evaluation. An eligible point requires independent new mixed and normal confirmation before promotion. Streaming silence after the final observation remains unsupported without a watermark; same-level normal repetition can still mask a stuck signal.

## Results and interpretation failure

| Repetition | Health threshold | Events /124 | Historical false windows | Lost baseline nonphysical events | New normal false windows |
|---|---:|---:|---:|---:|---:|
| Original v0.5 control | — | 71 | 4 | 11 | 25 |
| Off | .05 | 79 | 4 | 6 | 25 |
| Off | .10 | 77 | 4 | 6 | 25 |
| Off | .20 | 76 | 4 | 7 | 25 |
| On | .05 | 79 | 5 | 6 | 32 |
| On | .10 | 77 | 5 | 6 | 25 |
| On | .20 | 76 | 4 | 7 | 25 |

All six points fail the predeclared gate. Secondary seeds and new holdouts were skipped. No promotion. The .05 quality/availability point preserves the 15 original physical detections and leaves 25 false windows in 131.067 normal asset-hours (4.578/day); class rates remain below 10/day. But six baseline-detected loss events remain undetected. Loaded JSON reference scores exactly reproduce all four saved development score tables (repetition on/off, historical/new-normal).

**Source semantic mismatch discovered after evaluation:** pinned Lab `718babb` writes row-level `sampling_interval_ms` as the observed elapsed interval since the previous sample (`src/ot_lab/simulation.py:631`). Its run metadata stores the configured cadence separately (line 673). Example: PUMP flow in `dev-06::validation-00014` begins with row intervals 500, 501, 499, 500 ms and later contains larger gaps. The extraction contract assumed each row declared the configured interval. The strict change guard therefore disables sampling/repetition for ordinary source jitter and dropouts.

Consequently, sampling and repetition are available in only **94/1,965 historical** and **480/7,864 normal** windows; quality is available throughout. This is an invalid test of the intended availability channel, not evidence that availability detection cannot work. Preserve the result and provenance; do not silently overwrite it with a corrected run. The six remaining losses are signal/sample/communication-loss labels. Total recoveries cannot be attributed to repetition, which adds no event detections here.

Next correction: distinguish observed gap from configured cadence in the source adapter, retain both, verify nominal cadence against source metadata/scenario hashes per run, and explicitly require declared cadence for health extraction. Add jitter/dropout and true configuration-change tests. Repeat the fixed six development points under a separately recorded correction before considering fresh confirmation. Do not infer nominal cadence from future gaps or anomaly labels. [Numeric evidence and source hashes](../../../results/telemetry-health-20261005.json).
