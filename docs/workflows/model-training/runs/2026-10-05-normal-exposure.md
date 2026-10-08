# Normal exposure of four frozen controls

## Scope and provenance

The [protocol](../../../NORMAL_EXPOSURE_PROTOCOL.md), suite and freeze utility were committed as `238aa4b` before generating seed 920601. The freeze records model/threshold/anchor hashes and Lab commit `718babb7772c3a21f0b87c403f628540cbce58db`. Generation completed successfully: 180 one-hour runs, all four classes per run, 24,120,000 observations, no injected anomalies. Source data: OT Irregularity Lab 0.6.0, Mysthrala Kurogane Defense Labs, CC BY 4.0.

The evaluator was implemented after generation, before inspecting model outcomes on these data. It preserves the frozen scoring implementation, decisions and protocol. Its exact source hashes and the source manifest hash are recorded separately in `execution.json`; this does not claim that the evaluator itself existed at the earlier model freeze.

`scripts/evaluate_normal_exposure.py` processes one run at a time, verifies telemetry/truth/metadata byte hashes and the scenario semantic hash, and writes a prediction Parquet plus an atomic completion record. Resuming requires identical execution/source/model hashes and intact saved predictions. It never fits a model, calibrates a threshold or selects an operating point. Ground-truth regimes are attached only after scoring, for diagnostics.

## Validation and correction

The first evaluation attempt aborted before any inference: the Lab manifest's `scenario_sha256` hashes canonical JSON, not the serialized YAML file. The verifier now checks the canonical representation and records the YAML byte hash separately; Lab metadata is named `run_metadata.json`. The unsuccessful execution identity was preserved. No models, data, thresholds or candidate choices changed.

79 local tests pass, including semantic normalization, episode boundaries across normal/missing windows, assets and runs, overlap rejection, run-cluster bootstrap, paired identical controls, zero-count caution, and scenario hash tampering. Existing adapter tests also pass after extracting its unchanged semantic transformation for reuse. An explicit before/after comparison against commit 238aa4b also produces identical frames and event JSON in train, validation and test fixtures. Software tests do not establish model performance.

## Commands

```powershell
# In the pinned Lab environment, after freeze_normal_exposure.py succeeds:
python -m ot_lab.cli dataset create --suite NORMAL_SUITE --runs 180 --seed 920601 --workers 4 --resume --output NEW_SOURCE

# In the detector's CUDA environment:
python scripts/evaluate_normal_exposure.py --source NEW_SOURCE --output FROZEN_OUTPUT --baseline BASELINE --sample SAMPLE_MODELS --relational V05_BUNDLE --relationships RELATIONAL_FREEZE --device cuda:0
```

Progress is written atomically to `progress.json`; per-run provenance and predictions remain under `runs/`. These are batch evaluations of independent simulated runs, not an online service or industrial field trial.

## Completed evaluation

180/180 runs completed; 42,848 complete minute windows, **178.533 asset-hours per class**, 714.133 overall. All four classes exceed the declared 168-hour target. Both supplementary components are available on all 10,712 windows/class. These are aggregate one-hour runs, not a continuous week.

| Control | False windows | False windows/asset-day (run-cluster 95% CI) | Episodes | Episodes/asset-day |
|---|---:|---:|---:|---:|
| v0.4 | 156 | 5.24 [4.07, 6.55] | 156 | 5.24 |
| Original v0.5 relational | 171 | 5.75 [4.77, 6.69] | 169 | 5.68 |
| Baseline OR thermal | 226 | 7.60 [5.68, 9.58] | 183 | 6.15 |
| Rejected joint diagnostic | 346 | 11.63 [9.47, 13.73] | 300 | 10.08 |

Paired joint-minus-baseline false-window rate difference: **+6.39/day**, 95% CI [4.57, 8.33]. Thermal-minus-baseline: +2.35/day, CI [0.84, 4.10]. These are run-cluster intervals within the simulated population. They support increased false alarms here, not universal industrial rates.

| Class, false windows/asset-day | v0.4 | Original v0.5 | Thermal union | Joint diagnostic |
|---|---:|---:|---:|---:|
| CNC | 4.03 | 1.61 | 4.03 | 5.65 |
| Compressor | 2.96 | 7.12 | 11.56 | 12.10 |
| Conveyor | 7.66 | 6.45 | 8.20 | 16.00 |
| Pump | 6.32 | 7.80 | 6.59 | 12.77 |

The joint diagnostic exceeds the exploratory 10/day target overall and in three classes. Thermal exceeds it for compressors despite a lower overall rate. The original v0.5 result does not erase its previously measured integrity-detection losses; this normal-only dataset cannot assess event sensitivity.

## Diagnostic strata and next hypothesis

Among the thermal union's 70 extra false windows, 64 belong to compressors. Of those 64, 22 are entirely within cooldown and 29 span multiple truth regimes. For the joint decision, 52/62 extra conveyor false windows span regimes, as do 21/48 extra pump windows; 18 additional pump windows occur in cooldown. Mixed-regime windows are already common (5,881/10,712 per class), so these counts alone do not establish transition causality. Truth is diagnostic metadata, never an inference feature.

The baseline itself has 16.49 false windows/day in the planned-maintenance profile and 13.92 in startup-maintenance-shutdown, despite passing the overall target. Different profiles/cadences/regime visibility also have different rates; assignments are not a controlled causal ablation. All strata and denominators are in the [numeric evidence](../../../results/normal-exposure-20261005.json).

**Decision: keep both the development rejection and these normal-exposure failures. No threshold change, seed selection or model promotion.** The next development work should improve representation of normal stops, mixed windows and longer thermal evolution, using separately generated normal training/calibration runs. Preserving every baseline alert also preserves its false alarms; a future gate must explicitly distinguish preservation of labeled integrity-event detection from preservation of all normal-window decisions. Such a protocol change must be declared before a new experiment, never used to retroactively pass this one.

This entire exposure set is now consumed evaluation evidence. It can guide hypotheses but must not become training/calibration data or independent confirmation of a later improvement. New development and confirmation sources are required.

## Completion checks

The verified-resume command completed successfully over all 180 saved runs. An independent audit of prediction identities, decision counts, exposure sums and baseline-alert preservation accompanies the metric calculation. No failed run was omitted. The first technical failure occurred before scoring and is retained above. Raw telemetry, predictions and weights remain outside Git; public evidence includes the protocol, exact implementation hashes, summaries and all 180 prediction hashes. Local tests, remote CI and model promotion remain separate gates.
