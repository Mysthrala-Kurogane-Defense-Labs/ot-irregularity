# Cross-signal residual experiment, 2026-10-05

## Scope and diagnosis

Investigate missed physical deviations and test a small normal-only relational representation. Keep v0.4 baseline and all candidates on the same complete-tail one-minute windows. Historical train/validation only for diagnosis and selection; the **nine** evaluated seed batches at the start of this experiment are already consumed and are not reused for selection.

Development contains 35 physical events, four detected by v0.4. The descriptive raw audit finds at least one sample outside the pooled normal 99% reference interval in 22 events. None exceeds the 99th normal percentile of the aggregate marginal physical-feature distance. No physical event overlaps another event in this audit. These are different descriptive measures with different opportunities for extremes; they do not establish loss of information or detectability. There is no matched same-seed normal counterfactual, so the exact role of injection, regime changes, aggregation and reconstruction remains unresolved.

## Predeclared representation screen

- Six continuous mean roles per asset class, selected from the existing applicable-role schema. No product identifiers or ground-truth features enter predictors.
- Normal-only Ridge alpha=10 predicts each role from the other five, using normal-fit RobustScaler. A temporal variant also receives all six means from the previous contiguous minute, without crossing run/asset/gap boundaries. Static fallback at warmup/bad history.
- Quality gate: every current role must have good ratio >=0.95, coverage >=0.9, finite mean and missing=0. Unavailable residuals cannot assert normality; supplements preserve baseline scoring for these windows.
- Compare static and temporal residuals, direct top-two normalized magnitude and AE top-two normalized error, standalone and max-supplemented with v0.4: eight variants. AE latent 2, <=3,000 optimizer updates, CUDA, seed 20261005; Isolation Forest is fitted/persisted as a separate auxiliary detector and not mixed into the selected score.
- Preserve whole-run fit/early-stop/calibration/development separation. Ridge/scalers use fit only, residual scales and AE checkpoints use early-stop normals, CDF/thresholds use normal calibration. All validation rows are transformed before subsetting so observed past context remains causal.
- Gate: more physical events, no loss of total events, precision >=0.5, no increase in falsely alerted minutes. First q99 screen passes no candidate despite a large sensitivity increase; those negative operating points are retained.

## Adaptive development stage and confirmation

A second explicit development protocol compared q99, q99.5, q99.75 and q99.9, calibrated on normal data. Unresolvable thresholds above 1 were excluded. This is additional model selection, not independent evidence. The chosen `max-static-ae`, q99.75 detects **71/124 total and 15/35 physical events with four false minutes**, versus baseline **53/124, 4/35, four false minutes**. The direct static supplement ties at 15 physical but detects 68 total; the AE supplement is chosen for its higher total at the same accepted false-alert constraint.

Two additional AE seeds are trained before test, with Ridge, residual scaler, baseline and auxiliary IF fixed. Seeds 20261005/06/07 detect 71/75/74 total and 15/19/18 physical events, with 4/4/5 false minutes. The third seed exceeds the strict development false-minute baseline by one. No seed reselection: 20261005 remains primary. Reloaded initial research scores have maximum absolute difference **0.0** in every representation.

`frozen.json` records thresholds and artifact hashes before generation of holdout seeds 910531/32/33. Test controls include baseline at its original threshold, baseline at the matched normal-calibration quantile and the direct static residual supplement. The candidate can lose some individual events even when aggregate detection improves; report family results and tradeoffs.

## Reproduction and provenance

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
uv run --no-sync python scripts/research_relationships.py --dataset $Dataset --baseline $Baseline --raw-root $RawDevelopment --cache $MinuteCache --output $Research --device cuda:0
uv run --no-sync python scripts/relationship_operating_points.py --dataset $Dataset --baseline $Baseline --raw-root $RawDevelopment --cache $MinuteCache --research $Research
uv run --no-sync python scripts/validate_relationships.py fit --dataset $Dataset --baseline $Baseline --raw-root $RawDevelopment --cache $MinuteCache --research $Research --output $Validation
# Pinned Lab checkout; repeat with 910532 and 910533:
uv run ot-lab dataset create --suite suites/training-v0.2.yaml --runs 240 --seed 910531 --workers 4 --output $Raw1
uv run --no-sync python scripts/validate_relationships.py evaluate --baseline $Baseline --output $Validation --sources $Raw1 $Raw2 $Raw3
```

Data attribution: Mysthrala Kurogane Defense Labs, [OT Irregularity Lab](https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab), CC BY 4.0. Historical development: Lab 0.3.1. Fresh generation: clean Lab 0.6.0 commit `718babb7772c3a21f0b87c403f628540cbce58db`, same suite 0.2.0. Datasets, raw diagnostic rows and weights stay outside Git; publish aggregate evidence, source/truth hashes, configs and reproducible code.

## Independent result

Three new batches, 240 generated runs each; 36 test runs/batch evaluated. **108 runs, 1,418 complete minutes (23.6333 asset-hours), 96 original intervals, 33 physical events**. No event lacks observable windows. Same windows and tail policy for every model/control.

| Model | PR-AUC | Precision | All events / 96 | Physical / 33 | False minutes |
|---|---:|---:|---:|---:|---:|
| v0.4 complete | 0.5423 | 88.1% | 40 | 1 | 7 |
| Candidate, primary seed | **0.7036** | **97.6%** | **61** | **23** | **2** |
| Candidate, seed 20261006 | 0.7062 | 96.3% | 59 | 21 | 3 |
| Candidate, seed 20261007 | 0.7035 | 96.2% | 58 | 21 | 3 |
| Baseline at matched calibration quantile | 0.5423 | 87.7% | 42 | 2 | 8 |
| Direct residual supplement, no residual AE | 0.6688 | 97.3% | 54 | 19 | 2 |

The control key `v0.4-tight` is a historical script label: its threshold is actually **0.986792**, below the original 0.990566, because calibration excludes incomplete tails. It means matched normal-calibration quantile, not a stricter numerical cutoff. The selected supplemental threshold is 0.997821. All were frozen before test.

Primary event detection: **63.5%**; physical detection: **69.7%**; window recall 41.2%; false minutes/asset-day **2.03**, versus 7.11 baseline. The exploratory >=50% events, >=50% precision and <=10 false minutes/day targets pass in this sample for all three seeds. Short synthetic exposure does not establish operational acceptance.

Paired whole-run bootstrap: ΔPR-AUC +0.1613, 95% CI [0.0919, 0.2213], 500 resamples; Δevent detection +21.875 points, CI [8.08, 35.17]; Δphysical detection +66.67 points, CI [48.48, 83.79], 2,000 resamples. False-minute rate difference CI [-11.39, +1.03]/asset-day includes zero: the observed reduction is not a proven long-run false-alarm improvement.

### Family regressions remain visible

| Simulator family | Baseline | Candidate primary |
|---|---:|---:|
| bearing_degradation | 1/11 | 9/11 |
| cavitation | 0/8 | 8/8 |
| mechanical_overload | 0/7 | 6/7 |
| cooling_degradation | 0/7 | **0/7** |
| asset_communication_loss | 8/8 | **6/8** |
| missing_telemetry | 6/7 | **4/7** |
| quality_degradation | 7/8 | **3/8** |
| sensor_stuck | 8/9 | **5/9** |

The stricter supplemental threshold loses some integrity events; the aggregate gain does not erase these regressions. Keep the candidate experimental and v0.4 available. Do not silently restore baseline alarms after inspecting test: that would be another candidate requiring development selection and a new holdout. Next hypotheses are separate integrity/physical alert budgets and thermal context; neither is validated here.

## Reusable candidate and validation

`scripts/relationship_candidate.py package` creates a self-contained local research bundle with baseline, primary supplement, hashes, threshold, metrics, attribution and model card. `infer` emits JSONL or Parquet, preserves baseline AE/IF scores and exposes the relational AE separately. Unavailable relational scores are `null` with an explicit reason. Explanations separate baseline contributions from normalized relational reconstruction contributions and show observed versus relation-expected means; they do not assert fault causes.

```powershell
uv run python scripts/relationship_candidate.py package --baseline $Baseline --validation $Validation --output $Bundle
uv run python scripts/relationship_candidate.py infer --model $Bundle --dataset $PreparedTest --output predictions.jsonl
```

The generic `ot-irregularity` defaults are unchanged; the research bundle uses its explicit script. Do not load its root as an ordinary baseline artifact. Publication includes code and numeric evidence, not data or weights. No release tag, merge or field deployment.

**65 local tests passed.** Tests cover same-run/asset contiguous history, future exclusion, target exclusion from its current predictors, unavailable quality/nonfinite handling, serialization, static fallback and candidate integrity, alongside the existing pipeline suite. Two bundle JSONL inferences over 440 test windows are byte-identical (SHA-256 `965ee345ff9b0ad15006213315429cb9a68dac515522732ad7722a72e9778389`); all scores exactly match the frozen evaluation and Parquet output. A conflicting Parquet output filename was refused without overwriting the original frozen predictions, then successfully written to a fresh filename.

Code review and documentation link/diff checks passed locally. No remote CI is configured. Provenance rechecks were added to confirmation cache loading; source/cache hashes were independently verified in the preceding research stage. The raw marginal diagnostic remains an observation rather than a counterfactual explanation. Twelve test seed batches are now consumed; further selection needs new data.
