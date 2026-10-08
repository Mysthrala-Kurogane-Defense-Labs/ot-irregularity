# Separate alert budgets: development rejection

The predeclared [maintenance protocol](../../../V0.5_MAINTENANCE.md) was evaluated without new training or test access. Baseline and all three previously frozen relational seeds were hash-checked. Cache dataset/schema/config/content provenance was verified. Only normal calibration and development runs were used; original validation event intervals supplied ground truth.

Decision: `(baseline >= original_v04_threshold) OR (relational >= normal_calibrated_threshold)`. Components remain separate. This preserves every baseline alert window, and therefore every baseline event detection on identical windows, but also preserves every baseline false window. Additional relational false alerts cannot cancel them.

## Results

Same 1,965 complete minute windows, 124 events, 35 physical events. Baseline detects 53 events, 4 physical, with 4 false windows.

| AE seed | Relational quantile | Events | Physical | False windows | Gate |
|---|---:|---:|---:|---:|---|
| 20261005 (primary) | .995 | 87 | 20 | 9 | Fail |
| 20261005 (primary) | .9975 | 83 | 16 | 8 | Fail |
| 20261006 | .995 | 88 | 20 | 17 | Fail |
| 20261006 | .9975 | 87 | 20 | 8 | Fail |
| 20261007 | .995 | 88 | 20 | 17 | Fail |
| 20261007 | .9975 | 86 | 19 | 9 | Fail |

Quantile .999 is unresolvable for all three seeds: strict nextafter threshold exceeds 1. It is rejected rather than used to disable the supplement. No baseline alert window is lost at any resolvable operating point. All six points violate the predeclared false-window gate. **No candidate selected and no new test generated.** Passing a looser daily-rate target would not justify changing the agreed development gate retrospectively.

The primary shared-threshold v0.5 control retains its historical development result (71 events, 15 physical, 4 false windows), including its known integrity trade-off. Neither deployed decisions nor the bundle were changed. PR-AUC/ROC-AUC are deliberately omitted for binary decision unions: these decisions do not supply a new continuous ranking. Latencies, precision/recall and all family denominators remain in the [numeric evidence](../../../results/alert-budgets-20261005.json).

## Reproduction and checks

```powershell
uv run python scripts/research_alert_budgets.py --baseline $Baseline --validation $FrozenRelationalRun --dataset $DevelopmentDataset --cache $VerifiedWindowCache --raw-root $OriginalDevelopmentTruth --output $NewOutput --device cuda:0
```

The script refuses an existing output directory, writes its protocol before scoring, and incrementally writes `progress.json` for reuse in a monitor. Sources remain OT Irregularity Lab, CC BY 4.0, attribution Mysthrala Kurogane Defense Labs. Per-file hashes and cache provenance are in the numeric record; no raw telemetry or private paths are published.

Local software validation: **67 tests passed**, including union preservation, unavailable supplement handling and invalid inputs. The new revision requires its own remote CI; earlier green CI applies only to earlier commits. The screen ran with the existing CUDA environment; no retraining or cross-device equivalence is claimed.

Next: investigate thermal context and the features distinguishing telemetry integrity from physical residuals on development data. Any replacement decision rule needs a separately recorded protocol, selection and fresh holdout. The twelve previous test seed batches remain consumed.
