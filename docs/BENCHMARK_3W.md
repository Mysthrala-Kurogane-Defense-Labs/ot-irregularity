# Exploratory benchmark: Petrobras 3W 1.1.1

## Source and reuse

The data came from [Figshare, 3W Dataset 1.1.1](https://doi.org/10.6084/m9.figshare.29205947.v1), DOI `10.6084/m9.figshare.29205947.v1`, CC BY 4.0. The [Petrobras dataset page](https://github.com/petrobras/3W/blob/main/dataset/README.md) confirms the data license; the [structure guide](https://github.com/petrobras/3W/blob/main/3W_DATASET_STRUCTURE.md) and [citation guide](https://github.com/petrobras/3W/blob/main/CITATION.md) define the file layout and requested citation. Petrobras requests this citation for Dataset 1.x:

> Vaz Vargas, R. E., Munaro, C. J., Marques Ciarelli, P., Gonçalves Medeiros, A., Guberfain do Amaral, B., Centurion Barrionuevo, D., Dias de Araújo, J. C., Lins Ribeiro, J., & Pierezan Magalhães, L. “A realistic and public dataset with rare undesirable real events in oil wells.” *Journal of Petroleum Science and Engineering*, 181 (2019). https://doi.org/10.1016/j.petrol.2019.106223.

The archive SHA-256 used here is `fed2af7c6f607b46d4963fe7eb7ee7ada7df3cfde0885f205a6408d0c2adff69`. The archive and normalized partitions were kept outside the repository. This project contains the normalizer, checksum, source citation, and metrics, not the source observations.

## Preparation and split

Run the following from the repository root after downloading the Figshare archive:

```bash
uv run python scripts/prepare_3w.py \
  --archive /path/to/3w_dataset_1.1.1.zip \
  --output /path/to/3w-normalized

uv run ot-irregularity train \
  --dataset /path/to/3w-normalized \
  --config configs/benchmark-3w.yaml \
  --output /path/to/model-3w-v0.1

uv run ot-irregularity evaluate \
  --model /path/to/model-3w-v0.1 \
  --dataset /path/to/3w-normalized/test

uv run ot-irregularity infer \
  --model /path/to/model-3w-v0.1 \
  --dataset /path/to/3w-normalized/test \
  --output /path/to/predictions-test.jsonl
```

The normalizer checks the archive hash by default. It reshapes the eight source process columns into long format, excludes `class` from model inputs, derives only a binary `is_anomaly` evaluation label (`class != 0`), and assigns an event key for metric grouping. The source has no quality or unit fields: quality is recorded as `uncertain`, quality features are disabled, and units remain explicitly unknown. The 1-second sampling interval was checked from source timestamps.

The script selects complete instances using upstream `folds_clf_02.csv`: folds 0 and 1 for training, fold 2 for validation/calibration, and fold 4 for final evaluation. Training includes only source-directory class `0`. Fold 3 is unused. The script records missing paths referenced by the fold CSV and skips repeated references to the same instance. For this archive that meant 13 absent paths and three repeated fold entries. The resulting partitions contain:

| Partition | Instances | Normal instances | Event instances | Long rows |
| --- | ---: | ---: | ---: | ---: |
| Train | 18 | 18 | 0 | 1,925,903 |
| Validation | 24 | 6 | 18 | 2,482,656 |
| Test | 21 | 7 | 14 | 2,178,358 |

Repeated references to identical source files are skipped. No further duplicate rows were present in the normalized partitions. Details and file-level membership are in the generated external `source_manifest.json`.

## Result

Configuration: seed 42; 15-minute windows with a 1-minute stride; RobustScaler; 30-epoch cap with early stopping; 300-tree Isolation Forest; equal detector weights; decision threshold 0.95. The threshold is the existing baseline's 95th-percentile empirical normal calibration rule, not an optimized operating point.

| Split / metric | Validation | Test |
| --- | ---: | ---: |
| Windows | 5,877 | 5,227 |
| Labeled anomalous windows | 4,021 | 3,234 |
| Ensemble PR-AUC | 0.8400 | 0.7824 |
| Ensemble ROC-AUC | 0.8382 | 0.8256 |
| Precision at 0.95 | 0.4500 | 0.4462 |
| Recall at 0.95 | 0.0179 | 0.0257 |
| F1 at 0.95 | 0.0344 | 0.0485 |
| False-positive windows | 88 | 103 |
| Event-group detection at 0.95 | 16.7% (3/18) | 6.25% (1/16) |
| Mean latency for detected event groups | 5,659 s | 28 s |

On the held-out test fold, the Autoencoder scored PR-AUC 0.8126 (precision 0.4361, recall 0.0306 at 0.95); Isolation Forest scored PR-AUC 0.7532 (precision 0.4049, recall 0.0309). The ensemble traded away a small amount of recall for fewer false-positive windows, but detected only one of 16 event groups at that threshold. All three component scores remain in every inference record.

Test inference wrote 5,227 JSONL prediction records, each preserving the Autoencoder, Isolation Forest, and ensemble scores. The test result shows that the conservative threshold misses most labeled abnormal windows. PR-AUC is better than the positive-window prevalence (61.9%), but remains a preliminary result on one narrow asset domain.

## Limits

- The experiment validates the current loading, feature, training, persistence, inference, and evaluation path on a real industrial time-series source. It does not establish operational utility.
- This is oil-well telemetry, not a broad industrial benchmark. The fold file mixes real and simulated event instances; the model predicts binary deviation labels, not event causes.
- Fold separation is by whole instance (`run_id`), but wells recur between folds. It does not measure generalization to unseen wells and should not be represented as an asset-held-out result.
- Window labels are positive if any source row in the window is labeled nonzero. The source's event classes are intentionally collapsed; no class-level diagnosis is evaluated.
- The false-positive rates per asset-hour/day in the current metrics implementation aggregate exposure over all windows and repeated assets. The window counts are reported directly; do not use those rates for an operational alarm budget until the metric is redesigned around independent asset-time exposure.
- Model weights trained from CC BY data have not been reviewed for redistribution terms. Do not publish the weights or normalized data as part of the Apache-2.0 code release without a separate licensing review.

## Other open candidates

- [UCI AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i) is CC BY 4.0 and useful for a small schema/feature exercise, but UCI describes it as synthetic; it does not establish temporal behavior of a physical process.
- [SKAB](https://github.com/waico/skab) is a potentially useful multivariate pump corpus. Its repository is GPL-3.0; confirm file-level terms and implications for model redistribution before using it in a distributable training run.
