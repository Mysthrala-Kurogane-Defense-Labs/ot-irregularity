# 3W unseen-well and unknown-event benchmark

This benchmark measures whether a model trained on normal oil-well behavior can score labeled observations from wells and event patterns withheld from fitting. It does not evaluate fault diagnosis or cross-industry transfer.

## Sources and terms

- Petrobras, [3W Dataset 1.1.1](https://doi.org/10.6084/m9.figshare.29205947.v1), CC BY 4.0. Cite the dataset's 2019 paper as listed in the [upstream citation guide](https://github.com/petrobras/3W/blob/main/CITATION.md).
- Petrobras, [3W Dataset 2.0.0](https://doi.org/10.6084/m9.figshare.29205836.v1), CC BY 4.0. Its upstream notes describe new wells, signals and event label 9: [release notes](https://github.com/petrobras/3W/blob/main/dataset/README.md). Cite Vaz Vargas et al., “3W Dataset 2.0.0: a realistic and public dataset with rare undesirable real events in oil wells,” *Scientific Data* (2026), DOI [10.1038/s41597-026-07225-z](https://doi.org/10.1038/s41597-026-07225-z).
- Both source archives stay outside Git. `source_manifest.json` records the version DOI, license, checksum, source members, selected partitions and transformations. Dataset rights do not derive from this repository's Apache-2.0 code license.

## Reproduce

Download the versioned archives from Figshare, then run:

```powershell
uv run python scripts/prepare_3w_unseen.py `
  --v1 D:\data\3w_dataset_1.1.1.zip `
  --v2 D:\data\3w_dataset_2.0.0.zip `
  --output D:\data\ot-irregularity\3w-unseen

uv run ot-irregularity train `
  --dataset D:\data\ot-irregularity\3w-unseen `
  --config configs/benchmark-3w-unseen-wells.yaml `
  --output D:\data\ot-irregularity\model-unseen-wells-v0.1

uv run ot-irregularity evaluate --model D:\data\ot-irregularity\model-unseen-wells-v0.1 `
  --dataset D:\data\ot-irregularity\3w-unseen\test

uv run ot-irregularity evaluate --model D:\data\ot-irregularity\model-unseen-wells-v0.1 `
  --dataset D:\data\ot-irregularity\3w-unseen\challenge --challenge
```

The preparation script verifies SHA-256 for 1.1.1 and Figshare's published MD5 for 2.0.0 by default. The manifest notes the seven-signal mapping and the sampling and label rules. Do not commit the archives, normalized Parquet, predictions or trained artifacts.

## Normalization and leakage controls

The compared feature space uses seven shared 3W tags: P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, P-JUS-CKGL and QGL. These map to stable semantic `signal_class` names and the engineering units in the upstream [`dataset.ini`](https://github.com/petrobras/3W/blob/main/dataset/dataset.ini). Twenty signals present only in 2.0.0 are excluded so new schema coverage cannot itself appear as process novelty. Raw tag identity is retained for provenance.

The source `class` becomes only a Boolean evaluation label (`class != 0`); `state`, asset identity and run identity are not model features. Null labels are omitted rather than guessed. Runs are segmented at label transitions and timestamp gaps above 2.5 seconds so no window crosses an unlabeled region or normal/event boundary. Source data do not provide data-quality grades; `quality=uncertain` is recorded and quality-ratio features are disabled. A repeated exact value `-1.180116e42` occurs in `P-PDG` in nine selected 3W 1.1.1 validation runs (354,782 observations); the 3W mapping config treats this as an unavailable source value and removes those observations before windowing. This is a source-specific mapping, not a generic rule. The 1-second sampling interval is checked against source timestamps. Unit normalization precedes robust ML scaling.

The split is by physical `asset_id`; preparation fails on asset overlap across partitions:

| Partition | Source | Assets | Use |
|---|---|---|---|
| Train | 3W 1.1.1 | WELL-00001, 00002, 00004, 00005 | Labeled normal-only source instances |
| Validation | 3W 1.1.1 | WELL-00006, 00007, 00008 | Normal score calibration and threshold assessment; selected known events support supervised validation metrics |
| Test | 3W 2.0.0 | WELL-00019, 00021–00032 | New version; one normal source for WELL-00019 and selected known-event instances on the other wells |
| Challenge | 3W 2.0.0 | WELL-00014, 00016, 00020, 00037, 00040–00042 | Explicit holdout of label 9 / transient 109; never used by training or validation |

Prepared source volume is bounded and pinned in `source_manifest.json`:

| Partition | Source files | Label-contiguous segments | Long-format rows | Assets |
|---|---:|---:|---:|---:|
| Train | 48 | 48 | 4,257,605 | 4 |
| Validation | 41 | 51 | 6,312,427 | 3 |
| Test | 14 | 27 | 21,301,547 | 13 |
| Challenge | 7 | 14 | 821,065 | 7 |

Folder names alone are not treated as labels. A source Parquet is selected for the unknown-event challenge only when its `class` column actually contains 9 or 109. This check found seven real wells with class-9 labels in the release; the challenge reserves all seven assets from the other evaluation partitions. Challenge files contain labeled class-0 context, enabling within-run false-positive measurement, but not long-term normal-only data for each asset.

## Interpretation limits

- This is a bounded, deterministic, event-stratified subset, not a random prevalence sample. It cannot estimate deployment alert rates or production false positives per day.
- Only WELL-00019 has a separate class-0 source instance among the newly introduced test wells. Other test wells have class-0 context inside the selected event files, so their false positives are measured within those files; the rate does not estimate long-term normal operation.
- Event 9 is previously unseen to training, but its challenge files include normal context. Results can establish detection on these selected labeled intervals, not general performance across all hydrate events or assets.
- Both releases belong to the same Petrobras dataset family. This is held-out-well / held-out-event evidence, not validation on another industry, acquisition system or simulator.
- Report precision, recall, PR-AUC, false positives and event-level latency with counts and windows. Accuracy and raw scores alone are insufficient. The model outputs irregularity and supporting feature errors, not causal diagnosis.

## Results

Training used 10,262 normal windows from wells 1, 2, 4 and 5. Validation used 16,737 windows from wells 6–8, including 10,638 labeled-normal calibration windows and 6,099 event windows. The configured ensemble threshold was 0.95. On validation:

| Detector | Precision | Recall | F1 | PR-AUC | ROC-AUC | False positives / asset / day | Event detection | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Autoencoder | 0.000 | 0.000 | 0.000 | 0.309 | 0.320 | 45.77 | 0/20 | unavailable |
| Isolation Forest | 0.664 | 0.172 | 0.273 | 0.493 | 0.596 | 45.77 | 20/20 | 875.9 s |
| 50/50 ensemble | 0.000 | 0.000 | 0.000 | 0.293 | 0.398 | 0.086 | 0/20 | unavailable |

At this threshold the 50/50 ensemble does not improve on Isolation Forest, and neither its event recall nor false-positive rate is acceptable as a production alarm policy. The weight and threshold remain configurable; these baseline results do not establish an optimal setting.

The 3W 2.0.0 test contains 72,570 windows (58,959 positive; 81.2% prevalence, so a prevalence-only PR-AUC reference is about 0.812). Every selected new test asset has a labeled event, but only WELL-00019 also has a standalone normal file. At threshold 0.95:

| Detector | Precision | Recall | F1 | PR-AUC | ROC-AUC | False positives / asset / day | Event detection | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Autoencoder | 1.000 | 0.175 | 0.298 | 0.852 | 0.552 | 0.00 | 1/13 | 7,863 s |
| Isolation Forest | 0.825 | 0.592 | 0.690 | 0.798 | 0.512 | 146.84 | 13/13 | 26,319 s |
| 50/50 ensemble | 1.000 | 0.175 | 0.298 | 0.864 | 0.566 | 0.00 | 1/13 | 7,863 s |

The explicit class-9 challenge contains seven assets and 1,786 evaluable windows (614 positive). Six labeled event segments produce 15-minute windows; WELL-00020's selected event lasts under 15 minutes and is not window-evaluable at this configuration. The seven source files include labeled class-0 context. Results at threshold 0.95:

| Detector | Precision | Recall | F1 | PR-AUC | ROC-AUC | False positives / asset / day | Event detection | Mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Autoencoder | 0.000 | 0.000 | 0.000 | 0.342 | 0.493 | 0.00 | 0/6 | unavailable |
| Isolation Forest | 0.233 | 0.034 | 0.060 | 0.268 | 0.330 | 55.63 | 6/6 | 868 s |
| 50/50 ensemble | 0.000 | 0.000 | 0.000 | 0.264 | 0.325 | 0.00 | 0/6 | unavailable |

The model's ensemble fails this unknown-event challenge at its configured operating point. Isolation Forest detects each windowable event only by also producing many false positives. This baseline is research evidence, not an operational alerting model. These challenge files support within-file false-positive counts but do not represent long-term operational prevalence. Predictions on the challenge were byte-identical in two inference runs (SHA-256 `eeefd51f75b4b5ec334049154ec6ae750577628bfe52109ca413a9585123722e`).
