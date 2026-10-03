# Dataset sources and reuse

This project does not redistribute the source datasets. Downloaded research data lives outside the repository. The dataset license is separate from this project's Apache-2.0 code license.

## Selected: Petrobras 3W Dataset 1.1.1

- Source record: [Figshare, 3W Dataset 1.1.1](https://doi.org/10.6084/m9.figshare.29205947.v1), version 1.1.1, posted 2025-05-31, licensed CC BY 4.0.
- Publisher project and explicit data terms: [Petrobras 3W repository](https://github.com/petrobras/3W), section “Licenses”: its code is Apache-2.0 and files under `dataset/` are CC BY 4.0. Do not infer the dataset license from the code license.
- Citation requested by the project for 3W Dataset 1.x: Vaz Vargas, R. E., Munaro, C. J., Marques Ciarelli, P., Gonçalves Medeiros, A., Guberfain do Amaral, B., Centurion Barrionuevo, D., Dias de Araújo, J. C., Lins Ribeiro, J., & Pierezan Magalhães, L. “A realistic and public dataset with rare undesirable real events in oil wells.” *Journal of Petroleum Science and Engineering*, 181 (2019). https://doi.org/10.1016/j.petrol.2019.106223. See the [Petrobras citation guidance](https://github.com/petrobras/3W/blob/main/CITATION.md).
- Fit: multivariate oil-well process telemetry; source files are Parquet, observations are rows, timestamps are in the index, labels are columns, and each file is an instance. Official structure guide: [3W dataset structure](https://github.com/petrobras/3W/blob/main/3W_DATASET_STRUCTURE.md).
- Evaluation policy here: use only class `0` instances for normal training and normal calibration; hold out whole instances; collapse nonzero event classes to an anomaly flag only for evaluation. Do not feed event labels/classes into features, and do not report fault diagnosis.
- Attribution: retain the citation, DOI, version, CC BY 4.0 link, and note any transformation. The original archive stays outside Git; derived training metadata records its checksum. Review terms before redistributing transformed data or model weights.

The tested subset and measured results are recorded in [BENCHMARK_3W.md](BENCHMARK_3W.md). The normalizer pins the Figshare archive checksum and stores normalized files and artifacts outside Git.

## Candidate, limited fit: UCI AI4I 2020

- Source and license: [UCI dataset 601](https://archive.ics.uci.edu/dataset/601/ai4i), DOI https://doi.org/10.24432/C5HS5C, CC BY 4.0. Cite as *AI4I 2020 Predictive Maintenance Dataset* (UCI Machine Learning Repository, 2020).
- UCI describes it as synthetic, 10,000 observations with generated process variables and machine-failure labels. It is useful for a small feature/schema exercise, but it does not provide a real industrial process or an explicit acquisition timestamp in its published variable list; do not use its row order as real time for temporal-performance claims.

## Candidate with a different distribution license: SKAB

- [SKAB upstream repository](https://github.com/waico/skab) describes a multivariate water-pump testbed with timestamps, vibration, current, pressure, temperature, voltage and anomaly/changepoint labels. Its current repository license is GPL-3.0, not CC BY or Apache. It may be studied under that license, but we will not copy its files into this Apache-2.0 repository or redistribute them from it. Confirm the dataset files' applicable terms and downstream model implications before using SKAB in a distributable training run.
