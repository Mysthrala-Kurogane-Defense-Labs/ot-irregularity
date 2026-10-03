# Procedure: 3W unseen wells and unknown event

- Purpose and scope: reproducibly compare shared-channel behavior across Petrobras 3W 1.1.1 and 2.0.0 with disjoint assets and event 9 held out.
- Last validated: 2026-10-03, Windows / Python 3.12.10 / `uv`; trained artifact records source commit `aa606f14ddfe1b01cd9d45a18bcf813cbd49515a`.
- Evidence: [execution record](runs/2026-10-03.md); [benchmark report](../../BENCHMARK_3W_OOD.md).

## Prerequisites

- Python 3.12+, `uv`, project dependencies installed.
- Download original archives from the Figshare version pages cited in `docs/DATASET_SOURCES.md`; store them outside Git.
- Confirm archive hashes in `scripts/prepare_3w_unseen.py`. Do not bypass checksums for benchmark evidence.
- Normalized datasets and artifacts are written to a local external-data directory, not the repository.

## Execution

1. Check out the intended project revision and inspect `git status`.
2. Run `uv run pytest`.
3. Prepare data with `uv run python scripts/prepare_3w_unseen.py --v1 <archive-1.1.1.zip> --v2 <archive-2.0.0.zip> --output <external>/3w-unseen`.
4. Inspect `source_manifest.json`: verify source hashes, label counts, common tag mapping, partition asset IDs and no overlap. Challenge files must contain labels 9/109, not merely live in directory `9/`.
5. Train with `uv run ot-irregularity train --dataset <external>/3w-unseen --config configs/benchmark-3w-unseen-wells.yaml --output <external>/model-unseen-wells-v0.1`.
6. Evaluate the `test/` partition normally and `challenge/` only with `--challenge`. Record exact command output and model metadata. Keep the two metric sets separate.
7. Confirm predictions do not use source labels/state or identifiers as features. The manifest should say null labels were omitted and windows do not cross label/time gaps.

## Completion and recovery

- Completion requires successful checksum verification, no asset leakage, normal-only train windows, persisted model/scaler/config/schema/metrics, test and challenge metrics, and green project tests.
- If a source folder contains files with only null or normal labels, exclude them from positive challenge selection based on the actual label column.
- Treat `uncertain` and null quality as usable measurements unless the project's feature policy changes; source data without quality must not collapse to zero-valued signal statistics.
- A timeout during feature generation is not a metric result. Profile window counts/runtime, preserve the split, optimize implementation, then rerun from the beginning.
- Report selected-file and within-run results as bounded evidence, not estimates of production prevalence or cross-industry performance.
