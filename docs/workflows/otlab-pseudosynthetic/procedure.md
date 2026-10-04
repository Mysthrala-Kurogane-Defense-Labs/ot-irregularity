# Procedure: OT Irregularity Lab pseudo-synthetic benchmark

- Purpose: train the generic detector on event-free process simulations and evaluate on separate runs with ground-truth event intervals.
- Source: local OT Irregularity Lab v0.3.0 dataset release; code and dataset are independent of Kurogane Hub.
- Evidence: [execution record](runs/2026-10-03.md); [benchmark report](../../BENCHMARK_OT_IRREGULARITY_LAB.md).

## Steps

1. Verify both checkouts, intended source dataset release, source `dataset_manifest.json`, and clean/known worktree state. Do not modify the lab repository as part of preparation.
2. Prepare the dataset with `uv run python scripts/prepare_otlab.py --source <lab-dataset> --output <new-external-path>`. The output must not already exist.
3. Inspect output `dataset_manifest.json`: source hashes verified, training/calibration runs event-free, test run IDs disjoint, and event labels confined to test.
4. Train using `configs/benchmark-otlab.yaml`. The 5-minute window is required because source runs are 5–10 minutes long.
5. Evaluate only the output `test/` partition and record detector-specific metrics, event coverage, normal false positives, and deterministic inference hash.
6. For false-positive stress evidence, generate an event-free `normal-operation-v0.1` dataset using a new seed. If the suite has no dataset license, make an external suite copy and explicitly set `data_license: CC-BY-4.0` before generation; retain its exact suite hash, simulator version and seed.
7. Adapt the stress dataset with the same script and evaluate existing Lab-trained artifacts against only its `test/` partition. Since that partition has no positive events, report false-positive rates only; PR-AUC and recall are undefined.
8. Run project tests, inspect Git diff, and keep adapted data/model outside Git.

## Limits

- The source simulator's process models are simplified and uncalibrated against physical plant telemetry.
- Generated event prevalence, severity and type weights are suite choices, not field estimates.
- Results test generalization to new runs within a single simulator family, not cross-industry field transfer.
- Source labels remain evaluation-only. Never select training windows based on event fields passed as model features.
