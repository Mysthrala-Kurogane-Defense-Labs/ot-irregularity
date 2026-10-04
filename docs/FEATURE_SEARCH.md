# Reproducible feature search

`scripts/feature_search.py` runs a staged set of signal and feature-family ablations. It writes atomic `state.json` snapshots and an append-only `events.jsonl` stream after each phase, dataset, and training update. The local dashboard refreshes every 1.5 seconds.

## Start the local dashboard

Keep generated datasets, logs, checkpoints and model artifacts outside the repository:

```powershell
$runs = '$OT_IRREGULARITY_LOCAL\feature-search'
Start-Process -WindowStyle Hidden -FilePath '.\.venv\Scripts\python.exe' -ArgumentList @('scripts\feature_search.py','serve','--runs-root',$runs,'--host','127.0.0.1','--port','8765')
```

Open `http://127.0.0.1:8765/`. The server binds to loopback and serves read-only state plus static files from `web/feature-search/`.

## Run

```powershell
uv run python scripts/feature_search.py run `
  --simulator-project $OT_LAB_CHECKOUT `
  --runs-root $OT_IRREGULARITY_LOCAL\feature-search `
  --config configs\feature-search.yaml
```

For CUDA-specific environments, invoke the same CLI with the project interpreter directly so the selected torch build stays fixed: `.venv\Scripts\python.exe scripts\feature_search.py run ...`.

The simulator's Python source is imported with the model project's Python environment. The simulator checkout is read-only for this workflow. CLI flags can override repeat counts, run count, steps, workers, seed and device; `dataset-repeats` must equal `screening-repeats + finalist-repeats`.

## Experiment protocol

1. Generate independently seeded datasets from `suites/training-v0.2.yaml` (CC BY 4.0).
2. Prepare normal-only train, mixed labeled validation, and a disjoint test partition. The test partition is retained in every dataset and never loaded for candidate selection.
3. Screen the full continuous baseline, one broad signal-class ablation at a time, and the `statistical`, `slopes`, `quality`, and `sampling` feature-family ablations. Categorical `value_kind` rows are excluded until categorical features are implemented.
4. Rank on validation PR-AUC, event detection rate and false positives per asset-day. Finalists receive the configured additional independent datasets.
5. Train the baseline as a paired holdout control on those same additional datasets and seeds. This supports a direct comparison when screening selected an ablation ahead of the full model.
6. `scripts/analyze_feature_search.py --run-dir <completed-run>` computes mean/sample deviation and paired per-dataset 95% Student t intervals versus the baseline. These intervals describe seed variation in this generator only.
7. Save every config, training output, metrics file, data manifest hash, event and final report under the selected run directory.

This is a grouped ablation study, not an exhaustive search over all raw columns. Screening-based finalist selection introduces selection bias. The synthetic suite's event prevalence and process models do not represent field incident rates or prove performance on physical equipment. Thresholds remain operating points; the ranking guides the next experiment rather than operational deployment.

## JSON state and API

- `state.json`: current phase, percentage, host/device, data provenance, candidate ranking, active progress records, repeat metrics, failures and summary.
- `events.jsonl`: timestamped phase, dataset-ready, run-complete/failure and terminal events.
- `paired_analysis.json`: per-seed metric variation and paired differences, including confidence intervals.
- `GET /api/runs`: latest experiment snapshots.
- `GET /api/state?run_id=<id>`: one experiment snapshot.

Paths and generated telemetry are kept out of Git. Each candidate run has a saved config, metrics, feature schema, model/scaler files and log.

See [the 2026-10-04 run record](workflows/feature-search/runs/2026-10-04.md) for provenance, outcome and limitations.
