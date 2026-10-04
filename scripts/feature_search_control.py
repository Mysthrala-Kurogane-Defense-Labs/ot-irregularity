"""Add the paired baseline holdout control to a completed feature-search run."""
from __future__ import annotations

import concurrent.futures
import copy
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.feature_search import aggregate, digest, save_json, utc_now  # noqa: E402
from ot_irregularity.models import resolve_device  # noqa: E402


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "feature-search.yaml")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    state_path = run_dir / "state.json"
    event_path = run_dir / "events.jsonl"
    lock = threading.RLock()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "completed":
        raise ValueError("Paired baseline control requires a completed feature-search run")
    if any(row.get("stage") == "control" for row in state.get("runs", [])):
        raise ValueError("A baseline control already exists for this run")
    if digest(args.config) != state["config"]["config_file_sha256"]:
        raise ValueError("Config hash differs from the original experiment")
    device = resolve_device(state["config"]["device"])
    if str(device) != state["host"]["resolved_device"]:
        raise ValueError(f"Device changed since the experiment: {device}")
    dataset_repeats = int(state["config"]["dataset_repeats"])
    screening_repeats = int(state["config"]["screening_repeats"])
    finalist_repeats = int(state["config"]["finalist_repeats"])
    base_seed = int(state["config"]["seed"])
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    python = Path(sys.executable)
    jobs = list(range(screening_repeats, dataset_repeats))
    if len(jobs) != finalist_repeats:
        raise ValueError("Dataset count and finalist repeat count do not form paired holdout seeds")
    state["status"] = "running"
    state["phase"] = "control"
    state["progress"] = 0.96
    state["planned"]["control_runs"] = len(jobs)
    state["host"]["control_workers"] = args.workers
    state["updated_at"] = utc_now()
    save_json(state_path, state)
    with event_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp": utc_now(), "kind": "phase", "phase": "control",
                                "message": "Entrenamiento del baseline en los mismos 12 lotes holdout que los finalistas"}) + "\n")

    def execute(index: int) -> dict:
        dataset = run_dir / "datasets" / f"dev-{index + 1:02d}"
        dataset_meta = state["datasets"][index]
        cfg = copy.deepcopy(base)
        cfg["seed"] = base_seed + index * 997
        cfg["device"] = str(device)
        cfg["autoencoder"]["steps"] = int(state["config"]["steps"])
        cfg["autoencoder"]["epochs"] = max(int(cfg["autoencoder"].get("epochs", 10000)), cfg["autoencoder"]["steps"])
        cfg["autoencoder"]["patience"] = max(int(cfg["autoencoder"].get("patience", 10000)), cfg["autoencoder"]["steps"])
        cfg["autoencoder"]["log_interval"] = max(100, cfg["autoencoder"]["steps"] // 30)
        cfg["isolation_forest"]["random_state"] = cfg["seed"]
        run_id = f"control-baseline-d{index + 1:02d}"
        config_path = run_dir / "configs" / f"{run_id}.yaml"
        output = run_dir / "artifacts" / run_id
        log_path = run_dir / "logs" / f"{run_id}.log"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
        row = {"id": run_id, "stage": "control", "candidate_id": "baseline",
               "candidate_label": "Baseline (paired holdout control)", "dataset_id": dataset.name,
               "dataset_seed": dataset_meta["seed"], "training_seed": cfg["seed"],
               "config_sha256": digest(config_path), "status": "running", "started_at": utc_now(), "progress": 0.0}
        with lock:
            state["runs"].append(row)
            state["updated_at"] = utc_now()
            save_json(state_path, state)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            command = [str(python), "-m", "ot_irregularity.cli", "train", "--dataset", str(dataset),
                       "--config", str(config_path), "--output", str(output)]
            with log_path.open("w", encoding="utf-8") as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            metric_doc = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            supervised = metric_doc.get("supervised_validation", {})
            result = {"autoencoder": supervised.get("autoencoder"),
                      "isolation_forest": supervised.get("isolation_forest"),
                      "irregularity": supervised.get("irregularity")}
            with lock:
                row.update({"status": "completed", "completed_at": utc_now(), "progress": 1.0,
                            "metrics": result,
                            "feature_count": len(json.loads((output / "feature_schema.json").read_text(encoding="utf-8"))["features"])})
        except Exception as exc:
            with lock:
                row.update({"status": "failed", "completed_at": utc_now(), "error": str(exc)})
            with event_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"timestamp": utc_now(), "kind": "run_failed", "run_id": run_id,
                                         "error": str(exc), "log": log_path.name}) + "\n")
        return row

    futures: dict[concurrent.futures.Future, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index in jobs:
            futures[pool.submit(execute, index)] = index
        finished = 0
        while futures:
            for future in [item for item in futures if item.done()]:
                index = futures.pop(future)
                future.result()
                finished += 1
            active = [row for row in state["runs"] if row.get("stage") == "control" and row["status"] == "running"]
            with lock:
                for row in active:
                    progress_path = run_dir / "artifacts" / row["id"] / "training_progress.jsonl"
                    if progress_path.exists():
                        try:
                            event = json.loads(progress_path.read_text(encoding="utf-8").splitlines()[-1])
                            row["training"] = event
                            total = event.get("steps_requested") or state["config"]["steps"]
                            row["progress"] = min(0.99, event.get("steps_completed", 0) / max(total, 1))
                        except (OSError, ValueError, IndexError):
                            pass
                state["active_workers"] = active
                state["progress"] = min(0.995, 0.96 + 0.04 * finished / len(jobs))
                state["updated_at"] = utc_now()
                save_json(state_path, state)
            time.sleep(1)

    screening_rows = [row for row in state["runs"] if row.get("stage") == "screening"]
    paired_rows = [row for row in state["runs"] if row.get("stage") in {"finalist", "control"}]
    state["screening_ranking"] = aggregate(screening_rows)
    state["ranking"] = aggregate(paired_rows)
    all_success = all(row["status"] == "completed" for row in state["runs"])
    state["status"] = "completed" if all_success else "completed_with_errors"
    state["phase"] = "completed"
    state["progress"] = 1.0
    state["active_workers"] = []
    state["completed_at"] = utc_now()
    state["summary"]["successful_runs"] = sum(row["status"] == "completed" for row in state["runs"])
    state["summary"]["failed_runs"] = sum(row["status"] == "failed" for row in state["runs"])
    state["summary"]["control_baseline_runs"] = len(jobs)
    state["summary"]["paired_holdout_datasets"] = [f"dev-{index + 1:02d}" for index in jobs]
    state["summary"]["reserved_test_used"] = False
    state["updated_at"] = utc_now()
    save_json(state_path, state)
    with event_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp": utc_now(), "kind": "control_completed",
                                "successful_runs": state["summary"]["successful_runs"],
                                "failed_runs": state["summary"]["failed_runs"]}) + "\n")

    report = ["# Feature search results", "", f"Run: `{run_dir.name}`  ",
              f"Status: `{state['status']}`  ",
              f"Datasets: {dataset_repeats} independent seeds. Reserved test partition used: **no**.", "",
              "## Screening (3 independent seeds per candidate)", "",
              "The initial ranking selected five ablations. Screening values are selection evidence only; final comparisons below use 12 new paired holdout seeds.", "",
              "## Paired holdout comparison (12 seeds per model)", "",
              "The baseline and all five finalists were trained and evaluated on the same 12 holdout datasets, with paired stochastic seeds.", "",
              "| Rank | Candidate | Repeats | PR-AUC | Precision | Recall | Event detection | FP / asset-day | Latency (s) |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(state["ranking"], 1):
        fmt = lambda value: "—" if value is None else f"{value:.4f}"
        report.append(f"| {rank} | {row['label']} | {row['repeats_completed']} | {fmt(row['pr_auc'])} | {fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['event_detection_rate'])} | {fmt(row['false_positives_per_asset_day'])} | {fmt(row['latency_seconds'])} |")
    report += ["", "## Interpretation limits", "",
               "The suite is pseudo-synthetic, with event prevalence and simplified process models defined by its generator. These metrics do not establish performance on field data.",
               "Scores and ranks use validation labels only; test partitions remained reserved. The baseline is now paired with the finalists on identical holdout datasets and random seeds.",
               "", "See `state.json`, `events.jsonl`, dataset manifests, per-run configs, logs and `artifacts/*/metrics.json` for full evidence.", ""]
    (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"run_id": run_dir.name, "status": state["status"],
                      "control_runs": state["summary"]["control_baseline_runs"],
                      "report": str(run_dir / "report.md")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
