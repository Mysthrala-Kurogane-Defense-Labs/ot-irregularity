"""Generate seeded Lab development sets, run staged feature ablations and serve live JSON state."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as dt
import hashlib
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

import polars as pl
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_otlab import prepare  # noqa: E402
from ot_irregularity.models import resolve_device  # noqa: E402


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        for attempt in range(8):
            try:
                os.replace(temp, path)
                return
            except PermissionError:
                if attempt == 7: raise
                time.sleep(0.04 * (attempt + 1))
    finally:
        temp.unlink(missing_ok=True)


class Experiment:
    def __init__(self, run_dir: Path, args: argparse.Namespace):
        self.run_dir = run_dir
        self.args = args
        self.state_path = run_dir / "state.json"
        self.event_path = run_dir / "events.jsonl"
        self.lock = threading.RLock()
        self.state = {"run_id": run_dir.name, "created_at": utc_now(), "updated_at": utc_now(),
                      "status": "initializing", "phase": "initializing", "progress": 0.0,
                      "datasets": [], "candidates": [], "runs": [], "active_workers": [], "errors": [],
        "config": {"seed": args.seed, "config_file_sha256": digest(args.config), "dataset_repeats": args.dataset_repeats, "screening_repeats": args.screening_repeats,
                                 "finalist_repeats": args.finalist_repeats, "max_finalists": args.max_finalists,
                                 "runs_per_dataset": args.runs_per_dataset, "steps": args.steps, "device": args.device, "workers": args.workers}}
        self.event("created", message="Búsqueda de features preparada")
        self.flush()

    def event(self, kind: str, **data) -> None:
        item = {"timestamp": utc_now(), "kind": kind, **data}
        with self.lock:
            with self.event_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        with self.lock:
            self.state["updated_at"] = utc_now()
            save_json(self.state_path, self.state)

    def phase(self, name: str, message: str | None = None) -> None:
        self.state["phase"] = name
        if message:
            self.event("phase", phase=name, message=message)
        self.flush()


def build_candidates(classes: list[str]) -> list[dict]:
    candidates = [{"id": "baseline", "label": "Todas las señales y familias", "features": {}}]
    for name in classes:
        candidates.append({"id": "without-class-" + name, "label": f"Sin signal_class: {name}",
                           "features": {"exclude_signal_classes": [name]}})
    for family in ("statistical", "slopes", "quality", "sampling"):
        candidates.append({"id": "without-" + family, "label": f"Sin familia: {family}",
                           "features": {family: False}})
    return candidates


def aggregate(rows: list[dict]) -> list[dict]:
    by_id: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("status") == "completed" and row.get("metrics"):
            by_id.setdefault(row["candidate_id"], []).append(row)
    result = []
    for candidate_id, repeats in by_id.items():
        label = repeats[0]["candidate_label"]
        vals = [r["metrics"] for r in repeats]
        def mean(path):
            numbers = [m.get(path[0], {}).get(path[1]) for m in vals]
            numbers = [float(x) for x in numbers if x is not None]
            return sum(numbers) / len(numbers) if numbers else None
        result.append({"id": candidate_id, "label": label, "repeats_completed": len(repeats),
                       "pr_auc": mean(("irregularity", "pr_auc")),
                       "precision": mean(("irregularity", "precision")),
                       "recall": mean(("irregularity", "recall")),
                       "event_detection_rate": mean(("irregularity", "event_detection_rate")),
                       "false_positives_per_asset_day": mean(("irregularity", "false_positives_per_asset_day")),
                       "latency_seconds": mean(("irregularity", "mean_detection_latency_seconds"))})
    for metric, reverse in (("pr_auc", True), ("event_detection_rate", True), ("false_positives_per_asset_day", False)):
        ordered = sorted((c for c in result if c[metric] is not None), key=lambda c: c[metric], reverse=reverse)
        for rank, candidate in enumerate(ordered, 1):
            candidate.setdefault("rank_sum", 0)
            candidate.setdefault("rank_count", 0)
            candidate["rank_sum"] += rank
            candidate["rank_count"] += 1
    for candidate in result:
        candidate["mean_rank"] = candidate.get("rank_sum", 0) / candidate.get("rank_count", 1) if candidate.get("rank_sum") else None
        candidate.pop("rank_sum", None);candidate.pop("rank_count", None)
    return sorted(result, key=lambda c: (c["mean_rank"] is None, c["mean_rank"] or 1e9, -(c["pr_auc"] or 0), c["id"]))


def run_search(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    if device.type == "cuda":
        gpu_info = {"device_name": torch.cuda.get_device_name(device),
                    "memory_total_mb": round(torch.cuda.get_device_properties(device).total_memory / 1048576)}
    else:
        gpu_info = {"device_name": "CPU"}
    run_dir = (args.runs_root / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    exp = Experiment(run_dir, args)
    source_root = args.simulator_project.resolve()
    suite = source_root / "suites" / "training-v0.2.yaml"
    if not suite.is_file():
        raise FileNotFoundError(f"No se encuentra suite de generación: {suite}")
    python = Path(sys.executable)
    lab_source = source_root / "src"
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.dataset_repeats != args.screening_repeats + args.finalist_repeats:
        raise ValueError("dataset_repeats debe ser screening_repeats + finalist_repeats")
    exp.state.update({"status": "running", "source": {"suite": suite.name, "suite_sha256": digest(suite),
        "license": "CC-BY-4.0", "source": "OT Irregularity Lab generated pseudo-synthetic data"},
        "host": {"device_requested": args.device, "resolved_device": str(device), "gpu": gpu_info, "worker_limit": args.workers},
        "planned": {"datasets": args.dataset_repeats, "screening_runs": None, "finalist_runs": None}})
    exp.phase("generating", "Generando lotes con semillas independientes")
    datasets: list[Path] = []
    env = os.environ.copy()
    old_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(lab_source) + (os.pathsep + old_path if old_path else "")
    lab_code = "from ot_lab.cli import main; main()"
    for index in range(args.dataset_repeats):
        seed = args.seed + index * 7919
        raw = run_dir / "datasets" / f"raw-{index + 1:02d}"
        prepared = run_dir / "datasets" / f"dev-{index + 1:02d}"
        raw.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(python), "-c", lab_code, "dataset", "create", "--suite", str(suite), "--runs", str(args.runs_per_dataset),
               "--seed", str(seed), "--workers", str(min(4, args.workers)), "--output", str(raw)]
        log_path = run_dir / "logs" / f"generate-{index + 1:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(cmd, cwd=source_root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest = prepare(raw, prepared, max_train_runs=84, max_validation_runs=18, max_test_runs=18)
        datasets.append(prepared)
        exp.state["datasets"].append({"id": prepared.name, "seed": seed, "manifest_sha256": digest(prepared / "dataset_manifest.json"),
            "license": "CC-BY-4.0", "train_runs": manifest["partitions"]["train"]["runs"],
            "validation_runs": manifest["partitions"]["validation"]["runs"], "test_runs_reserved": manifest["partitions"]["test"]["runs"],
            "test_used": False})
        exp.state["progress"] = 0.1 * len(datasets) / args.dataset_repeats
        exp.event("dataset_ready", dataset_id=prepared.name, seed=seed, index=index + 1, total=args.dataset_repeats)
        exp.flush()
        if index == 0:
            tr = pl.read_parquet(prepared / "train" / "telemetry.parquet", columns=["signal_class", "value_kind"])
            if "value_kind" in tr.columns:
                tr = tr.filter(pl.col("value_kind").is_null() | (pl.col("value_kind") == "continuous"))
            classes = sorted(str(x) for x in tr["signal_class"].drop_nulls().unique().to_list())
            candidates = build_candidates(classes)
            exp.state["candidates"] = [{"id": c["id"], "label": c["label"], "status": "queued"} for c in candidates]
            exp.state["planned"].update({"screening_runs": len(candidates) * args.screening_repeats,
                                         "finalist_runs": args.max_finalists * args.finalist_repeats})
            exp.flush()

    candidates = build_candidates(classes)
    exp.phase("screening", "Cribado de configuraciones en los tres primeros lotes")
    all_rows: list[dict] = []

    def execute(candidate: dict, dataset_index: int, stage_name: str) -> dict:
        repeat_id = f"{stage_name}-{candidate['id']}-d{dataset_index + 1:02d}"
        item = {"id": repeat_id, "stage": stage_name, "candidate_id": candidate["id"], "candidate_label": candidate["label"],
                "dataset_id": datasets[dataset_index].name, "dataset_seed": exp.state["datasets"][dataset_index]["seed"],
                "status": "running", "started_at": utc_now(), "progress": 0.0}
        out = run_dir / "artifacts" / repeat_id
        cfg = copy.deepcopy(base)
        # Pair each candidate on the same data split with the same stochastic seed.
        cfg["seed"] = args.seed + dataset_index * 997
        cfg["device"] = args.device
        cfg["autoencoder"]["steps"] = args.steps
        cfg["autoencoder"]["epochs"] = max(int(cfg["autoencoder"].get("epochs", 10000)), args.steps)
        cfg["autoencoder"]["patience"] = max(int(cfg["autoencoder"].get("patience", 10000)), args.steps)
        cfg["autoencoder"]["log_interval"] = max(100, args.steps // 30)
        cfg["isolation_forest"]["random_state"] = cfg["seed"]
        cfg["features"].update(candidate["features"])
        cfg_path = run_dir / "configs" / f"{repeat_id}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
        item["training_seed"] = cfg["seed"]
        item["config_sha256"] = digest(cfg_path)
        with exp.lock:
            exp.state["runs"].append(item)
            exp.flush()
        output_log = run_dir / "logs" / f"{repeat_id}.log"
        output_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            cmd = [str(python), "-m", "ot_irregularity.cli", "train", "--dataset", str(datasets[dataset_index]),
                   "--config", str(cfg_path), "--output", str(out)]
            with output_log.open("w", encoding="utf-8") as stream:
                subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            metrics_file = out / "metrics.json"
            metrics_doc = json.loads(metrics_file.read_text(encoding="utf-8"))
            result = metrics_doc.get("supervised_validation", {})
            metrics = {"autoencoder": result.get("autoencoder"), "isolation_forest": result.get("isolation_forest"),
                       "irregularity": result.get("irregularity")}
            with exp.lock:
                item.update({"status": "completed", "completed_at": utc_now(), "progress": 1.0, "metrics": metrics,
                             "feature_count": len(json.loads((out / "feature_schema.json").read_text(encoding="utf-8"))["features"])})
        except Exception as exc:
            with exp.lock:
                item.update({"status": "failed", "completed_at": utc_now(), "error": str(exc)})
            exp.event("run_failed", run_id=repeat_id, error=str(exc), log=output_log.name)
        return item

    def run_batch(job_list: list[tuple[dict, int, str]], progress_base: float, progress_span: float):
        futures = {}
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for candidate, dataset_index, stage_name in job_list:
                future = pool.submit(execute, candidate, dataset_index, stage_name)
                futures[future] = (candidate, dataset_index)
            active_ids: set[str] = set()
            while futures:
                completed = [f for f in futures if f.done()]
                for future in completed:
                    candidate, dataset_index = futures.pop(future)
                    row = future.result()
                    all_rows.append(row)
                    done += 1
                    if row["status"] == "completed": exp.event("run_completed", run_id=row["id"], candidate_id=candidate["id"], metrics=row["metrics"])
                with exp.lock:
                    active = [r for r in exp.state["runs"] if r["status"] == "running"]
                    exp.state["active_workers"] = active
                    exp.state["progress"] = min(0.98, progress_base + progress_span * done / len(job_list))
                    for row in active:
                        progress_file = run_dir / "artifacts" / row["id"] / "training_progress.jsonl"
                        if progress_file.exists():
                            try:
                                last = json.loads(progress_file.read_text(encoding="utf-8").splitlines()[-1])
                                row["training"] = last
                                steps = last.get("steps_requested") or args.steps
                                row["progress"] = min(0.99, float(last.get("steps_completed", 0)) / max(steps, 1))
                            except (ValueError, IndexError, OSError):
                                pass
                    exp.state["ranking"] = aggregate(all_rows)
                    exp.flush()
                time.sleep(1.0)

    screen_jobs = [(candidate, index, "screening") for candidate in candidates for index in range(args.screening_repeats)]
    run_batch(screen_jobs, 0.1, 0.5)
    ranking = aggregate(all_rows)
    finalist_ids = [row["id"] for row in ranking[:args.max_finalists]]
    finalists = [candidate for candidate in candidates if candidate["id"] in finalist_ids]
    for row in exp.state["candidates"]:
        row["status"] = "finalist" if row["id"] in finalist_ids else "screened"
    exp.state["ranking"] = ranking
    exp.state["finalists"] = finalist_ids
    exp.flush()
    exp.phase("finalists", f"Repeticiones independientes de {len(finalists)} finalistas")
    final_jobs = [(candidate, index, "finalist") for candidate in finalists
                  for index in range(args.screening_repeats, args.screening_repeats + args.finalist_repeats)]
    run_batch(final_jobs, 0.6, 0.38)
    final_rows = [r for r in all_rows if r["stage"] == "finalist"]
    exp.state["ranking"] = aggregate(final_rows)
    exp.state["status"] = "completed" if all(row["status"] == "completed" for row in exp.state["runs"]) else "completed_with_errors"
    exp.state["phase"] = "completed"
    exp.state["progress"] = 1.0
    exp.state["completed_at"] = utc_now()
    exp.state["summary"] = {"screening_candidates": len(candidates), "screening_repeats_per_candidate": args.screening_repeats,
        "finalist_ids": finalist_ids, "finalist_repeats_per_candidate": args.finalist_repeats,
        "successful_runs": sum(r["status"] == "completed" for r in exp.state["runs"]),
        "failed_runs": sum(r["status"] == "failed" for r in exp.state["runs"]),
        "reserved_test_used": False, "selection_metrics": ["PR-AUC", "event detection rate", "false positives per asset-day"],
        "dataset_sha256": {d["id"]: d["manifest_sha256"] for d in exp.state["datasets"]}}
    exp.event("search_completed", **exp.state["summary"])
    exp.flush()
    report = "# Feature search results\n\n" + f"Run: `{run_dir.name}`  \nStatus: `{exp.state['status']}`  \n" + \
        f"Datasets: {args.dataset_repeats} independent seeds. Reserved test partition used: **no**.\n\n" + \
        "## Finalist ranking\n\n| Rank | Candidate | Repeats | PR-AUC | Precision | Recall | Event detection | FP / asset-day | Latency (s) |\n|---:|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for rank, row in enumerate(exp.state["ranking"], 1):
        fmt = lambda value: "—" if value is None else f"{value:.4f}"
        report += f"| {rank} | {row['label']} | {row['repeats_completed']} | {fmt(row['pr_auc'])} | {fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['event_detection_rate'])} | {fmt(row['false_positives_per_asset_day'])} | {fmt(row['latency_seconds'])} |\n"
    report += "\nAll evidence is from generated data using the OT Irregularity Lab suite (CC BY 4.0); results do not establish real-plant performance. See `state.json`, `events.jsonl`, dataset manifests, and per-run `metrics.json` for details.\n"
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"run_id": run_dir.name, "status": exp.state["status"], "report": str(run_dir / "report.md")}, ensure_ascii=False), flush=True)


def serve(args: argparse.Namespace) -> None:
    runs_root = args.runs_root.resolve()
    web_root = ROOT / "web" / "feature-search"
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/runs":
                rows = []
                for path in runs_root.iterdir() if runs_root.exists() else []:
                    state = path / "state.json"
                    if path.is_dir() and state.exists():
                        try: rows.append(json.loads(state.read_text(encoding="utf-8")))
                        except (OSError, ValueError): pass
                rows.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                return self.send_json(rows)
            if parsed.path == "/api/state":
                run_id = urllib.parse.parse_qs(parsed.query).get("run_id", [""])[0]
                if not run_id or Path(run_id).name != run_id:
                    return self.send_error(400, "run_id inválido")
                path = runs_root / run_id / "state.json"
                if not path.is_file(): return self.send_error(404)
                return self.send_json(json.loads(path.read_text(encoding="utf-8")))
            assets = {"/": "index.html", "/app.css": "app.css", "/app.js": "app.js"}
            name = assets.get(parsed.path)
            if not name: return self.send_error(404)
            content_type = "text/html; charset=utf-8" if name.endswith("html") else "text/css; charset=utf-8" if name.endswith("css") else "text/javascript; charset=utf-8"
            self.send_response(200); self.send_header("Content-Type", content_type); self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str((web_root / name).stat().st_size)); self.end_headers()
            self.wfile.write((web_root / name).read_bytes())
        def send_json(self, value):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def log_message(self, fmt, *values):
            if args.verbose: super().log_message(fmt, *values)
    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--simulator-project", type=Path, required=True)
    run.add_argument("--runs-root", type=Path, required=True)
    run.add_argument("--config", type=Path, default=ROOT / "configs" / "feature-search.yaml")
    run.add_argument("--run-id", default=dt.datetime.now().strftime("feature-search-%Y%m%d-%H%M%S"))
    run.add_argument("--seed", type=int, default=20261004)
    run.add_argument("--runs-per-dataset", type=int, default=120)
    run.add_argument("--dataset-repeats", type=int, default=15)
    run.add_argument("--screening-repeats", type=int, default=3)
    run.add_argument("--finalist-repeats", type=int, default=12)
    run.add_argument("--max-finalists", type=int, default=5)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--steps", type=int, default=30000)
    run.add_argument("--device", default="cuda:0")
    web = sub.add_parser("serve")
    web.add_argument("--runs-root", type=Path, required=True)
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.command == "run": run_search(args)
    else: serve(args)
