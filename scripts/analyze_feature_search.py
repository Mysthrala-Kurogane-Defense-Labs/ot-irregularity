"""Add paired-seed dispersion and baseline-difference intervals to a completed search."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from feature_search import aggregate, save_json

METRICS = {
    "pr_auc": "pr_auc",
    "precision": "precision",
    "recall": "recall",
    "event_detection_rate": "event_detection_rate",
    "false_positives_per_asset_day": "false_positives_per_asset_day",
    "latency_seconds": "mean_detection_latency_seconds",
}


def metric_value(row: dict, name: str) -> float | None:
    value = (row.get("metrics") or {}).get("irregularity", {}).get(METRICS[name])
    return None if value is None else float(value)


def summarize(values: list[float]) -> dict:
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def paired_summary(values: list[float], baseline: list[float]) -> dict:
    differences = np.asarray(values, dtype=float) - np.asarray(baseline, dtype=float)
    n = len(differences)
    mean = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1)) if n > 1 else 0.0
    half_width = float(student_t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else None
    return {"n": n, "mean_difference": mean, "sample_sd": sd,
            "ci95_low": mean - half_width if half_width is not None else None,
            "ci95_high": mean + half_width if half_width is not None else None,
            "per_dataset_differences": differences.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "completed" or int(state.get("summary", {}).get("failed_runs", -1)) != 0:
        raise ValueError("Analysis requires a completed run with zero failed trainings")
    rows = [r for r in state["runs"] if r.get("stage") in {"finalist", "control"} and r.get("status") == "completed"]
    by_candidate: dict[str, list[dict]] = {}
    for row in rows: by_candidate.setdefault(row["candidate_id"], []).append(row)
    if "baseline" not in by_candidate: raise ValueError("Missing paired baseline control")
    holdout_ids = sorted({r["dataset_id"] for r in by_candidate["baseline"]})
    expected = int(state["config"]["finalist_repeats"])
    if len(holdout_ids) != expected or any(len(by_candidate.get(candidate, [])) != expected for candidate in by_candidate):
        raise ValueError("Holdout models do not all have the same number of paired datasets")
    datasets_by_candidate = {candidate: {r["dataset_id"]: r for r in candidate_rows}
                             for candidate, candidate_rows in by_candidate.items()}
    if any(set(data) != set(holdout_ids) for data in datasets_by_candidate.values()):
        raise ValueError("Holdout datasets differ between baseline and candidate")
    values = {}
    for candidate, dataset_rows in datasets_by_candidate.items():
        values[candidate] = {}
        for metric in METRICS:
            samples = [metric_value(dataset_rows[dataset_id], metric) for dataset_id in holdout_ids]
            samples = [value for value in samples if value is not None]
            if samples: values[candidate][metric] = summarize(samples)
    baseline_rows = datasets_by_candidate["baseline"]
    paired = {}
    for candidate, dataset_rows in datasets_by_candidate.items():
        if candidate == "baseline": continue
        paired[candidate] = {}
        for metric in METRICS:
            candidate_values = [metric_value(dataset_rows[dataset_id], metric) for dataset_id in holdout_ids]
            baseline_values = [metric_value(baseline_rows[dataset_id], metric) for dataset_id in holdout_ids]
            if all(value is not None for value in candidate_values + baseline_values):
                paired[candidate][metric] = paired_summary(candidate_values, baseline_values)
    labels = {row["id"]: row["label"] for row in state["ranking"]}
    analysis = {"method": "paired per-dataset differences; two-sided Student t 95% confidence interval across independently generated seeds",
                "baseline_candidate_id": "baseline", "holdout_dataset_ids": holdout_ids,
                "holdout_repeats": expected, "metric_dispersion": values,
                "paired_differences_vs_baseline": paired}
    for row in state["ranking"]:
        key = row["id"]
        row["standard_deviation"] = {metric: summary["sample_sd"] for metric, summary in values.get(key, {}).items()}
    state["analysis"] = analysis
    state["updated_at"] = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    save_json(state_path, state)
    (run_dir / "paired_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_path = run_dir / "report.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "## Interpretation limits"
    if marker not in report: raise ValueError("Final report is missing interpretation section")
    extra = ["## Variability over paired holdout seeds", "",
             "Values below show mean ± sample standard deviation across the same 12 independent generated datasets.", "",
             "| Model | PR-AUC | Precision | Recall | Event detection | FP / asset-day | Latency (s) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in state["ranking"]:
        name = row["label"]
        summary = values.get(row["id"], {})
        cells = []
        for metric in METRICS:
            item = summary.get(metric)
            cells.append("—" if item is None else f"{item['mean']:.4f} ± {item['sample_sd']:.4f}")
        extra.append(f"| {name} | " + " | ".join(cells) + " |")
    extra += ["", "## Paired differences versus baseline", "",
              "Positive differences increase the metric; negative differences reduce it. The intervals describe variation across 12 simulator seeds, not uncertainty over real industrial plants.", "",
              "| Candidate | Δ PR-AUC (95% CI) | Δ event detection (95% CI) | Δ FP / asset-day (95% CI) |",
              "|---|---:|---:|---:|"]
    for row in state["ranking"]:
        if row["id"] == "baseline": continue
        diffs = paired.get(row["id"], {})
        cells = []
        for metric in ("pr_auc", "event_detection_rate", "false_positives_per_asset_day"):
            value = diffs.get(metric)
            cells.append("—" if value is None else f"{value['mean_difference']:+.4f} [{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}]")
        extra.append(f"| {row['label']} | " + " | ".join(cells) + " |")
    report_path.write_text(report.replace(marker, "\n".join(extra) + "\n\n" + marker), encoding="utf-8")
    print(json.dumps({"paired_datasets": len(holdout_ids), "models": len(values),
                      "analysis": str(run_dir / "paired_analysis.json"), "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
