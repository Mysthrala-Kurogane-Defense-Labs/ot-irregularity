"""Prepare OT Irregularity Lab runs as a separated pseudo-synthetic benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _epoch_us(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"Ground-truth timestamp must include a UTC offset: {value!r}")
    return int(parsed.timestamp() * 1_000_000)


def _annotate(frame: pl.DataFrame, events: list[dict]) -> pl.DataFrame:
    """Add evaluation-only labels from event intervals; never expose causes as features."""
    time_us = pl.col("timestamp").dt.timestamp("us")
    anomaly = pl.lit(False)
    event_id = pl.lit(None, dtype=pl.String)
    event_start_us = pl.lit(None, dtype=pl.Int64)
    for event in sorted(events, key=lambda item: (item["start"], item["event_id"])):
        start = _epoch_us(event["start"])
        end = _epoch_us(event["end"])
        asset_rows = frame.filter(pl.col("asset_id") == event["asset_id"])
        asset_times = asset_rows.select(pl.col("timestamp").dt.timestamp("us")).to_series().to_list()
        observed_during_event = any(start <= int(value) <= end for value in asset_times)
        fallback_us = None
        if asset_times and not observed_during_event:
            # Whole-asset communication loss can remove every measurement inside
            # an event. Attach evaluation metadata to the closest surviving row
            # so its containing window is still evaluated against ground truth.
            midpoint = (start + end) // 2
            fallback_us = min((int(value) for value in asset_times), key=lambda value: abs(value - midpoint))
        interval = (time_us >= start) & (time_us <= end)
        if fallback_us is not None:
            interval = interval | (time_us == fallback_us)
        condition = (pl.col("asset_id") == event["asset_id"]) & interval
        anomaly = anomaly | condition
        # In the rare case of overlapping events, retain the first interval's ID.
        first = condition & event_id.is_null()
        event_id = pl.when(first).then(pl.lit(event["event_id"])).otherwise(event_id)
        event_start_us = pl.when(first).then(pl.lit(start)).otherwise(event_start_us)
    return frame.with_columns(
        anomaly.alias("is_anomaly"),
        event_id.alias("event_id"),
        event_start_us.alias("event_start_us"),
    )


def normalize_observations(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply Lab semantics without fitting ML transforms or retaining event labels."""
    if "tag_id" in frame.columns:
        frame = frame.with_columns(pl.col("tag_id").cast(pl.String).alias("measurement_role"))
    if "unit" in frame.columns:
        frame = frame.with_columns(
            pl.when(pl.col("unit").cast(pl.String).str.to_lowercase().is_in(["code", "enum", "state"]))
            .then(pl.lit("categorical")).otherwise(pl.lit("continuous")).alias("value_kind"))
    frame = frame.drop([name for name in ("is_anomaly", "event_id", "event_start_us") if name in frame.columns])
    if "quality" in frame.columns:
        frame = frame.with_columns(pl.col("quality").cast(pl.String).str.to_lowercase())
    return frame


def prepare(source: Path, output: Path, *, max_train_runs: int | None = None,
            max_validation_runs: int | None = None, max_test_runs: int | None = None) -> dict:
    """Verify a Lab dataset and emit normal-only train plus mixed validation and labeled test."""
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new path: {output}")
    manifest_path = source / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Source must be a dataset root containing dataset_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("synthetic") is not True or manifest.get("generated") is not True:
        raise ValueError("Source manifest must identify synthetic, generated data")
    if manifest.get("customer_data") is not False:
        raise ValueError("Source manifest must explicitly declare customer_data=false")
    if manifest.get("data_license") != "CC-BY-4.0":
        raise ValueError("Expected source dataset license CC-BY-4.0")
    source_dataset_id = manifest["dataset_id"]
    release_url = (
        "https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab/releases/tag/dataset-v0.3.0"
        if source_dataset_id == "ot-irregularity-training-v0.3-release"
        else "https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab"
    )
    attribution = (
        f"{source_dataset_id}, generated by Mysthrala Kurogane Defense Labs' OT Irregularity Lab "
        f"simulator {manifest['simulator_version']}; CC BY 4.0."
    )

    source_runs = manifest.get("runs", [])
    selected: dict[str, list[tuple[dict, dict]]] = {key: [] for key in ("train", "validation", "test")}
    normal_counts: dict[str, int] = {}
    for partition in selected:
        entries = [entry for entry in source_runs if entry.get("partition") == partition]
        normal = []
        anomalous = []
        for entry in entries:
            run_dir = source / partition / entry["run_id"]
            telemetry = run_dir / "telemetry.parquet"
            truth_path = run_dir / "ground_truth.json"
            if not telemetry.is_file() or not truth_path.is_file():
                raise ValueError(f"Missing telemetry or ground truth for {partition}/{entry['run_id']}")
            if _sha256(telemetry) != entry.get("telemetry_sha256"):
                raise ValueError(f"Telemetry hash mismatch for {partition}/{entry['run_id']}")
            truth = json.loads(truth_path.read_text(encoding="utf-8"))
            if truth.get("run_id") != entry["run_id"]:
                raise ValueError(f"Ground-truth run_id mismatch for {partition}/{entry['run_id']}")
            pair = (entry, truth)
            (normal if not truth.get("events") else anomalous).append(pair)
        normal_counts[partition] = len(normal)
        if partition == "train":
            limit = max_train_runs
            if not normal:
                raise ValueError(f"No event-free runs available in source {partition}")
            selected[partition] = normal[:limit] if limit is not None else normal
        elif partition == "validation":
            # Keep mixed validation runs for feature selection; calibration still uses its normal windows.
            limit = max_validation_runs
            validation_runs = sorted(normal + anomalous, key=lambda pair: pair[0]["run_id"])
            if limit is None:
                selected[partition] = validation_runs
            else:
                chosen = validation_runs[:limit]
                if limit >= 2 and normal and anomalous:
                    if not any(not truth.get("events") for _, truth in chosen): chosen[-1] = normal[0]
                    if not any(truth.get("events") for _, truth in chosen): chosen[-1] = anomalous[0]
                    selected[partition] = sorted(chosen, key=lambda pair: pair[0]["run_id"])
                else:
                    selected[partition] = chosen
        else:
            # Keep the full independent test partition so its declared scenario mix is visible.
            test_runs = sorted(normal + anomalous, key=lambda pair: pair[0]["run_id"])
            selected[partition] = test_runs[:max_test_runs] if max_test_runs is not None else test_runs

    run_ids: dict[str, set[str]] = {
        name: {entry["run_id"] for entry, _ in runs} for name, runs in selected.items()
    }
    if run_ids["train"] & run_ids["validation"] or run_ids["train"] & run_ids["test"] or run_ids["validation"] & run_ids["test"]:
        raise ValueError("Dataset leakage: source run IDs overlap between output partitions")
    if not selected["test"]:
        raise ValueError("No test runs selected")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        partition_stats = {}
        for partition, runs in selected.items():
            frames = []
            interval_events = []
            observation_count = 0
            event_count = 0
            for entry, truth in runs:
                telemetry_path = source / partition / entry["run_id"] / "telemetry.parquet"
                frame = normalize_observations(pl.read_parquet(telemetry_path))
                events = truth.get("events", []) if partition in ("validation", "test") else []
                # Lab event IDs are local to a run (evt-1, evt-2); namespace them
                # because this project's evaluator compares IDs across partitions.
                namespaced_events = [
                    {**event, "event_id": f"{entry['run_id']}/{event['event_id']}"}
                    for event in events
                ]
                interval_events.extend({"run_id":entry["run_id"],"asset_id":event["asset_id"],
                    "event_id":event["event_id"],"start_us":_epoch_us(event["start"]),
                    "end_us":_epoch_us(event["end"])} for event in namespaced_events)
                frame = _annotate(frame, namespaced_events)
                frames.append(frame)
                observation_count += frame.height
                event_count += len(events)
            if not frames:
                raise ValueError(f"No runs selected for {partition}")
            dest = stage / partition
            dest.mkdir()
            pl.concat(frames, how="diagonal_relaxed").write_parquet(dest / "telemetry.parquet", compression="zstd")
            (dest / "events.json").write_text(json.dumps({"schema_version":"1","events":interval_events},indent=2)+"\n",encoding="utf-8")
            partition_stats[partition] = {
                "runs": len(runs),
                "normal_runs": sum(not truth.get("events") for _, truth in runs),
                "event_bearing_runs": sum(bool(truth.get("events")) for _, truth in runs),
                "events": event_count,
                "observations": observation_count,
                "run_ids": sorted(entry["run_id"] for entry, _ in runs),
                "parquet_sha256": _sha256(dest / "telemetry.parquet"),
                "events_sha256": _sha256(dest / "events.json"),
            }
        result = {
            "dataset_id": output.name,
            "synthetic": True,
            "pseudo_synthetic": True,
            "source_dataset_id": source_dataset_id,
            "source_simulator_version": manifest["simulator_version"],
            "source_schema_version": manifest["schema_version"],
            "source_master_seed": manifest.get("master_seed"),
            "source_suite_id": manifest.get("suite_id"),
            "source_suite_version": manifest.get("suite_version"),
            "source_suite_sha256": manifest.get("suite_sha256"),
            "source_manifest_sha256": _sha256(manifest_path),
            "source_license": manifest["data_license"],
            "source_url": release_url,
            "source_attribution": attribution,
            "label_policy": "Train uses event-free runs only. Validation retains a deterministic mixed selection for feature selection and threshold calibration uses only its normal windows. Test is a separate reserved source partition and is not used for candidate selection; is_anomaly, event_id, and event_start_us are evaluation-only ground-truth fields. For an event interval with no surviving asset observation (for example, complete communication loss), metadata is attached to the nearest surviving observation so overlapping windows remain evaluable.",
            "normal_runs_available": normal_counts,
            "partitions": partition_stats,
            "run_ids_disjoint": True,
            "limitations": [
                "Simplified simulator process models are not field-calibrated digital twins.",
                "Synthetic event labels and prevalence reflect benchmark scenario design, not industrial incident rates.",
                "This tests cross-run behavior within one simulator family, not cross-industry or real-plant generalization.",
            ],
        }
        (stage / "dataset_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (stage / "DATASET_LICENSE.txt").write_text(
            f"Adapted from {source_dataset_id}, generated by Mysthrala Kurogane Defense Labs' OT Irregularity Lab simulator {manifest['simulator_version']}.\n"
            "Source data license: Creative Commons Attribution 4.0 International (CC BY 4.0).\n"
            f"Source: {release_url}\n"
            "Changes: selected event-free runs for training/calibration, preserved disjoint run partitions, and added evaluation-only event labels to test telemetry from separate ground-truth files.\n",
            encoding="utf-8",
        )
        os.replace(stage, output)
        return result
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="OT Irregularity Lab dataset root")
    parser.add_argument("--output", type=Path, required=True, help="new output directory, preferably outside Git")
    parser.add_argument("--max-train-runs", type=int)
    parser.add_argument("--max-validation-runs", type=int)
    parser.add_argument("--max-test-runs", type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, max_train_runs=args.max_train_runs,
                             max_validation_runs=args.max_validation_runs, max_test_runs=args.max_test_runs), indent=2))


if __name__ == "__main__":
    main()
