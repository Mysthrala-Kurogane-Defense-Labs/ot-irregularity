"""Join the fixed partitions from the paired feature-search batches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assemble(source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    batches = sorted(p for p in (source / "datasets").glob("dev-*") if p.is_dir())
    if len(batches) != 15:
        raise ValueError(f"Expected 15 independent dev batches, found {len(batches)}")
    attributions = {batch.name: (batch / "DATASET_LICENSE.txt").read_text(encoding="utf-8")
                   for batch in batches}
    if any("CC BY 4.0" not in text for text in attributions.values()):
        raise ValueError("Every source batch must provide CC BY 4.0 attribution")
    license_text = "\n".join(text.rstrip() for text in attributions.values()) + "\n"

    schemas: dict[str, tuple] = {}
    output.mkdir(parents=True)
    partitions: dict[str, dict] = {}
    run_ids: dict[str, set[str]] = {name: set() for name in ("train", "validation", "test")}
    for partition in run_ids:
        frames = []
        rows = 0
        anomalies = 0
        for batch in batches:
            path = batch / partition / "telemetry.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pl.read_parquet(path)
            schema = tuple(frame.schema.items())
            if partition in schemas and schemas[partition] != schema:
                raise ValueError(f"Schema drift in {batch.name}/{partition}")
            schemas[partition] = schema
            required = {"run_id", "asset_id", "tag_id", "timestamp", "value", "is_anomaly", "event_id"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{batch.name}/{partition} missing fields: {sorted(missing)}")
            natural_key = ["run_id", "asset_id", "tag_id", "timestamp"]
            duplicate_rows = frame.select(pl.struct(natural_key).is_duplicated().sum()).item()
            if duplicate_rows:
                raise ValueError(f"{batch.name}/{partition} has {duplicate_rows} duplicate measurement keys")
            if frame["is_anomaly"].null_count():
                raise ValueError(f"Null anomaly labels in {batch.name}/{partition}")
            source_ids = frame["run_id"].drop_nulls().unique().to_list()
            if not source_ids or len(source_ids) != frame["run_id"].n_unique():
                raise ValueError(f"Null or empty run identifiers in {batch.name}/{partition}")
            namespaced_runs = [f"{batch.name}::{value}" for value in source_ids]
            duplicate = run_ids[partition].intersection(namespaced_runs)
            if duplicate:
                raise ValueError(f"Duplicate run IDs in {partition}: {sorted(duplicate)[:3]}")
            run_ids[partition].update(namespaced_runs)
            frame = frame.with_columns(
                pl.col("run_id").map_elements(lambda value: f"{batch.name}::{value}", return_dtype=pl.String),
                pl.col("event_id").map_elements(
                    lambda value: f"{batch.name}::{value}" if value is not None else None,
                    return_dtype=pl.String,
                ),
                pl.lit(batch.name).alias("source_dataset_id"),
            )
            rows += frame.height
            if partition == "train" and "is_anomaly" in frame.columns:
                anomalies += int(frame["is_anomaly"].sum())
            frames.append(frame)
        if partition == "train" and anomalies:
            raise ValueError(f"Training partition contains {anomalies} anomalous observations")
        target = output / partition
        target.mkdir()
        parquet = target / "telemetry.parquet"
        pl.concat(frames, how="vertical").write_parquet(parquet, compression="zstd")
        partitions[partition] = {"rows": rows, "runs": len(run_ids[partition]),
                                "parquet_sha256": sha256(parquet)}

    overlaps = {f"{a}/{b}": sorted(run_ids[a] & run_ids[b])
                for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    if any(overlaps.values()):
        raise ValueError(f"Run leakage between partitions: {overlaps}")
    manifest = {
        "dataset_id": output.name,
        "source_run_id": source.name,
        "source": "OT Irregularity Lab generated pseudo-synthetic batches",
        "source_url": "https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab",
        "license": "CC-BY-4.0",
        "source_attributions": attributions,
        "source_dataset_manifests": {
            batch.name: sha256(batch / "dataset_manifest.json") for batch in batches
        },
        "partition_policy": "all 15 normal-only train partitions, mixed validation partitions, and untouched test partitions; run and event IDs namespaced by source batch",
        "partitions": partitions,
        "run_ids_disjoint": True,
    }
    (output / "DATASET_LICENSE.txt").write_text(license_text, encoding="utf-8")
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.search_run, args.output), indent=2))


if __name__ == "__main__":
    main()
