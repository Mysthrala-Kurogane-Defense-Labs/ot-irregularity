"""Prepare a reproducible, external 3W benchmark without bundling source data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import polars as pl

EXPECTED_SHA256 = "fed2af7c6f607b46d4963fe7eb7ee7ada7df3cfde0885f205a6408d0c2adff69"
PARTITIONS = {"train": {0, 1}, "validation": {2}, "test": {4}}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(archive: Path, output: Path, verify_hash: bool = True) -> dict:
    actual = digest(archive)
    if verify_hash:
        if actual != EXPECTED_SHA256:
            raise ValueError(f"Unexpected 3W archive SHA-256: {actual}")

    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        folds = list(csv.DictReader(io.TextIOWrapper(bundle.open("folds/folds_clf_02.csv"), encoding="utf-8-sig")))
        inventory = set(bundle.namelist())
        summary = {"archive_sha256": actual, "source_hash_verified": verify_hash, "split": {}, "source": "Petrobras 3W Dataset 1.1.1, folds_clf_02.csv", "missing_fold_references": [], "duplicate_fold_entries_skipped": [], "quality_policy": "Source CSV does not provide quality; normalized as uncertain, and quality features are disabled in configs/benchmark-3w.yaml."}

        for partition, selected_folds in PARTITIONS.items():
            rows = [row for row in folds if int(row["fold"]) in selected_folds]
            if partition == "train":
                rows = [row for row in rows if row["instancia"].startswith("0/")]
            frames = []
            instances = []
            duplicate_rows_removed = 0
            seen_instances = set()
            for row in rows:
                name = row["instancia"]
                if name in seen_instances:
                    summary["duplicate_fold_entries_skipped"].append({"partition": partition, "path": name, "fold": int(row["fold"])})
                    continue
                seen_instances.add(name)
                if name not in inventory:
                    summary["missing_fold_references"].append(name)
                    continue
                run_id = name.replace("/", "__").removesuffix(".csv")
                instance_class = name.split("/", 1)[0]
                wide = pl.read_csv(bundle.open(name), try_parse_dates=True, null_values=[""])
                if "timestamp" not in wide.columns or "class" not in wide.columns:
                    raise ValueError(f"Unexpected source schema in {name}")
                signal_columns = [column for column in wide.columns if column not in {"timestamp", "class"}]
                if not signal_columns:
                    raise ValueError(f"No sensor columns in {name}")
                wide = wide.with_columns(pl.col("class").cast(pl.Int64, strict=True))
                frame = wide.unpivot(index=["timestamp", "class"], on=signal_columns, variable_name="signal_class", value_name="value")
                frame = frame.filter(pl.col("value").is_not_null()).with_columns(
                    pl.lit(run_id).alias("run_id"),
                    pl.lit(Path(name).stem.split("_")[0]).alias("asset_id"),
                    pl.col("signal_class").alias("tag_id"),
                    pl.lit("oil_well").alias("asset_class"),
                    pl.lit("unknown").alias("operating_regime"),
                    pl.lit("uncertain").alias("quality"),
                    pl.lit(1000.0).alias("sampling_interval_ms"),
                    (pl.col("class") != 0).alias("is_anomaly"),
                    pl.when(pl.col("class") != 0).then(pl.concat_str([pl.lit(run_id), pl.col("class").cast(pl.String)], separator=":" )).otherwise(None).alias("event_id"),
                    pl.lit("unknown (not specified in source file)").alias("unit"),
                ).select("run_id", "asset_id", "asset_class", "timestamp", "tag_id", "signal_class", "value", "unit", "quality", "sampling_interval_ms", "operating_regime", "is_anomaly", "event_id")
                unique = frame.unique(subset=["run_id", "asset_id", "timestamp", "tag_id"], maintain_order=True)
                duplicate_rows_removed += frame.height - unique.height
                frame = unique
                frames.append(frame)
                instances.append({"path": name, "fold": int(row["fold"]), "is_ova": row["is_ova"].lower() == "true", "source_directory_class": int(instance_class), "rows_wide": wide.height, "signals": signal_columns})
            if not frames:
                raise ValueError(f"No selected source instances for {partition}")
            data = pl.concat(frames, how="vertical_relaxed")
            (output / partition).mkdir(parents=True, exist_ok=True)
            data.write_parquet(output / partition / "3w.parquet", compression="zstd")
            summary["split"][partition] = {
                "rows_long": data.height,
                "identical_duplicate_rows_removed": duplicate_rows_removed,
                "run_count": len(instances),
                "normal_runs": sum(item["source_directory_class"] == 0 for item in instances),
                "anomalous_runs": sum(item["source_directory_class"] != 0 for item in instances),
                "instances": instances,
            }

    (output / "source_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Path to the downloaded 3W Dataset 1.1.1 ZIP")
    parser.add_argument("--output", type=Path, required=True, help="External output directory for normalized partitions")
    parser.add_argument("--skip-hash-check", action="store_true", help="Allow another archive revision (its SHA-256 is still recorded)")
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output, not args.skip_hash_check), indent=2))


if __name__ == "__main__":
    main()
