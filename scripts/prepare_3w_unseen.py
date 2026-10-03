"""Prepare a bounded, deterministic 3W unseen-well benchmark outside Git."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import polars as pl
import yaml

V1_SHA256 = "fed2af7c6f607b46d4963fe7eb7ee7ada7df3cfde0885f205a6408d0c2adff69"
V2_MD5 = "2c23b87b60c5d19ed9cf9559efa6ffa7"
SIGNALS = {
    "P-PDG": ("downhole_pressure", "Pa"),
    "P-TPT": ("subsea_tree_pressure", "Pa"),
    "T-TPT": ("subsea_tree_temperature", "degC"),
    "P-MON-CKP": ("production_choke_upstream_pressure", "Pa"),
    "T-JUS-CKP": ("production_choke_downstream_temperature", "degC"),
    "P-JUS-CKGL": ("gas_lift_choke_downstream_pressure", "Pa"),
    "QGL": ("gas_lift_flow", "m3/s"),
}
INVALID_VALUES = {"P-PDG": (-1.180116e42,)}
WELL_RE = re.compile(r"WELL-(\d+)", re.I)


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _well(path: str) -> int | None:
    match = WELL_RE.search(path)
    return int(match.group(1)) if match else None


def _spread(paths: list[str], limit: int) -> list[str]:
    paths = sorted(paths)
    if len(paths) <= limit:
        return paths
    if limit == 1:
        return [paths[len(paths) // 2]]
    return [paths[round(i * (len(paths) - 1) / (limit - 1))] for i in range(limit)]


def _select(v1: zipfile.ZipFile, v2: zipfile.ZipFile) -> dict[str, list[str]]:
    n1 = [n for n in v1.namelist() if n.lower().endswith(".csv") and n.split("/", 1)[0].isdigit()]
    n2 = [n for n in v2.namelist() if n.lower().endswith(".parquet") and n.split("/", 1)[0].isdigit()]
    selected: dict[str, list[str]] = {name: [] for name in ("train", "validation", "test", "challenge")}
    # WELL-00003 has files below class 0 but their class label is null in the
    # released files; exclude them instead of treating unlabeled data as normal.
    for well in (1, 2, 4, 5):
        candidates = [n for n in n1 if n.startswith("0/") and _well(n) == well]
        selected["train"].extend(_spread(candidates, 12))
    for well in range(6, 9):
        candidates = [n for n in n1 if n.startswith("0/") and _well(n) == well]
        selected["validation"].extend(_spread(candidates, 12))
        for cls in range(1, 9):
            selected["validation"].extend(n for n in n1 if n.startswith(f"{cls}/") and _well(n) == well)
    # 3W 2.0.0 contributes one new well with normal data, and event-only
    # records for other new wells. Include one representative event instance
    # per well, preferring earlier lexical paths for deterministic selection.
    normal19 = [n for n in n2 if n.startswith("0/") and _well(n) == 19]
    selected["test"].extend(_spread(normal19, 1))
    for well in [19, *range(21, 33)]:
        events = [n for n in n2 if n.split("/", 1)[0] in {str(c) for c in range(1, 9)} and _well(n) == well]
        selected["test"].extend(_spread(events, 1))
    # Folder membership alone is not a label: some files in 9/ contain only
    # class 0. Require a confirmed class 9 or transient class 109 observation.
    candidates_by_well: dict[int, list[tuple[int, str]]] = {}
    for name in n2:
        well = _well(name)
        if not name.startswith("9/") or well not in {14, 16, 20, 37, 40, 41, 42}:
            continue
        labels = pl.read_parquet(io.BytesIO(v2.read(name)), columns=["class"])["class"].drop_nulls().unique().to_list()
        if any(int(value) in {9, 109} for value in labels):
            candidates_by_well.setdefault(well, []).append((v2.getinfo(name).file_size, name))
    for well in (14, 16, 20, 37, 40, 41, 42):
        candidates = sorted(candidates_by_well.get(well, []))
        if not candidates:
            raise ValueError(f"No confirmed class-9 event instance for WELL-{well:05d}")
        selected["challenge"].append(candidates[0][1])
    for part, names in selected.items():
        selected[part] = sorted(set(names))
        if not selected[part]:
            raise ValueError(f"No source instances selected for {part}")
    return selected


def _read_member(bundle: zipfile.ZipFile, name: str) -> pl.DataFrame:
    raw = bundle.read(name)
    if name.endswith(".parquet"):
        return pl.read_parquet(io.BytesIO(raw))
    return pl.read_csv(io.BytesIO(raw), try_parse_dates=True, null_values=[""])


def _normalize_member(bundle: zipfile.ZipFile, name: str, version: str, invalid_values: dict | None = None, signals_map: dict | None = None) -> tuple[list[pl.DataFrame], dict]:
    source = _read_member(bundle, name)
    if "timestamp" not in source.columns or "class" not in source.columns:
        raise ValueError(f"Unexpected source schema: {version}:{name}")
    signals_map = signals_map or SIGNALS
    available = [tag for tag in signals_map if tag in source.columns]
    if len(available) < 4:
        raise ValueError(f"Too few common channels in {version}:{name}: {available}")
    invalid_values = invalid_values or INVALID_VALUES
    sentinel_count = 0
    signal_exprs = []
    for tag in available:
        expr = pl.col(tag).cast(pl.Float64, strict=True)
        sentinels = invalid_values.get(tag, [])
        if sentinels:
            labeled_sentinel = expr.is_in(sentinels) & pl.col("class").is_not_null()
            sentinel_count += int(source.select(labeled_sentinel.sum()).item() or 0)
            expr = pl.when(expr.is_in(sentinels)).then(None).otherwise(expr)
        signal_exprs.append(expr.alias(tag))
    source = source.with_columns(
        pl.col("timestamp").cast(pl.Datetime("us"), strict=True),
        pl.col("class").cast(pl.Int64, strict=False).alias("_label"),
        *signal_exprs,
    ).sort("timestamp")
    # Null labels denote unlabelled transition/context samples in 3W. Do not
    # infer their status or bridge across them; only labeled runs are scored.
    source = source.filter(pl.col("_label").is_not_null())
    if source.is_empty():
        return [], {"path": name, "rows_labeled": 0, "segments": 0, "signals": available, "invalid_sentinel_observations": sentinel_count}
    positive = pl.col("_label") != 0
    source = source.with_columns(positive.cast(pl.Int8).alias("_binary"))
    times = source["timestamp"].dt.timestamp("ms").to_list()
    labels = source["_binary"].to_list()
    breaks = [0]
    for i in range(1, len(times)):
        if labels[i] != labels[i - 1] or times[i] - times[i - 1] > 2500:
            breaks.append(i)
    breaks.append(len(times))
    asset = f"WELL-{_well(name):05d}" if _well(name) is not None else "unknown"
    segments: list[pl.DataFrame] = []
    for index, (start, end) in enumerate(zip(breaks, breaks[1:])):
        piece = source.slice(start, end - start)
        if piece.height < 2:
            continue
        anomaly = bool(labels[start])
        run_id = f"{version}__{name.replace('/', '__')}__segment-{index:03d}"
        event_id = run_id if anomaly else None
        wide = piece.select("timestamp", "_binary", *available)
        long = wide.unpivot(index=["timestamp", "_binary"], on=available, variable_name="tag_id", value_name="value")
        long = long.filter(pl.col("value").is_not_null())
        long = long.with_columns(
            pl.lit(run_id).alias("run_id"), pl.lit(asset).alias("asset_id"),
            pl.lit("oil_well").alias("asset_class"), pl.lit("unknown").alias("operating_regime"),
            pl.col("tag_id").replace_strict({tag: signals_map[tag][0] for tag in available}).alias("signal_class"),
            pl.col("tag_id").replace_strict({tag: signals_map[tag][1] for tag in available}).alias("unit"),
            pl.lit("uncertain").alias("quality"), pl.lit(1000.0).alias("sampling_interval_ms"),
            (pl.col("_binary") == 1).alias("is_anomaly"), pl.lit(event_id, dtype=pl.String).alias("event_id"),
        ).select("run_id", "asset_id", "asset_class", "timestamp", "tag_id", "signal_class", "value", "unit", "quality", "sampling_interval_ms", "operating_regime", "is_anomaly", "event_id")
        segments.append(long)
    return segments, {"path": name, "rows_labeled": source.height, "segments": len(segments), "signals": available, "invalid_sentinel_observations": sentinel_count}


def prepare(v1_path: Path, v2_path: Path, output: Path, verify_hash: bool = True, mapping_path: Path | None = None) -> dict:
    sha = digest(v1_path, "sha256")
    md5 = digest(v2_path, "md5")
    if verify_hash and sha != V1_SHA256:
        raise ValueError(f"Unexpected 3W 1.1.1 SHA-256: {sha}")
    if verify_hash and md5 != V2_MD5:
        raise ValueError(f"Unexpected 3W 2.0.0 MD5: {md5}")
    mapping_path = mapping_path or Path(__file__).resolve().parents[1] / "configs" / "3w-common-signals.yaml"
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    signals = {tag: (spec["signal_class"], spec["unit"]) for tag, spec in mapping["signals"].items()}
    invalid_values = {tag: [float(value) for value in values] for tag, values in mapping.get("invalid_values", {}).items()}
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(v1_path) as one, zipfile.ZipFile(v2_path) as two:
        selected = _select(one, two)
        details: dict[str, list[dict]] = {}
        for partition, names in selected.items():
            bundle = one if partition in {"train", "validation"} else two
            version = "3W-1.1.1" if bundle is one else "3W-2.0.0"
            frames = []
            entries = []
            for name in names:
                parts, entry = _normalize_member(bundle, name, version, invalid_values, signals)
                frames.extend(parts)
                entries.append(entry)
            if not frames:
                raise ValueError(f"No labeled segments produced for {partition}")
            data = pl.concat(frames, how="vertical_relaxed").sort(["run_id", "timestamp", "tag_id"])
            target = output / partition
            target.mkdir(parents=True, exist_ok=True)
            data.write_parquet(target / "3w-unseen.parquet", compression="zstd")
            details[partition] = {
                "source_version": version, "source_instances": len(names),
                "segments": len(frames), "rows_long": data.height,
                "assets": sorted(data["asset_id"].unique().to_list()),
                "normal_rows": data.filter(~pl.col("is_anomaly" )).height,
                "anomaly_rows": data.filter(pl.col("is_anomaly")).height,
                "normal_asset_counts": data.filter(~pl.col("is_anomaly")).group_by("asset_id").len().sort("asset_id").to_dicts(),
                "event_asset_counts": data.filter(pl.col("is_anomaly")).group_by("asset_id").len().sort("asset_id").to_dicts(),
                "instances": entries,
            }
        parts = {name: set(summary["assets"]) for name, summary in details.items()}
        for left, right in (("train", "validation"), ("train", "test"), ("train", "challenge"), ("validation", "test"), ("validation", "challenge")):
            overlap = parts[left] & parts[right]
            if overlap:
                raise ValueError(f"Asset leakage between {left}/{right}: {sorted(overlap)}")
    manifest = {
        "sources": [
            {"title": "Petrobras 3W Dataset 1.1.1", "doi": "10.6084/m9.figshare.29205947.v1", "license": "CC BY 4.0", "archive_sha256": sha},
            {"title": "Petrobras 3W Dataset 2.0.0", "doi": "10.6084/m9.figshare.29205836.v1", "license": "CC BY 4.0", "archive_md5": md5},
        ],
        "normalization": {
            "schema_version": "3w-common-signals-v1", "source_signal_count": 7,
            "signal_mapping": {tag: {"signal_class": value[0], "canonical_unit": value[1]} for tag, value in signals.items()},
            "excluded_from_model": ["class", "state", "well ID", "run ID", "20 signals present only in 3W 2.0.0"],
            "label_policy": "Drop null-class observations; binary evaluation label is class != 0, including transient codes. Split continuity at label change or >2.5s timestamp gap. No diagnosis label enters features.",
            "unit_policy": "3W dataset.ini engineering units mapped to canonical SI units; quality unavailable in source and represented as uncertain; quality features disabled.",
            "invalid_source_values": {tag: values for tag, values in invalid_values.items()},
            "invalid_source_value_policy": "Exact configured values are converted to null before unpivoting so fixed source sentinels are treated as unavailable samples, not numeric process readings.",
            "sampling_interval_ms": 1000,
        },
        "selection_policy": "Deterministic stratified source-file selection; old-version labeled-normal files spread lexically per well, one normal and one known-event file for WELL-00019, one known-event file per other selected new well except WELL-00020, reserved for challenge, and the smallest confirmed class-9 file for each challenge well. This is a bounded research subset, not a prevalence sample.",
        "split": details,
        "limitations": [
            "Only WELL-00019 has a separate class-0 source instance among newly introduced test wells. Other test wells have class-0 context inside selected event files; their false-positive rates are measurable within those files but do not estimate long-term normal operation.",
            "Challenge event 9 is confirmed by labels 9/109 in seven real wells; selected files also contain class-0 context for within-run false-positive measurement, but not separate long-term normal-operation files for each challenge well.",
            "The same underlying Petrobras dataset family is used across versions; this is well and event holdout, not cross-industry validation.",
        ],
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.v1, args.v2, args.output, not args.skip_hash_check)
    print(json.dumps({"sources": manifest["sources"], "split_summary": {k: {key: value for key, value in v.items() if key != "instances"} for k, v in manifest["split"].items()}, "limitations": manifest["limitations"]}, indent=2))


if __name__ == "__main__":
    main()
