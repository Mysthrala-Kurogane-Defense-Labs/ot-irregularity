import json

import polars as pl
import pytest

from scripts.assemble_feature_search_dataset import assemble


def _source(tmp_path):
    root = tmp_path / "search"
    for index in range(1, 16):
        batch = root / "datasets" / f"dev-{index:02d}"
        batch.mkdir(parents=True)
        (batch / "DATASET_LICENSE.txt").write_text("Source data license: CC BY 4.0\n")
        (batch / "dataset_manifest.json").write_text("{}\n")
        for partition in ("train", "validation", "test"):
            target = batch / partition
            target.mkdir()
            anomalous = partition != "train"
            pl.DataFrame({
                "run_id": [f"{partition}-1", f"{partition}-1"],
                "asset_id": ["pump-1", "pump-1"],
                "tag_id": ["temperature", "temperature"],
                "timestamp": pl.datetime_range(
                    __import__("datetime").datetime(2026, 1, 1),
                    __import__("datetime").datetime(2026, 1, 1, 0, 1),
                    interval="1m", eager=True,
                ),
                "value": [1.0, 2.0],
                "is_anomaly": [False, anomalous],
                "event_id": [None, "event-1" if anomalous else None],
            }).write_parquet(target / "telemetry.parquet")
    return root


def test_assembly_namespaces_runs_and_events_and_preserves_disjoint_splits(tmp_path):
    source = _source(tmp_path)
    out = tmp_path / "combined"
    result = assemble(source, out)
    train = pl.read_parquet(out / "train/telemetry.parquet")
    validation = pl.read_parquet(out / "validation/telemetry.parquet")
    test = pl.read_parquet(out / "test/telemetry.parquet")
    assert result["run_ids_disjoint"]
    assert result["partitions"]["train"]["runs"] == 15
    assert not train["is_anomaly"].any()
    assert train["run_id"].n_unique() == validation["run_id"].n_unique() == test["run_id"].n_unique() == 15
    assert validation["event_id"].drop_nulls().n_unique() == 15
    assert json.loads((out / "dataset_manifest.json").read_text())["license"] == "CC-BY-4.0"
    with pytest.raises(FileExistsError):
        assemble(source, out)
