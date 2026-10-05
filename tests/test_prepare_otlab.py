import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from scripts.prepare_otlab import prepare


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(root: Path, *, license: str = "CC-BY-4.0") -> Path:
    root.mkdir()
    entries = []
    for partition, run_id, events in (
        ("train", "train-normal", []),
        ("train", "train-event", [{"event_id": "evt-1", "asset_id": "pump-1", "type": "sensor_bias",
                                     "start": "2025-01-01T00:00:01Z", "end": "2025-01-01T00:00:02Z"}]),
        ("validation", "validation-normal", []),
        ("validation", "validation-event", [{"event_id": "evt-val", "asset_id": "pump-1", "type": "sensor_bias",
                                                "start": "2025-01-01T00:00:01Z", "end": "2025-01-01T00:00:02Z"}]),
        ("test", "test-run", [{"event_id": "evt-test", "asset_id": "pump-1", "type": "thermal_deviation",
                                  "start": "2025-01-01T00:00:01Z", "end": "2025-01-01T00:00:02Z"}]),
    ):
        run = root / partition / run_id
        run.mkdir(parents=True)
        telemetry = run / "telemetry.parquet"
        pl.DataFrame({
            "run_id": [run_id] * 3,
            "asset_id": ["pump-1"] * 3,
            "timestamp": [datetime(2025, 1, 1, 0, 0, second, tzinfo=timezone.utc) for second in range(3)],
            "tag_id": ["temperature"] * 3,
            "signal_class": ["temperature"] * 3,
            "value": [20.0, 21.0, 22.0],
            "quality": ["GOOD"] * 3,
        }).write_parquet(telemetry)
        (run / "ground_truth.json").write_text(json.dumps({"run_id": run_id, "events": events}), encoding="utf-8")
        entries.append({"partition": partition, "run_id": run_id, "telemetry_sha256": _sha256(telemetry)})
    (root / "dataset_manifest.json").write_text(json.dumps({
        "dataset_id": "fixture", "simulator_version": "0.3.0", "schema_version": "1.0.0",
        "data_license": license, "synthetic": True, "generated": True, "customer_data": False, "runs": entries,
    }), encoding="utf-8")
    return root


def test_prepare_keeps_normal_train_and_labeled_validation_test(tmp_path):
    source = _source(tmp_path / "source")
    output = tmp_path / "prepared"

    manifest = prepare(source, output)

    assert manifest["run_ids_disjoint"] is True
    assert manifest["partitions"]["train"]["run_ids"] == ["train-normal"]
    assert manifest["partitions"]["validation"]["run_ids"] == ["validation-event", "validation-normal"]
    assert manifest["partitions"]["test"]["event_bearing_runs"] == 1
    train = pl.read_parquet(output / "train" / "telemetry.parquet")
    validation = pl.read_parquet(output / "validation" / "telemetry.parquet")
    test = pl.read_parquet(output / "test" / "telemetry.parquet")
    assert train["is_anomaly"].sum() == 0
    assert validation["is_anomaly"].sum() == 2
    assert validation["event_id"].drop_nulls().unique().to_list() == ["validation-event/evt-val"]
    assert test["is_anomaly"].to_list() == [False, True, True]
    assert test["event_id"].to_list() == [None, "test-run/evt-test", "test-run/evt-test"]
    assert test["quality"].to_list() == ["good"] * 3
    assert test["event_start_us"].to_list() == [None, 1735689601000000, 1735689601000000]
    event_doc=json.loads((output/"test"/"events.json").read_text())
    assert event_doc["events"][0]["event_id"]=="test-run/evt-test"
    assert event_doc["events"][0]["start_us"]==1735689601000000
    assert manifest["partitions"]["test"]["events_sha256"]==hashlib.sha256((output/"test"/"events.json").read_bytes()).hexdigest()
    assert "fixture" in (output / "DATASET_LICENSE.txt").read_text(encoding="utf-8")
    assert (output / "DATASET_LICENSE.txt").is_file()


def test_completely_missing_event_interval_gets_window_evaluation_anchor():
    from scripts.prepare_otlab import _annotate

    frame = pl.DataFrame({
        "asset_id": ["pump-1"] * 3,
        "timestamp": [datetime(2025, 1, 1, 0, 0, second, tzinfo=timezone.utc) for second in (0, 1, 9)],
    })
    result = _annotate(frame, [{"event_id": "run/comm-loss", "asset_id": "pump-1",
                                "start": "2025-01-01T00:00:03Z", "end": "2025-01-01T00:00:06Z"}])

    assert result["is_anomaly"].to_list() == [False, True, False]
    assert result["event_id"].to_list() == [None, "run/comm-loss", None]
    assert result["event_start_us"][1] == 1735689603000000


@pytest.mark.parametrize("updates", [
    {"synthetic": False}, {"generated": False}, {"customer_data": True},
])
def test_prepare_rejects_unverified_data_source(tmp_path, updates):
    source = _source(tmp_path / "source")
    path = source / "dataset_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(updates)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        prepare(source, tmp_path / "prepared")


def test_prepare_rejects_non_cc_by_source_and_hash_mismatch(tmp_path):
    source = _source(tmp_path / "source", license="UNSPECIFIED")
    with pytest.raises(ValueError, match="CC-BY-4.0"):
        prepare(source, tmp_path / "rejected-license")

    source = _source(tmp_path / "other-source")
    with (source / "train" / "train-normal" / "telemetry.parquet").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare(source, tmp_path / "rejected-hash")
