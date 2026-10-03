import io
import zipfile
import datetime

import polars as pl

from scripts.prepare_3w_unseen import _normalize_member, _spread, _well


def test_3w_segments_exclude_unlabeled_and_split_event_boundaries(tmp_path):
    frame = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                pl.datetime(2020, 1, 1), pl.datetime(2020, 1, 1, 0, 0, 7),
                interval="1s", eager=True,
            ),
            "class": [None, 0, 0, 109, 109, 0, 0, None],
            "P-PDG": [100.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "P-TPT": [90.0] * 8,
            "T-TPT": [20.0] * 8,
            "P-MON-CKP": [80.0] * 8,
        }
    )
    archive = tmp_path / "source.zip"
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("9/WELL-00020_20200101.parquet", buffer.getvalue())
    with zipfile.ZipFile(archive) as bundle:
        segments, detail = _normalize_member(bundle, "9/WELL-00020_20200101.parquet", "3W-2.0.0")

    assert detail["rows_labeled"] == 6
    assert len(segments) == 3
    rows = pl.concat(segments)
    assert rows["asset_id"].unique().to_list() == ["WELL-00020"]
    assert rows.filter(pl.col("is_anomaly"))["event_id"].n_unique() == 1
    assert rows.filter(pl.col("is_anomaly")).height == 8  # 2 timestamps x 4 common tags
    assert rows.filter(~pl.col("is_anomaly")).height == 16
    assert set(rows["signal_class"].unique().to_list()) == {
        "downhole_pressure", "subsea_tree_pressure", "subsea_tree_temperature", "production_choke_upstream_pressure"
    }


def test_source_selection_helpers_are_deterministic():
    assert _well("8/WELL-00042_20141218.parquet") == 42
    assert _well("8/SIMULATED_00001.parquet") is None
    assert _spread(["c", "a", "b", "d"], 2) == ["a", "d"]


def test_configured_source_sentinel_becomes_missing_observation(tmp_path):
    frame = pl.DataFrame(
        {
            "timestamp": [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 1, 0, 0, 1)],
            "class": [0, 0], "P-PDG": [-1.180116e42, 100.0], "P-TPT": [90.0, 90.0],
            "T-TPT": [20.0, 20.0], "P-MON-CKP": [80.0, 80.0],
        }
    )
    archive = tmp_path / "source.zip"
    buffer = io.BytesIO()
    frame.write_parquet(buffer)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("0/WELL-00001_20200101.parquet", buffer.getvalue())
    with zipfile.ZipFile(archive) as bundle:
        segments, detail = _normalize_member(
            bundle, "0/WELL-00001_20200101.parquet", "3W-2.0.0", {"P-PDG": [-1.180116e42]}
        )
    assert detail["invalid_sentinel_observations"] == 1
    pressure = pl.concat(segments).filter(pl.col("tag_id") == "P-PDG")
    assert pressure["value"].to_list() == [100.0]
