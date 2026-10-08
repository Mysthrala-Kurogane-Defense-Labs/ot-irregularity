import json

import pytest
from pydantic import ValidationError

from ot_irregularity.specialists import SpecialistResult, combine_specialists, telemetry_specialists


def result(name="physical", **changes):
    fields = dict(run_id="run", asset_id="pump", asset_class="PUMP", window_start_us=0,
        window_end_us=60_000_000, specialist=name, model_version="1", calibration_version="cal1",
        status="available", raw_score=2., calibrated_score=.8, threshold=.9)
    return SpecialistResult(**(fields | changes))


def combine(rows, **kw):
    return combine_specialists(rows, expected_versions={r.specialist:("1", "cal1") for r in rows},
        policy_version="1", **kw)


def test_available_scores_roundtrip_and_threshold_boundary():
    row = result(calibrated_score=.9)
    assert SpecialistResult.model_validate_json(row.model_dump_json()) == row
    combined = combine([row])
    assert combined["rank"] == .5 and combined["detected"]
    assert json.loads(json.dumps(combined, allow_nan=False))["specialists"][0]["raw_score"] == 2.


def test_unavailable_is_not_normal_and_partial_requires_explicit_policy():
    absent = result("thermal", status="unavailable", reason="missing_temperature", raw_score=None, calibrated_score=None)
    rows = [result(calibrated_score=1), absent]
    assert combine(rows)["detected"] is None
    partial = combine(rows, missing_policy="available_only")
    assert partial["status"] == "partial" and partial["detected"]
    assert partial["unavailable"] == {"thermal":"missing_temperature"}
    assert combine([absent], missing_policy="available_only")["rank"] is None


@pytest.mark.parametrize("changes", [dict(calibrated_score=float("nan")), dict(raw_score=float("inf")),
    dict(window_end_us=0), dict(status="unavailable", reason="missing"),
    dict(contributions={"temp":-1}), dict(calibrated_score=1.1), dict(reason="missing")])
def test_invalid_result_rejected(changes):
    with pytest.raises(ValidationError): result(**changes)


def test_combination_rejects_identity_duplicates_versions_and_missing_inventory():
    with pytest.raises(ValueError, match="aligned"): combine([result(),result("thermal",run_id="other")])
    with pytest.raises(ValueError, match="Exactly"): combine([result(),result()])
    with pytest.raises(ValueError, match="version"): combine([result(model_version="2")])
    with pytest.raises(ValueError, match="Exactly"):
        combine_specialists([result()],expected_versions={"physical":("1","cal1"),"thermal":("1","cal1")},policy_version="1")


def test_telemetry_adapter_preserves_scores_and_separates_observations():
    row=dict(run_id="r",asset_id="p",asset_class="PUMP",window_start=0,window_end=60,
        quality_score=.12,sampling_score=None,observations=["quality_deviation","signal_loss"])
    values=telemetry_specialists(row,model_version="1",calibration_version="cal1",quality_threshold=.05,sampling_threshold=.05)
    assert values[0].raw_score == values[0].calibrated_score == .12
    assert values[0].observations == ("quality_deviation",) and values[0].detected
    assert values[1].status == "unavailable" and values[1].observations == () and values[1].detected is None


def test_extreme_valid_threshold_has_finite_rank():
    combined = combine([result(calibrated_score=1., threshold=5e-324)])
    assert combined["rank"] == 1. and combined["detected"]
    json.dumps(combined, allow_nan=False)
