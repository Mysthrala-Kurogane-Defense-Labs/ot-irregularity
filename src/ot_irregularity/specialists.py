"""Versioned specialist outputs and explicit, availability-aware decisions."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SpecialistResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    asset_id: str = Field(min_length=1)
    asset_class: str = Field(min_length=1)
    window_start_us: int
    window_end_us: int
    specialist: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    calibration_version: str = Field(min_length=1)
    status: Literal["available", "unavailable", "out_of_scope"]
    reason: str | None = None
    raw_score: float | None = Field(default=None, ge=0)
    calibrated_score: float | None = Field(default=None, ge=0, le=1)
    threshold: float = Field(gt=0, le=1)
    observations: tuple[str, ...] = ()
    contributions: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_state(self):
        if self.window_end_us <= self.window_start_us:
            raise ValueError("Positive window duration required")
        if self.status == "available":
            if self.raw_score is None or self.calibrated_score is None or self.reason is not None:
                raise ValueError("Available result requires scores and no unavailable reason")
        elif self.raw_score is not None or self.calibrated_score is not None or not self.reason:
            raise ValueError("Unavailable result requires a reason and null scores")
        if self.status != "available" and (self.observations or self.contributions):
            raise ValueError("Unavailable result cannot assert observed deviations")
        if any(not math.isfinite(v) or v < 0 for v in self.contributions.values()):
            raise ValueError("Finite nonnegative contributions required")
        return self

    @property
    def identity(self):
        return (self.run_id, self.asset_id, self.asset_class, self.window_start_us, self.window_end_us)

    @property
    def detected(self):
        return None if self.status != "available" else self.calibrated_score >= self.threshold


def combine_specialists(results: list[SpecialistResult], *, expected_versions: dict[str, tuple[str, str]],
                        policy_version: str, missing_policy: Literal["require_all", "available_only"] = "require_all"):
    """Maximum threshold-relative margin; a ranking, never a failure probability.

    Partial operation is opt-in and remains visible. Each detector already owns
    its calibrated threshold. This policy does not claim a joint false-alarm rate.
    """
    if not policy_version or not expected_versions or missing_policy not in ("require_all", "available_only"):
        raise ValueError("Explicit versioned policy and specialist inventory required")
    if not results or len({r.identity for r in results}) != 1:
        raise ValueError("One aligned window required")
    by_name = {r.specialist: r for r in results}
    if len(by_name) != len(results) or set(by_name) != set(expected_versions):
        raise ValueError("Exactly one result for every declared specialist required")
    for name, row in by_name.items():
        if (row.model_version, row.calibration_version) != expected_versions[name]:
            raise ValueError("Specialist artifact version differs from policy")
    available = [r for r in results if r.status == "available"]
    unavailable = {r.specialist: r.reason for r in results if r.status != "available"}
    evaluable = bool(available) and (not unavailable or missing_policy == "available_only")
    rank = max(r.calibrated_score/(r.calibrated_score+r.threshold) for r in available) if evaluable else None
    return {"schema_version": 1, "policy_version": policy_version, "missing_policy": missing_policy,
            "run_id": results[0].run_id, "asset_id": results[0].asset_id,
            "asset_class": results[0].asset_class, "window_start_us": results[0].window_start_us,
            "window_end_us": results[0].window_end_us,
            "status": ("partial" if unavailable else "available") if evaluable else "unavailable",
            "rank": rank,
            "detected": any(r.detected for r in available) if evaluable else None,
            "unavailable": unavailable,
            "specialists": [r.model_dump(mode="json") | {"detected": r.detected}
                            for r in sorted(results, key=lambda r: r.specialist)]}


def telemetry_specialists(row: dict, *, model_version: str, calibration_version: str,
                          quality_threshold: float, sampling_threshold: float) -> list[SpecialistResult]:
    """Adapt a TelemetryHealthReference aggregate without changing its scores.

    Detailed role explanations remain in the health output; unavailable reasons
    here deliberately describe aggregate availability only.
    """
    identity = {k: row[k] for k in ("run_id", "asset_id", "asset_class")}
    identity.update(window_start_us=row["window_start"], window_end_us=row["window_end"])
    result = []
    for name, column, threshold, allowed in (
        ("telemetry_quality", "quality_score", quality_threshold, {"quality_deviation"}),
        ("telemetry_sampling", "sampling_score", sampling_threshold, {"sampling_degradation", "signal_loss"}),
    ):
        score = row[column]
        available = score is not None
        result.append(SpecialistResult(**identity, specialist=name, model_version=model_version,
            calibration_version=calibration_version, status="available" if available else "unavailable",
            reason=None if available else "no_evaluable_role", raw_score=score, calibrated_score=score,
            threshold=threshold, observations=tuple(sorted(set(row.get("observations", [])) & allowed)) if available else ()))
    return result


def relational_specialist(row: dict, *, margin: float | None, model_version: str,
                           calibration_version: str) -> SpecialistResult:
    """Expose an already calibrated class-relative reconstruction margin.

    The monotonic bounded rank margin/(1+margin) has decision threshold .5.
    Missing members remain unavailable. This adapter does not refit references
    or claim that the rank is a probability or an empirical percentile.
    """
    if margin is not None and (not math.isfinite(margin) or margin < 0):
        raise ValueError("Finite nonnegative margin or explicit None required")
    identity = {k: row[k] for k in ("run_id", "asset_id", "asset_class")}
    identity.update(window_start_us=row["window_start"], window_end_us=row["window_end"])
    return SpecialistResult(**identity, specialist="relational_magnitude", model_version=model_version,
        calibration_version=calibration_version, status="available" if margin is not None else "unavailable",
        reason=None if margin is not None else "relational_members_unavailable", raw_score=margin,
        calibrated_score=None if margin is None else margin/(1+margin), threshold=.5,
        observations=("multivariate_novelty",) if margin is not None and margin >= 1 else ())
