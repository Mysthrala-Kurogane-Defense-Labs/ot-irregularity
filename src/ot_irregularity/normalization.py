"""Engineering-unit normalization, kept separate from ML feature scaling."""
from __future__ import annotations

import polars as pl


def _key(unit: str) -> str:
    return str(unit).strip().casefold().replace(" ", "").replace("°", "deg").replace("³", "3").replace("^", "")


# scale and offset convert source values to each dimension's canonical unit.
_UNITS = {
    "pa": ("pressure", 1.0, 0.0), "kpa": ("pressure", 1e3, 0.0),
    "mpa": ("pressure", 1e6, 0.0), "bar": ("pressure", 1e5, 0.0),
    "mbar": ("pressure", 100.0, 0.0),
    "pa(abs)": ("pressure_absolute", 1.0, 0.0), "kpa(abs)": ("pressure_absolute", 1e3, 0.0),
    "bar(abs)": ("pressure_absolute", 1e5, 0.0), "bar(a)": ("pressure_absolute", 1e5, 0.0),
    "psia": ("pressure_absolute", 6894.757293168, 0.0),
    "pa(g)": ("pressure_gauge", 1.0, 0.0), "kpa(g)": ("pressure_gauge", 1e3, 0.0),
    "bar(g)": ("pressure_gauge", 1e5, 0.0), "psig": ("pressure_gauge", 6894.757293168, 0.0),
    "degc": ("temperature", 1.0, 0.0), "c": ("temperature", 1.0, 0.0),
    "oc": ("temperature", 1.0, 0.0), "celsius": ("temperature", 1.0, 0.0),
    "k": ("temperature", 1.0, -273.15), "kelvin": ("temperature", 1.0, -273.15),
    "degf": ("temperature", 5.0 / 9.0, -32.0 * 5.0 / 9.0), "f": ("temperature", 5.0 / 9.0, -32.0 * 5.0 / 9.0),
    "m3/s": ("flow", 1.0, 0.0), "m3/h": ("flow", 1.0 / 3600.0, 0.0),
    "l/s": ("flow", 1e-3, 0.0), "l/min": ("flow", 1.0 / 60000.0, 0.0),
    "l/h": ("flow", 1.0 / 3_600_000.0, 0.0),
    "1": ("ratio", 1.0, 0.0), "fraction": ("ratio", 1.0, 0.0), "ratio": ("ratio", 1.0, 0.0),
    "%": ("ratio", 0.01, 0.0), "percent": ("ratio", 0.01, 0.0), "percentage": ("ratio", 0.01, 0.0),
    "v": ("voltage", 1.0, 0.0), "mv": ("voltage", 1e-3, 0.0), "kv": ("voltage", 1e3, 0.0),
    "a": ("current", 1.0, 0.0), "ma": ("current", 1e-3, 0.0), "ka": ("current", 1e3, 0.0),
    "w": ("power", 1.0, 0.0), "kw": ("power", 1e3, 0.0), "mw": ("power", 1e6, 0.0),
    "hz": ("frequency", 1.0, 0.0), "khz": ("frequency", 1e3, 0.0),
    "rpm": ("rotational_speed", 1.0, 0.0),
    "m": ("length", 1.0, 0.0), "cm": ("length", 1e-2, 0.0), "mm": ("length", 1e-3, 0.0),
}


def normalize_units(frame: pl.DataFrame, canonical_units: dict[str, str] | None) -> pl.DataFrame:
    """Convert configured signal classes to canonical units before windowing.

    Signals without a configured target are passed through unchanged. A target
    requires a non-null, recognized source unit on every observation.
    """
    if not canonical_units:
        return frame
    if "unit" not in frame.columns:
        raise ValueError("normalization.canonical_units requires a unit column")
    out = frame
    signals = set(frame["signal_class"].drop_nulls().cast(pl.String).unique().to_list())
    for signal, target in canonical_units.items():
        if signal not in signals:
            continue
        target_key = _key(target)
        if target_key not in _UNITS:
            raise ValueError(f"Unsupported canonical unit {target!r} for signal {signal!r}")
        target_dimension, target_scale, target_offset = _UNITS[target_key]
        subset = out.filter(pl.col("signal_class") == signal)
        if subset["unit"].null_count():
            raise ValueError(f"Signal {signal!r} has observations without units")
        source_units = subset["unit"].cast(pl.String).unique().to_list()
        expression = pl.col("value").cast(pl.Float64)
        for source in source_units:
            source_key = _key(source)
            definition = _UNITS.get(source_key)
            if definition is None:
                raise ValueError(f"Unsupported source unit {source!r} for signal {signal!r}")
            dimension, source_scale, source_offset = definition
            if dimension != target_dimension:
                raise ValueError(f"Incompatible units {source!r} and {target!r} for signal {signal!r}")
            factor = source_scale / target_scale
            offset = (source_offset - target_offset) / target_scale
            expression = pl.when((pl.col("signal_class") == signal) & (pl.col("unit").cast(pl.String) == source)).then(pl.col("value") * factor + offset).otherwise(expression)
        out = out.with_columns(
            expression.alias("value"),
            pl.when(pl.col("signal_class") == signal).then(pl.lit(target)).otherwise(pl.col("unit")).alias("unit"),
        )
    return out
