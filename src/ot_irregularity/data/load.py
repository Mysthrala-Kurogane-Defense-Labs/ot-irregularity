from pathlib import Path
import polars as pl
REQUIRED = {"run_id", "asset_id", "timestamp", "tag_id", "signal_class", "value"}
def load_dataset(path):
    path=Path(path)
    files=[path] if path.is_file() else sorted([*path.rglob("*.parquet"), *path.rglob("*.csv")])
    if not files: raise ValueError(f"No CSV or Parquet files found: {path}")
    frames=[pl.read_parquet(f) if f.suffix==".parquet" else pl.read_csv(f) for f in files]
    df=pl.concat(frames, how="diagonal_relaxed")
    missing=REQUIRED-set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.schema["timestamp"] == pl.String:
        df=df.with_columns(pl.col("timestamp").str.to_datetime(strict=True, time_zone="UTC"))
    else:
        df=df.with_columns(pl.col("timestamp").cast(pl.Datetime, strict=True))
    df=df.with_columns(pl.col("value").cast(pl.Float64,strict=True))
    if df.select(pl.any_horizontal([pl.col(c).is_null() for c in REQUIRED]).any()).item():
        raise ValueError("Required schema fields cannot be null")
    if "quality" in df.columns and df.filter(~pl.col("quality").is_null() & ~pl.col("quality").is_in(["good","uncertain","bad"])).height:
        raise ValueError("quality must be good, uncertain, bad, or null")
    if df.select((pl.col("value").is_nan() | pl.col("value").is_infinite()).any()).item(): raise ValueError("value must be finite")
    if "is_anomaly" in df.columns:
        try: df=df.with_columns(pl.col("is_anomaly").cast(pl.Boolean,strict=True))
        except Exception as exc: raise ValueError("is_anomaly labels must be boolean or 0/1") from exc
    keys=["run_id","asset_id","timestamp","tag_id"]
    if df.select(pl.struct(keys).is_duplicated().any()).item(): raise ValueError("Duplicate (run_id, asset_id, timestamp, tag_id) observations")
    return df
