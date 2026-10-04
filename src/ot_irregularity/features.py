import hashlib
import numpy as np
import polars as pl

def _filter_feature_options(features, options):
    statistical={"mean","median","min","max","range","std","mad","last","delta"}
    quality={"good_ratio","uncertain_ratio","bad_ratio"}
    sampling={"sample_count","expected_sample_count","coverage_ratio","missing"}
    def has_suffix(key, suffixes):
        return any(key.endswith("_" + suffix) for suffix in suffixes)
    return {key:value for key,value in features.items()
            if (options.get("statistical",True) or not has_suffix(key,statistical))
            and (options.get("slopes",True) or not key.endswith("_slope"))
            and (options.get("quality",True) or not has_suffix(key,quality))
            and (options.get("sampling",True) or not has_suffix(key,sampling))}

def _signal_features(name, sg, width, default_interval, options):
    if sg.is_empty():
        expected=width/(default_interval*1000) if default_interval and default_interval>0 else 0.
        feats={f"{name}_mean":0.,f"{name}_median":0.,f"{name}_min":0.,f"{name}_max":0.,f"{name}_range":0.,f"{name}_std":0.,f"{name}_mad":0.,f"{name}_last":0.,f"{name}_delta":0.,f"{name}_slope":0.,f"{name}_sample_count":0.,f"{name}_expected_sample_count":expected,f"{name}_coverage_ratio":0.,f"{name}_good_ratio":0.,f"{name}_uncertain_ratio":0.,f"{name}_bad_ratio":1.,f"{name}_missing":1.}
        return _filter_feature_options(feats,options)
    usable=sg.filter(pl.col("value").is_not_null()&pl.col("value").cast(pl.Float64).is_finite())
    if "quality" in sg.columns:usable=usable.filter(pl.col("quality").is_null()|(pl.col("quality")!="bad"))
    vals=usable["value"].to_numpy();allv=sg["value"].to_numpy();vals=vals if len(vals) else np.array([0.]);t=usable["_us"].to_numpy().astype(float);v=vals.astype(float);relative_seconds=(t-t[0])/1_000_000 if len(t) else t
    feats={f"{name}_mean":float(np.mean(vals)),f"{name}_median":float(np.median(vals)),f"{name}_min":float(np.min(vals)),f"{name}_max":float(np.max(vals)),f"{name}_range":float(np.ptp(vals)),f"{name}_std":float(np.std(vals)),f"{name}_mad":float(np.median(np.abs(vals-np.median(vals)))),f"{name}_last":float(vals[-1]),f"{name}_delta":float(vals[-1]-vals[0]),f"{name}_slope":float(np.polyfit(relative_seconds,v,1)[0]) if len(v)>1 and np.ptp(relative_seconds)>0 else 0.,f"{name}_sample_count":float(len(allv))}
    interval=sg["sampling_interval_ms"][0] if "sampling_interval_ms" in sg.columns else default_interval;expected=width/(float(interval)*1000) if interval and interval>0 else float(len(allv));feats[f"{name}_expected_sample_count"]=expected;feats[f"{name}_coverage_ratio"]=min(len(allv)/expected,1.) if expected else 0.
    q=[v if v is not None else "uncertain" for v in sg["quality"].to_list()] if "quality" in sg.columns else ["good"]*len(allv)
    for label in ("good","uncertain","bad"):feats[f"{name}_{label}_ratio"]=q.count(label)/len(q) if q else 0.
    feats[f"{name}_missing"]=0.
    return _filter_feature_options(feats,options)

def make_windows(df,size_minutes=15,stride_minutes=1,signal_classes=None,sampling_intervals=None,options=None):
    width=size_minutes*60*1_000_000;step=stride_minutes*60*1_000_000;options=options or {}
    channel_column="measurement_role" if "measurement_role" in df.columns else "signal_class"
    if "value_kind" in df.columns:
        # Categorical encodings need a dedicated vocabulary and transition features.
        df=df.filter(pl.col("value_kind").is_null() | (pl.col("value_kind")=="continuous"))
    if df.is_empty():raise ValueError("No continuous measurement channels available for window generation")
    df=df.sort(["run_id","asset_id","timestamp","signal_class"]).with_columns(pl.col("timestamp").dt.timestamp("us").alias("_us"));out=[]
    classes=list(signal_classes or sorted(str(x) for x in df[channel_column].unique().to_list()));intervals=sampling_intervals or {}
    unexpected=set(str(x) for x in df[channel_column].unique().to_list())-set(classes)
    if signal_classes is not None and unexpected:raise ValueError(f"Untrained measurement channels: {sorted(unexpected)}")
    for (run,asset),assetdf in df.group_by(["run_id","asset_id"],maintain_order=True):
        time_us=assetdf["_us"].to_numpy().astype(np.int64,copy=False)
        signal_frames={name:assetdf.filter(pl.col(channel_column)==name) for name in classes}
        signal_times={name:signal_frames[name]["_us"].to_numpy().astype(np.int64,copy=False) for name in classes}
        constant_context={}
        for context in ("asset_class","operating_regime"):
            if context in assetdf.columns:
                values=[str(v) for v in assetdf[context].drop_nulls().unique().to_list()]
                constant_context[context]=values[0] if len(values)==1 else None
        anomaly_values=assetdf["is_anomaly"].cast(pl.Boolean).unique().to_list() if "is_anomaly" in assetdf.columns else []
        constant_anomaly=bool(anomaly_values[0]) if len(anomaly_values)==1 else None
        event_ids=[x for x in assetdf["event_id"].drop_nulls().unique().to_list() if x] if "event_id" in assetdf.columns else []
        event_has_null=assetdf["event_id"].null_count()>0 if "event_id" in assetdf.columns else False
        constant_event= str(event_ids[0]) if len(event_ids)==1 and not event_has_null else (False if event_ids else None)
        constant_event_start=None
        if isinstance(constant_event,str):
            event_rows=assetdf.filter(pl.col("event_id")==constant_event)
            starts=event_rows["event_start_us"].drop_nulls() if "event_start_us" in event_rows.columns else []
            constant_event_start=int(starts.min()) if len(starts) else int(event_rows["_us"].min())
        low=int(time_us[0]);high=int(time_us[-1])
        last_start=high-width+step
        for start in range((low//step)*step,last_start+1,step):
            # Rows are already sorted by timestamp. Binary-search bounds avoid
            # rescanning every observation for every overlapping window.
            lo=int(np.searchsorted(time_us,start,side="left"));hi=int(np.searchsorted(time_us,start+width,side="left"))
            g=assetdf.slice(lo,hi-lo)
            if g.is_empty():continue
            feats={"run_id":run,"asset_id":asset,"window_start":start,"window_end":start+width}
            for context in ("asset_class","operating_regime"):
                if context in g.columns:
                    value=constant_context.get(context)
                    if value is None:
                        values=[str(v) for v in g[context].drop_nulls().unique().to_list()];value=values[0] if len(values)==1 else ("mixed" if values else "unknown")
                    feats[context]=value
            if "is_anomaly" in g.columns:feats["is_anomaly"]=constant_anomaly if constant_anomaly is not None else bool(g["is_anomaly"].cast(pl.Boolean).any())
            if "event_id" in g.columns:
                if constant_event is False:
                    ids=[x for x in g["event_id"].drop_nulls().unique().to_list() if x];event_id=str(ids[0]) if ids else None
                    event_rows=g.filter(pl.col("event_id")==event_id) if event_id else g.head(0)
                    starts=event_rows["event_start_us"].drop_nulls() if "event_start_us" in event_rows.columns else []
                    event_start=int(starts.min()) if len(starts) else (int(event_rows["_us"].min()) if not event_rows.is_empty() else None)
                else:event_id=constant_event;event_start=constant_event_start
                feats["event_id"]=event_id;feats["event_start_us"]=event_start
            for name in classes:
                signal_lo=int(np.searchsorted(signal_times[name],start,side="left"));signal_hi=int(np.searchsorted(signal_times[name],start+width,side="left"))
                sg=signal_frames[name].slice(signal_lo,signal_hi-signal_lo);feats.update(_signal_features(name,sg,width,intervals.get(name),options))
            out.append(feats)
    if not out:raise ValueError("No windows generated")
    return pl.DataFrame(out,infer_schema_length=None).sort(["run_id","asset_id","window_start"])

def encode_context(frame, vocabulary=None):
    vocab=vocabulary or {key:sorted(str(x) for x in frame[key].drop_nulls().unique().to_list()) if key in frame.columns else [] for key in ("asset_class","operating_regime")}
    out=frame
    for context in ("asset_class","operating_regime"):
        values=out[context].cast(pl.String).fill_null("unknown").to_list() if context in out.columns else ["unknown"]*out.height
        known=set(vocab.get(context,[]));unknown=[v not in known for v in values]
        for value in vocab.get(context,[]):
            suffix=hashlib.sha1(value.encode()).hexdigest()[:8]
            out=out.with_columns(pl.Series(f"context_{context}_{suffix}",[v==value for v in values]))
        if context=="operating_regime":out=out.with_columns(pl.Series("context_regime_unknown",unknown))
    return out,vocab

def feature_columns(frame): return [c for c in frame.columns if c not in {"run_id","asset_id","asset_class","operating_regime","window_start","window_end","is_anomaly","event_id","event_start_us"}]
