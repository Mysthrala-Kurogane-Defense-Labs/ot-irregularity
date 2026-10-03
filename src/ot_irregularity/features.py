import hashlib
import numpy as np
import polars as pl

def _signal_features(name, sg, width, default_interval, options):
    if sg.is_empty():
        expected=width/(default_interval*1000) if default_interval and default_interval>0 else 0.
        return {f"{name}_mean":0.,f"{name}_median":0.,f"{name}_min":0.,f"{name}_max":0.,f"{name}_range":0.,f"{name}_std":0.,f"{name}_mad":0.,f"{name}_last":0.,f"{name}_delta":0.,f"{name}_slope":0.,f"{name}_sample_count":0.,f"{name}_expected_sample_count":expected,f"{name}_coverage_ratio":0.,f"{name}_good_ratio":0.,f"{name}_uncertain_ratio":0.,f"{name}_bad_ratio":1.,f"{name}_missing":1.}
    good=sg.filter(pl.col("quality")=="good") if "quality" in sg.columns else sg
    vals=good["value"].to_numpy();allv=sg["value"].to_numpy();vals=vals if len(vals) else np.array([0.]);t=sg["_us"].to_numpy().astype(float);v=sg["value"].to_numpy().astype(float)
    feats={f"{name}_mean":float(np.mean(vals)),f"{name}_median":float(np.median(vals)),f"{name}_min":float(np.min(vals)),f"{name}_max":float(np.max(vals)),f"{name}_range":float(np.ptp(vals)),f"{name}_std":float(np.std(vals)),f"{name}_mad":float(np.median(np.abs(vals-np.median(vals)))),f"{name}_last":float(vals[-1]),f"{name}_delta":float(vals[-1]-vals[0]),f"{name}_slope":float(np.polyfit(t,v,1)[0]) if len(v)>1 and np.ptp(t)>0 else 0.,f"{name}_sample_count":float(len(allv))}
    interval=sg["sampling_interval_ms"][0] if "sampling_interval_ms" in sg.columns else default_interval;expected=width/(float(interval)*1000) if interval and interval>0 else float(len(allv));feats[f"{name}_expected_sample_count"]=expected;feats[f"{name}_coverage_ratio"]=min(len(allv)/expected,1.) if expected else 0.
    q=[v if v is not None else "uncertain" for v in sg["quality"].to_list()] if "quality" in sg.columns else ["good"]*len(allv)
    for label in ("good","uncertain","bad"):feats[f"{name}_{label}_ratio"]=q.count(label)/len(q) if q else 0.
    feats[f"{name}_missing"]=0.
    statistical={"mean","median","min","max","range","std","mad","last","delta"}
    quality={"good_ratio","uncertain_ratio","bad_ratio"}
    sampling={"sample_count","expected_sample_count","coverage_ratio","missing"}
    return {key:value for key,value in feats.items() if (options.get("statistical",True) or key.rsplit("_",1)[-1] not in statistical) and (options.get("slopes",True) or not key.endswith("_slope")) and (options.get("quality",True) or key.rsplit("_",1)[-1] not in quality) and (options.get("sampling",True) or key.rsplit("_",1)[-1] not in sampling)}

def make_windows(df,size_minutes=15,stride_minutes=1,signal_classes=None,sampling_intervals=None,options=None):
    width=size_minutes*60*1_000_000;step=stride_minutes*60*1_000_000;options=options or {}
    df=df.sort(["run_id","asset_id","signal_class","timestamp"]).with_columns(pl.col("timestamp").dt.timestamp("us").alias("_us"));out=[]
    classes=list(signal_classes or sorted(str(x) for x in df["signal_class"].unique().to_list()));intervals=sampling_intervals or {}
    unexpected=set(str(x) for x in df["signal_class"].unique().to_list())-set(classes)
    if signal_classes is not None and unexpected:raise ValueError(f"Untrained signal classes: {sorted(unexpected)}")
    for (run,asset),assetdf in df.group_by(["run_id","asset_id"],maintain_order=True):
        low=int(assetdf["_us"].min());high=int(assetdf["_us"].max())
        last_start=high-width+step
        for start in range((low//step)*step,last_start+1,step):
            g=assetdf.filter((pl.col("_us")>=start)&(pl.col("_us")<start+width))
            if g.is_empty():continue
            feats={"run_id":run,"asset_id":asset,"window_start":start,"window_end":start+width}
            for context in ("asset_class","operating_regime"):
                if context in g.columns:
                    values=[str(v) for v in g[context].drop_nulls().unique().to_list()];feats[context]=values[0] if len(values)==1 else ("mixed" if values else "unknown")
            if "is_anomaly" in g.columns:feats["is_anomaly"]=bool(g["is_anomaly"].cast(pl.Boolean).any())
            if "event_id" in g.columns:
                ids=[x for x in g["event_id"].drop_nulls().unique().to_list() if x];feats["event_id"]=str(ids[0]) if ids else None;event_rows=g.filter(pl.col("event_id").is_not_null());feats["event_start_us"]=int(event_rows["_us"].min()) if not event_rows.is_empty() else None
            for name in classes:
                sg=g.filter(pl.col("signal_class")==name);feats.update(_signal_features(name,sg,width,intervals.get(name),options))
            out.append(feats)
    if not out:raise ValueError("No windows generated")
    return pl.DataFrame(out).sort(["run_id","asset_id","window_start"])

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
