from __future__ import annotations
import datetime, hashlib, json, random, subprocess
from pathlib import Path
import joblib, numpy as np, torch, yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler
from .data.load import load_dataset
from .features import encode_context, feature_columns, feature_signal, make_windows
from .metrics import evaluate_scores
from .models import ae_errors, load_ae, resolve_device, train_ae
from .normalization import normalize_units

def cdf_calibrate(reference, values):
    ref=np.sort(np.asarray(reference,dtype=float))
    if not len(ref): raise ValueError("Calibration reference cannot be empty")
    return np.searchsorted(ref,np.asarray(values,dtype=float),side="right")/len(ref)

def _hash_paths(paths):
    h=hashlib.sha256()
    for root in sorted(map(Path,paths)):
        fs=[root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for f in fs:
            h.update(f.relative_to(root).as_posix().encode() if root.is_dir() else f.name.encode());h.update(f.read_bytes())
    return h.hexdigest()

def _git_commit():
    try:return subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError):return None

def _git_working_tree_dirty():
    try:return bool(subprocess.run(["git","status","--porcelain","--untracked-files=all"],capture_output=True,text=True,check=True).stdout.strip())
    except (OSError,subprocess.CalledProcessError):return None

def _duration_minutes(value):
    text=str(value).strip().lower();units={"m":1,"min":1,"h":60,"s":1/60}
    for suffix,mult in units.items():
        if text.endswith(suffix): return max(1,int(float(text[:-len(suffix)])*mult))
    raise ValueError(f"Unsupported window duration {value!r}; use s, m, min, or h")

def _windows(path,cfg,signal_schema=None):
    d=normalize_units(load_dataset(path),cfg.get("normalization",{}).get("canonical_units",{}));d=_exclude_feature_inputs(d,cfg);w=cfg.get("window",{});signals=((signal_schema.get("measurement_roles") or signal_schema.get("signal_classes")) if signal_schema else None);intervals=signal_schema.get("sampling_intervals_ms",{}) if signal_schema else None
    return make_windows(d,_duration_minutes(w.get("size","15m")),_duration_minutes(w.get("stride","1m")),signals,intervals,cfg.get("features",{}))

def _exclude_feature_inputs(frame,cfg):
    options=cfg.get("features",{})
    if "signal_class" in frame.columns and options.get("exclude_signal_classes"):
        frame=frame.filter(~__import__("polars").col("signal_class").cast(__import__("polars").String).is_in(options["exclude_signal_classes"]))
    if "measurement_role" in frame.columns and options.get("exclude_measurement_roles"):
        frame=frame.filter(~__import__("polars").col("measurement_role").cast(__import__("polars").String).is_in(options["exclude_measurement_roles"]))
    return frame

def _scores(modeldir,w):
    if (Path(modeldir)/"contextual_models.json").is_file():
        from .contextual import score_contextual
        return score_contextual(modeldir,w)
    d=Path(modeldir);schema=json.loads((d/"feature_schema.json").read_text());cols=schema["features"]
    if set(cols)-set(w.columns):raise ValueError("Inference dataset is missing trained feature signals")
    produced=set(feature_columns(w))
    if produced-set(cols):raise ValueError(f"Inference dataset has untrained features: {sorted(produced-set(cols))}")
    x=joblib.load(d/"scaler.joblib").transform(w.select(cols).to_numpy().astype(float));cfg=yaml.safe_load((d/"training_config.yaml").read_text());device=resolve_device(cfg.get("device","auto"));ae=load_ae(d/"autoencoder.pt",device);iso=joblib.load(d/"isolation_forest.joblib");errors=ae_errors(ae,x);cal=json.loads((d/"thresholds.json").read_text())
    ae_score=cdf_calibrate(cal["autoencoder_reference"],errors.mean(axis=1));iso_score=cdf_calibrate(cal["isolation_reference"],-iso.score_samples(x));ens=cfg.get("ensemble",{});a=float(ens.get("autoencoder_weight",.5));b=float(ens.get("isolation_weight",.5))
    if a<0 or b<0 or a+b<=0:raise ValueError("Ensemble weights must be nonnegative and sum to a positive value")
    return cols,errors,ae_score,iso_score,(a*ae_score+b*iso_score)/(a+b),float(ens.get("decision_threshold",.95))

def _prediction_records(w,cols,errors,sa,si,se,threshold,signal_applicability=None,model_version="0.1.0"):
    records=[]
    for i,row in enumerate(w.iter_rows(named=True)):
        contrib=errors[i]/max(float(errors[i].sum()),1e-12);by_signal={}
        for name,value in zip(cols,contrib):
            signal=feature_signal(name)
            if signal is None or value<=0 or name[len(signal)+1:] not in {"mean","median","min","max","range","std","mad","last","delta","slope"}:continue
            by_signal[signal]=by_signal.get(signal,0.)+float(value)
        observations=[]
        # Missing optional regime context is not evidence of a mismatched regime.
        if row.get("context_regime_unknown",False) and row.get("operating_regime") is not None:observations.append("regime_mismatch")
        applicable=set((signal_applicability or {}).get(str(row.get("asset_class")),[]))
        # Legacy schemas lack this map; do not mistake a global signal union
        # for channels that should exist on every asset class.
        missing_signals=[c[:-8] for c in cols if c.endswith("_missing") and c[:-8] in applicable and row.get(c,0)>0]
        if missing_signals:observations.append("signal_loss")
        if any(c.endswith("_coverage_ratio") and c[:-15] in applicable and row.get(c,1)<.8 for c in cols):observations.append("sampling_degradation")
        if se[i]>=threshold:
            if any(name.endswith(("_bad_ratio","_uncertain_ratio")) and value>0 and row.get(name,0)>0 for name,value in zip(cols,contrib)):
                observations.append("quality_degradation")
            for signal,_ in sorted(by_signal.items(),key=lambda pair:pair[1],reverse=True)[:3]:
                n=signal.lower()
                if "vibr" in n:observations.append("vibration_deviation")
                elif "temp" in n or "therm" in n:observations.append("thermal_deviation")
                elif "press" in n:observations.append("pressure_deviation")
            if not observations:observations=["multivariate_novelty"]
        records.append({"run_id":row["run_id"],"asset_id":row["asset_id"],"window_start":__import__("datetime").datetime.fromtimestamp(row["window_start"]/1e6,__import__("datetime").timezone.utc).isoformat(),"window_end":__import__("datetime").datetime.fromtimestamp(row["window_end"]/1e6,__import__("datetime").timezone.utc).isoformat(),"scores":{"autoencoder":float(sa[i]),"isolation_forest":float(si[i]),"irregularity":float(se[i])},"observations":list(dict.fromkeys(observations)),"feature_contributions":dict(zip(cols,map(float,contrib))),"model_version":model_version})
    return records

def _record_progress(path,phase,**fields):
    event={"phase":phase,"timestamp":datetime.datetime.now(datetime.timezone.utc).timestamp(),**fields}
    with Path(path).open("a",encoding="utf-8") as stream:stream.write(json.dumps(event)+"\n")

def train(dataset,config,output):
    requested_cfg=yaml.safe_load(Path(config).read_text())
    if requested_cfg.get("model_family")=="contextual" and not all((Path(dataset)/part).is_dir() for part in ("train","validation")):
        raise ValueError("Contextual training requires explicit train/ and validation/ directories")
    if requested_cfg.get("model_family")=="contextual" and Path(output).exists() and any(Path(output).iterdir()):
        raise FileExistsError("Contextual training requires an empty output directory; preserve previous artifacts")
    out=Path(output);out.mkdir(parents=True,exist_ok=True);progress=out/"training_progress.jsonl";progress.write_text("",encoding="utf-8");_record_progress(progress,"loading_data")
    cfg=yaml.safe_load(Path(config).read_text());seed=int(cfg.get("seed",42));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    root=Path(dataset);trainp=root/"train";valp=root/"validation"
    if not trainp.exists():trainp=root
    if not valp.exists():raise ValueError("A separate validation/ partition is required for calibration")
    group=cfg.get("split_group","run_id");units=cfg.get("normalization",{}).get("canonical_units",{});rawtr=_exclude_feature_inputs(normalize_units(load_dataset(trainp),units),cfg);rawva=_exclude_feature_inputs(normalize_units(load_dataset(valp),units),cfg)
    _record_progress(progress,"data_loaded",training_rows=rawtr.height,validation_rows=rawva.height)
    if group not in rawtr.columns or group not in rawva.columns:raise ValueError(f"Split group {group!r} is not present in both partitions")
    overlap=set(rawtr[group].drop_nulls().unique().to_list())&set(rawva[group].drop_nulls().unique().to_list())
    if overlap:raise ValueError(f"Data leakage: overlapping {group} between train and validation: {sorted(map(str,overlap))[:5]}")
    channel_column="measurement_role" if "measurement_role" in rawtr.columns else "signal_class"
    if "value_kind" in rawtr.columns:
        rawtr=rawtr.filter(__import__("polars").col("value_kind").is_null()|(__import__("polars").col("value_kind")=="continuous"))
        rawva=rawva.filter(__import__("polars").col("value_kind").is_null()|(__import__("polars").col("value_kind")=="continuous"))
    signal_classes=sorted(str(x) for x in rawtr[channel_column].unique().to_list())
    intervals={}
    if "sampling_interval_ms" in rawtr.columns:
        for signal,sub in rawtr.group_by(channel_column):
            vals=sub["sampling_interval_ms"].drop_nulls().to_list()
            if vals:intervals[str(signal[0] if isinstance(signal,tuple) else signal)]=float(np.median(vals))
    signal_schema={"signal_classes":signal_classes if channel_column=="signal_class" else [],"measurement_roles":signal_classes if channel_column=="measurement_role" else [],"sampling_intervals_ms":intervals}
    tr=_windows(trainp,cfg,signal_schema);va=_windows(valp,cfg,signal_schema);_record_progress(progress,"features_ready",training_windows=tr.height,validation_windows=va.height)
    if "is_anomaly" in tr.columns and tr["is_anomaly"].any():raise ValueError("Training partition must contain normal-only windows")
    tr,context_vocab=encode_context(tr);va,context_vocab=encode_context(va,context_vocab)
    if cfg.get("model_family")=="contextual":
        from .contextual import train_contextual_windows
        applicability={}
        if "asset_class" in rawtr.columns:
            for key,sub in rawtr.group_by("asset_class"):
                applicability[str(key[0])]=sorted(str(v) for v in sub[channel_column].drop_nulls().unique().to_list())
        schema={**signal_schema,"context_vocabulary":context_vocab,"signals_by_asset_class":applicability,"canonical_units":units}
        metadata={"model_version":str(cfg.get("model_version","0.4.0-candidate")),"training_date":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "git_commit":_git_commit(),"git_working_tree_dirty":_git_working_tree_dirty(),"dataset_hash":_hash_paths([trainp,valp]),
                  "dataset_schema_version":"1","stride_seconds":_duration_minutes(cfg.get("window",{}).get("stride","1m"))*60}
        return train_contextual_windows(tr,va,cfg,out,schema,metadata)
    normal=va.filter(~__import__("polars").col("is_anomaly")) if "is_anomaly" in va.columns else va
    if normal.is_empty():raise ValueError("Validation must contain at least one normal window for score calibration")
    cols=feature_columns(tr);val_cols=set(feature_columns(va));missing=set(cols)-val_cols;extra=val_cols-set(cols)
    if missing or extra:raise ValueError(f"Validation feature mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    scaler_name=cfg.get("scaler",{}).get("type","robust");scaler=RobustScaler() if scaler_name=="robust" else StandardScaler() if scaler_name=="standard" else None
    if scaler is None:raise ValueError("scaler.type must be robust or standard")
    train_x=tr.select(cols).to_numpy().astype(float);val_x=va.select(cols).to_numpy().astype(float);norm_x=normal.select(cols).to_numpy().astype(float);scaler.fit(train_x);train_x=scaler.transform(train_x);val_x=scaler.transform(val_x);norm_x=scaler.transform(norm_x)
    ae_cfg={**cfg.get("autoencoder",{}),"device":cfg.get("device","auto")};_record_progress(progress,"autoencoder_start",device=cfg.get("device","auto"),training_windows=len(train_x),validation_windows=len(norm_x));ae=train_ae(train_x,norm_x,ae_cfg,seed,out/"autoencoder.pt",progress)
    _record_progress(progress,"isolation_forest_start");ic=cfg.get("isolation_forest",{});iso=IsolationForest(n_estimators=int(ic.get("n_estimators",300)),max_samples=ic.get("max_samples","auto"),contamination=ic.get("contamination","auto"),max_features=ic.get("max_features",1.),random_state=int(ic.get("random_state",seed))).fit(train_x);joblib.dump(iso,out/"isolation_forest.joblib");joblib.dump(scaler,out/"scaler.joblib");_record_progress(progress,"calibrating_scores")
    normal_ae=ae_errors(ae,norm_x).mean(axis=1);normal_iso=-iso.score_samples(norm_x);all_errors=ae_errors(ae,val_x);ae_score=cdf_calibrate(normal_ae,all_errors.mean(axis=1));iso_score=cdf_calibrate(normal_iso,-iso.score_samples(val_x));ens=cfg.get("ensemble",{});a=float(ens.get("autoencoder_weight",.5));b=float(ens.get("isolation_weight",.5))
    if a<0 or b<0 or a+b<=0:raise ValueError("Ensemble weights must be nonnegative and sum to a positive value")
    ensemble=(a*ae_score+b*iso_score)/(a+b);threshold=float(ens.get("decision_threshold",.95));metrics={"validation_windows":len(va),"calibration_normal_windows":len(normal),"validation_normal_only":bool(not va["is_anomaly"].any()) if "is_anomaly" in va.columns else None,"validation_normal_assumed":"is_anomaly" not in va.columns,"autoencoder_score_median":float(np.median(ae_score)),"isolation_score_median":float(np.median(iso_score)),"irregularity_score_median":float(np.median(ensemble)),"decision_threshold":threshold}
    if "is_anomaly" in va.columns:
        win_minutes=_duration_minutes(cfg.get("window",{}).get("stride","1m"));labels=va["is_anomaly"].cast(__import__("polars").Boolean).to_numpy().astype(int);metric_args={"asset_ids":va["asset_id"].to_numpy(),"event_ids":va["event_id"].to_numpy() if "event_id" in va.columns else None,"event_start_us":va["event_start_us"].to_list() if "event_start_us" in va.columns else None,"window_end_us":va["window_end"].to_numpy(),"window_seconds":win_minutes*60}
        metrics["supervised_validation"]={"autoencoder":evaluate_scores(labels,ae_score,threshold,**metric_args),"isolation_forest":evaluate_scores(labels,iso_score,threshold,**metric_args),"irregularity":evaluate_scores(labels,ensemble,threshold,**metric_args)}
    canonical_units=cfg.get("normalization",{}).get("canonical_units",{})
    signals_by_asset_class={}
    if "asset_class" in rawtr.columns:
        for asset_class,sub in rawtr.group_by("asset_class"):
            key=asset_class[0] if isinstance(asset_class,tuple) else asset_class
            signals_by_asset_class[str(key)]=sorted(str(value) for value in sub[channel_column].drop_nulls().unique().to_list())
    model_version=str(cfg.get("model_version","0.1.0"))
    (out/"feature_schema.json").write_text(json.dumps({"version":"1","features":cols,"context_vocabulary":context_vocab,"signal_classes":signal_classes if channel_column=="signal_class" else [],"measurement_roles":signal_classes if channel_column=="measurement_role" else [],"signals_by_asset_class":signals_by_asset_class,"sampling_intervals_ms":intervals,"canonical_units":canonical_units},indent=2));(out/"training_config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=True));(out/"thresholds.json").write_text(json.dumps({"calibration":"empirical validation-normal CDF; lower sklearn score_samples is more anomalous","decision_threshold":threshold,"autoencoder_raw_p99":float(np.quantile(normal_ae,.99)),"isolation_raw_p99":float(np.quantile(normal_iso,.99)),"autoencoder_reference":normal_ae.tolist(),"isolation_reference":normal_iso.tolist()},indent=2));(out/"metrics.json").write_text(json.dumps(metrics,indent=2));(out/"training_metadata.json").write_text(json.dumps({"model_version":model_version,"training_date":datetime.datetime.now(datetime.timezone.utc).isoformat(),"git_commit":_git_commit(),"git_working_tree_dirty":_git_working_tree_dirty(),"dataset_hash":_hash_paths([trainp,valp]),"dataset_schema_version":"1","seed":seed,"feature_schema_version":"1","autoencoder_parameters":ae_cfg,"autoencoder_training":ae.training_metadata,"isolation_forest_parameters":ic,"scaler":scaler_name,"canonical_units":canonical_units,"training_samples":len(train_x),"validation_samples":len(val_x),"training_normal_assumed":"is_anomaly" not in tr.columns},indent=2));model_card=(f"# Model card\n\n- Version: {model_version}\n- Training date: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n- Git commit: {_git_commit()} (working tree dirty: {_git_working_tree_dirty()})\n- Intended use: unsupervised operational behavior irregularity screening\n- Out of scope: fault diagnosis, causal claims and safety control\n- Dataset SHA-256 (train + validation): {_hash_paths([trainp,valp])}\n- Split group: {group}\n- Signal classes: {", ".join(signal_classes)}\n- Window and stride: {cfg.get("window",{}).get("size","15m")} / {cfg.get("window",{}).get("stride","1m")}\n- Scaler: {scaler_name}\n- Autoencoder training device/updates: {ae.training_metadata["device"]} / {ae.training_metadata["steps_completed"]} of {ae.training_metadata["steps_requested"]} gradient steps over {ae.training_metadata["epochs_completed"]} epochs (best epoch {ae.training_metadata["best_epoch"]})\n- Calibration: empirical CDF on {len(normal)} normal validation windows\n- Alert threshold: {threshold} (configurable baseline)\n- Metrics: see metrics.json\n- Limitations: synthetic evidence does not establish generalization; scores describe deviations without causal attribution\n- Human review: investigate supporting trends, coverage, regime and raw measurements before acting\n")
    (out/"model_card.md").write_text(model_card);_record_progress(progress,"completed",metrics_file=str(out/"metrics.json"),model_version=model_version)
    return metrics

def infer(modeldir,dataset,output):
    cfg=yaml.safe_load((Path(modeldir)/"training_config.yaml").read_text());schema=json.loads((Path(modeldir)/"feature_schema.json").read_text());w=_windows(dataset,cfg,schema);w,_=encode_context(w,schema.get("context_vocabulary",{}));cols,errors,sa,si,se,threshold=_scores(modeldir,w);records=_prediction_records(w,cols,errors,sa,si,se,threshold,schema.get("signals_by_asset_class"),str(cfg.get("model_version","0.1.0")))
    if (Path(modeldir)/"contextual_models.json").exists():
        for record in records:record["contribution_basis"]="share of top-k normalized squared reconstruction error"
    if output and str(output).lower().endswith(".parquet"):
        import polars as pl;pl.DataFrame(records).write_parquet(output)
    elif output:Path(output).write_text("\n".join(json.dumps(r) for r in records)+"\n")
    return records
