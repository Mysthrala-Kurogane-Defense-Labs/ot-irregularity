import datetime,json,shutil,sys
from pathlib import Path
import numpy as np
import polars as pl
import pytest,yaml
from ot_irregularity.data.load import dataset_has_column,load_dataset
from ot_irregularity.features import encode_context,make_windows
from ot_irregularity.metrics import evaluate_scores
from ot_irregularity.pipeline import _hash_paths,cdf_calibrate,train,infer
from ot_irregularity.normalization import normalize_units
from ot_irregularity.cli import main as cli_main

def test_cdf_calibration_order_and_bounds():
 scores=cdf_calibrate([1,2,3],[0,2,4]);assert np.allclose(scores,[0,2/3,1]);assert np.all(np.diff(scores)>=0)
def test_engineering_units_are_normalized_before_feature_scaling():
 d=pl.DataFrame({"signal_class":["pressure","pressure","temperature","flow","valve"],"value":[1.,100.,32.,60.,50.],"unit":["bar","kPa","degF","L/min","%"]})
 out=normalize_units(d,{"pressure":"Pa","temperature":"degC","flow":"m3/s","valve":"ratio"})
 assert np.allclose(out["value"],[100000.,100000.,0.,.001,.5])
 assert out["unit"].to_list()==["Pa","Pa","degC","m3/s","ratio"]
def test_unit_normalization_rejects_unknown_or_missing_units():
 d=pl.DataFrame({"signal_class":["pressure"],"value":[1.],"unit":["furlong"]})
 with pytest.raises(ValueError,match="Unsupported source unit"):normalize_units(d,{"pressure":"Pa"})
 d=d.with_columns(pl.lit(None,dtype=pl.String).alias("unit"))
 with pytest.raises(ValueError,match="without units"):normalize_units(d,{"pressure":"Pa"})
def test_window_statistics_and_labels():
 d=pl.DataFrame({"run_id":["a"]*4,"asset_id":["p"]*4,"timestamp":pl.datetime_range(datetime.datetime(2026,1,1),datetime.datetime(2026,1,1,0,3),interval="1m",eager=True),"tag_id":["x"]*4,"signal_class":["temp"]*4,"value":[1.,2.,9.,10.],"is_anomaly":[False,False,True,True],"event_id":[None,None,"e1","e1"]});w=make_windows(d,2,1);assert w.height>=3 and "temp_mean" in w.columns;assert w["is_anomaly"].any();assert "event_start_us" in w.columns
def test_uncertain_quality_values_still_contribute_to_signal_statistics():
 timestamps=[datetime.datetime(2020,1,1),datetime.datetime(2020,1,1,0,0,30)]
 d=pl.DataFrame({"run_id":["r","r"],"asset_id":["a","a"],"asset_class":["pump","pump"],"operating_regime":["steady","steady"],"timestamp":timestamps,"signal_class":["temperature"]*2,"tag_id":["T1"]*2,"value":[10.,30.],"quality":["uncertain"]*2,"sampling_interval_ms":[30000.,30000.]})
 w=make_windows(d,size_minutes=1,stride_minutes=1,options={"quality":False})
 assert w["temperature_mean"].to_list()==[20.]
 assert w["temperature_last"].to_list()==[30.]
 assert w["temperature_slope"].to_list()==pytest.approx([2/3])
def test_late_event_ids_do_not_break_window_schema_inference():
 timestamps=pl.datetime_range(datetime.datetime(2026,1,1),datetime.datetime(2026,1,1,1,45),interval="1m",eager=True)
 n=len(timestamps)
 d=pl.DataFrame({"run_id":["a"]*n,"asset_id":["p"]*n,"timestamp":timestamps,"tag_id":["x"]*n,"signal_class":["temp"]*n,"value":np.arange(n,dtype=float),"is_anomaly":[False]*(n-6)+[True]*6,"event_id":[None]*(n-6)+["event-1"]*6})
 windows=make_windows(d,1,1)
 assert windows["event_id"].drop_nulls().to_list()==["event-1"]*6
def test_missing_signal_and_unseen_regime_are_explicit():
 ts=pl.datetime_range(datetime.datetime(2026,1,1),datetime.datetime(2026,1,1,0,3),interval="1m",eager=True)
 d=pl.DataFrame({"run_id":["a"]*6,"asset_id":["p"]*6,"timestamp":[ts[0],ts[0],ts[1],ts[1],ts[2],ts[3]],"tag_id":["t","v","t","v","t","t"],"signal_class":["temp","vibration","temp","vibration","temp","temp"],"value":[1.,2.,2.,2.,3.,4.],"asset_class":["pump"]*6,"operating_regime":["steady"]*6})
 w=make_windows(d,2,1,["temp","vibration"],{"temp":60000,"vibration":60000});assert w["vibration_missing"].max()==1
 encoded,vocab=encode_context(w);novel,_=encode_context(w.with_columns(pl.lit("startup").alias("operating_regime")),vocab);assert novel["context_regime_unknown"].all()
def test_schema_rejects_missing_fields(tmp_path):
 f=tmp_path/"bad.csv";f.write_text("asset_id,value\np,1\n");
 with pytest.raises(ValueError,match="Missing required"):load_dataset(f)
 assert dataset_has_column(f,"asset_id") and not dataset_has_column(f,"run_id")
def test_metrics_and_event_latency():
 result=evaluate_scores([0,1,1],[.1,.99,.98],.95,["p"]*3,[None,"e","e"],[None,1_000_000,1_000_000],[1_000_000,3_000_000,4_000_000],60);assert result["precision"]==1 and result["recall"]==1 and result["event_detection_rate"]==1 and result["mean_detection_latency_seconds"]==2
def _fast_config(root,tmp_path):
 cfg=yaml.safe_load((root/"configs/baseline.yaml").read_text());cfg["autoencoder"].update(epochs=3,patience=2);cfg["isolation_forest"]["n_estimators"]=40;f=tmp_path/"config.yaml";f.write_text(yaml.safe_dump(cfg));return f
def test_training_rejects_group_leakage(tmp_path):
 root=Path(__file__).parents[1];data=tmp_path/"data";shutil.copytree(root/"examples/synthetic/train",data/"train");shutil.copytree(root/"examples/synthetic/validation",data/"validation");tr=pl.read_csv(data/"train/observations.csv").with_columns(pl.when(pl.col("run_id")=="r0").then(pl.lit("r100")).otherwise(pl.col("run_id")).alias("run_id"));tr.write_csv(data/"train/observations.csv");
 with pytest.raises(ValueError,match="Data leakage"):train(data,_fast_config(root,tmp_path),tmp_path/"out")
def test_end_to_end_and_reproducible_inference(tmp_path,capsys):
 root=Path(__file__).parents[1];out=tmp_path/"model";metrics=train(root/"examples/synthetic",_fast_config(root,tmp_path),out);assert metrics["validation_windows"]>0 and metrics["calibration_normal_windows"]<metrics["validation_windows"];assert json.loads((out/"training_metadata.json").read_text())["dataset_hash"]==_hash_paths([root/"examples/synthetic/train",root/"examples/synthetic/validation"]);assert metrics["supervised_validation"]["irregularity"]["pr_auc"] is not None and metrics["supervised_validation"]["autoencoder"]["pr_auc"] is not None
 expected={"autoencoder.pt","isolation_forest.joblib","scaler.joblib","feature_schema.json","training_config.yaml","training_metadata.json","thresholds.json","metrics.json","model_card.md"};assert expected <= {p.name for p in out.iterdir()}
 a=infer(out,root/"examples/synthetic/test",tmp_path/"a.jsonl");b=infer(out,root/"examples/synthetic/test",tmp_path/"b.jsonl");assert a==b and (tmp_path/"a.jsonl").read_bytes()==(tmp_path/"b.jsonl").read_bytes();assert a and isinstance(a[0]["window_start"],str) and a[0]["scores"]
 challenge=root/"examples/synthetic/challenge";challenge_predictions=infer(out,challenge,tmp_path/"challenge.jsonl");assert any("regime_mismatch" in r["observations"] for r in challenge_predictions)
 old_argv=sys.argv;sys.argv=["ot-irregularity","evaluate","--model",str(out),"--dataset",str(challenge)]
 try:
  with pytest.raises(SystemExit,match="requires --challenge"):cli_main()
 finally:sys.argv=old_argv
 sys.argv=["ot-irregularity","evaluate","--model",str(out),"--dataset",str(challenge/"observations.csv")]
 try:
  with pytest.raises(SystemExit,match="requires --challenge"):cli_main()
 finally:sys.argv=old_argv
 sys.argv=["ot-irregularity","evaluate","--model",str(out),"--dataset",str(challenge),"--challenge"]
 try:cli_main()
 finally:sys.argv=old_argv
 evaluation=json.loads(capsys.readouterr().out)
 assert set(evaluation)=={"autoencoder","isolation_forest","irregularity","window_count"}
 assert all(evaluation[name]["pr_auc"] is not None for name in ("autoencoder","isolation_forest","irregularity"))
 assert a[0]["scores"]["irregularity"]==pytest.approx((a[0]["scores"]["autoencoder"]+a[0]["scores"]["isolation_forest"])/2)
