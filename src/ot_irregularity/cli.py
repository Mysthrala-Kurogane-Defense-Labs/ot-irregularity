import argparse,json
from pathlib import Path
import numpy as np
from .data.load import dataset_has_column
from .features import encode_context,make_windows
from .metrics import evaluate_scores,evaluate_event_intervals
from .pipeline import _duration_minutes,_model_config,_prediction_records,_scores,_windows,infer,train
import yaml

def main():
 p=argparse.ArgumentParser(prog="ot-irregularity");s=p.add_subparsers(dest="cmd",required=True)
 t=s.add_parser("train");t.add_argument("--dataset",required=True);t.add_argument("--config",required=True);t.add_argument("--output",required=True)
 i=s.add_parser("infer");i.add_argument("--model",required=True);i.add_argument("--dataset",required=True);i.add_argument("--output",default="predictions.jsonl")
 e=s.add_parser("evaluate");e.add_argument("--model",required=True);e.add_argument("--dataset",required=True);e.add_argument("--challenge",action="store_true");e.add_argument("--labels",default="is_anomaly")
 e.add_argument("--events",help="JSON with an events list (run_id, asset_id, event_id, start_us, end_us); counts overlapping and unobservable events")
 x=s.add_parser("inspect");x.add_argument("--model",required=True)
 a=p.parse_args()
 if a.cmd=="train":print(json.dumps(train(a.dataset,a.config,a.output),indent=2))
 elif a.cmd=="infer":infer(a.model,a.dataset,a.output);print(f"Wrote {a.output}")
 elif a.cmd=="inspect":
  d=Path(a.model);th=json.loads((d/"thresholds.json").read_text());th.pop("autoencoder_reference",None);th.pop("isolation_reference",None);print(json.dumps({"metadata":json.loads((d/"training_metadata.json").read_text()),"features":json.loads((d/"feature_schema.json").read_text()),"metrics":json.loads((d/"metrics.json").read_text()),"thresholds":th},indent=2))
 elif a.cmd=="evaluate":
  path=Path(a.dataset).resolve()
  if any(part.name.casefold()=="challenge" for part in (path,*path.parents)) and not a.challenge:raise SystemExit("Challenge evaluation requires --challenge")
  cfg=_model_config(a.model);has_labels=dataset_has_column(path,a.labels)
  schema=json.loads((Path(a.model)/"feature_schema.json").read_text())
  interval_events=json.loads(Path(a.events).read_text(encoding="utf-8"))["events"] if a.events else None
  if not has_labels and interval_events is None:print(json.dumps({"windows":_windows(path,cfg,schema).height,"metrics":None,"reason":f"No ground-truth column {a.labels!r}; metrics omitted."},indent=2));return
  w=_windows(path,cfg,schema);w,_=encode_context(w,schema.get("context_vocabulary",{}));cols,errors,sa,si,scores,threshold=_scores(a.model,w);assets=w["asset_id"].to_numpy();events=w["event_id"].to_numpy() if "event_id" in w.columns else None;secs=_duration_minutes(cfg["window"]["stride"])*60
  if has_labels:labels=w[a.labels].cast(__import__('polars').Boolean).to_numpy().astype(int)
  else:
   labels=np.zeros(len(w),dtype=int)
   for event in interval_events:
    mask=(w["run_id"].to_numpy()==event["run_id"])&(assets==event["asset_id"])&(w["window_start"].to_numpy()<=int(event["end_us"]))&(w["window_end"].to_numpy()>int(event["start_us"]))
    labels[mask]=1
  metric_args={"asset_ids":assets,"event_ids":events,"event_start_us":w["event_start_us"].to_list() if "event_start_us" in w.columns else None,"window_end_us":w["window_end"].to_numpy(),"window_seconds":secs}
  result={"autoencoder":evaluate_scores(labels,sa,threshold,**metric_args),"isolation_forest":evaluate_scores(labels,si,threshold,**metric_args),"irregularity":evaluate_scores(labels,scores,threshold,**metric_args),"window_count":len(w)}
  if a.events:
   result["event_intervals"]={name:evaluate_event_intervals(interval_events,values,threshold,w["run_id"].to_numpy(),assets,w["window_start"].to_numpy(),w["window_end"].to_numpy()) for name,values in (("autoencoder",sa),("isolation_forest",si),("irregularity",scores))}
  print(json.dumps(result,indent=2))
if __name__=="__main__":main()
