import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score,average_precision_score,roc_auc_score
def evaluate_scores(y_true,scores,threshold=.95,asset_ids=None,event_ids=None,event_start_us=None,window_end_us=None,window_seconds=60):
    y=np.asarray(y_true,dtype=int);s=np.asarray(scores,dtype=float);pred=s>=threshold
    result={"threshold":float(threshold),"samples":int(len(y)),"positive_samples":int(y.sum()),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"pr_auc":float(average_precision_score(y,s)) if y.sum() else None,"roc_auc":float(roc_auc_score(y,s)) if len(np.unique(y))>1 else None,"false_positive_windows":int((pred&(y==0)).sum())}
    if asset_ids is not None and len(y):
        duration_hours=len(y)*window_seconds/3600;result["false_positives_per_asset_hour"]=float(result["false_positive_windows"]/max(duration_hours,1e-12));result["false_positives_per_asset_day"]=float(result["false_positive_windows"]/max(duration_hours/24,1e-12))
    if event_ids is not None:
        events=np.asarray(event_ids,dtype=object);starts=np.asarray(event_start_us,dtype=object) if event_start_us is not None else None;ends=np.asarray(window_end_us,dtype=float) if window_end_us is not None else None
        known=[]
        for event in set(events):
            if event is None or event=="":continue
            if isinstance(event,(float,np.floating)) and np.isnan(event):continue
            known.append(event)
        detected=0;latencies=[]
        for event in known:
            ids=np.where(events==event)[0];hit=ids[pred[ids]]
            if len(hit):
                detected+=1
                if starts is not None and ends is not None:
                    valid=[starts[i] for i in ids if starts[i] is not None];onset=min(valid) if valid else None
                    if onset is not None:latencies.append(max(0.,(ends[hit[0]]-float(onset))/1e6))
        result["event_count"]=len(known);result["event_detection_rate"]=float(detected/len(known)) if known else None;result["mean_time_to_first_detection_seconds"]=float(np.mean(latencies)) if latencies else None;result["mean_detection_latency_seconds"]=result["mean_time_to_first_detection_seconds"]
    return result
