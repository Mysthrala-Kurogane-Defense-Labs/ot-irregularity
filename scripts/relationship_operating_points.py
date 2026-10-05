"""Explicit second development stage: normal-calibrated operating points, no test."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import polars as pl

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_detection import write_json
from scripts.research_physical import load_events,summarize
from scripts.research_relationships import closed
from ot_irregularity.contextual import run_bucket,tail_errors
from ot_irregularity.models import load_ae,ae_errors
from ot_irregularity.pipeline import _scores,cdf_calibrate


def residual_scores(root,frame,device='auto'):
    arrays={k:np.zeros(len(frame)) for k in ('direct','ae')}
    for group in sorted(frame['asset_class'].unique()):
        path=root/group;model=joblib.load(path/'detector.joblib')
        mask=frame['asset_class'].to_numpy()==group
        residual,available,_=model['relationship'].transform(frame.filter(pl.col('asset_class')==group))
        raw_direct=np.sort(np.abs(residual)/model['residual_scale'],axis=1)[:,-2:].mean(axis=1)
        ae=load_ae(path/'autoencoder.pt',device)
        raw_ae,_=tail_errors(ae_errors(ae,model['scaler'].transform(residual)),model['ae_scale'],2)
        for kind,raw in [('direct',raw_direct),('ae',raw_ae)]:
            score=cdf_calibrate(model['references'][kind],raw);score[~available]=0.
            arrays[kind][mask]=score
    return arrays


def run(args):
    output=args.research/'operating_points.json'
    if output.exists():raise FileExistsError('Operating points already evaluated')
    write_json(args.research/'operating_point_protocol.json',{'stage':'Second adaptive development stage, after representation screening; no test used',
        'quantiles':[.99,.995,.9975,.999],'selection':'Original no-regression gate unchanged: more physical events, total detections >= baseline, precision >=0.5, false minutes <= baseline. Unresolvable cutoffs excluded.'})
    va=closed(pl.read_parquet(args.cache/'validation.parquet'),pl.read_parquet(args.dataset/'validation/telemetry.parquet'))
    cal_mask=np.array([run_bucket(r,2)==0 for r in va['run_id']])&~va['is_anomaly'].to_numpy()
    dev_mask=np.array([run_bucket(r,2)==1 for r in va['run_id']]);dev=va.filter(pl.Series(dev_mask))
    events,_=load_events(args.raw_root,dev)
    _,_,_,_,base,threshold=_scores(args.baseline,va)
    b=summarize(dev,base[dev_mask],threshold,events)
    rows=[];repeatability={}
    original=pl.read_parquet(args.research/'development_predictions.parquet')
    for mode in ('static','temporal'):
        values=residual_scores(args.research/mode,va,args.device)
        for kind,score in values.items():
            name=mode+'-'+kind
            repeatability[name]=float(np.max(np.abs(original[name].to_numpy()-score[dev_mask])))
            for supplement in (False,True):
                s=np.maximum(score,base) if supplement else score
                for q in (.99,.995,.9975,.999):
                    th=float(np.nextafter(np.quantile(s[cal_mask],q,method='higher'),np.inf))
                    if th>1:continue
                    m=summarize(dev,s[dev_mask],th,events)
                    eligible=(m['physical_events']['detected']>b['physical_events']['detected'] and m['detected_events']>=b['detected_events'] and m['precision']>=.5 and m['false_positive_windows']<=b['false_positive_windows'])
                    rows.append({'id':('max-' if supplement else '')+name,'quantile':q,'metrics':m,'eligible':bool(eligible)})
    write_json(output,{'rows':rows,'baseline':b,'reload_score_max_difference':repeatability,'test_used':False})
    print(json.dumps([{'id':r['id'],'q':r['quantile'],'events':r['metrics']['detected_events'],'physical':r['metrics']['physical_events']['detected'],'fp':r['metrics']['false_positive_windows'],'eligible':r['eligible']} for r in rows],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','baseline','raw-root','cache','research'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');run(p.parse_args())
