"""Diagnose physical misses and screen normal-only relationship residual detectors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import polars as pl
import torch
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_detection import cache_windows,write_json
from scripts.research_physical import PHYSICAL,VALUE_SUFFIXES,load_events,summarize
from scripts.validate_contextual import ui_state
from ot_irregularity.contextual import run_bucket,select_columns,tail_errors
from ot_irregularity.models import ae_errors,train_ae
from ot_irregularity.pipeline import _scores,cdf_calibrate
from ot_irregularity.relationships import RelationshipResiduals


def closed(frame,raw):
    bounds=raw.group_by(['run_id','asset_id']).agg(pl.col('timestamp').max().dt.timestamp('us').alias('_watermark'))
    return frame.join(bounds,on=['run_id','asset_id']).filter(pl.col('window_end')<=pl.col('_watermark')).drop('_watermark').sort(['run_id','asset_id','window_start'])


def diagnostic(raw_train,raw_val,fit,cal,dev,schema,baseline,events,out):
    """Marginal visibility audit; no counterfactual/causal interpretation."""
    normal=raw_train.filter(pl.col('run_id').is_in(fit['run_id'].unique().to_list()))
    role='measurement_role' if 'measurement_role' in normal.columns else 'signal_class'
    normal=normal.filter((pl.col('quality')=='good')&pl.col('value').is_finite())
    bounds=normal.group_by(['asset_class',role]).agg(pl.col('value').quantile(.005).alias('lo'),pl.col('value').quantile(.995).alias('hi'))
    refs={(r['asset_class'],r[role]):(r['lo'],r['hi']) for r in bounds.to_dicts()}
    feature_score=np.zeros(len(dev))
    for group in sorted(fit['asset_class'].unique()):
        cols=[c for c in select_columns(fit,schema,group) if c.endswith(VALUE_SUFFIXES)]
        scaler=RobustScaler().fit(fit.filter(pl.col('asset_class')==group).select(cols).to_numpy())
        a=np.abs(scaler.transform(cal.filter(pl.col('asset_class')==group).select(cols).to_numpy()))
        mask=dev['asset_class'].to_numpy()==group
        b=np.abs(scaler.transform(dev.filter(pl.col('asset_class')==group).select(cols).to_numpy()))
        feature_score[mask]=cdf_calibrate(np.sort(a,axis=1)[:,-4:].mean(axis=1),np.sort(b,axis=1)[:,-4:].mean(axis=1))
    _,_,_,_,score,threshold=_scores(baseline,dev)
    raw_val=raw_val.with_columns(pl.col('timestamp').dt.timestamp('us').alias('_us'))
    rows=[]
    for event in events:
        if event['family'] not in PHYSICAL:continue
        overlap=(dev['run_id'].to_numpy()==event['run_id'])&(dev['asset_id'].to_numpy()==event['asset_id'])&(dev['window_start'].to_numpy()<=event['end_us'])&(dev['window_end'].to_numpy()>event['start_us'])
        raw=raw_val.filter((pl.col('run_id')==event['run_id'])&(pl.col('asset_id')==event['asset_id'])&(pl.col('_us')>=event['start_us'])&(pl.col('_us')<=event['end_us']))
        outside=0;usable=0
        for (group,signal),part in raw.group_by(['asset_class',role]):
            limits=refs.get((group,signal))
            if limits:
                v=part.filter((pl.col('quality')=='good')&pl.col('value').is_finite())['value'].to_numpy()
                usable+=len(v);outside+=int(((v<limits[0])|(v>limits[1])).sum())
        peak=float(feature_score[overlap].max()) if overlap.any() else None
        ae=float(score[overlap].max()) if overlap.any() else None
        rows.append({**event,'raw_usable_samples':usable,'raw_outside_normal_99pct':outside,
            'raw_outside_fraction':outside/usable if usable else None,'feature_peak_normal_percentile':peak,
            'ae_peak_normal_percentile':ae,'detected':bool(overlap.any() and (score[overlap]>=threshold).any()),
            'cooccurring_events':sum(e['event_id']!=event['event_id'] and e['run_id']==event['run_id'] and e['asset_id']==event['asset_id'] and e['start_us']<=event['end_us'] and e['end_us']>=event['start_us'] for e in events)})
    summary={'physical_events':len(rows),'detected':sum(r['detected'] for r in rows),
        'with_any_raw_marginal_extreme':sum(r['raw_outside_normal_99pct']>0 for r in rows),
        'with_feature_percentile_ge_99':sum((r['feature_peak_normal_percentile'] or 0)>=.99 for r in rows),
        'feature_extreme_but_not_detected':sum((r['feature_peak_normal_percentile'] or 0)>=.99 and not r['detected'] for r in rows),
        'cooccurring':sum(r['cooccurring_events']>0 for r in rows)}
    write_json(out/'diagnostic.json',{'summary':summary,'events':rows,
        'limit':'Marginal extremes across all applicable roles are descriptive only. Multiple comparisons, regime/load changes and concurrent events can explain them. No same-seed normal counterfactual was generated; raw physical detectability is unresolved.'})
    print('DIAGNOSTIC',json.dumps(summary),flush=True)


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    schema=json.loads((args.baseline/'feature_schema.json').read_text());cfg=yaml.safe_load((args.baseline/'training_config.yaml').read_text())
    write_json(out/'protocol.json',{'seed':20261005,'ridge_alpha':10,'ae_latent':2,'ae_steps':3000,
        'variants':['static-direct','static-ae','temporal-direct','temporal-ae'],'supplements':'max of each variant and v0.4, recalibrated on normal calibration windows',
        'tail_policy':'complete for every candidate and baseline','quantile':.99,
        'selection':'More physical events, no loss of all events, precision >=0.5 and no increase in false alert minutes versus unchanged v0.4. Eligible candidates require confirmation across seeds and a fresh frozen holdout.',
        'history':'only previous contiguous minute, same run and asset; static fallback at warmup or bad previous quality',
        'unavailable':'any role with good_ratio<0.95, coverage<0.9 or missing: residual detector unavailable, score sentinel zero; baseline preserved in supplements',
        'test_used':False,'baseline_manifest_sha256':hashlib.sha256((args.baseline/'contextual_models.json').read_bytes()).hexdigest()})
    frames=cache_windows(args.dataset,schema,cfg,args.cache)
    raws={p:pl.read_parquet(args.dataset/p/'telemetry.parquet') for p in frames}
    tr,va=[closed(frames[p],raws[p]) for p in ('train','validation')]
    if tr['is_anomaly'].null_count() or tr['is_anomaly'].any():raise ValueError('Non-normal training')
    if set(tr['run_id'])&set(va['run_id']):raise ValueError('Run leakage')
    fit=tr.filter(pl.Series([run_bucket(r,5)!=0 for r in tr['run_id']]))
    stop=tr.filter(pl.Series([run_bucket(r,5)==0 for r in tr['run_id']]))
    cal_mask=np.array([run_bucket(r,2)==0 for r in va['run_id']])&~va['is_anomaly'].to_numpy()
    dev_mask=np.array([run_bucket(r,2)==1 for r in va['run_id']])
    cal,dev=va.filter(pl.Series(cal_mask)),va.filter(pl.Series(dev_mask));events,truth_hashes=load_events(args.raw_root,dev)
    diagnostic(raws['train'],raws['validation'],fit,cal,dev,schema,args.baseline,events,out)
    _,_,_,_,base_cal,threshold=_scores(args.baseline,cal);_,_,_,_,base_dev,_=_scores(args.baseline,dev)
    rows=[{'id':'v0.4-complete','metrics':summarize(dev,base_dev,threshold,events)}]
    predictions=dev.select(['run_id','asset_id','asset_class','window_start','window_end','is_anomaly','event_id','event_start_us']).with_columns(pl.Series('v0.4-complete',base_dev))
    all_scores={};coverage={}
    for temporal in (False,True):
        name='temporal' if temporal else 'static';arrays={k:np.zeros(len(va)) for k in ('direct','ae')}
        coverage[name]={};ui_state(out,'screening',.2 if not temporal else .6,rows,name)
        for group in sorted(fit['asset_class'].unique()):
            root=out/name/group;root.mkdir(parents=True)
            f=fit.filter(pl.col('asset_class')==group);e=stop.filter(pl.col('asset_class')==group);v=va.filter(pl.col('asset_class')==group)
            roles=sorted(schema['signals_by_asset_class'][group]);relation=RelationshipResiduals(roles,temporal).fit(f)
            rf,vf,_=relation.transform(f);re,ve,_=relation.transform(e);rv,vv,history=relation.transform(v)
            if min(vf.sum(),ve.sum())<20:raise ValueError('Insufficient valid residual training')
            residual_scale=np.maximum(np.quantile(np.abs(re[ve]),.95,axis=0),1e-6)
            raw_direct=np.sort(np.abs(rv)/residual_scale,axis=1)[:,-2:].mean(axis=1)
            scaler=RobustScaler().fit(rf[vf]);xf,xe,xv=[scaler.transform(r) for r in (rf[vf],re[ve],rv)]
            ae=train_ae(xf,xe,{**cfg['autoencoder'],'latent_dim':2,'device':args.device},20261005,root/'autoencoder.pt')
            iso=IsolationForest(n_estimators=200,random_state=20261005,n_jobs=2).fit(xf)
            ae_scale=np.maximum(np.quantile(ae_errors(ae,xe),.95,axis=0),1e-8)
            raw_ae,_=tail_errors(ae_errors(ae,xv),ae_scale,2)
            local_cal=np.array([run_bucket(r,2)==0 for r in v['run_id']])&~v['is_anomaly'].to_numpy()&vv
            if local_cal.sum()<20:raise ValueError('Insufficient residual calibration')
            group_mask=va['asset_class'].to_numpy()==group
            references={}
            for kind,raw in [('direct',raw_direct),('ae',raw_ae)]:
                references[kind]=raw[local_cal];scores=cdf_calibrate(references[kind],raw);scores[~vv]=0.
                arrays[kind][group_mask]=scores
            joblib.dump({'relationship':relation,'scaler':scaler,'isolation':iso,'residual_scale':residual_scale,
                'ae_scale':ae_scale,'references':references},root/'detector.joblib')
            coverage[name][group]={'fit':relation.fit_counts_,'validation_windows':len(v),'available':int(vv.sum()),'history':int(history.sum()),'roles':roles}
            print(name,group,'complete',flush=True)
        for kind,values in arrays.items():
            variant=name+'-'+kind;c,d=values[cal_mask],values[dev_mask];all_scores[variant]=(c,d)
    for variant,(c,d) in list(all_scores.items()):
        all_scores['max-'+variant]=(np.maximum(c,base_cal),np.maximum(d,base_dev))
    for variant,(c,d) in all_scores.items():
        th=float(np.nextafter(np.quantile(c,.99,method='higher'),np.inf))
        row={'id':variant,'metrics':summarize(dev,d,th,events)};rows.append(row)
        predictions=predictions.with_columns(pl.Series(variant,d));print(json.dumps(row),flush=True)
    b=rows[0]['metrics']
    eligible=[r['id'] for r in rows[1:] if r['metrics']['physical_events']['detected']>b['physical_events']['detected'] and
        r['metrics']['detected_events']>=b['detected_events'] and r['metrics']['precision']>=.5 and
        r['metrics']['false_positive_windows']<=b['false_positive_windows']]
    write_json(out/'results.json',{'rows':rows,'eligible':eligible,'test_used':False,'coverage':coverage,'truth_hashes':truth_hashes,
        'counts':{k:len(f) for k,f in [('fit',fit),('stop',stop),('cal',cal),('dev',dev)]}})
    predictions.write_parquet(out/'development_predictions.parquet');ui_state(out,'completed',1.,rows)
    print('ELIGIBLE',eligible,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','baseline','raw-root','cache','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');run(p.parse_args())
