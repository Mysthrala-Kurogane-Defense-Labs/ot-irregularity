"""Predeclared development screen of normal thermal dynamics, without test access."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.research_alert_budgets import decision_metrics, union_decisions
from scripts.research_detection import cache_windows, write_json
from scripts.research_physical import load_events, summarize
from scripts.research_relationships import closed
from scripts.validate_physical import hashes
from ot_irregularity.contextual import run_bucket
from ot_irregularity.metrics import evaluate_event_intervals
from ot_irregularity.pipeline import _scores, cdf_calibrate
from ot_irregularity.thermal import ThermalDynamics


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    quantiles=[.99,.995,.9975]
    write_json(out/'protocol.json', {'stage':'development only','alpha':1.,'window':'complete 1m',
        'targets':'temperature roles, separately; endpoint change in degC',
        'drivers':'previous endpoint temperature and current mean nonthermal/nonvibration operating roles; excludes photoeye counts',
        'calibration':'per-target absolute residual normal CDF, maximum across available targets; pooled normal quantile',
        'quantiles':quantiles,'candidates':['thermal-alone','max-baseline','union-baseline'],
        'gate':'More cooling events than baseline, no baseline event detection lost, false windows <= baseline, precision >= .5',
        'selection':'most cooling events, then physical events, then total events, then fewer false windows; deterministic Ridge no seed sweep',
        'test_used':False})
    cfg=yaml.safe_load((args.baseline/'training_config.yaml').read_text())
    if cfg['window']!={'size':'1m','stride':'1m'}:raise ValueError('This experiment requires baseline one-minute nonoverlapping windows')
    schema=json.loads((args.baseline/'feature_schema.json').read_text())
    cached=cache_windows(args.dataset,schema,cfg,args.cache)
    frames={p:closed(cached[p],pl.read_parquet(args.dataset/p/'telemetry.parquet')) for p in ('train','validation')}
    tr,va=frames['train'],frames['validation']
    if set(tr['run_id'])&set(va['run_id']):raise ValueError('Run leakage')
    fit=tr.filter(pl.Series([run_bucket(r,5)!=0 for r in tr['run_id']]))
    cal=np.array([run_bucket(r,2)==0 for r in va['run_id']])&~va['is_anomaly'].to_numpy()
    dm=np.array([run_bucket(r,2)==1 for r in va['run_id']]);dev=va.filter(pl.Series(dm))
    events,truth_hashes=load_events(args.raw_root,dev)
    score=np.zeros(len(va));available=np.zeros(len(va),dtype=bool);diagnostic=[];models={}
    for group,roles in sorted(schema['signals_by_asset_class'].items()):
        targets=[r for r in roles if 'temperature' in r]
        drivers=[r for r in roles if r not in targets and 'vibration' not in r and 'photoeye' not in r]
        gf=fit.filter(pl.col('asset_class')==group);mask=va['asset_class'].to_numpy()==group;gv=va.filter(pl.Series(mask))
        gc=cal[mask];gd=dm[mask];local=np.zeros(len(gv));ga=np.zeros(len(gv),dtype=bool)
        for target in targets:
            model=ThermalDynamics(target,drivers).fit(gf)
            residual,valid=model.transform(gv);reference=np.abs(residual[gc&valid])
            if len(reference)<20:raise ValueError('Insufficient valid thermal calibration')
            s=cdf_calibrate(reference,np.abs(residual));s[~valid]=0.
            local=np.maximum(local,s);ga|=valid
            key=group+'--'+target
            joblib.dump({'dynamics':model,'reference':reference},out/(key+'.joblib'))
            models[key]={'target':target,'drivers':drivers,'fit_windows':model.fit_windows_,
                'calibration_windows':len(reference),'normal_abs_residual_quantiles_degC':np.quantile(reference,[.5,.95,.99,1]).tolist()}
            for event in events:
                if event['family']!='cooling_degradation':continue
                overlap=(gv['run_id'].to_numpy()==event['run_id'])&(gv['asset_id'].to_numpy()==event['asset_id'])&(gv['window_start'].to_numpy()<=event['end_us'])&(gv['window_end'].to_numpy()>event['start_us'])&gd
                if not overlap.any():continue
                good=overlap&valid
                diagnostic.append({**event,'target':target,'overlap_windows':int(overlap.sum()),'available_windows':int(good.sum()),
                    'peak_abs_residual_degC':float(np.abs(residual[good]).max()) if good.any() else None,
                    'peak_normal_percentile':float(s[good].max()) if good.any() else None})
        score[mask]=local;available[mask]=ga
    _,_,_,_,base,bth=_scores(args.baseline,va)
    baseline=summarize(dev,base[dm],bth,events)
    def detections(alert):
        return evaluate_event_intervals(events,alert.astype(float),.5,dev['run_id'].to_numpy(),dev['asset_id'].to_numpy(),dev['window_start'].to_numpy(),dev['window_end'].to_numpy())['events']
    before=detections(base[dm]>=bth);rows=[]
    for kind in ('thermal-alone','max-baseline','union-baseline'):
        combined=np.maximum(score,base) if kind=='max-baseline' else score
        for q in quantiles:
            th=float(np.nextafter(np.quantile(combined[cal],q,method='higher'),np.inf))
            row={'id':kind,'quantile':q,'threshold':th,'eligible':False}
            if th>1:row['rejected_reason']='Unresolvable calibration'
            else:
                alert=union_decisions(base[dm],score[dm],bth,th) if kind=='union-baseline' else combined[dm]>=th
                m=decision_metrics(dev,alert,events)
                lost=[a['event_id'] for a,b in zip(before,detections(alert)) if a['detected'] and not b['detected']]
                eligible=(not lost and m['false_positive_windows']<=baseline['false_positive_windows'] and m['precision']>=.5
                          and m['by_family']['cooling_degradation']['detected']>baseline['by_family']['cooling_degradation']['detected'])
                row.update(metrics=m,lost_baseline_events=lost,eligible=bool(eligible))
            rows.append(row);write_json(out/'progress.json',{'phase':'development','completed':len(rows),'total':9,'rows':rows})
            print(json.dumps(row),flush=True)
    eligible=[r for r in rows if r['eligible']]
    selected=max(eligible,key=lambda r:(r['metrics']['by_family']['cooling_degradation']['detected'],r['metrics']['physical_events']['detected'],r['metrics']['detected_events'],-r['metrics']['false_positive_windows'])) if eligible else None
    write_json(out/'results.json',{'baseline':baseline,'rows':rows,'selected':selected,'models':models,'thermal_events':diagnostic,
        'development_windows':len(dev),'available_development_windows':int(available[dm].sum()),'test_used':False,
        'truth_hashes':truth_hashes,'baseline_files':hashes(args.baseline),'cache_manifest':json.loads((args.cache/'cache_manifest.json').read_text())})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','dataset','cache','raw-root','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
