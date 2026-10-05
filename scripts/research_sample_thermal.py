"""Development-only sample-first thermal screen with explicit initial-context ablation."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import polars as pl
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_alert_budgets import decision_metrics,union_decisions
from scripts.research_detection import cache_windows,write_json
from scripts.research_physical import load_events
from scripts.research_relationships import closed
from scripts.validate_physical import hashes
from ot_irregularity.contextual import run_bucket
from ot_irregularity.pipeline import _scores,cdf_calibrate
from ot_irregularity.sample_thermal import SampleThermal,aggregate_residuals


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    quantiles=[.99,.995,.9975]
    protocol={'stage':'development only, sample-first dynamics','alpha':1.,'initial_context':[False,True],
        'scores':['abs_time_weighted_mean_rate','q95_abs_rate'],'quantiles':quantiles,
        'alignment':'Exact timestamps only; no interpolation or future values; gap <=1.5*training schema nominal interval',
        'quality':'Good finite required roles in both sample endpoints, >=90% expected pairs per complete minute and >=20 pairs',
        'initial_context_limit':'First fully usable target value per run/asset, retained causally. Full run batch only; not an online chunk API or a measured ambient temperature.',
        'combination':'Union with unchanged baseline threshold; no baseline alert can be lost',
        'gate':'More cooling events, false windows <= baseline, precision >= .5',
        'selection':'Most cooling, physical, total events; then fewer false windows and preference for no initial context',
        'test_used':False}
    write_json(out/'protocol.json',protocol)
    cfg=yaml.safe_load((args.baseline/'training_config.yaml').read_text());schema=json.loads((args.baseline/'feature_schema.json').read_text())
    if cfg['window']!={'size':'1m','stride':'1m'}:raise ValueError('This screen requires complete nonoverlapping minute windows')
    cached=cache_windows(args.dataset,schema,cfg,args.cache)
    raws={p:pl.read_parquet(args.dataset/p/'telemetry.parquet') for p in ('train','validation')}
    tr,va=[closed(cached[p],raws[p]) for p in ('train','validation')]
    if set(tr['run_id'])&set(va['run_id']):raise ValueError('Run leakage')
    fit_runs=sorted({r for r in tr['run_id'] if run_bucket(r,5)!=0})
    rawfit=raws['train'].filter(pl.col('run_id').is_in(fit_runs))
    cal=np.array([run_bucket(r,2)==0 for r in va['run_id']])&~va['is_anomaly'].to_numpy()
    dm=np.array([run_bucket(r,2)==1 for r in va['run_id']]);dev=va.filter(pl.Series(dm))
    events,truth_hashes=load_events(args.raw_root,dev)
    _,_,_,_,base,bth=_scores(args.baseline,va)
    baseline=decision_metrics(dev,base[dm]>=bth,events)
    rows=[];model_info={};predictions=va.select(['run_id','asset_id','window_start','window_end'])
    for initial in (False,True):
        scores=np.zeros((len(va),2));availability=np.zeros(len(va),dtype=bool)
        for group,roles in sorted(schema['signals_by_asset_class'].items()):
            targets=[r for r in roles if 'temperature' in r]
            drivers=[r for r in roles if r not in targets and 'vibration' not in r and 'photoeye' not in r]
            gf=rawfit.filter(pl.col('asset_class')==group);gv=raws['validation'].filter(pl.col('asset_class')==group)
            mask=va['asset_class'].to_numpy()==group;windows=va.filter(pl.Series(mask));local=np.zeros((len(windows),2));valid_any=np.zeros(len(windows),dtype=bool)
            for target in targets:
                interval=schema['sampling_intervals_ms'][target]/1000
                model=SampleThermal(target,drivers,initial,max_gap_seconds=1.5*interval).fit(gf)
                values,valid=aggregate_residuals(model.transform(gv),windows,interval)
                reference=values[cal[mask]&valid]
                if len(reference)<20:raise ValueError('Insufficient normal calibration')
                refs=[]
                for j in range(2):
                    s=cdf_calibrate(reference[:,j],values[:,j]);s[~valid]=0.
                    local[:,j]=np.maximum(local[:,j],s);refs.append(reference[:,j])
                valid_any|=valid
                key=f'initial-{initial}-{group}-{target}'
                joblib.dump({'dynamics':model,'references':refs,'sampling_interval_seconds':interval},out/(key+'.joblib'))
                model_info[key]={'fit_samples':model.fit_samples_,'calibration_windows':len(reference),
                    'development_available':int(valid[dm[mask]].sum()),'normal_quantiles':np.quantile(reference,[.5,.95,.99],axis=0).tolist()}
                write_json(out/'progress.json',{'phase':'fit_and_calibrate','model':key,'models_completed':len(model_info),'models_total':10})
                print('MODEL',key,flush=True)
            scores[mask]=local;availability[mask]=valid_any
        for j,name in enumerate(protocol['scores']):
            predictions=predictions.with_columns(pl.Series(f'{initial}-{name}',scores[:,j]))
            for q in quantiles:
                th=float(np.nextafter(np.quantile(scores[cal,j],q,method='higher'),np.inf))
                row={'initial_context':initial,'score':name,'quantile':q,'threshold':th,'eligible':False,
                     'available_development_windows':int(availability[dm].sum())}
                if th>1:row['rejected_reason']='Unresolvable calibration'
                else:
                    alert=union_decisions(base[dm],scores[dm,j],bth,th);m=decision_metrics(dev,alert,events)
                    row.update(metrics=m,eligible=bool(m['by_family']['cooling_degradation']['detected']>baseline['by_family']['cooling_degradation']['detected']
                        and m['false_positive_windows']<=baseline['false_positive_windows'] and m['precision']>=.5),
                        lost_baseline_alert_windows=int(np.sum((base[dm]>=bth)&~alert)))
                rows.append(row);print(json.dumps(row),flush=True)
    eligible=[r for r in rows if r['eligible']]
    selected=max(eligible,key=lambda r:(r['metrics']['by_family']['cooling_degradation']['detected'],r['metrics']['physical_events']['detected'],
        r['metrics']['detected_events'],-r['metrics']['false_positive_windows'],-int(r['initial_context']))) if eligible else None
    predictions.write_parquet(out/'validation_scores.parquet')
    write_json(out/'results.json',{'protocol':protocol,'baseline':baseline,'rows':rows,'selected':selected,'models':model_info,
        'development_windows':len(dev),'events':len(events),'truth_hashes':truth_hashes,'baseline_files':hashes(args.baseline),
        'cache_manifest':json.loads((args.cache/'cache_manifest.json').read_text()),'test_used':False})
    write_json(out/'progress.json',{'phase':'completed','rows':rows,'selected':selected})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','dataset','cache','raw-root','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
