"""Explicit adaptive development stage: preserve thermal excess beyond empirical CDF saturation."""
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
from ot_irregularity.pipeline import _scores
from ot_irregularity.sample_thermal import aggregate_residuals,bounded_tail_score


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    protocol={'stage':'Adaptive development after sample-first CDF screen; no test selection',
        'transform':'magnitude / (magnitude + normal calibration q99 magnitude); floor 1e-9 degC/s',
        'interpretation':'bounded relative magnitude, NOT a normal percentile or fault probability',
        'initial_context':[False,True],'scores':['abs_time_weighted_mean_rate','q95_abs_rate'],
        'quantiles':[.99,.995,.9975,.999],'new_fit':False,
        'gate':'Unchanged: preserve every baseline alert, more cooling, false windows <= baseline, precision >= .5',
        'selection':'Most cooling, physical, total events, fewer false windows, then prefer no initial context',
        'test_used':False}
    write_json(out/'protocol.json',protocol)
    source=json.loads((args.research/'results.json').read_text())
    if 'r2' not in source['protocol'].get('revision',''):raise ValueError('Requires corrected per-run cadence experiment')
    if hashes(args.baseline)!=source['baseline_files']:raise ValueError('Baseline changed')
    source_hashes=hashes(args.research)
    cfg=yaml.safe_load((args.baseline/'training_config.yaml').read_text());schema=json.loads((args.baseline/'feature_schema.json').read_text())
    cached=cache_windows(args.dataset,schema,cfg,args.cache);raw=pl.read_parquet(args.dataset/'validation/telemetry.parquet')
    va=closed(cached['validation'],raw)
    cal=np.array([run_bucket(r,2)==0 for r in va['run_id']])&~va['is_anomaly'].to_numpy()
    dm=np.array([run_bucket(r,2)==1 for r in va['run_id']]);dev=va.filter(pl.Series(dm));events,truth_hashes=load_events(args.raw_root,dev)
    _,_,_,_,base,bth=_scores(args.baseline,va);baseline=decision_metrics(dev,base[dm]>=bth,events)
    rows=[];anchors={};predictions=va.select(['run_id','asset_id','window_start','window_end'])
    for initial in (False,True):
        scores=np.zeros((len(va),2));available=np.zeros(len(va),dtype=bool)
        for group,roles in sorted(schema['signals_by_asset_class'].items()):
            mask=va['asset_class'].to_numpy()==group;windows=va.filter(pl.Series(mask));gv=raw.filter(pl.col('asset_class')==group)
            local=np.zeros((len(windows),2));valid_any=np.zeros(len(windows),dtype=bool)
            for target in (r for r in roles if 'temperature' in r):
                key=f'initial-{initial}-{group}-{target}';bundle=joblib.load(args.research/(key+'.joblib'))
                values,valid=aggregate_residuals(bundle['dynamics'].transform(gv),windows)
                anchor=[max(float(np.quantile(ref,.99)),1e-9) for ref in bundle['references']];anchors[key]=anchor
                for j in range(2):
                    s=bounded_tail_score(values[:,j],anchor[j]);s[~valid]=0
                    local[:,j]=np.maximum(local[:,j],s)
                valid_any|=valid
            scores[mask]=local;available[mask]=valid_any
        for j,name in enumerate(protocol['scores']):
            predictions=predictions.with_columns(pl.Series(f'{initial}-{name}',scores[:,j]))
            for q in protocol['quantiles']:
                th=float(np.nextafter(np.quantile(scores[cal,j],q,method='higher'),np.inf))
                row={'initial_context':initial,'score':name,'quantile':q,'threshold':th,'eligible':False,
                     'available_development_windows':int(available[dm].sum())}
                if th>1:row['rejected_reason']='Unresolvable calibration'
                else:
                    alert=union_decisions(base[dm],scores[dm,j],bth,th);m=decision_metrics(dev,alert,events)
                    row.update(metrics=m,eligible=bool(m['by_family']['cooling_degradation']['detected']>baseline['by_family']['cooling_degradation']['detected']
                        and m['false_positive_windows']<=baseline['false_positive_windows'] and m['precision']>=.5),
                        lost_baseline_alert_windows=int(np.sum((base[dm]>=bth)&~alert)))
                rows.append(row);write_json(out/'progress.json',{'phase':'development','completed':len(rows),'total':16,'rows':rows})
                print(json.dumps(row),flush=True)
    eligible=[r for r in rows if r['eligible']]
    selected=max(eligible,key=lambda r:(r['metrics']['by_family']['cooling_degradation']['detected'],r['metrics']['physical_events']['detected'],
        r['metrics']['detected_events'],-r['metrics']['false_positive_windows'],-int(r['initial_context']))) if eligible else None
    predictions.write_parquet(out/'validation_scores.parquet')
    write_json(out/'results.json',{'protocol':protocol,'baseline':baseline,'rows':rows,'selected':selected,'anchors':anchors,
        'development_windows':len(dev),'events':len(events),'truth_hashes':truth_hashes,'source_hashes':source_hashes,
        'baseline_files':source['baseline_files'],'test_used':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','dataset','cache','raw-root','research','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
