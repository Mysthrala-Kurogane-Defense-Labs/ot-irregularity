"""Evaluate the frozen thermal component once on explicitly reserved fresh batches."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import polars as pl

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_otlab import prepare
from scripts.relationship_operating_points import residual_scores
from scripts.research_alert_budgets import decision_metrics,union_decisions
from scripts.research_detection import write_json
from scripts.research_physical import PHYSICAL,summarize
from scripts.validate_contextual import bootstrap
from scripts.validate_physical import hashes
from ot_irregularity.metrics import evaluate_event_intervals
from ot_irregularity.pipeline import _model_config,_scores,_windows
from ot_irregularity.sample_thermal import aggregate_residuals,bounded_tail_score


def run(args):
    out=args.output;frozen=json.loads((out/'frozen.json').read_text())
    if (out/'results.json').exists():raise FileExistsError('Holdout already evaluated; do not select again')
    for path,key in [(args.baseline,'baseline_files'),(args.sample,'sample_files'),(args.relational,'relational_control_files')]:
        if hashes(path)!=frozen[key]:raise ValueError('Frozen artifact mismatch: '+key)
    repo=Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git','-C',str(repo),'diff',frozen['git_commit'],'--','src'],text=True):
        raise ValueError('Inference implementation changed since freeze')
    cfg=_model_config(args.baseline);cfg['window']['tail_policy']='complete'
    schema=json.loads((args.baseline/'feature_schema.json').read_text());parts=[];raws=[];events=[];sources=[]
    for source in args.sources:
        seed=json.loads((source/'dataset_manifest.json').read_text())['master_seed']
        if seed not in frozen['holdout_seeds'] or seed in [s['seed'] for s in sources]:raise ValueError('Unexpected or repeated holdout seed')
        dest=out/('prepared-'+source.name)
        manifest=prepare(source,dest)
        raw=pl.read_parquet(dest/'test/telemetry.parquet');w=_windows(dest/'test',cfg,schema)
        def namespace(frame):
            return frame.with_columns((pl.lit(source.name+'::')+pl.col('run_id')).alias('run_id'),
                pl.when(pl.col('event_id').is_not_null()).then(pl.lit(source.name+'::')+pl.col('event_id')).otherwise(None).alias('event_id'))
        raws.append(namespace(raw));parts.append(namespace(w));truth_hashes={}
        for path in sorted((source/'test').glob('*/ground_truth.json')):
            doc=json.loads(path.read_text());run_id=source.name+'::'+doc['run_id']
            truth_hashes[path.relative_to(source).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
            for e in doc['events']:
                def micros(text):return int(dt.datetime.fromisoformat(text.replace('Z','+00:00')).timestamp()*1e6)
                events.append({'run_id':run_id,'asset_id':e['asset_id'],'event_id':run_id+'/'+e['event_id'],'family':e['type'],
                               'start_us':micros(e['start']),'end_us':micros(e['end'])})
        sources.append({'seed':seed,'manifest_sha256':hashlib.sha256((source/'dataset_manifest.json').read_bytes()).hexdigest(),
            'truth_hashes':truth_hashes,'license':manifest['source_license'],'attribution':manifest['source_attribution'],
            'simulator_version':manifest['source_simulator_version'],'test_runs':manifest['partitions']['test']['runs']})
        write_json(out/'progress.json',{'phase':'preparing','completed':len(parts),'total':3})
    if set(s['seed'] for s in sources)!=set(frozen['holdout_seeds']):raise ValueError('Incomplete holdout')
    w=pl.concat(parts);raw=pl.concat(raws);selected=frozen['selected'];initial=selected['initial_context']
    index=['abs_time_weighted_mean_rate','q95_abs_rate'].index(selected['score'])
    thermal=np.zeros(len(w));available=np.zeros(len(w),dtype=bool)
    for group,roles in sorted(schema['signals_by_asset_class'].items()):
        mask=w['asset_class'].to_numpy()==group;gw=w.filter(pl.Series(mask));gr=raw.filter(pl.col('asset_class')==group)
        local=np.zeros(len(gw));valid_any=np.zeros(len(gw),dtype=bool)
        for target in (r for r in roles if 'temperature' in r):
            key=f'initial-{initial}-{group}-{target}';m=joblib.load(args.sample/(key+'.joblib'))
            values,valid=aggregate_residuals(m['dynamics'].transform(gr),gw)
            score=bounded_tail_score(values[:,index],frozen['anchors'][key][index]);score[~valid]=0
            local=np.maximum(local,score);valid_any|=valid
        thermal[mask]=local;available[mask]=valid_any
    _,_,_,_,base,bth=_scores(args.baseline,w)
    if bth!=frozen['baseline_threshold']:raise ValueError('Baseline threshold changed')
    th=selected['threshold'];alert=union_decisions(base,thermal,bth,th)
    # Ranking is the maximum relative decision margin, not a calibrated probability.
    margin=np.maximum(base/bth,thermal/th);rank=margin/(1+margin)
    if not np.array_equal(rank>=.5,alert):raise ValueError('Ranking and frozen union decisions disagree')
    relational=np.maximum(base,residual_scores(args.relational/'relationships',w,args.device)['ae'])
    rth=json.loads((args.relational/'candidate.json').read_text())['threshold']
    specs=[('v0.4-complete',base,bth),('thermal-union',rank,.5),('v0.5-relational',relational,rth)]
    rows=[{'id':name,'metrics':summarize(w,score,threshold,events)} for name,score,threshold in specs]
    table=[]
    for _,score,threshold in specs[:2]:
        table.append(evaluate_event_intervals(events,score,threshold,w['run_id'].to_numpy(),w['asset_id'].to_numpy(),w['window_start'].to_numpy(),w['window_end'].to_numpy())['events'])
    lost=[a['event_id'] for a,b in zip(*table) if a['detected'] and not b['detected']]
    families={(e['run_id'],e['asset_id'],e['event_id']):e['family'] for e in events};runs=sorted(set(w['run_id'])|{e['run_id'] for e in events})
    counts=[]
    for entries in table:
        counts.append(np.array([[sum(e['run_id']==run and (kind=='all' or families[(e['run_id'],e['asset_id'],e['event_id'])]=='cooling_degradation') and (not detected or e['detected']) for e in entries)
            for kind in ('all','cooling') for detected in (False,True)] for run in runs]))
    rng=np.random.default_rng(20261005);deltas={'all':[],'cooling':[]}
    for _ in range(2000):
        picks=rng.integers(0,len(runs),len(runs));a,b=[v[picks].sum(axis=0) for v in counts]
        for kind,n,d in [('all',1,0),('cooling',3,2)]:
            if a[d] and b[d]:deltas[kind].append(b[n]/b[d]-a[n]/a[d])
    window_ci=bootstrap(w,[base,rank],[bth,.5]);window_ci.pop('event_detection_rate',None)
    bm,cm=rows[0]['metrics'],rows[1]['metrics']
    gate=(not lost and cm['by_family'].get('cooling_degradation',{}).get('detected',0)>bm['by_family'].get('cooling_degradation',{}).get('detected',0)
          and cm['precision']>=.5 and cm['event_detection_rate']>=.5 and cm['false_positives_per_asset_day']<=10)
    result={'rows':rows,'sources':sources,'windows':len(w),'events':len(events),'available_thermal_windows':int(available.sum()),
        'lost_baseline_events':lost,'primary_gate_passed':bool(gate),'paired_event_delta_ci95':{k:np.quantile(v,[.025,.975]).tolist() if v else None for k,v in deltas.items()},
        'paired_window_delta_ci95':window_ci,'ranking_definition':'max(baseline/baseline_threshold,thermal/thermal_threshold), mapped m/(1+m); decision threshold .5',
        'selection_after_test':False,'limit':'Thermal component; compare v0.5 relational families before any overall promotion; new seeds of the same simulator suite only'}
    write_json(out/'events.json',events);write_json(out/'results.json',result)
    w.select(['run_id','asset_id','window_start','window_end','is_anomaly']).with_columns(pl.Series('baseline',base),pl.Series('thermal',thermal),
        pl.Series('thermal_available',available),pl.Series('thermal_union_ranking',rank),pl.Series('relational',relational)).write_parquet(out/'predictions.parquet')
    write_json(out/'progress.json',{'phase':'completed','rows':rows,'primary_gate_passed':bool(gate)})
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','baseline','sample','relational'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sources',type=Path,nargs='+',required=True);p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
