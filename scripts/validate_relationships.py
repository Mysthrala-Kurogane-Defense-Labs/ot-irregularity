"""Confirm the selected static residual AE across seeds, freeze, then test once."""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import polars as pl
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_otlab import prepare
from scripts.relationship_operating_points import residual_scores
from scripts.research_detection import cache_windows,write_json
from scripts.research_physical import PHYSICAL,load_events,summarize
from scripts.research_relationships import closed
from scripts.validate_contextual import bootstrap,ui_state
from scripts.validate_physical import hashes
from ot_irregularity.contextual import run_bucket,tail_errors
from ot_irregularity.metrics import evaluate_event_intervals
from ot_irregularity.models import train_ae,ae_errors
from ot_irregularity.pipeline import _scores,_windows,_model_config


def fit(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    protocol={'candidate':'max-static-ae','quantile':.9975,'primary_seed':20261005,'seeds':[20261005,20261006,20261007],
        'selected_before_test':True,'selection_reason':'Among eligible development points, static AE and static direct tie at 15 physical events; AE preserves more total events (71 versus 68), within baseline false-minute count.',
        'policy':'Same complete-tail windows for all; primary chosen before test, secondary seeds report variability; no reselection after test.'}
    write_json(out/'selection.json',protocol)
    operations=json.loads((args.research/'operating_points.json').read_text())
    selected=next(r for r in operations['rows'] if r['id']==protocol['candidate'] and r['quantile']==protocol['quantile'])
    if not selected['eligible']:raise ValueError('Selected point failed development gate')
    cfg=yaml.safe_load((args.baseline/'training_config.yaml').read_text())
    schema=json.loads((args.baseline/'feature_schema.json').read_text())
    cached=cache_windows(args.dataset,schema,cfg,args.cache)
    frames={p:closed(cached[p],pl.read_parquet(args.dataset/p/'telemetry.parquet')) for p in ('train','validation')}
    tr,va=frames['train'],frames['validation']
    fit_frame=tr.filter(pl.Series([run_bucket(r,5)!=0 for r in tr['run_id']]))
    stop=tr.filter(pl.Series([run_bucket(r,5)==0 for r in tr['run_id']]))
    cal=np.array([run_bucket(r,2)==0 for r in va['run_id']])&~va['is_anomaly'].to_numpy()
    devmask=np.array([run_bucket(r,2)==1 for r in va['run_id']]);dev=va.filter(pl.Series(devmask))
    events,_=load_events(args.raw_root,dev);_,_,_,_,base,base_th=_scores(args.baseline,va)
    rows=[];models=[]
    for seed in protocol['seeds']:
        root=out/f'seed-{seed}';shutil.copytree(args.research/'static',root)
        ui_state(out,'finalists',(seed-protocol['primary_seed'])/6,rows,str(seed))
        if seed!=protocol['primary_seed']:
            for group in sorted(fit_frame['asset_class'].unique()):
                folder=root/group;m=joblib.load(folder/'detector.joblib')
                relation=m['relationship']
                rf,vf,_=relation.transform(fit_frame.filter(pl.col('asset_class')==group))
                re,ve,_=relation.transform(stop.filter(pl.col('asset_class')==group))
                v=va.filter(pl.col('asset_class')==group);rv,vv,_=relation.transform(v)
                xf,xe,xv=[m['scaler'].transform(r) for r in (rf[vf],re[ve],rv)]
                ae=train_ae(xf,xe,{**cfg['autoencoder'],'latent_dim':2,'device':args.device},seed,folder/'autoencoder.pt')
                m['ae_scale']=np.maximum(np.quantile(ae_errors(ae,xe),.95,axis=0),1e-8)
                raw,_=tail_errors(ae_errors(ae,xv),m['ae_scale'],2)
                c=np.array([run_bucket(r,2)==0 for r in v['run_id']])&~v['is_anomaly'].to_numpy()&vv
                m['references']['ae']=raw[c];joblib.dump(m,folder/'detector.joblib')
        supplementary=residual_scores(root,va,args.device)['ae'];score=np.maximum(base,supplementary)
        threshold=float(np.nextafter(np.quantile(score[cal],protocol['quantile'],method='higher'),np.inf))
        if threshold>1:raise ValueError('Unresolvable calibration')
        row={'id':f'candidate-{seed}','metrics':summarize(dev,score[devmask],threshold,events)};rows.append(row)
        models.append({'directory':root.name,'seed':seed,'threshold':threshold,'files':hashes(root)})
        print(json.dumps(row),flush=True)
    # Controls distinguish the effect of the residual AE from threshold-only changes and direct regression error.
    direct=np.maximum(base,residual_scores(out/'seed-20261005',va,args.device)['direct'])
    control_threshold=float(np.nextafter(np.quantile(direct[cal],protocol['quantile'],method='higher'),np.inf))
    tight=float(np.nextafter(np.quantile(base[cal],protocol['quantile'],method='higher'),np.inf))
    write_json(out/'frozen.json',{**protocol,'frozen_at':dt.datetime.now(dt.timezone.utc).isoformat(),
        'baseline_files':hashes(args.baseline),'models':models,'development':rows,
        'baseline_threshold':base_th,'tight_baseline_threshold':tight,'direct_threshold':control_threshold,
        'holdout_seeds':[910531,910532,910533],'test_used_for_selection':False})
    ui_state(out,'control',.5,rows,'Frozen; awaiting new holdouts')


def evaluate(args):
    out=args.output;frozen=json.loads((out/'frozen.json').read_text())
    if (out/'results.json').exists():raise FileExistsError('Holdout already evaluated')
    if hashes(args.baseline)!=frozen['baseline_files']:raise ValueError('Baseline changed')
    for m in frozen['models']:
        if hashes(out/m['directory'])!=m['files']:raise ValueError('Candidate changed after freeze')
    cfg=_model_config(args.baseline);cfg['window']['tail_policy']='complete'
    schema=json.loads((args.baseline/'feature_schema.json').read_text());parts=[];events=[];sources=[]
    for source in args.sources:
        dest=out/('prepared-'+source.name);manifest=prepare(source,dest)
        seed=manifest['source_master_seed']
        if seed not in frozen['holdout_seeds'] or seed in [s['seed'] for s in sources]:raise ValueError('Unexpected or duplicate seed')
        w=_windows(dest/'test',cfg,schema)
        w=w.with_columns((pl.lit(source.name+'::')+pl.col('run_id')).alias('run_id'),
            pl.when(pl.col('event_id').is_not_null()).then(pl.lit(source.name+'::')+pl.col('event_id')).otherwise(None).alias('event_id'))
        parts.append(w);truth_hashes={}
        for truth in sorted((source/'test').glob('*/ground_truth.json')):
            doc=json.loads(truth.read_text());truth_hashes[truth.relative_to(source).as_posix()]=hashlib.sha256(truth.read_bytes()).hexdigest()
            for e in doc['events']:
                def micros(s):return int(dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()*1e6)
                run=source.name+'::'+doc['run_id']
                events.append({'run_id':run,'asset_id':e['asset_id'],'event_id':run+'/'+e['event_id'],'family':e['type'],'start_us':micros(e['start']),'end_us':micros(e['end'])})
        sources.append({'seed':seed,'manifest_sha256':hashlib.sha256((source/'dataset_manifest.json').read_bytes()).hexdigest(),
            'truth_hashes':truth_hashes,'license':manifest['source_license'],'attribution':manifest['source_attribution'],
            'source_url':manifest['source_url'],'simulator_version':manifest['source_simulator_version'],'test_runs':manifest['partitions']['test']['runs']})
    if set(s['seed'] for s in sources)!=set(frozen['holdout_seeds']):raise ValueError('Incomplete holdout')
    w=pl.concat(parts);_,_,_,_,base,threshold=_scores(args.baseline,w)
    scores={'v0.4-complete':base};thresholds={'v0.4-complete':threshold}
    for model in frozen['models']:
        values=residual_scores(out/model['directory'],w,args.device)
        name=f"candidate-{model['seed']}";scores[name]=np.maximum(base,values['ae']);thresholds[name]=model['threshold']
        if model['seed']==frozen['primary_seed']:
            direct=np.maximum(base,values['direct'])
    scores['v0.4-tight']=base;thresholds['v0.4-tight']=frozen['tight_baseline_threshold']
    scores['direct-control']=direct;thresholds['direct-control']=frozen['direct_threshold']
    rows=[{'id':name,'metrics':summarize(w,score,thresholds[name],events)} for name,score in scores.items()]
    names=list(scores);event_results=[]
    for name in names[:2]:
        event_results.append(evaluate_event_intervals(events,scores[name],thresholds[name],w['run_id'].to_numpy(),w['asset_id'].to_numpy(),w['window_start'].to_numpy(),w['window_end'].to_numpy())['events'])
    family={e['event_id']:e['family'] for e in events};runs=sorted(set(w['run_id'])|{e['run_id'] for e in events})
    counts=[np.array([[sum(e['run_id']==r for e in table),sum(e['run_id']==r and e['detected'] for e in table),
        sum(e['run_id']==r and family[e['event_id']] in PHYSICAL for e in table),sum(e['run_id']==r and family[e['event_id']] in PHYSICAL and e['detected'] for e in table)] for r in runs]) for table in event_results]
    rng=np.random.default_rng(20261005);delta={'all':[],'physical':[]}
    for _ in range(2000):
        picks=rng.integers(0,len(runs),len(runs));a,b=[c[picks].sum(axis=0) for c in counts]
        for key,n,d in [('all',1,0),('physical',3,2)]:
            if a[d] and b[d]:delta[key].append(b[n]/b[d]-a[n]/a[d])
    ci={k:np.quantile(v,[.025,.975]).tolist() for k,v in delta.items()}
    window_ci=bootstrap(w,[scores[n] for n in names[:2]],[thresholds[n] for n in names[:2]])
    window_ci.pop('event_detection_rate',None)
    write_json(out/'results.json',{'rows':rows,'sources':sources,'windows':len(w),'event_count':len(events),'tail_policy':'complete',
        'primary_seed':frozen['primary_seed'],'paired_event_delta_ci95':ci,'paired_window_delta_ci95':window_ci,'selection_after_test':False})
    pred=w.select(['run_id','asset_id','asset_class','window_start','window_end','is_anomaly','event_id','event_start_us'])
    pred.with_columns([pl.Series(k,v) for k,v in scores.items()]).write_parquet(out/'predictions.parquet')
    write_json(out/'events.json',events);ui_state(out,'completed',1.,rows);print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['fit','evaluate'])
    for key in ('baseline','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('research','cache','dataset','raw-root'):p.add_argument('--'+key,type=Path)
    p.add_argument('--sources',type=Path,nargs='+');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.command=='fit' and any(getattr(a,k) is None for k in ('research','cache','dataset','raw_root')):p.error('fit requires research, cache, dataset, raw-root')
    if a.command=='evaluate' and not a.sources:p.error('evaluate requires sources')
    (fit if a.command=='fit' else evaluate)(a)
