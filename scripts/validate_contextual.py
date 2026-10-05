"""Freeze contextual candidates, then explicitly evaluate newly generated holdouts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_otlab import prepare
from scripts.research_detection import write_json, metrics, cache_windows
from ot_irregularity.contextual import train_contextual_windows
from ot_irregularity.features import encode_context
from ot_irregularity.pipeline import _hash_paths, _scores, _windows, _git_commit, _git_working_tree_dirty


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ui_state(out,phase,progress,rows,active=None):
    ranking=[]
    for row in rows:
        m=row['metrics']
        ranking.append({'id':row['id'],'label':row['id'],'repeats_completed':1,
                        **m,'latency_seconds':m.get('mean_detection_latency_seconds')})
    ranking.sort(key=lambda row:row['pr_auc'] or 0,reverse=True)
    state={'run_id':out.name,'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),
           'updated_at':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'completed' if phase=='completed' else 'running',
           'phase':phase,'progress':progress,'config':{'device':'cuda:0','dataset_repeats':3},
           'datasets':[{'id':i} for i in range(3)],'planned':{'screening_runs':3,'control_runs':1},
           'runs':[dict(r,status='completed') for r in rows], 'ranking':ranking,
           'active_workers':[{'id':active,'status':phase,'progress':0}] if active else []}
    write_json(out/'state.json',state)


def fit(args):
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    cfg=yaml.safe_load(args.config.read_text())
    base=json.loads((args.baseline/'feature_schema.json').read_text())
    cache=args.research/'cache'
    # Recheck source bytes before reusing a window cache of this exact baseline schema.
    datahash=_hash_paths([args.dataset/'train',args.dataset/'validation'])
    baseline_meta=json.loads((args.baseline/'training_metadata.json').read_text())
    if datahash!=baseline_meta['dataset_hash']:
        raise ValueError('Source dataset changed since the baseline/cache was produced')
    baseline_cfg=yaml.safe_load((args.baseline/'training_config.yaml').read_text())
    if any(cfg.get(k,{})!=baseline_cfg.get(k,{}) for k in ('window','features','normalization')):
        raise ValueError('Cached finalist validation requires the same window/features/normalization as its baseline')
    frames=cache_windows(args.dataset,base,baseline_cfg,cache)
    tr,vocab=encode_context(frames['train'])
    va,_=encode_context(frames['validation'],vocab)
    rows=[];artifacts=[]
    for i,seed in enumerate(args.seeds):
        model=out/f'model-{seed}'
        model.mkdir()
        cfg['seed']=seed
        ui_state(out,'finalists',i/len(args.seeds)*.5,rows,f'CUDA seed {seed}')
        metadata={'model_version':cfg['model_version'],'training_date':dt.datetime.now(dt.timezone.utc).isoformat(),
                  'git_commit':_git_commit(),'git_working_tree_dirty':_git_working_tree_dirty(),'dataset_hash':datahash,
                  'dataset_schema_version':'1','stride_seconds':60}
        schema={k:v for k,v in base.items() if k not in ('features','version')}
        schema['context_vocabulary']=vocab
        result=train_contextual_windows(tr,va,cfg,model,schema,metadata)
        rows.append({'id':f'development-seed-{seed}','metrics':result['supervised_validation']['irregularity']})
        artifacts.append({'seed':seed,'directory':model.name,'manifest_sha256':digest(model/'contextual_models.json'),
                          'files':{p.relative_to(model).as_posix():digest(p) for p in model.rglob('*') if p.is_file()}})
        print(json.dumps(rows[-1]),flush=True)
    write_json(out/'frozen.json',{'frozen_at':dt.datetime.now(dt.timezone.utc).isoformat(),'primary_seed':args.seeds[0],
        'config_sha256':digest(args.config),'dataset_hash':datahash,'models':artifacts,'development':rows,
        'holdout_used_for_selection':False,'baseline_manifest_sha256':digest(args.baseline/'feature_schema.json'),
        'acceptance_target':{'event_detection_rate_min':.5,'precision_min':.5,'false_positive_windows_per_asset_day_max':10}})
    ui_state(out,'control',.5,rows,'Awaiting explicit holdout evaluation')


def bootstrap(frame,scores,thresholds,repeats=500):
    """Paired cluster bootstrap over whole source runs; duplicate events get unique IDs."""
    rng=np.random.default_rng(20261005)
    ids=frame['run_id'].to_numpy()
    clusters=[np.flatnonzero(ids==r) for r in sorted(set(ids))]
    deltas={name:[] for name in ('pr_auc','precision','event_detection_rate','false_positives_per_asset_day')}
    for _ in range(repeats):
        picks=rng.integers(0,len(clusters),len(clusters))
        indices=np.concatenate([clusters[p] for p in picks])
        parts=[]
        for occurrence,pick in enumerate(picks):
            sub=frame[clusters[pick]]
            sub=sub.with_columns(pl.when(pl.col('event_id').is_not_null()).then(pl.lit(str(occurrence)+'::')+pl.col('event_id')).otherwise(None).alias('event_id'))
            parts.append(sub)
        sampled=pl.concat(parts)
        a=metrics(sampled,scores[0][indices],thresholds[0]);b=metrics(sampled,scores[1][indices],thresholds[1])
        for key in deltas:
            if a.get(key) is not None and b.get(key) is not None:deltas[key].append(b[key]-a[key])
    return {key:{'lower':float(np.quantile(v,.025)),'upper':float(np.quantile(v,.975)),'replicates':len(v)} for key,v in deltas.items()}


def evaluate(args):
    out=args.output
    frozen=json.loads((out/'frozen.json').read_text())
    if (out/'holdout_results.json').exists():
        raise FileExistsError('Holdout already evaluated; inspect the recorded result rather than selecting again')
    for model in frozen['models']:
        for name,sha in model['files'].items():
            if digest(out/model['directory']/name)!=sha:raise ValueError('Candidate changed after freeze')
    baseline_schema=json.loads((args.baseline/'feature_schema.json').read_text())
    baseline_cfg=yaml.safe_load((args.baseline/'training_config.yaml').read_text())
    parts=[];sources=[]
    for source in args.holdouts:
        prepared=out/('prepared-'+source.name)
        manifest=prepare(source,prepared,max_train_runs=None,max_validation_runs=None,max_test_runs=None)
        w=_windows(prepared/'test',baseline_cfg,baseline_schema)
        w=w.with_columns((pl.lit(source.name+'::')+pl.col('run_id')).alias('run_id'),
                         pl.when(pl.col('event_id').is_not_null()).then(pl.lit(source.name+'::')+pl.col('event_id')).otherwise(None).alias('event_id'))
        parts.append(w)
        sources.append({'batch':source.name,'manifest_sha256':digest(source/'dataset_manifest.json'),
                        'prepared_manifest_sha256':digest(prepared/'dataset_manifest.json'),
                        'license':manifest['source_license'],'attribution':manifest['source_attribution'],
                        'source_url':manifest['source_url'],'source_simulator_version':manifest['source_simulator_version'],
                        'source_suite_version':manifest['source_suite_version'],'seed':manifest['source_master_seed'],
                        'ground_truth_events':manifest['partitions']['test']['events'],'test_runs':manifest['partitions']['test']['runs']})
    w=pl.concat(parts)
    w,_=encode_context(w,baseline_schema['context_vocabulary'])
    rows=[];arrays=[];thresholds=[]
    predictions=w.select(['run_id','asset_id','asset_class','window_start','window_end','is_anomaly','event_id','event_start_us'])
    models=[('baseline-v0.3',args.baseline)]+[(f"candidate-{m['seed']}",out/m['directory']) for m in frozen['models']]
    for i,(name,model) in enumerate(models):
        ui_state(out,'control',.5+i/(len(models)+1)*.4,rows,name)
        _,_,sa,si,se,th=_scores(model,w)
        arrays.append(se);thresholds.append(th)
        row={'id':name,'metrics':metrics(w,se,th),
             'autoencoder':metrics(w,sa,th),'isolation_forest':metrics(w,si,th)}
        rows.append(row)
        predictions=predictions.with_columns(pl.Series(name,se))
        print(json.dumps(row),flush=True)
    predictions.write_parquet(out/'holdout_predictions.parquet')
    # Calibration-budget baseline shows whether improvement is only a threshold change.
    # Frozen baseline's published threshold remains the main comparison; matched budget
    # must be learned from historical development calibration, never this holdout.
    stratified={}
    for group in sorted(w['asset_class'].unique()):
        mask=w['asset_class'].to_numpy()==group
        stratified[group]={name:metrics(w.filter(pl.Series(mask)),s[mask],t) for (name,_),s,t in zip(models,arrays,thresholds)}
    write_json(out/'holdout_results.json',{'rows':rows,'sources':sources,'by_asset_class':stratified,
        'test_windows':len(w),'ground_truth_events':sum(s['ground_truth_events'] for s in sources),
        'primary_seed':frozen['primary_seed'],'paired_cluster_bootstrap':bootstrap(w,arrays[:2],thresholds[:2]),
        'selection_after_test':False,'limit':'New seeds of the same suite; not real industrial validation'})
    ui_state(out,'completed',1.,rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['fit','evaluate-holdout'])
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--research',type=Path)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--config',type=Path,default=Path('configs/model-v0.4-contextual.yaml'))
    p.add_argument('--seeds',type=int,nargs='+',default=[20261005,20261006,20261007])
    p.add_argument('--holdouts',type=Path,nargs='+')
    args=p.parse_args()
    if args.command=='fit' and (args.dataset is None or args.research is None):
        p.error('fit requires --dataset and --research')
    if args.command=='evaluate-holdout' and not args.holdouts:
        p.error('evaluate-holdout requires --holdouts')
    if args.command=='fit':fit(args)
    else:evaluate(args)
