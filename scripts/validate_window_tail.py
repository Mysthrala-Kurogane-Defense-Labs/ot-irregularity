"""Freeze and evaluate a batch-window closure policy without changing model scores."""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_otlab import prepare
from scripts.research_detection import write_json
from scripts.research_physical import summarize
from scripts.validate_contextual import ui_state
from scripts.validate_physical import hashes
from ot_irregularity.data.load import load_dataset
from ot_irregularity.pipeline import _exclude_feature_inputs, _model_config, _scores, _windows


def freeze(args):
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'frozen.json',{'frozen_at':dt.datetime.now(dt.timezone.utc).isoformat(),
        'baseline_files':hashes(args.baseline),'holdout_seeds':args.seeds,
        'policy':'window.tail_policy: complete; window_end <= last observed timestamp of its run/asset',
        'weights_and_threshold':'unchanged',
        'acceptance':'No loss of detected events or physical detections, fewer false alert minutes. Report excluded coverage and events without windows. No physical-learning improvement claim.',
        'test_used_for_selection':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    ui_state(args.output,'control',0.,[],'Window-tail policy frozen')


def evaluate(args):
    root=args.output;frozen=json.loads((root/'frozen.json').read_text())
    if (root/'results.json').exists():raise FileExistsError('Holdout already evaluated')
    if hashes(args.baseline)!=frozen['baseline_files']:raise ValueError('Model changed after freeze')
    cfg=_model_config(args.baseline);schema=json.loads((args.baseline/'feature_schema.json').read_text())
    parts=[];events=[];sources=[]
    for source in args.sources:
        dest=root/('prepared-'+source.name);manifest=prepare(source,dest)
        seed=manifest['source_master_seed']
        if seed not in frozen['holdout_seeds'] or seed in [s['seed'] for s in sources]:raise ValueError('Unexpected seed')
        raw=_exclude_feature_inputs(load_dataset(dest/'test'),cfg)
        if 'value_kind' in raw.columns:
            raw=raw.filter(pl.col('value_kind').is_null()|(pl.col('value_kind')=='continuous'))
        bounds=raw.group_by(['run_id','asset_id']).agg(pl.col('timestamp').max().dt.timestamp('us').alias('watermark'),
            pl.col('timestamp').min().dt.timestamp('us').alias('first_observation'))
        w=_windows(dest/'test',cfg,schema).join(bounds,on=['run_id','asset_id'])
        keep=(w['window_end']<=w['watermark']).to_numpy()
        # Verify the implemented generator reproduces the analytical policy exactly.
        complete_cfg=copy.deepcopy(cfg);complete_cfg['window']['tail_policy']='complete'
        generated=_windows(dest/'test',complete_cfg,schema)
        expected=w.filter(pl.Series(keep)).drop('watermark','first_observation')
        comparison=[c for c in generated.columns if c not in ('event_id','event_start_us')]
        if not generated.select(comparison).sort(['run_id','asset_id','window_start']).equals(expected.select(comparison).sort(['run_id','asset_id','window_start'])):
            raise ValueError('Complete generator and filtered reference differ')
        _,_,_,_,scores,th=_scores(args.baseline,w)
        # Bounds are evaluation-only: never include them in model feature arrays.
        w=w.with_columns(pl.Series('score',scores),pl.Series('retained',keep),
            (pl.lit(source.name+'::')+pl.col('run_id')).alias('run_id'),
            pl.when(pl.col('event_id').is_not_null()).then(pl.lit(source.name+'::')+pl.col('event_id')).otherwise(None).alias('event_id'))
        parts.append(w.select(['run_id','asset_id','asset_class','window_start','window_end','is_anomaly','event_id',
            'event_start_us','score','retained','watermark','first_observation']))
        truth_hashes={}
        for truth in sorted((source/'test').glob('*/ground_truth.json')):
            doc=json.loads(truth.read_text());truth_hashes[truth.relative_to(source).as_posix()]=hashlib.sha256(truth.read_bytes()).hexdigest()
            for e in doc['events']:
                def micros(s):return int(dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()*1e6)
                run=source.name+'::'+doc['run_id']
                events.append({'run_id':run,'asset_id':e['asset_id'],'event_id':run+'/'+e['event_id'],
                    'family':e['type'],'start_us':micros(e['start']),'end_us':micros(e['end'])})
        sources.append({'seed':seed,'manifest_sha256':hashlib.sha256((source/'dataset_manifest.json').read_bytes()).hexdigest(),
            'truth_hashes':truth_hashes,'license':manifest['source_license'],'attribution':manifest['source_attribution'],
            'source_url':manifest['source_url'],'simulator_version':manifest['source_simulator_version'],'test_runs':manifest['partitions']['test']['runs']})
    if set(s['seed'] for s in sources)!=set(frozen['holdout_seeds']):raise ValueError('Incomplete holdout')
    w=pl.concat(parts);closed=w.filter(pl.col('retained'))
    baseline=summarize(w,w['score'].to_numpy(),th,events)
    candidate=summarize(closed,closed['score'].to_numpy(),th,events)
    exposure=w.unique(['run_id','asset_id']).select(((pl.col('watermark')-pl.col('first_observation'))/3_600_000_000).sum()).item()
    for metrics in (baseline,candidate):
        metrics['false_alert_minutes_per_observed_asset_day']=metrics['false_positive_windows']/(exposure/24)
    # Paired run bootstrap for false alert counts; zero padding here means no emitted
    # alert on incomplete tails, not an imputed normal score or measured true negative.
    counts=w.group_by('run_id').agg(
        ((pl.col('score')>=th)&~pl.col('is_anomaly')).sum().alias('before'),
        ((pl.col('score')>=th)&~pl.col('is_anomaly')&pl.col('retained')).sum().alias('after'))
    delta=(counts['after'].cast(pl.Int64)-counts['before'].cast(pl.Int64)).to_numpy();rng=np.random.default_rng(20261005)
    draws=[float(delta[rng.integers(0,len(delta),len(delta))].sum()) for _ in range(2000)]
    rows=[{'id':'v0.4-legacy','metrics':baseline},{'id':'v0.4-complete-tail','metrics':candidate}]
    passed=(candidate['detected_events']>=baseline['detected_events'] and
        candidate['physical_events']['detected']>=baseline['physical_events']['detected'] and
        candidate['false_positive_windows']<baseline['false_positive_windows'])
    result={'rows':rows,'sources':sources,'event_count':len(events),'threshold':th,
        'coverage':{'original_minutes':len(w),'retained_minutes':len(closed),'excluded_minutes':len(w)-len(closed),
            'observed_asset_hours':exposure,'retained_window_asset_hours':len(closed)/60,
            'excluded_positive_minutes':int(w.filter(~pl.col('retained'))['is_anomaly'].sum())},
        'paired_false_alert_count_delta_ci95':np.quantile(draws,[.025,.975]).tolist(),'bootstrap_replicates':2000,
        'gate_passed':bool(passed),'selection_after_test':False,
        'limit':'Batch endpoint correction with reduced emitted coverage; no improvement in learned physical sensitivity. PR-AUC populations differ.'}
    write_json(root/'results.json',result);write_json(root/'events.json',events)
    w.write_parquet(root/'predictions.parquet');ui_state(root,'completed',1.,rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','evaluate'])
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[910521,910522,910523]);p.add_argument('--sources',type=Path,nargs='+')
    a=p.parse_args()
    if a.command=='evaluate' and not a.sources:p.error('evaluate requires sources')
    (freeze if a.command=='freeze' else evaluate)(a)
