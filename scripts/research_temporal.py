"""Compare subminute detectors on a shared minute grid using development only."""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import torch
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_detection import cache_windows, write_json
from scripts.research_physical import load_events, summarize
from scripts.validate_contextual import ui_state
from ot_irregularity.contextual import run_bucket, train_contextual_windows
from ot_irregularity.metrics import evaluate_event_intervals
from ot_irregularity.pipeline import _scores

KEYS=['run_id','asset_id','window_start']
MINUTE=60_000_000


def complete_minutes(windows, seconds):
    """Find complete subwindow tilings; never invent scores for missing telemetry."""
    w=windows.with_columns((pl.col('window_start')//MINUTE*MINUTE).alias('minute_start'))
    w=w.filter((pl.col('window_end')<=pl.col('minute_start')+MINUTE)&
               (pl.col('window_end')-pl.col('window_start')==seconds*1_000_000)&
               ((pl.col('window_start')-pl.col('minute_start'))%(seconds*1_000_000)==0))
    keys=w.group_by(['run_id','asset_id','minute_start']).agg(
        pl.col('window_start').n_unique().alias('n'),pl.len().alias('rows'))
    return keys.filter((pl.col('n')==60//seconds)&(pl.col('rows')==pl.col('n'))).select(
        'run_id','asset_id',pl.col('minute_start').alias('window_start'))


def minute_scores(windows, scores, reference):
    """Align max short-window score with each frozen reference minute."""
    grouped=windows.with_columns(pl.Series('score',scores),
        (pl.col('window_start')//MINUTE*MINUTE).alias('minute_start')).group_by(
        ['run_id','asset_id','minute_start']).agg(pl.col('score').max())
    grouped=grouped.rename({'minute_start':'window_start'})
    aligned=reference.select(KEYS).with_row_index('order').join(grouped,on=KEYS,how='left').sort('order')
    if aligned['score'].null_count(): raise ValueError('Missing subwindow scores in comparison grid')
    return aligned['score'].to_numpy()


def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    schema=json.loads((args.baseline/'feature_schema.json').read_text())
    basecfg=yaml.safe_load((args.baseline/'training_config.yaml').read_text())
    write_json(out/'protocol.json',{'created_at':dt.datetime.now(dt.timezone.utc).isoformat(),
        'durations_seconds':[30,15],'seed':20261005,'test_used':False,
        'hypothesis':'Short windows preserve local physical deviations diluted by a minute.',
        'metric_contract':'Max score per common complete minute for AP, precision, false alert minutes and interval detection; fine-window event localization reported separately.',
        'threshold':'q99 of max minute scores on normal calibration minutes only; strict nextafter cutoff',
        'selection':'More physical events, no loss of total events, precision >=0.5, no increase in false alert minutes. Require fine-window physical detections >= baseline physical detections as a localization guard. Rank eligible by physical then total events then fewer false minutes.',
        'baseline_manifest_sha256':hashlib.sha256((args.baseline/'contextual_models.json').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    # Provenance-checked baseline cache may be reused; short-window caches are separate.
    minute=cache_windows(args.dataset,schema,basecfg,args.minute_cache)
    frames={};configs={};common={p:minute[p] for p in minute}
    for seconds in (30,15):
        cfg=copy.deepcopy(basecfg);cfg['window']={'size':f'{seconds}s','stride':f'{seconds}s'}
        cfg['device']=args.device;cfg['seed']=20261005
        cfg['model_version']=f'0.4-temporal-{seconds}s-research'
        configs[seconds]=cfg
        ui_state(out,'features',0.,[],f'Generate {seconds}s windows')
        frames[seconds]=cache_windows(args.dataset,schema,cfg,out/f'cache-{seconds}s')
        for part in common:
            common[part]=common[part].join(complete_minutes(frames[seconds][part],seconds),on=KEYS,how='semi')
    va=common['validation']
    cal=va.filter(pl.Series([run_bucket(r,2)==0 for r in va['run_id']])).filter(~pl.col('is_anomaly'))
    dev=va.filter(pl.Series([run_bucket(r,2)==1 for r in va['run_id']]))
    events,truth_hashes=load_events(args.raw_root,dev)
    _,_,_,_,scores,threshold=_scores(args.baseline,dev)
    rows=[{'id':'v0.4','metrics':summarize(dev,scores,threshold,events)}]
    pred=dev.with_columns(pl.Series('v0.4',scores))
    for index,seconds in enumerate((30,15)):
        cfg=configs[seconds];tr,va=frames[seconds]['train'],frames[seconds]['validation']
        if tr['is_anomaly'].null_count() or tr['is_anomaly'].any(): raise ValueError('Train must be explicitly normal')
        if set(tr['run_id']) & set(va['run_id']): raise ValueError('Run leakage')
        model=out/f'model-{seconds}s';model.mkdir()
        ui_state(out,'finalists',.2+index*.3,rows,f'CUDA {seconds}s')
        manifest=json.loads((out/f'cache-{seconds}s'/'cache_manifest.json').read_text())
        train_contextual_windows(tr,va,cfg,model,copy.deepcopy(schema),
            {'model_version':cfg['model_version'],'dataset_hash':manifest['provenance']['dataset_hash'],
             'stride_seconds':seconds,'training_date':dt.datetime.now(dt.timezone.utc).isoformat()})
        _,_,_,_,s,_=_scores(model,va)
        cal_scores=minute_scores(va,s,cal);dev_scores=minute_scores(va,s,dev)
        th=float(np.nextafter(np.quantile(cal_scores,.99,method='higher'),np.inf))
        if th>1: raise ValueError('Unresolvable minute calibration quantile')
        # Common minute results provide equal time resolution and opportunity for both models.
        result=summarize(dev,dev_scores,th,events)
        fine=va.with_columns(pl.Series('score',s),(pl.col('window_start')//MINUTE*MINUTE).alias('minute_start'))
        fine=fine.join(dev.select(KEYS).rename({'window_start':'minute_start'}),on=['run_id','asset_id','minute_start'],how='semi')
        localized=summarize(fine,fine['score'].to_numpy(),th,events)
        # The helper assumes minute exposure: fine-window false-rate fields are deliberately excluded.
        localization={k:localized[k] for k in ('event_count','detected_events','event_detection_rate',
            'physical_events','by_family','mean_detection_latency_seconds')}
        rows.append({'id':f'{seconds}s','metrics':result,'fine_window_localization':localization})
        pred=pred.with_columns(pl.Series(f'{seconds}s',dev_scores))
        write_json(model/'minute_decision.json',{'threshold':th,'quantile':.99,
            'contract':'Alert if any short-window score >= threshold; evaluate false alert minutes on the shared grid',
            'calibration_minutes':len(cal),'localization_seconds':seconds})
        print(json.dumps(rows[-1]),flush=True)
    b=rows[0]['metrics']
    eligible=[r for r in rows[1:] if r['metrics']['physical_events']['detected']>b['physical_events']['detected'] and
        r['metrics']['detected_events']>=b['detected_events'] and r['metrics']['precision']>=.5 and
        r['metrics']['false_positive_windows']<=b['false_positive_windows'] and
        r['fine_window_localization']['physical_events']['detected']>=b['physical_events']['detected']]
    eligible.sort(key=lambda r:(-r['metrics']['physical_events']['detected'],-r['metrics']['detected_events'],r['metrics']['false_positive_windows']))
    write_json(out/'results.json',{'rows':rows,'eligible':[r['id'] for r in eligible],
        'selection':eligible[0]['id'] if eligible else None,'test_used':False,'truth_hashes':truth_hashes,
        'coverage':{p:{'original_minutes':len(minute[p]),'common_minutes':len(common[p])} for p in common},
        'calibration_minutes':len(cal),'development_minutes':len(dev),'events':len(events)})
    pred.write_parquet(out/'development_predictions.parquet')
    ui_state(out,'completed',1.,rows)
    print('ELIGIBLE',[r['id'] for r in eligible],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','baseline','raw-root','minute-cache','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
