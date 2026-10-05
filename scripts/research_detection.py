"""Reproducible development comparison; never reads the historical test partition."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler

from ot_irregularity.features import encode_context, feature_columns, feature_signal
from ot_irregularity.metrics import evaluate_scores
from ot_irregularity.models import ae_errors, train_ae
from ot_irregularity.pipeline import _hash_paths, _windows, cdf_calibrate


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def bucket(run_id, modulo=5):
    return int(hashlib.sha256(str(run_id).encode()).hexdigest()[:8], 16) % modulo


def metrics(frame, score, threshold):
    return evaluate_scores(frame['is_anomaly'].to_numpy(), score, threshold,
                           asset_ids=frame['asset_id'].to_numpy(),
                           event_ids=frame['event_id'].to_numpy(),
                           event_start_us=frame['event_start_us'].to_list(),
                           window_end_us=frame['window_end'].to_numpy(), window_seconds=60)


def cache_windows(dataset, schema, cfg, output):
    output.mkdir(parents=True, exist_ok=True)
    provenance={'dataset_hash':_hash_paths([dataset/'train',dataset/'validation']),
                'schema_sha256':hashlib.sha256(json.dumps(schema,sort_keys=True).encode()).hexdigest(),
                'feature_config':{k:cfg.get(k,{}) for k in ('window','features','normalization')}}
    manifest_path=output/'cache_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['provenance']!=provenance:
            raise ValueError('Window cache dataset/schema/config mismatch')
        for name,sha in manifest['files'].items():
            if hashlib.sha256((output/name).read_bytes()).hexdigest()!=sha:
                raise ValueError('Window cache content hash mismatch')
    elif any(output.glob('*.parquet')):
        raise ValueError('Window cache has no provenance; regenerate in a new output directory')
    frames = {}
    for part in ('train', 'validation'):
        path = output / f'{part}.parquet'
        if not path.exists():
            print(f'features {part}', flush=True)
            _windows(dataset / part, cfg, schema).write_parquet(path)
        frames[part] = pl.read_parquet(path)
    write_json(manifest_path,{'provenance':provenance,'files':{f'{part}.parquet':hashlib.sha256((output/f'{part}.parquet').read_bytes()).hexdigest() for part in frames}})
    return frames


def select_columns(frame, schema, asset_class, compact):
    cols = feature_columns(frame)
    if asset_class != '__pooled__':
        roles = schema['signals_by_asset_class'][asset_class]
        cols = [c for c in cols if feature_signal(c) in roles]
    if compact:
        suffixes = ('_mean', '_std', '_range', '_delta', '_slope', '_coverage_ratio',
                    '_uncertain_ratio', '_bad_ratio', '_missing')
        cols = [c for c in cols if c.endswith(suffixes)]
    return cols


def experiment(args):
    torch.set_num_threads(2)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output/'state.json').exists():
        raise FileExistsError('Research state already exists; inspect it or choose a fresh output directory')
    schema = json.loads((args.baseline / 'feature_schema.json').read_text())
    cfg = yaml.safe_load((args.baseline / 'training_config.yaml').read_text())
    frames = cache_windows(args.dataset, schema, cfg, output / 'cache')
    train, vocab = encode_context(frames['train'])
    val, _ = encode_context(frames['validation'], vocab)
    # Split whole runs. Fit/early stopping, calibration, and selection are distinct.
    fit = train.filter(pl.Series([bucket(r) != 0 for r in train['run_id']]))
    stop = train.filter(pl.Series([bucket(r) == 0 for r in train['run_id']]))
    cal = val.filter(pl.Series([bucket(r, 2) == 0 for r in val['run_id']])).filter(~pl.col('is_anomaly'))
    dev = val.filter(pl.Series([bucket(r, 2) == 1 for r in val['run_id']]))
    state = {'status': 'running', 'phase': 'screening', 'progress': 0., 'runs': [],
             'selection': 'validation run hash bucket 1; normal calibration bucket 0',
             'historical_test_used': False, 'seed': args.seed, 'device': args.device,
             'sample_counts': {k:len(v) for k,v in [('fit',fit),('early_stop',stop),('calibration',cal),('development',dev)]}}
    write_json(output / 'state.json', state)
    variants = [('pooled-robust-all',False,False,'robust'),
                ('pooled-standard-compact',False,True,'standard'),
                ('class-robust-all',True,False,'robust'),
                ('class-robust-compact',True,True,'robust'),
                ('class-standard-compact',True,True,'standard')]
    for vid, grouped, compact, scaling in variants:
        root = output / vid
        root.mkdir(exist_ok=True)
        state['active_variant'] = vid
        write_json(output / 'state.json',state)
        scores = {k:np.zeros(len(dev)) for k in ('mean-ae','tail-ae','isolation','mean-ensemble','tail-ensemble')}
        refs = {k:np.zeros(len(cal)) for k in scores}
        groups = sorted(fit['asset_class'].unique()) if grouped else ['__pooled__']
        for group in groups:
            print(f'{vid} / {group}',flush=True)
            masks = [np.ones(len(f),dtype=bool) if group=='__pooled__' else f['asset_class'].to_numpy()==group for f in (fit,stop,cal,dev)]
            cols=select_columns(fit,schema,group,compact)
            arrays=[f.select(cols).to_numpy().astype(float)[mask] for f,mask in zip((fit,stop,cal,dev),masks)]
            scaler=(RobustScaler() if scaling=='robust' else StandardScaler()).fit(arrays[0])
            x,xe,xc,xd=[scaler.transform(a) for a in arrays]
            groupdir=root/group
            groupdir.mkdir(exist_ok=True)
            ae=train_ae(x,xe,{'device':args.device,'epochs':10000,'steps':args.steps,'patience':40,
                            'latent_dim':8,'batch_size':128,'log_interval':500},args.seed,groupdir/'autoencoder.pt')
            iso=IsolationForest(n_estimators=200,random_state=args.seed,n_jobs=2).fit(x)
            ec,ed,ee=ae_errors(ae,xc),ae_errors(ae,xd),ae_errors(ae,xe)
            # Scale residuals with independent normal early-stop runs, not labels.
            residual_scale=np.maximum(np.quantile(ee,.95,axis=0),1e-8)
            def tail(e):
                values=e/residual_scale
                return np.sort(values,axis=1)[:,-min(4,values.shape[1]):].mean(axis=1)
            raw={'mean-ae':(ec.mean(axis=1),ed.mean(axis=1)),
                 'tail-ae':(tail(ec),tail(ed)),
                 'isolation':(-iso.score_samples(xc),-iso.score_samples(xd))}
            local={k:(cdf_calibrate(c,c),cdf_calibrate(c,d)) for k,(c,d) in raw.items()}
            for kind,ae_name in [('mean-ensemble','mean-ae'),('tail-ensemble','tail-ae')]:
                c=(local[ae_name][0]+local['isolation'][0])/2
                d=(local[ae_name][1]+local['isolation'][1])/2
                local[kind]=(c,d)
            for kind,(c,d) in local.items():
                scores[kind][masks[3]]=d
                refs[kind][masks[2]]=c
            joblib.dump({'scaler':scaler,'isolation':iso,'features':cols,'residual_scale':residual_scale,
                         'references':{k:c for k,(c,d) in raw.items()}},groupdir/'detectors.joblib')
        for name,score in scores.items():
            # A 0.5% normal calibration alert-window budget. Threshold frozen before dev scoring.
            threshold=float(np.nextafter(np.quantile(refs[name],.995,method='higher'),np.inf))
            row={'candidate':vid+'/'+name,'threshold':threshold,**metrics(dev,score,threshold)}
            state['runs'].append(row)
            print(json.dumps(row),flush=True)
        pred=dev.select(['run_id','asset_id','asset_class','window_start','window_end','is_anomaly','event_id','event_start_us'])
        pred.with_columns([pl.Series(k,v) for k,v in scores.items()]).write_parquet(root/'development_scores.parquet')
        write_json(root/'calibration.json',{k:v.tolist() for k,v in refs.items()})
        state['progress']=(variants.index((vid,grouped,compact,scaling))+1)/len(variants)
        write_json(output/'state.json',state)
    state['status']='completed'
    state['phase']='completed'
    state['runs'].sort(key=lambda r:r['pr_auc'],reverse=True)
    write_json(output/'state.json',state)


def operating_points(output):
    """Development-only weight/threshold grid; normal calibration determines each threshold."""
    rows=[]
    for variant in ('class-robust-all','class-robust-compact','class-standard-compact'):
        dev=pl.read_parquet(output/variant/'development_scores.parquet')
        refs=json.loads((output/variant/'calibration.json').read_text())
        for ae_weight in (1.,.9,.75,.5):
            cal=ae_weight*np.array(refs['tail-ae'])+(1-ae_weight)*np.array(refs['isolation'])
            score=ae_weight*dev['tail-ae'].to_numpy()+(1-ae_weight)*dev['isolation'].to_numpy()
            for q in (.98,.99,.995):
                threshold=float(np.nextafter(np.quantile(cal,q,method='higher'),np.inf))
                rows.append({'variant':variant,'ae_weight':ae_weight,'quantile':q,**metrics(dev,score,threshold)})
    write_json(output/'operating_points.json',rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--seed',type=int,default=20261005)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--operating-points-only',action='store_true')
    args=parser.parse_args()
    if args.operating_points_only:operating_points(args.output)
    else:experiment(args)
