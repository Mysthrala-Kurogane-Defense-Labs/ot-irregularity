"""Package and infer the frozen relational research candidate without changing defaults."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import polars as pl

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_detection import write_json
from scripts.validate_physical import hashes
from ot_irregularity.contextual import tail_errors
from ot_irregularity.features import encode_context
from ot_irregularity.models import ae_errors,load_ae
from ot_irregularity.pipeline import _model_config,_prediction_records,_scores,_windows,cdf_calibrate


def package(args):
    frozen=json.loads((args.validation/'frozen.json').read_text())
    results=json.loads((args.validation/'results.json').read_text())
    selected=next(m for m in frozen['models'] if m['seed']==frozen['primary_seed'])
    if hashes(args.baseline)!=frozen['baseline_files'] or hashes(args.validation/selected['directory'])!=selected['files']:
        raise ValueError('Artifacts changed after freeze')
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(args.baseline,args.output/'baseline')
    shutil.copytree(args.validation/selected['directory'],args.output/'relationships')
    write_json(args.output/'candidate.json',{'format_version':1,'model_version':'0.5.0-candidate',
        'primary_seed':frozen['primary_seed'],'threshold':selected['threshold'],'tail_policy':'complete',
        'baseline_files':hashes(args.output/'baseline'),'relationship_files':hashes(args.output/'relationships'),
        'data_license':'CC-BY-4.0','attribution':'Mysthrala Kurogane Defense Labs, OT Irregularity Lab generated pseudo-synthetic telemetry',
        'source_url':'https://github.com/Mysthrala-Kurogane-Defense-Labs/ot-irregularity-lab',
        'formula':'max(baseline irregularity CDF, relational AE residual CDF)',
        'availability':'All roles good_ratio>=0.95, coverage>=0.9, finite means, missing=0. Otherwise relational score is unavailable.',
        'frozen_manifest_sha256':hashlib.sha256((args.validation/'frozen.json').read_bytes()).hexdigest()})
    write_json(args.output/'metrics.json',results)
    (args.output/'model_card.md').write_text('# Relational research candidate 0.5.0\n\n'
        'Frozen primary seed 20261005. v0.4 normal-only AE/IF plus a second AE on six normal cross-signal Ridge residuals per class. '
        'Complete-tail batch windows; q99.75 normal calibration of maximum score. No causal diagnosis. '
        'See metrics.json for independent pseudo-synthetic evaluation and uncertainty. Cooling degradation remains missed. '
        'Not validated on plants or continuous monitoring; tail silence needs an external watermark. '
        'Scores are reference percentiles, not probabilities of failure. Baseline IF remains separately reported.\n',encoding='utf-8')


def infer_records(model,dataset,device='auto'):
    manifest=json.loads((model/'candidate.json').read_text())
    if manifest['format_version']!=1:raise ValueError('Unsupported candidate format')
    if hashes(model/'baseline')!=manifest['baseline_files'] or hashes(model/'relationships')!=manifest['relationship_files']:
        raise ValueError('Candidate artifact integrity mismatch')
    cfg=_model_config(model/'baseline');cfg['window']['tail_policy']='complete'
    schema=json.loads((model/'baseline'/'feature_schema.json').read_text())
    windows=_windows(dataset,cfg,schema)
    windows,_=encode_context(windows,schema.get('context_vocabulary',{}))
    cols,errors,sa,si,base,_=_scores(model/'baseline',windows);threshold=manifest['threshold']
    records=_prediction_records(windows,cols,errors,sa,si,base,threshold,schema.get('signals_by_asset_class'),manifest['model_version'])
    relational=np.zeros(len(windows));available_all=np.zeros(len(windows),dtype=bool);contexts=[None]*len(windows)
    for group in sorted(windows['asset_class'].unique()):
        folder=model/'relationships'/group;m=joblib.load(folder/'detector.joblib');relation=m['relationship']
        indices=np.flatnonzero(windows['asset_class'].to_numpy()==group);frame=windows[indices]
        residual,available,_=relation.transform(frame);scaled=m['scaler'].transform(residual)
        raw,contributions=tail_errors(ae_errors(load_ae(folder/'autoencoder.pt',device),scaled),m['ae_scale'],2)
        values=cdf_calibrate(m['references']['ae'],raw);values[~available]=0.
        relational[indices]=values;available_all[indices]=available
        actual=frame.select([role+'_mean' for role in relation.roles]).to_numpy()
        expected=actual-residual*relation.scaler_.scale_
        for local,index in enumerate(indices):
            if not available[local]:
                contexts[index]={'available':False,'reason':'insufficient current signal quality or coverage'};continue
            denom=contributions[local].sum()
            contexts[index]={'available':True,'standardized_residual':dict(zip(relation.roles,map(float,residual[local]))),
                'normal_relation_expected_mean':dict(zip(relation.roles,map(float,expected[local]))),
                'observed_mean':dict(zip(relation.roles,map(float,actual[local]))),
                'reconstruction_contributions':dict(zip(relation.roles,map(float,contributions[local]/denom if denom>0 else np.zeros(len(relation.roles))))) }
    for i,record in enumerate(records):
        record['scores']['baseline_irregularity']=float(base[i])
        record['scores']['relationship_autoencoder']=float(relational[i]) if available_all[i] else None
        record['scores']['irregularity']=float(max(base[i],relational[i]))
        record['decision_threshold']=threshold
        record['window_tail_policy']='complete'
        record['relationship_context']=contexts[i]
        record['contribution_basis']='feature_contributions: baseline normalized AE error; relationship_context: relational AE error; neither is causal'
        if available_all[i] and relational[i]>=threshold:
            record['observations']=sorted(set(record['observations'])|{'multivariate_relationship_deviation'})
        record['is_irregular']=record['scores']['irregularity']>=threshold
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    export=commands.add_parser('package')
    for name in ('baseline','validation','output'):export.add_argument('--'+name,type=Path,required=True)
    infer=commands.add_parser('infer')
    for name in ('model','dataset','output'):infer.add_argument('--'+name,type=Path,required=True)
    infer.add_argument('--device',default='auto',help='Residual AE device; baseline retains its artifact device setting');args=p.parse_args()
    if args.command=='package':package(args)
    else:
        if args.output.exists():raise FileExistsError('Prediction output exists; choose a new file')
        records=infer_records(args.model,args.dataset,args.device);args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.suffix.lower()=='.parquet':pl.from_dicts(records,infer_schema_length=None).write_parquet(args.output)
        else:args.output.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),encoding='utf-8',newline='\n')
