"""Verify and score normal-only balanced development with frozen specialists."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
import polars as pl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_otlab import _sha256, normalize_observations, _annotate
from scripts.health_cadence import declared_cadence, attach_declared_cadence
from scripts.research_detection import write_json
from scripts.research_class_ensemble import member_errors
from scripts.research_telemetry_health import align_scores
from scripts.validate_physical import hashes
from ot_irregularity.pipeline import _model_config, _windows, _scores, _git_commit
from ot_irregularity.telemetry_health import TelemetryHealthReference, extract_health, KEYS
from ot_irregularity.class_ensemble import ClassMagnitudeEnsemble

REGIMES=['off','idle','warmup','low_load','normal_load','high_load','cooldown','maintenance','transitions']


def verify_manifest(doc,seed,suite_hash,lock_hash,seen):
    if (doc.get('master_seed')!=seed or doc.get('suite_sha256')!=suite_hash or doc.get('uv_lock_sha256')!=lock_hash
        or doc.get('data_license')!='CC-BY-4.0' or doc.get('simulator_version')!='0.6.0'
        or doc.get('synthetic') is not True or doc.get('generated') is not True or doc.get('customer_data') is not False
        or doc.get('generation_complete') is not True or doc.get('partition_counts')!={'train':12,'validation':12,'test':0}):
        raise ValueError('Balanced source protocol mismatch')
    runs=doc['runs']; seeds=[r['seed'] for r in runs]
    if len(runs)!=24 or len({r['run_id'] for r in runs})!=24 or len(set(seeds))!=24 or set(seeds)&seen:
        raise ValueError('Run population or seed leakage')
    if any(sum(r['partition']==p for r in runs)!=12 for p in ('train','validation')):
        raise ValueError('Actual partition counts differ')
    seen.update(seeds)
    return runs


def run(args):
    root=args.root;out=args.output;repo=Path(__file__).resolve().parents[1]
    freeze=json.loads((root/'class-confirmation-20261005/frozen.json').read_text())
    baseline=root/'validation-20261005/model-20261005';relations=root/'relationships-holdout-20261005'
    health_path=root/'telemetry-health-20261005-r2/health-reference.json';class_path=root/'class-confirmation-20261005/class-reference.json'
    if hashes(baseline)!=freeze['baseline_files'] or _sha256(health_path)!=freeze['health_reference_sha256'] or _sha256(class_path)!=freeze['class_reference_sha256']:
        raise ValueError('Frozen artifact mismatch')
    for m in freeze['members']:
        if hashes(relations/m['directory'])!=m['files']: raise ValueError('Frozen member mismatch')
    seen=set(freeze['excluded_run_seeds'])
    for name in freeze['source_names']:
        doc=json.loads((root/name/'dataset_manifest.json').read_text());seen.update(r['seed'] for r in doc['runs'])
    sources=[];source_hashes={}
    for i,regime in enumerate(REGIMES):
        seed=930101+i;source=root/f'balanced-regime-{seed}';path=source/'dataset_manifest.json'
        doc=json.loads(path.read_text());source_hashes[str(seed)]=_sha256(path)
        runs=verify_manifest(doc,seed,_sha256(repo/f'configs/balanced-regime-development/{regime}.yaml'),freeze['lab_uv_lock_sha256'],seen)
        sources.append((regime,seed,source,runs))
    inputs={'source_manifests':source_hashes,'frozen_artifacts_sha256':_sha256(root/'class-confirmation-20261005/frozen.json'),
        'runtime':{p.relative_to(repo).as_posix():_sha256(p) for folder in ('scripts','src') for p in sorted((repo/folder).rglob('*.py'))},
        'protocol_sha256':_sha256(repo/'docs/BALANCED_REGIME_PROTOCOL.md')}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'inputs.json').exists():
        if json.loads((out/'inputs.json').read_text())!=inputs: raise ValueError('Resume inputs changed')
    else: write_json(out/'inputs.json',inputs)
    cfg=_model_config(baseline,tail_policy='complete');schema=json.loads((baseline/'feature_schema.json').read_text())
    health=TelemetryHealthReference.load(health_path);reference=ClassMagnitudeEnsemble.load(class_path)
    frames={'train':[],'validation':[]};records=[];completed=0
    for regime,seed,source,entries in sources:
        for entry in entries:
            folder,period,source_files=declared_cadence(source,entry)
            if _sha256(folder/'ground_truth.json')!=entry['ground_truth_sha256']: raise ValueError('Truth hash changed')
            truth=json.loads((folder/'ground_truth.json').read_text())
            if truth['events'] or truth['run_id']!=entry['run_id']: raise ValueError('Explicitly normal run required')
            source_files['ground_truth.json']=entry['ground_truth_sha256']
            dest=out/'runs'/str(seed)/entry['run_id'];dest.mkdir(parents=True,exist_ok=True)
            path=dest/'windows.parquet';marker=dest/'record.json'
            if marker.exists():
                record=json.loads(marker.read_text())
                if record['source_files']!=source_files or record['sha256']!=_sha256(path): raise ValueError('Completed run changed')
                w=pl.read_parquet(path)
            else:
                raw=pl.read_parquet(folder/'telemetry.parquet')
                if set(raw['run_id'].unique())!={entry['run_id']}: raise ValueError('Raw identity mismatch')
                raw=_annotate(normalize_observations(raw),[]).with_columns((pl.lit(f'balanced-{seed}::')+pl.col('run_id')).alias('run_id'))
                with tempfile.TemporaryDirectory(prefix='balanced-regime-') as temporary:
                    raw_path=Path(temporary)/'raw.parquet';raw.write_parquet(raw_path);w=_windows(raw_path,cfg,schema)
                if not len(w) or w['is_anomaly'].any() or set(w['asset_class'])!={'CNC','COMPRESSOR','CONVEYOR','PUMP'}:
                    raise ValueError('Invalid normal feature coverage')
                errors,cdfs=member_errors(relations,freeze['members'],w,args.device)
                _,_,ae,iso,base,_=_scores(baseline,w)
                features=extract_health(attach_declared_cadence(raw,period),w,schema['signals_by_asset_class'])
                hs,_=health.score(features,repetition=False);hs=align_scores(w,hs)
                margin,available=reference.score(errors,w['asset_class'].to_numpy())
                primary_margin=np.maximum(np.maximum(base,cdfs[:,0])/freeze['members'][0]['threshold'],np.nan_to_num(hs['health_score'].to_numpy(),nan=0)/.05)
                w=w.with_columns(pl.Series('class_margin',margin),pl.Series('baseline_score',base),pl.Series('baseline_ae',ae),pl.Series('baseline_if',iso),
                    pl.Series('health',hs['health_score']),pl.Series('primary',primary_margin/(1+primary_margin)),pl.lit(regime).alias('source_profile'),
                    *[pl.Series('raw_'+str(m['seed']),errors[:,i]) for i,m in enumerate(freeze['members'])])
                w.write_parquet(path);record={'source_files':source_files,'sha256':_sha256(path),'windows':len(w),'partition':entry['partition']}
                write_json(marker,record)
            frames[entry['partition']].append(w);records.append(record);completed+=1
            write_json(out/'progress.json',{'phase':'score_normal_development','completed_runs':completed,'total_runs':216})
    for partition,parts in frames.items(): pl.concat(parts,how='diagonal_relaxed').write_parquet(out/(partition+'.parquet'))
    write_json(out/'manifest.json',{'input_sha256':_sha256(out/'inputs.json'),'files':{p:_sha256(out/(p+'.parquet')) for p in frames},'runs':216,'test_used':False,'git_commit':_git_commit()})
    write_json(out/'progress.json',{'phase':'completed','runs':216})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');run(p.parse_args())
