"""Six predeclared class-calibrated raw-error ensembles; development only."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import polars as pl
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_otlab import _sha256
from scripts.research_detection import cache_windows, write_json
from scripts.research_relationships import closed
from scripts.research_physical import load_events, summarize, PHYSICAL
from scripts.research_normal_coverage import event_table, integrity_losses, normal_metrics
from scripts.research_telemetry_health import align_scores, candidate_gate
from scripts.validate_physical import hashes
from ot_irregularity.class_ensemble import ClassMagnitudeEnsemble
from ot_irregularity.contextual import run_bucket, tail_errors
from ot_irregularity.models import ae_errors, load_ae
from ot_irregularity.pipeline import _scores, _git_commit, cdf_calibrate
from ot_irregularity.telemetry_health import KEYS, TelemetryHealthReference


def member_errors(root, specs, frame, device):
    raw=np.full((len(frame),3),np.nan); cdf=np.zeros((len(frame),3))
    for i,spec in enumerate(specs):
        for group in sorted(frame['asset_class'].unique()):
            path=root/spec['directory']/group; model=joblib.load(path/'detector.joblib')
            mask=frame['asset_class'].to_numpy()==group
            residual,available,_=model['relationship'].transform(frame.filter(pl.Series(mask)))
            values,_=tail_errors(ae_errors(load_ae(path/'autoencoder.pt',device),model['scaler'].transform(residual)),model['ae_scale'],2)
            ranks=cdf_calibrate(model['references']['ae'],values); ranks[~available]=0
            values=values.copy(); values[~available]=np.nan
            raw[mask,i]=values; cdf[mask,i]=ranks
    return raw,cdf


def run(args):
    repo=Path(__file__).resolve().parents[1]; out=args.output; out.mkdir(parents=True,exist_ok=False)
    public=json.loads((repo/'docs/results/telemetry-health-20261005-r2.json').read_text())
    inputs=json.loads((args.health_study/'inputs.json').read_text())
    if _sha256(args.health_study/'inputs.json') != public['verification']['inputs_sha256'] or inputs['test_used']:
        raise ValueError('Corrected health provenance differs from recorded development')
    if _sha256(args.health_study/'health-reference.json') != public['health_reference_sha256']:
        raise ValueError('Corrected health reference changed')
    frozen=json.loads((args.relationships/'frozen.json').read_text())
    if frozen['models']!=inputs['relational_models'] or hashes(args.baseline)!=frozen['baseline_files']:
        raise ValueError('Frozen members/baseline changed')
    if [m['seed'] for m in frozen['models']] != [20261005,20261006,20261007]:
        raise ValueError('Exactly the three declared members required')
    for m in frozen['models']:
        if hashes(args.relationships/m['directory'])!=m['files']:raise ValueError('Member bytes changed')
    nc=json.loads((args.normal_cache/'manifest.json').read_text())
    if _sha256(args.normal_cache/'manifest.json')!=inputs['normal_cache_manifest_sha256'] or nc['source_seed']!=920611 or nc['test_used']:
        raise ValueError('Only verified normal development seed 920611 allowed')
    if _sha256(args.normal_source/'dataset_manifest.json')!=nc['source_manifest_sha256']:
        raise ValueError('Normal source manifest changed')
    for name,sha in nc['files'].items():
        if _sha256(args.normal_cache/name)!=sha:raise ValueError('Normal cache changed')
    hc=args.health_study/'health-cache'; manifest=json.loads((hc/'manifest.json').read_text())
    if manifest['inputs']!=inputs:raise ValueError('Health cache provenance mismatch')
    for name,sha in manifest['files'].items():
        if _sha256(hc/name)!=sha:raise ValueError('Health cache changed')
    cfg=yaml.safe_load((args.baseline/'training_config.yaml').read_text())
    schema=json.loads((args.baseline/'feature_schema.json').read_text())
    old=cache_windows(args.dataset,schema,cfg,args.cache)
    historical=closed(old['validation'],pl.read_parquet(args.dataset/'validation/telemetry.parquet'))
    normal=pl.read_parquet(args.normal_cache/'validation.parquet')
    if set(historical['run_id']) & set(normal['run_id']) or set(old['train']['run_id']) & set(historical['run_id']):
        raise ValueError('Run leakage')
    frames={'historical':historical,'normal':normal}; health_reference=TelemetryHealthReference.load(args.health_study/'health-reference.json')
    data={}; calibration=[]; cal_classes=[]; cal_runs=set(); dev_runs=set()
    control=next(r for r in public['rows'] if r['seed']==20261005 and not r['repetition'] and r['health_threshold']==.05)
    for kind,frame in frames.items():
        raw,cdf=member_errors(args.relationships,frozen['models'],frame,args.device)
        base=_scores(args.baseline,frame)[4]
        features=pl.read_parquet(hc/('old-validation.parquet' if kind=='historical' else 'new-validation.parquet'))
        scored,_=health_reference.score(features,repetition=False); scored=align_scores(frame,scored)
        cal=np.array([run_bucket(r,2)==0 for r in frame['run_id']]) & ~frame['is_anomaly'].to_numpy()
        dev=np.array([run_bucket(r,2)==1 for r in frame['run_id']])
        cal_runs.update(frame.filter(pl.Series(cal))['run_id']); dev_runs.update(frame.filter(pl.Series(dev))['run_id'])
        calibration.append(raw[cal]); cal_classes.extend(frame.filter(pl.Series(cal))['asset_class'].to_list())
        w=frame.filter(pl.Series(dev)); health=scored['health_score'].to_numpy()[dev]
        original=np.maximum(base[dev],cdf[dev,0]); shared=frozen['models'][0]['threshold']
        saved=align_scores(w,pl.read_parquet(args.health_study/f'original-20261005-{kind}.parquet'))
        if not np.array_equal(original,saved['original_score'].to_numpy()):raise ValueError('Fresh original control differs')
        if not scored.filter(pl.Series(dev)).equals(pl.read_parquet(args.health_study/f'health-{kind}-repetition-False.parquet')):
            raise ValueError('Reloaded health differs')
        margin=np.maximum(original/shared,np.nan_to_num(health,nan=0)/.05)
        data[kind]={'frame':w,'raw':raw[dev],'base':base[dev],'health':health,'control':margin/(1+margin)}
        frame.select(KEYS).with_columns(*[pl.Series('raw_'+str(m['seed']),raw[:,i]) for i,m in enumerate(frozen['models'])],
            pl.Series('calibration',cal),pl.Series('development',dev)).write_parquet(out/(kind+'-raw.parquet'))
        write_json(out/'progress.json',{'phase':'raw_errors','population':kind})
    if cal_runs & dev_runs:raise ValueError('Calibration/development run leakage')
    raw_cal=np.vstack(calibration); classes=np.array(cal_classes)
    if set(classes)!=set(schema['signals_by_asset_class']):raise ValueError('Incomplete class calibration')
    source=json.loads((args.normal_source/'dataset_manifest.json').read_text())
    profiles=yaml.safe_load((repo/'configs/normal-coverage-development.yaml').read_text())['generation']['regime_profiles']
    profile={'normal-dev-920611::'+e['run_id']:next(p['profile_id'] for p in profiles if p['shift_pattern']==e['configured_shift_pattern']) for e in source['runs']}
    attrs={r['run_id']:{'profile':profile[r['run_id']],'sampling_interval_ms':r['sampling_interval_ms'],'expose_operating_regime':r['expose_operating_regime']} for r in nc['runs']}
    h=data['historical']; n=data['normal']; w=h['frame']; events,truth_hashes=load_events(args.raw_root,w)
    bth=_scores(args.baseline,w)[5]; baseline=summarize(w,h['base'],bth,events)
    baseline_events=event_table(events,w,h['base'],bth); primary_events=event_table(events,w,h['control'],.5)
    key=lambda e:(e['run_id'],e['asset_id'],e['event_id']); families={key(e):e['family'] for e in events}
    required={key(e) for e in primary_events if e['detected'] and families[key(e)] in PHYSICAL}
    rows=[]; available_counts={}; reload_checks={}
    for aggregation in ('mean','median'):
        for q in (.99,.995,.9975):
            name=f'{aggregation}-{q}'; model=ClassMagnitudeEnsemble(aggregation,q).fit(raw_cal,classes,np.zeros(len(classes),dtype=bool))
            model.save(out/(name+'.json')); loaded=ClassMagnitudeEnsemble.load(out/(name+'.json')); ranks={}
            for kind,d in data.items():
                margin,available=model.score(d['raw'],d['frame']['asset_class'].to_numpy())
                repeated=loaded.score(d['raw'],d['frame']['asset_class'].to_numpy())[0]
                if not np.array_equal(margin,repeated,equal_nan=True):raise ValueError('Class reference reload differs')
                combined=np.maximum.reduce([d['base']/shared,np.nan_to_num(margin,nan=0),np.nan_to_num(d['health'],nan=0)/.05])
                rank=combined/(1+combined)
                if not np.array_equal(rank>=.5,combined>=1):raise ValueError('Ranking/decision mismatch')
                ranks[kind]=rank; available_counts[kind]=int(available.sum())
                d['frame'].select(KEYS).with_columns(pl.Series('relational_margin',margin),pl.Series('health',d['health']),
                    pl.Series('baseline_score',d['base']),pl.Series('rank',rank)).write_parquet(out/(name+'-'+kind+'.parquet'))
            reload_checks[name]=True
            metrics=summarize(w,ranks['historical'],.5,events); nm=normal_metrics(n['frame'],ranks['normal'],.5,attrs)
            table=event_table(events,w,ranks['historical'],.5); found={key(e) for e in table if e['detected']}
            lost=integrity_losses(baseline_events,table,families); physical_lost=sorted(required-found)
            eligible=candidate_gate(metrics,nm,lost,physical_lost,baseline,control['historical'],control['new_normal'])
            row={'aggregation':aggregation,'quantile':q,'historical':metrics,'new_normal':nm,
                'lost_baseline_integrity':lost,'lost_primary_corrected_physical':physical_lost,
                'lost_primary_corrected_events':sorted(key(e) for e in primary_events if e['detected'] and key(e) not in found),'eligible':eligible}
            rows.append(row);print(json.dumps({'point':name,'events':metrics['detected_events'],'fp':metrics['false_positive_windows'],
                'normal_fp':nm['overall']['false_windows'],'lost_integrity':len(lost),'lost_physical':len(physical_lost),'eligible':eligible}),flush=True)
    eligible=[r for r in rows if r['eligible']]
    selected=max(eligible,key=lambda r:(r['historical']['detected_events'],-r['new_normal']['overall']['false_windows'],
        -r['historical']['false_positive_windows'],r['aggregation']=='median',r['quantile'])) if eligible else None
    write_json(out/'results.json',{'rows':rows,'selected':selected,'calibration_windows':len(raw_cal),'calibration_runs':len(cal_runs),
        'available_development':available_counts,'reference_reload_exact':reload_checks,'test_used':False,'network_retraining':False,
        'git_commit':_git_commit(),'protocol_sha256':_sha256(repo/'docs/CLASS_CALIBRATED_ENSEMBLE_PROTOCOL.md'),
        'health_inputs_sha256':_sha256(args.health_study/'inputs.json'),'baseline_files':frozen['baseline_files'],
        'members':frozen['models'],'truth_hashes':truth_hashes,'primary_control':control})
    write_json(out/'progress.json',{'phase':'completed','selected':selected})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','health-study','baseline','relationships','dataset','cache','normal-source','normal-cache','raw-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');run(p.parse_args())
