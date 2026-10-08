"""One predeclared normal-only context calibration; development evaluation only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_otlab import _sha256
from scripts.research_physical import load_events, summarize
from scripts.normal_exposure_metrics import alarm_counts
from scripts.research_detection import write_json
from ot_irregularity.telemetry_health import KEYS
from ot_irregularity.specialists import relational_specialist
from ot_irregularity.contextual import run_bucket
from ot_irregularity.context_calibration import ContextMagnitudeCalibration
from ot_irregularity.class_ensemble import ClassMagnitudeEnsemble
from scripts.freeze_class_ensemble import verified_reference
from scripts.research_normal_coverage import event_table, integrity_losses
from scripts.research_physical import PHYSICAL
from ot_irregularity.pipeline import _scores, _git_commit


def run(args):
    out=args.output
    if out.exists(): raise FileExistsError('New diagnostic output required')
    study=args.root/'class-ensemble-20261005'; health=args.root/'telemetry-health-20261005-r2'
    result=json.loads((study/'results.json').read_text())
    if result['test_used']: raise ValueError('Development study required')
    inputs={}; populations={}; frames={}; parity=0
    reference=ClassMagnitudeEnsemble.load(verified_reference(study))
    calibration=[]
    for kind,cache,manifest_name in (
        ('historical',args.root/'physical-20261005/cache','cache_manifest.json'),
        ('normal',args.root/'normal-coverage-20261005/normal-cache','manifest.json')):
        metadata=json.loads((cache/manifest_name).read_text())
        if _sha256(cache/'validation.parquet')!=metadata['files']['validation.parquet']: raise ValueError('Calibration cache changed')
        rawpath=study/(kind+'-raw.parquet'); raw=pl.read_parquet(rawpath).filter(pl.col('calibration'))
        f=raw.join(pl.read_parquet(cache/'validation.parquet').select(KEYS+['operating_regime','is_anomaly']),on=KEYS,how='left',validate='1:1')
        if f['is_anomaly'].null_count() or f['is_anomaly'].any() or any(run_bucket(r,2)!=0 for r in f['run_id']): raise ValueError('Normal calibration partition required')
        margins=reference.score(f.select(['raw_'+str(seed) for seed in (20261005,20261006,20261007)]).to_numpy(),f['asset_class'].to_numpy())[0]
        calibration.append(f.select(KEYS+['operating_regime']).with_columns(pl.Series('margin',margins)))
        inputs[kind+'_raw_calibration']=_sha256(rawpath)
    cal=pl.concat(calibration); context=ContextMagnitudeCalibration().fit(cal['margin'].to_numpy(),cal['asset_class'].to_numpy(),cal['operating_regime'].to_numpy(),np.zeros(len(cal),dtype=bool))
    routes={}

    for kind, cache, manifest_name in (
        ('historical',args.root/'physical-20261005/cache','cache_manifest.json'),
        ('normal',args.root/'normal-coverage-20261005/normal-cache','manifest.json')):
        manifest=json.loads((cache/manifest_name).read_text())
        path=cache/'validation.parquet'
        if _sha256(path)!=manifest['files']['validation.parquet']: raise ValueError('Feature cache changed')
        if kind=='normal' and (manifest['source_seed']!=920611 or manifest['test_used']): raise ValueError('Only normal development allowed')
        inputs[kind+'_cache']=_sha256(path)
        candidate_path=study/('median-0.9975-'+kind+'.parquet')
        original_path=health/('original-20261005-'+kind+'.parquet')
        candidate=pl.read_parquet(candidate_path); original=pl.read_parquet(original_path)
        if candidate.select(KEYS).is_duplicated().any() or not candidate.select(KEYS).sort(KEYS).equals(original.select(KEYS).sort(KEYS)):
            raise ValueError('Score identities differ')
        if any(run_bucket(r,2)!=1 for r in candidate['run_id']): raise ValueError('Selection partition required')
        frame=candidate.join(pl.read_parquet(path),on=KEYS,how='left',validate='1:1',maintain_order='left')
        if frame['is_anomaly'].null_count(): raise ValueError('Unmatched feature window')
        frame=frame.join(original,on=KEYS,validate='1:1',maintain_order='left')
        if kind=='normal' and frame['is_anomaly'].any(): raise ValueError('Normal development labels required')
        for row in frame.iter_rows(named=True):
            margin=row['relational_margin']; margin=None if margin is None or np.isnan(margin) else margin
            adapted=relational_specialist(row,margin=margin,model_version='frozen-three',calibration_version='class-median-.9975')
            assert adapted.detected == (None if margin is None else margin>=1)
            assert adapted.calibrated_score == (None if margin is None else margin/(1+margin))
            parity+=1
        primary_margin=np.maximum(frame['original_score'].to_numpy()/frame['original_threshold'].to_numpy(),np.nan_to_num(frame['health'].to_numpy(),nan=0)/.05)
        primary=primary_margin/(1+primary_margin); candidate_score=frame['rank'].to_numpy()
        contextual,route=context.score(frame['relational_margin'].to_numpy(),frame['asset_class'].to_numpy(),frame['operating_regime'].to_numpy())
        combined=np.maximum.reduce([frame['baseline_score'].to_numpy()/frame['original_threshold'].to_numpy(),np.nan_to_num(contextual,nan=0),np.nan_to_num(frame['health'].to_numpy(),nan=0)/.05])
        union=combined/(1+combined); routes[kind]={str(k):int(v) for k,v in zip(*np.unique(route,return_counts=True))}
        frame=frame.with_columns(pl.Series('primary',primary),pl.Series('candidate',candidate_score),pl.Series('contextual',union))

        frames[kind]=frame
        inputs[kind+'_candidate_scores']=_sha256(candidate_path);inputs[kind+'_primary_scores']=_sha256(original_path)
    events,truth_hashes=load_events(args.raw_root,frames['historical'])
    for kind,frame in frames.items():
        if kind=='historical': populations[kind]={k:summarize(frame,frame[k].to_numpy(),.5,events) for k in ('primary','candidate','contextual')}
        else: populations[kind]={group:{k:alarm_counts(f,f[k].to_numpy()>=.5) for k in ('primary','candidate','contextual')}
            for (group,),f in frame.partition_by('asset_class',as_dict=True).items()}
    table=event_table(events,frames['historical'],frames['historical']['contextual'].to_numpy(),.5)
    primary_table=event_table(events,frames['historical'],frames['historical']['primary'].to_numpy(),.5)
    key=lambda e:(e['run_id'],e['asset_id'],e['event_id'])
    families={key(e):e['family'] for e in events}; detected={key(e) for e in table if e['detected']}
    lost_physical=sorted(key(e) for e in primary_table if e['detected'] and families[key(e)] in PHYSICAL and key(e) not in detected)
    bth=_scores(args.root/'validation-20261005/model-20261005',frames['historical'])[5]
    baseline_table=event_table(events,frames['historical'],frames['historical']['baseline_score'].to_numpy(),bth)
    lost_integrity=integrity_losses(baseline_table,table,families)
    metric=populations['historical']['contextual']; normals=[v['contextual'] for v in populations['normal'].values()]
    eligible=not lost_physical and not lost_integrity and metric['detected_events']>=86 and metric['false_positive_windows']<=4 and metric['precision']>=.5 and sum(v['false_windows'] for v in normals)<=25 and all(v['false_windows']*24/v['asset_hours']<=10 for v in normals)
    out.mkdir(parents=True)
    context.save(out/'context-reference.json')
    reloaded=ContextMagnitudeCalibration.load(out/'context-reference.json')
    for frame in frames.values():
        arguments=(frame['relational_margin'].to_numpy(),frame['asset_class'].to_numpy(),frame['operating_regime'].to_numpy())
        assert np.array_equal(context.score(*arguments)[0],reloaded.score(*arguments)[0],equal_nan=True)

    for kind,frame in frames.items(): frame.select(KEYS+['primary','candidate','contextual']).write_parquet(out/(kind+'.parquet'))
    write_json(out/'results.json',{'test_used':False,'fit_used':'normal-only tail calibration','selection':True,'eligible':bool(eligible),'routes':routes,'lost_primary_physical':lost_physical,'lost_baseline_integrity':lost_integrity,'lost_primary_events':sorted(key(e) for e in primary_table if e['detected'] and key(e) not in detected),'reference_reload_exact':True,'execution_commit':_git_commit(),'protocol_sha256':_sha256(Path(__file__).resolve().parents[1]/'docs/CONTEXT_CALIBRATION_PROTOCOL.md'),'input_hashes':inputs,
        'truth_hashes':truth_hashes,'adapter_parity_windows':parity,'populations':populations})
    print(json.dumps({'eligible':bool(eligible),'routes':routes,'lost_physical':lost_physical,'lost_integrity':lost_integrity,'adapter_parity_windows':parity,'historical':{k:(v['detected_events'],v['false_positive_windows']) for k,v in populations['historical'].items()},'normal':populations['normal']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','raw-root','output'): p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
