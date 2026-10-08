"""One fixed development-only union diagnostic; no fitting or test access."""
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


def run(args):
    out=args.output
    if out.exists(): raise FileExistsError('New diagnostic output required')
    study=args.root/'class-ensemble-20261005'; health=args.root/'telemetry-health-20261005-r2'
    result=json.loads((study/'results.json').read_text())
    if result['test_used']: raise ValueError('Development study required')
    inputs={}; populations={}; frames={}; parity=0
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
        primary=primary_margin/(1+primary_margin); candidate_score=frame['rank'].to_numpy(); union=np.maximum(primary,candidate_score)
        frame=frame.with_columns(pl.Series('primary',primary),pl.Series('candidate',candidate_score),pl.Series('union',union))
        assert np.all((primary<.5)|(union>=.5))
        frames[kind]=frame
        inputs[kind+'_candidate_scores']=_sha256(candidate_path);inputs[kind+'_primary_scores']=_sha256(original_path)
    events,truth_hashes=load_events(args.raw_root,frames['historical'])
    for kind,frame in frames.items():
        if kind=='historical': populations[kind]={k:summarize(frame,frame[k].to_numpy(),.5,events) for k in ('primary','candidate','union')}
        else: populations[kind]={group:{k:alarm_counts(f,f[k].to_numpy()>=.5) for k in ('primary','candidate','union')}
            for (group,),f in frame.partition_by('asset_class',as_dict=True).items()}
    out.mkdir(parents=True)
    for kind,frame in frames.items(): frame.select(KEYS+['primary','candidate','union']).write_parquet(out/(kind+'.parquet'))
    write_json(out/'results.json',{'test_used':False,'fit_used':False,'selection':False,'input_hashes':inputs,
        'truth_hashes':truth_hashes,'adapter_parity_windows':parity,'populations':populations})
    print(json.dumps({'adapter_parity_windows':parity,'historical':{k:(v['detected_events'],v['false_positive_windows']) for k,v in populations['historical'].items()},'normal':populations['normal']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','raw-root','output'): p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
