"""Verify frozen members and evaluate the predeclared majority plus health."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_otlab import _sha256
from scripts.research_detection import cache_windows, write_json
from scripts.research_relationships import closed
from scripts.relationship_operating_points import residual_scores
from scripts.research_physical import PHYSICAL, load_events, summarize
from scripts.research_normal_coverage import integrity_losses, event_table, normal_metrics
from scripts.research_telemetry_health import align_scores, candidate_gate
from scripts.validate_physical import hashes
from ot_irregularity.contextual import run_bucket
from ot_irregularity.pipeline import _scores, _git_commit
from ot_irregularity.consensus import consensus_scores
from ot_irregularity.telemetry_health import KEYS, TelemetryHealthReference


def run(args):
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    h = args.health_study
    prior = json.loads((h/'results.json').read_text())
    inputs = json.loads((h/'inputs.json').read_text())
    verified = json.loads((h/'verification.json').read_text())
    if _sha256(h/'inputs.json') != verified['inputs_sha256'] or inputs['test_used']:
        raise ValueError('Corrected health provenance changed')
    if _sha256(h/'health-reference.json') != prior['health_reference_sha256']:
        raise ValueError('Health reference changed')
    frozen = json.loads((args.relationships/'frozen.json').read_text())
    if hashes(args.baseline) != frozen['baseline_files'] or frozen['models'] != inputs['relational_models']:
        raise ValueError('Frozen controls changed')
    if [m['seed'] for m in frozen['models']] != [20261005,20261006,20261007]:
        raise ValueError('Fixed ensemble members required')
    for spec in frozen['models']:
        if hashes(args.relationships/spec['directory']) != spec['files']:
            raise ValueError('Member artifact changed')
    cache_manifest = json.loads((h/'health-cache/manifest.json').read_text())
    if cache_manifest['inputs'] != inputs:
        raise ValueError('Health cache provenance differs')
    for name, sha in cache_manifest['files'].items():
        if _sha256(h/'health-cache'/name) != sha:
            raise ValueError('Health cache changed')
    nc = json.loads((args.normal_cache/'manifest.json').read_text())
    if _sha256(args.normal_cache/'manifest.json') != inputs['normal_cache_manifest_sha256'] or nc['test_used']:
        raise ValueError('Normal cache provenance changed')
    for name, sha in nc['files'].items():
        if _sha256(args.normal_cache/name) != sha:
            raise ValueError('Normal window cache changed')
    if _sha256(args.normal_source/'dataset_manifest.json') != nc['source_manifest_sha256']:
        raise ValueError('Normal source manifest changed')
    config = yaml.safe_load((args.baseline/'training_config.yaml').read_text())
    schema = json.loads((args.baseline/'feature_schema.json').read_text())
    cached = cache_windows(args.dataset, schema, config, args.cache)
    historical = closed(cached['validation'], pl.read_parquet(args.dataset/'validation/telemetry.parquet'))
    normal = pl.read_parquet(args.normal_cache/'validation.parquet')
    frames = {k:f.filter(pl.Series([run_bucket(r,2)==1 for r in f['run_id']]))
              for k,f in [('historical',historical),('normal',normal)]}
    profiles = yaml.safe_load((root/'configs/normal-coverage-development.yaml').read_text())['generation']['regime_profiles']
    source = json.loads((args.normal_source/'dataset_manifest.json').read_text())
    profile = {'normal-dev-920611::'+e['run_id']:next(p['profile_id'] for p in profiles if p['shift_pattern']==e['configured_shift_pattern']) for e in source['runs']}
    attributes = {r['run_id']:{'profile':profile[r['run_id']], 'sampling_interval_ms':r['sampling_interval_ms'],
                             'expose_operating_regime':r['expose_operating_regime']} for r in nc['runs']}
    reference = TelemetryHealthReference.load(h/'health-reference.json')
    predictions = {}; sources = {}; thresholds = [s['threshold'] for s in frozen['models']]
    for kind, frame in frames.items():
        _, _, _, _, base, bth = _scores(args.baseline, frame)
        members = []
        for spec in frozen['models']:
            name = f"original-{spec['seed']}-{kind}.parquet"
            saved = align_scores(frame, pl.read_parquet(h/name))
            values = np.maximum(base, residual_scores(args.relationships/spec['directory'], frame, args.device)['ae'])
            if not np.array_equal(values, saved['original_score'].to_numpy()) or not np.all(saved['original_threshold'].to_numpy()==spec['threshold']):
                raise ValueError('Fresh member inference differs from corrected health study')
            members.append(values); sources[name] = _sha256(h/name)
        feature_name = 'old-validation.parquet' if kind=='historical' else 'new-validation.parquet'
        features = pl.read_parquet(h/'health-cache'/feature_name).join(frame.select(KEYS),on=KEYS,how='semi')
        health, _ = reference.score(features,repetition=False)
        health = align_scores(frame,health)
        if not health.equals(pl.read_parquet(h/f'health-{kind}-repetition-False.parquet')):
            raise ValueError('Reloaded health differs from corrected study')
        health_values = health['health_score'].to_numpy()
        scores = np.column_stack(members)
        predictions[kind] = {'base':base, 'bth':bth, 'members':scores, 'health':health_values,
                             'modes':{mode:consensus_scores(scores,thresholds,health_values,mode=mode) for mode in ('majority','unanimity','union')}}
        saved = frame.select(KEYS).with_columns(pl.Series('health_score',health_values))
        saved = saved.with_columns([pl.Series('member_'+str(s['seed']),scores[:,i]) for i,s in enumerate(frozen['models'])])
        saved = saved.with_columns([pl.Series(mode+'_rank',v['rank']) for mode,v in predictions[kind]['modes'].items()])
        saved.write_parquet(out/(kind+'-predictions.parquet'))
    frame = frames['historical']; pred = predictions['historical']
    events, truth_hashes = load_events(args.raw_root,frame)
    families = {(e['run_id'],e['asset_id'],e['event_id']):e['family'] for e in events}
    key = lambda e:(e['run_id'],e['asset_id'],e['event_id'])
    baseline = summarize(frame,pred['base'],pred['bth'],events)
    baseline_events = event_table(events,frame,pred['base'],pred['bth'])
    corrected_events = []
    for i in range(3):
        margin = np.maximum(pred['members'][:,i]/thresholds[i],np.nan_to_num(pred['health'],nan=0)/.05)
        corrected_events.append(event_table(events,frame,margin,1.))
    physical_required = {key(e) for e in corrected_events[0] if e['detected'] and families[key(e)] in PHYSICAL}
    control = next(r for r in prior['rows'] if r['seed']==20261005 and not r['repetition'] and r['health_threshold']==.05)
    rows = []
    for mode in ('majority','unanimity','union'):
        values = pred['modes'][mode]
        metrics = summarize(frame,values['rank'],.5,events)
        nm = normal_metrics(frames['normal'],predictions['normal']['modes'][mode]['rank'],.5,attributes)
        table = event_table(events,frame,values['rank'],.5)
        found = {key(e) for e in table if e['detected']}
        lost = integrity_losses(baseline_events,table,families)
        physical_lost = sorted(physical_required-found)
        row = {'mode':mode, 'historical':metrics, 'new_normal':nm,'lost_baseline_nonphysical_events':lost,
               'lost_primary_corrected_physical_events':physical_lost,
               'lost_from_corrected_members':{str(20261005+i):sorted(key(e) for e in t if e['detected'] and key(e) not in found) for i,t in enumerate(corrected_events)},
               'eligible':mode=='majority' and candidate_gate(metrics,nm,lost,physical_lost,baseline,control['historical'],control['new_normal'])}
        rows.append(row)
        print(json.dumps({'mode':mode,'events':metrics['detected_events'],'fp':metrics['false_positive_windows'],
                         'normal_fp':nm['overall']['false_windows'],'lost_integrity':len(lost),'lost_physical':len(physical_lost),'eligible':row['eligible']}),flush=True)
    write_json(out/'results.json',{'rows':rows,'selected':'majority' if rows[0]['eligible'] else None,'test_used':False,
        'git_commit':_git_commit(),'protocol_sha256':_sha256(root/'docs/RELATIONAL_CONSENSUS_PROTOCOL.md'),
        'health_inputs_sha256':_sha256(h/'inputs.json'),'health_reference_sha256':_sha256(h/'health-reference.json'),
        'verified_score_files':sources,'member_thresholds':thresholds,'truth_hashes':truth_hashes,
        'fresh_member_inference_exact':True,'health_reload_exact':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','health-study','baseline','relationships','dataset','cache','normal-cache','normal-source','raw-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');run(p.parse_args())
