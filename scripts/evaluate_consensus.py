"""Frozen, resumable mixed and long-normal consensus confirmation."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import polars as pl
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.consensus_confirmation_support import source_manifest, acceptance, paired_intervals
from scripts.freeze_consensus import runtime_hashes
from scripts.health_cadence import declared_cadence, attach_declared_cadence
from scripts.evaluate_normal_exposure import verify_run
from scripts.normal_exposure_metrics import alarm_counts, summarize as normal_summary
from scripts.prepare_otlab import _sha256, _annotate, _epoch_us, normalize_observations
from scripts.research_detection import write_json
from scripts.research_physical import summarize, PHYSICAL
from scripts.research_normal_coverage import event_table, integrity_losses
from scripts.research_telemetry_health import align_scores
from scripts.relationship_operating_points import residual_scores
from scripts.validate_physical import hashes
from ot_irregularity.pipeline import _model_config, _windows, _scores
from ot_irregularity.telemetry_health import KEYS, extract_health, TelemetryHealthReference
from ot_irregularity.consensus import consensus_scores
from ot_irregularity.class_ensemble import ClassMagnitudeEnsemble
from scripts.research_class_ensemble import member_errors

MODELS = ['primary-health','majority','v04','v05']
CLASSES = {'CNC','COMPRESSOR','CONVEYOR','PUMP'}


def run(args):
    root = Path(__file__).resolve().parents[1]; out = args.output
    frozen = json.loads((out/'frozen.json').read_text()); freeze_hash = _sha256(out/'frozen.json')
    candidate_name = frozen['decision']
    if candidate_name not in ('majority', 'class-ensemble'):
        raise ValueError('Unsupported frozen candidate')
    models = MODELS + (['class-ensemble'] if candidate_name == 'class-ensemble' else [])
    class_model = None
    if candidate_name == 'class-ensemble':
        if _sha256(out/'class-reference.json') != frozen['class_reference_sha256']:
            raise ValueError('Frozen class reference changed')
        class_model = ClassMagnitudeEnsemble.load(out/'class-reference.json')
    if runtime_hashes(root) != frozen['runtime_files'] or _sha256(root/frozen.get('protocol_file','docs/CONSENSUS_CONFIRMATION_PROTOCOL.md')) != frozen['protocol_sha256']:
        raise ValueError('Frozen execution code/protocol changed')
    if hashes(args.baseline) != frozen['baseline_files'] or _sha256(args.health_reference) != frozen['health_reference_sha256']:
        raise ValueError('Frozen baseline or health changed')
    for m in frozen['members']:
        if hashes(args.relationships/m['directory']) != m['files']:
            raise ValueError('Frozen relational member changed')
    manifests = []; seen_seeds = set(); master_seeds = set()
    for source in args.sources:
        doc, entries, normal = source_manifest(source,frozen,seen_seeds)
        if doc['master_seed'] in master_seeds:
            raise ValueError('Duplicate master seed')
        master_seeds.add(doc['master_seed']); manifests.append((source,doc,entries,normal))
    if master_seeds != set([*frozen['mixed_seeds'],frozen['normal_seed']]):
        raise ValueError('All declared sources are required')
    cfg = _model_config(args.baseline); cfg['window']['tail_policy']='complete'
    schema = json.loads((args.baseline/'feature_schema.json').read_text())
    health_model = TelemetryHealthReference.load(args.health_reference)
    profiles = yaml.safe_load((root/'configs/normal-exposure-v05.yaml').read_text())['generation']['regime_profiles']
    availability_columns = ['health_score','quality_score','sampling_score'] + (['class_margin'] if class_model is not None else [])
    records = []; mixed_frames = []; events = []; completed = 0; availability = {}
    total = sum(len(e) for _,_,e,_ in manifests)
    for source, manifest, entries, normal in manifests:
        source_hash = _sha256(source/'dataset_manifest.json')
        for entry in entries:
            folder, period, source_files = declared_cadence(source,entry,allow_test=True)
            if _sha256(folder/'ground_truth.json') != entry['ground_truth_sha256']:
                raise ValueError('Ground truth hash changed')
            source_files['ground_truth.json'] = entry['ground_truth_sha256']
            truth = json.loads((folder/'ground_truth.json').read_text())
            if truth['run_id'] != entry['run_id']:
                raise ValueError('Truth identity mismatch')
            if normal:
                _, _, scenario, _ = verify_run(source,entry)
            else:
                scenario = yaml.safe_load((folder/'scenario.yaml').read_text())
            prefix = 'seed-'+str(manifest['master_seed'])+'::'
            identity = prefix+entry['run_id']
            run_dir = out/'runs'/str(manifest['master_seed'])/entry['run_id']; run_dir.mkdir(parents=True,exist_ok=True)
            marker = run_dir/'record.json'; pred_path = run_dir/'predictions.parquet'
            expected = {'freeze_sha256':freeze_hash,'source_manifest_sha256':source_hash,'source_files':source_files}
            if marker.exists():
                record = json.loads(marker.read_text())
                if record['inputs'] != expected or record['predictions_sha256'] != _sha256(pred_path) or record['run_id'] != identity or record['normal'] != normal:
                    raise ValueError('Completed run provenance changed')
                prediction = pl.read_parquet(pred_path)
            else:
                raw = pl.read_parquet(folder/'telemetry.parquet')
                if set(raw['run_id'].unique()) != {entry['run_id']}:
                    raise ValueError('Raw run identity mismatch')
                annotated = _annotate(normalize_observations(raw),truth['events']).with_columns(
                    (pl.lit(prefix)+pl.col('run_id')).alias('run_id'),
                    pl.when(pl.col('event_id').is_not_null()).then(pl.lit(identity+'/')+pl.col('event_id')).otherwise(None).alias('event_id'))
                with tempfile.TemporaryDirectory(prefix='consensus-confirmation-') as temporary:
                    path = Path(temporary)/'telemetry.parquet'; annotated.write_parquet(path)
                    windows = _windows(path,cfg,schema)
                if not len(windows) or (normal and (set(windows['asset_class'].unique()) != CLASSES or windows['is_anomaly'].any())):
                    raise ValueError('Missing windows/classes or unexpected normal labels')
                _, _, ae, isolation, base, bth = _scores(args.baseline,windows)
                raw_errors = None
                if class_model is not None:
                    raw_errors, member_cdfs = member_errors(args.relationships,frozen['members'],windows,args.device)
                    members = np.maximum(base[:,None],member_cdfs)
                else:
                    members = np.column_stack([np.maximum(base,residual_scores(args.relationships/m['directory'],windows,args.device)['ae']) for m in frozen['members']])
                thresholds = np.array([m['threshold'] for m in frozen['members']])
                features = extract_health(attach_declared_cadence(annotated,period),windows,schema['signals_by_asset_class'])
                health, _ = health_model.score(features,repetition=False); health = align_scores(windows,health)
                hv = health['health_score'].to_numpy()
                majority = consensus_scores(members,thresholds,hv,health_threshold=frozen['health_threshold'])
                margin = np.maximum(members[:,0]/thresholds[0],np.nan_to_num(hv,nan=0)/frozen['health_threshold'])
                prediction = windows.select(KEYS+['is_anomaly','event_id','event_start_us']).with_columns(
                    pl.Series('baseline_autoencoder',ae),pl.Series('baseline_isolation_forest',isolation),
                    pl.Series('v04',base/bth/(1+base/bth)),pl.Series('v05',(members[:,0]/thresholds[0])/(1+members[:,0]/thresholds[0])),
                    pl.Series('primary-health',margin/(1+margin)),pl.Series('majority',majority['rank']),
                    *[pl.Series('member_'+str(m['seed']),members[:,i]) for i,m in enumerate(frozen['members'])])
                if class_model is not None:
                    class_margin, class_available = class_model.score(raw_errors,windows['asset_class'].to_numpy())
                    combined = np.maximum.reduce([base/thresholds[0],np.nan_to_num(class_margin,nan=0),np.nan_to_num(hv,nan=0)/frozen['health_threshold']])
                    prediction = prediction.with_columns(
                        pl.Series('class-ensemble',combined/(1+combined)),
                        pl.Series('class_margin',class_margin,nan_to_null=True),
                        pl.Series('class_available',class_available),
                        *[pl.Series('raw_'+str(m['seed']),raw_errors[:,i],nan_to_null=True) for i,m in enumerate(frozen['members'])])
                prediction = prediction.join(health,on=KEYS,maintain_order='left')
                prediction.write_parquet(pred_path)
                record = {'inputs':expected,'run_id':identity,'normal':normal,'predictions_sha256':_sha256(pred_path),
                    'availability':{c:len(prediction)-prediction[c].null_count() for c in availability_columns},
                    'windows':len(prediction)}
                write_json(marker,record)
            population = 'normal' if normal else 'mixed'
            for (group,), gf in prediction.partition_by('asset_class',as_dict=True).items():
                counts = availability.setdefault(population,{}).setdefault(group,{'windows':0,**{c:0 for c in availability_columns}})
                counts['windows'] += len(gf)
                for c in availability_columns:
                    counts[c] += len(gf)-gf[c].null_count()
            if normal:
                profile = next(p['profile_id'] for p in profiles if p['shift_pattern']==scenario['shift_pattern'])
                for (group,), frame in prediction.partition_by('asset_class',as_dict=True).items():
                    records.append({'run_id':identity,'asset_class':group,'profile':profile,'sampling_interval_ms':period,
                        'expose_operating_regime':scenario['expose_operating_regime'],
                        'models':{m:alarm_counts(frame,frame[m].to_numpy()>=.5) for m in models}})
            else:
                mixed_frames.append(prediction)
                events.extend({'run_id':identity,'asset_id':e['asset_id'],'event_id':identity+'/'+e['event_id'],'family':e['type'],
                               'start_us':_epoch_us(e['start']),'end_us':_epoch_us(e['end'])} for e in truth['events'])
            completed += 1
            write_json(out/'progress.json',{'phase':'evaluate','completed_runs':completed,'total_runs':total,'run_id':identity})
    write_json(out/'progress.json',{'phase':'aggregate','completed_runs':completed,'total_runs':total})
    frame = pl.concat(mixed_frames,how='diagonal_relaxed')
    metrics = {m:summarize(frame,frame[m].to_numpy(),.5,events) for m in models}
    tables = {m:event_table(events,frame,frame[m].to_numpy(),.5) for m in models}
    key = lambda e:(e['run_id'],e['asset_id'],e['event_id'])
    families = {key(e):e['family'] for e in events}
    found = {key(e) for e in tables[candidate_name] if e['detected']}
    lost = integrity_losses(tables['v04'],tables[candidate_name],families)
    physical_lost = sorted(key(e) for e in tables['primary-health'] if e['detected'] and families[key(e)] in PHYSICAL and key(e) not in found)
    normal_result = {'overall':normal_summary(records,models),
        'by_class':{g:normal_summary([r for r in records if r['asset_class']==g],models) for g in sorted(CLASSES)},
        'strata':{k:{str(v):normal_summary([r for r in records if r[k]==v],models) for v in sorted({r[k] for r in records})}
                  for k in ('profile','sampling_interval_ms','expose_operating_regime')}}
    result = {'mixed':metrics,'normal':normal_result,'lost_baseline_integrity':lost,'lost_primary_physical':physical_lost,
        'gate':acceptance(metrics,normal_result,lost,physical_lost,frozen['minimum_asset_hours_per_class'],candidate_name=candidate_name),
        'paired_mixed_intervals':paired_intervals(frame,events,tables['primary-health'],tables[candidate_name],frozen['bootstrap_repetitions'],frozen['bootstrap_seed'],candidate_name=candidate_name),
        'freeze_sha256':freeze_hash,'evaluated_runs':completed,'mixed_windows':len(frame),'original_events':len(events),'selection_after_test':False,'component_availability':availability,
        'candidate':candidate_name,
        'lost_primary_events':sorted(key(e) for e in tables['primary-health'] if e['detected'] and key(e) not in found),
        'sources':[{'master_seed':d['master_seed'],'manifest_sha256':_sha256(s/'dataset_manifest.json'),'test_runs':len(e)} for s,d,e,_ in manifests]}
    write_json(out/'events.json',events); write_json(out/'event_decisions.json',tables)
    write_json(out/'normal_records.json',records); write_json(out/'results.json',result)
    write_json(out/'progress.json',{'phase':'completed','gate':result['gate'],'evaluated_runs':completed})
    print(json.dumps(result['gate']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','baseline','relationships','health-reference'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sources',type=Path,nargs='+',required=True); p.add_argument('--device',default='cuda:0');run(p.parse_args())
