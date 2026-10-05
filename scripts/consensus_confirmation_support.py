"""Population and acceptance contracts for independent consensus confirmation."""
import json

import numpy as np
from sklearn.metrics import average_precision_score

from scripts.research_physical import PHYSICAL


def source_manifest(source, frozen, seen_run_seeds):
    doc = json.loads((source/'dataset_manifest.json').read_text())
    seed = doc.get('master_seed'); normal = seed == frozen['normal_seed']
    if seed not in [*frozen['mixed_seeds'],frozen['normal_seed']]:
        raise ValueError('Undeclared master seed')
    expected_hash = frozen['normal_suite_sha256'] if normal else frozen['mixed_suite_sha256']
    if (doc.get('suite_sha256') != expected_hash or doc.get('simulator_version') != '0.6.0'
        or doc.get('data_license') != 'CC-BY-4.0' or doc.get('generated') is not True
        or doc.get('synthetic') is not True or doc.get('customer_data') is not False
        or doc.get('uv_lock_sha256') != frozen['lab_uv_lock_sha256']):
        raise ValueError('Source provenance differs from frozen protocol')
    runs = doc['runs']; expected = frozen['normal_runs'] if normal else frozen['mixed_runs_per_seed']
    if len(runs) != expected or len({r['run_id'] for r in runs}) != expected:
        raise ValueError('Incomplete or duplicate run population')
    if any(r['partition'] not in ('train','validation','test') for r in runs):
        raise ValueError('Unexpected source partition')
    seeds = [r['seed'] for r in runs]
    if len(set(seeds)) != len(seeds) or set(seeds) & (set(frozen['excluded_run_seeds']) | seen_run_seeds):
        raise ValueError('Run seed leakage between development or test sources')
    entries = [r for r in runs if r['partition']=='test']
    if len(entries) != (frozen['normal_runs'] if normal else frozen['mixed_test_runs_per_seed']):
        raise ValueError('Unexpected test population')
    seen_run_seeds.update(seeds)
    return doc, entries, normal


def acceptance(mixed, normal, lost_integrity, lost_physical, minimum_hours):
    candidate, control = mixed['majority'], mixed['primary-health']
    mixed_ok = (not lost_integrity and not lost_physical
        and candidate['detected_events'] >= control['detected_events']
        and candidate['false_positive_windows'] <= control['false_positive_windows']
        and candidate['precision'] >= .5 and candidate['event_detection_rate'] >= .5)
    normal_ok = (normal['overall']['majority']['false_windows'] <= normal['overall']['primary-health']['false_windows']
        and len(normal['by_class']) == 4
        and all(v['majority']['asset_hours'] >= minimum_hours and v['majority']['false_windows_per_asset_day'] <= 10
                for v in normal['by_class'].values()))
    return {'mixed_passed':bool(mixed_ok),'normal_passed':bool(normal_ok),'exploratory_confirmation_passed':bool(mixed_ok and normal_ok)}


def paired_intervals(frame, events, control_table, candidate_table, repetitions=2000, seed=20261005):
    runs = sorted(set(frame['run_id']) | {e['run_id'] for e in events})
    families = {(e['run_id'],e['asset_id'],e['event_id']):e['family'] for e in events}
    key = lambda e:(e['run_id'],e['asset_id'],e['event_id'])
    counts = []
    for table in (control_table,candidate_table):
        counts.append(np.array([[sum(e['run_id']==r and (not physical or families[key(e)] in PHYSICAL)
            and (not detected or e['detected']) for e in table)
            for physical in (False,True) for detected in (False,True)] for r in runs]))
    cluster = [np.flatnonzero(frame['run_id'].to_numpy()==r) for r in runs]
    labels = frame['is_anomaly'].to_numpy()
    scores = [frame[c].to_numpy() for c in ('primary-health','majority')]
    rng = np.random.default_rng(seed); deltas = {'event_detection':[],'physical_detection':[],'pr_auc':[],'false_windows_per_asset_day':[]}
    for _ in range(repetitions):
        picks = rng.integers(0,len(runs),len(runs)); a,b = [c[picks].sum(axis=0) for c in counts]
        for name, total, hit in [('event_detection',0,1),('physical_detection',2,3)]:
            if a[total] > 0:
                deltas[name].append(float((b[hit]-a[hit])/a[total]))
        ix = np.concatenate([cluster[i] for i in picks])
        if not len(ix):
            continue
        y = labels[ix]; x,z = [s[ix] for s in scores]
        if np.any(y) and np.any(~y):
            deltas['pr_auc'].append(float(average_precision_score(y,z)-average_precision_score(y,x)))
        hours = float((frame['window_end'].to_numpy()[ix]-frame['window_start'].to_numpy()[ix]).sum()/3.6e9)
        if hours > 0:
            deltas['false_windows_per_asset_day'].append(float((np.sum((z>=.5)&~y)-np.sum((x>=.5)&~y))*24/hours))
    return {name:{'ci95':np.quantile(v,[.025,.975]).tolist() if v else None,'replicates':len(v)} for name,v in deltas.items()}
