"""Reconcile frozen predictions against every original event interval, without refitting."""
import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.research_detection import write_json
from ot_irregularity.metrics import evaluate_event_intervals


def audit(output, sources):
    predictions=pl.read_parquet(output/'holdout_predictions.parquet')
    report=json.loads((output/'holdout_results.json').read_text())
    events=[];families={};source_hashes={}
    for source in sources:
        for truth in sorted((source/'test').glob('*/ground_truth.json')):
            document=json.loads(truth.read_text())
            source_hashes[source.name+'/'+truth.relative_to(source).as_posix()]=hashlib.sha256(truth.read_bytes()).hexdigest()
            for e in document['events']:
                eid=source.name+'::'+document['run_id']+'/'+e['event_id']
                def micros(text):return int(dt.datetime.fromisoformat(text.replace('Z','+00:00')).timestamp()*1e6)
                events.append({'run_id':source.name+'::'+document['run_id'],'asset_id':e['asset_id'],
                               'event_id':eid,'start_us':micros(e['start']),'end_us':micros(e['end'])})
                families[eid]=e.get('type',e.get('event_type',e.get('family','unknown')))
    write_json(output/'events.json',{'schema_version':'1','events':events})
    rows={};by_family=defaultdict(dict)
    for row in report['rows']:
        name=row['id']
        result=evaluate_event_intervals(events,predictions[name].to_numpy(),row['metrics']['threshold'],
            predictions['run_id'].to_numpy(),predictions['asset_id'].to_numpy(),
            predictions['window_start'].to_numpy(),predictions['window_end'].to_numpy())
        for family in sorted(set(families.values())):
            chosen=[e for e in result['events'] if families[e['event_id']]==family]
            by_family[family][name]={'events':len(chosen),'detected':sum(e['detected'] for e in chosen)}
        rows[name]=result
    # Paired cluster bootstrap on complete run-level event denominators.
    names=list(rows);run_ids=sorted(set(predictions['run_id'].to_list())|{e['run_id'] for e in events})
    def counts(name):
        return np.array([[sum(e['run_id']==run for e in rows[name]['events']),
                          sum(e['run_id']==run and e['detected'] for e in rows[name]['events'])] for run in run_ids])
    base,candidate=counts(names[0]),counts(names[1]);rng=np.random.default_rng(20261005);delta=[]
    for _ in range(2000):
        picks=rng.integers(0,len(run_ids),len(run_ids));a=base[picks].sum(axis=0);b=candidate[picks].sum(axis=0)
        if a[0] and b[0]:delta.append(b[1]/b[0]-a[1]/a[0])
    result={'event_count':len(events),'models':rows,'by_family':dict(by_family),
            'paired_event_detection_delta_ci95':list(map(float,np.quantile(delta,[.025,.975]))),
            'bootstrap_replicates':len(delta),'truth_hashes':source_hashes,
            'metric_policy':'All original intervals, including multiple events in a window; no refitting or threshold change'}
    write_json(output/'event_interval_audit.json',result)
    # Reconcile the live table with the authoritative interval denominator.
    state_path=output/'state.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
        for row in state.get('ranking',[]):
            interval=rows.get(row['id'])
            if interval:
                row['event_detection_rate']=interval['event_detection_rate']
                row['latency_seconds']=interval['mean_detection_latency_seconds']
        state['evaluation_note']=f"Holdout independiente: {len(events)} intervalos originales; modelos y umbrales congelados antes del test."
        write_json(state_path,state)
    print(json.dumps({k:{key:value for key,value in v.items() if key!='events'} for k,v in rows.items()},indent=2))
    print(json.dumps(dict(by_family),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sources',type=Path,nargs='+',required=True)
    a=p.parse_args();audit(a.output,a.sources)
