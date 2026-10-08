"""Replay development cooling events with matching Lab source, then zero only their severity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl


def run(args):
    sys.path.insert(0,str(args.lab_src.resolve()))
    from ot_lab import __version__
    from ot_lab.simulation import read_scenario,simulate
    args.output.mkdir(parents=True,exist_ok=False)
    diagnostic=json.loads(args.diagnostic.read_text())
    events={e['event_id']:e for e in diagnostic['thermal_events']}
    results=[]
    for event in events.values():
        batch,run_id=event['run_id'].split('::')
        root=args.raw_root/batch.replace('dev-','raw-')/'validation'/run_id
        metadata=json.loads((root/'run_metadata.json').read_text())
        if metadata['simulator_version']!=__version__ and not args.allow_version_mismatch:
            raise ValueError('Simulator version mismatch; explicit override still requires exact replay')
        scenario=read_scenario(root/'scenario.yaml')
        actual,_,_=simulate(scenario,metadata['seed'])
        columns=['timestamp','asset_id','tag_id','value','quality']
        keys=columns[:3]
        persisted=pl.read_parquet(root/'telemetry.parquet').select(columns).sort(keys)
        replay=actual.select(columns).sort(keys)
        if not persisted.equals(replay):raise ValueError(f'Exact original replay failed: {event["event_id"]}')
        index=int(event['event_id'].rsplit('evt-',1)[1])-1
        if scenario.anomalies[index].type!='cooling_degradation':raise ValueError('Wrong intervention event')
        counter=scenario.model_copy(deep=True)
        counter.anomalies[index].severity=0.
        normal,_,_=simulate(counter,metadata['seed'])
        paired=replay.join(normal.select(columns).rename({'value':'counter_value','quality':'counter_quality'}),on=keys,validate='1:1')
        if len(paired)!=len(replay):raise ValueError('Counterfactual sampling changed')
        paired=paired.with_columns((pl.col('value')-pl.col('counter_value')).alias('effect'),pl.col('timestamp').dt.timestamp('us').alias('us'))
        active=paired.filter((pl.col('asset_id')==event['asset_id'])&(pl.col('us')>=event['start_us'])&(pl.col('us')<=event['end_us']))
        summaries=[]
        for (tag,),part in active.group_by('tag_id'):
            effect=part['effect'].to_numpy()
            if not np.any(effect):continue
            means=paired.filter((pl.col('asset_id')==event['asset_id'])&(pl.col('tag_id')==tag)).group_by((pl.col('us')//60_000_000).alias('minute')).agg(pl.col('effect').mean())
            summaries.append({'tag':tag,'samples':len(effect),'peak_effect_degC':float(np.max(np.abs(effect))),
                              'peak_minute_mean_effect_degC':float(means['effect'].abs().max())})
        nonthermal=paired.filter(~pl.col('tag_id').str.contains('temperature'))
        if nonthermal['effect'].abs().max()!=0:raise ValueError('Intervention changed nonthermal signals')
        if not (paired['quality']==paired['counter_quality']).all():raise ValueError('Intervention changed quality')
        entry={'event_id':event['event_id'],'run_id':event['run_id'],'duration_s':(event['end_us']-event['start_us'])/1e6,
               'severity':scenario.anomalies[index].severity,'parameters':scenario.anomalies[index].parameters,
               'original_replay_exact':True,'declared_simulator_version':metadata['simulator_version'],
               'nonthermal_unchanged':True,'thermal_effects':summaries,
               'scenario_sha256':hashlib.sha256((root/'scenario.yaml').read_bytes()).hexdigest(),
               'telemetry_sha256':hashlib.sha256((root/'telemetry.parquet').read_bytes()).hexdigest()}
        results.append(entry)
        (args.output/'progress.json').write_text(json.dumps({'completed':len(results),'total':len(events),'events':results},indent=2),encoding='utf-8')
        print(json.dumps(entry),flush=True)
    doc={'simulator_version':__version__,'lab_source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.lab_src/'ot_lab').glob('*.py')},
         'events':results,'test_used':False,'intervention':'Only cooling severity set to zero; other parameters, anomalies and random seed retained',
         'limit':'Paired simulator intervention, not evidence of causal diagnosis in real equipment. Minute means include boundary dilution.',
         'attribution':'OT Irregularity Lab, Mysthrala Kurogane Defense Labs, CC BY 4.0'}
    (args.output/'results.json').write_text(json.dumps(doc,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('lab-src','raw-root','diagnostic','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--allow-version-mismatch',action='store_true',help='Permit a different declared version only if exact telemetry replay succeeds')
    run(p.parse_args())
