from copy import deepcopy
import json

import polars as pl
import pytest

from scripts.consensus_confirmation_support import source_manifest, acceptance, paired_intervals


def source_fixture(tmp_path):
    frozen = {'mixed_seeds':[1], 'normal_seed':2, 'normal_suite_sha256':'normal', 'mixed_suite_sha256':'mixed',
        'normal_runs':2,'mixed_runs_per_seed':3,'mixed_test_runs_per_seed':1,'lab_uv_lock_sha256':'lock','excluded_run_seeds':[10]}
    doc = {'master_seed':2,'suite_sha256':'normal','simulator_version':'0.6.0','data_license':'CC-BY-4.0',
        'generated':True,'synthetic':True,'customer_data':False,'uv_lock_sha256':'lock',
        'runs':[{'run_id':'test-00001','partition':'test','seed':20},{'run_id':'test-00002','partition':'test','seed':21}]}
    (tmp_path/'dataset_manifest.json').write_text(json.dumps(doc))
    return frozen,doc


def test_confirmation_population_and_seed_leakage(tmp_path):
    frozen,doc = source_fixture(tmp_path)
    seen=set(); _, entries, normal=source_manifest(tmp_path,frozen,seen)
    assert normal and len(entries)==2 and seen=={20,21}
    with pytest.raises(ValueError,match='leakage'):
        source_manifest(tmp_path,frozen,seen)
    for mutate in ('seed','partition','count','license','identity'):
        changed=deepcopy(doc)
        if mutate=='seed': changed['runs'][0]['seed']=10
        if mutate=='partition': changed['runs'][0]['partition']='challenge'
        if mutate=='count': changed['runs'].pop()
        if mutate=='license': changed['data_license']='unknown'
        if mutate=='identity': changed['runs'][1]['run_id']='test-00001'
        (tmp_path/'dataset_manifest.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError): source_manifest(tmp_path,frozen,set())


@pytest.mark.parametrize('candidate_name', ['majority', 'class-ensemble'])
def test_confirmation_gate_requires_both_populations_and_exact_events(candidate_name):
    m={'detected_events':8,'false_positive_windows':2,'precision':.8,'event_detection_rate':.8}
    mixed={candidate_name:m,'primary-health':m.copy()}
    normal={'overall':{candidate_name:{'false_windows':10},'primary-health':{'false_windows':12}},
        'by_class':{g:{candidate_name:{'asset_hours':170,'false_windows_per_asset_day':4}} for g in ('CNC','PUMP','COMPRESSOR','CONVEYOR')}}
    assert acceptance(mixed,normal,[],[],168,candidate_name=candidate_name)['exploratory_confirmation_passed']
    assert not acceptance(mixed,normal,['lost'],[],168,candidate_name=candidate_name)['mixed_passed']
    assert not acceptance(mixed,normal,[],['physical'],168,candidate_name=candidate_name)['mixed_passed']
    changed=deepcopy(normal); changed['by_class']['CNC'][candidate_name]['asset_hours']=167
    assert not acceptance(mixed,changed,[],[],168,candidate_name=candidate_name)['normal_passed']
    changed=deepcopy(normal); changed['by_class']['PUMP'][candidate_name]['false_windows_per_asset_day']=11
    assert not acceptance(mixed,changed,[],[],168,candidate_name=candidate_name)['normal_passed']
    changed=deepcopy(mixed); changed[candidate_name]['detected_events']=7
    assert not acceptance(changed,normal,[],[],168,candidate_name=candidate_name)['mixed_passed']


@pytest.mark.parametrize('candidate_name', ['majority', 'class-ensemble'])
def test_confirmation_bootstrap_identical_controls_include_no_window_events(candidate_name):
    frame=pl.DataFrame({'run_id':['a','a','b','b'],'is_anomaly':[True,False,True,False],
        'primary-health':[.8,.1,.7,.2],candidate_name:[.8,.1,.7,.2],
        'window_start':[0,60,0,60],'window_end':[60,120,60,120]}).with_columns(
            pl.col('window_start')*1_000_000,pl.col('window_end')*1_000_000)
    events=[{'run_id':r,'asset_id':'p','event_id':r+'/e','family':'bearing_degradation'} for r in ('a','b','no-window')]
    table=[{**e,'detected':e['run_id']!='no-window'} for e in events]
    result=paired_intervals(frame,events,table,table,repetitions=50,candidate_name=candidate_name)
    assert all(v['ci95']==[0.,0.] for v in result.values())
    assert result['event_detection']['replicates']==50
