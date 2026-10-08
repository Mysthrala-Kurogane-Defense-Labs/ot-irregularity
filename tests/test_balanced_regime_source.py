from copy import deepcopy
import pytest
from scripts.prepare_balanced_regimes import verify_manifest


def document():
    return dict(master_seed=930101,suite_sha256='suite',uv_lock_sha256='lock',data_license='CC-BY-4.0',
        simulator_version='0.6.0',synthetic=True,generated=True,customer_data=False,generation_complete=True,
        partition_counts={'train':12,'validation':12,'test':0},runs=[dict(run_id=f'{p}-{i}',partition=p,seed=k*12+i) for k,p in enumerate(['train','validation']) for i in range(12)])


def test_balanced_source_disjoint_and_complete():
    d=document();seen=set();assert len(verify_manifest(d,930101,'suite','lock',seen))==24
    with pytest.raises(ValueError,match='leakage'):verify_manifest(d,930101,'suite','lock',seen)


@pytest.mark.parametrize('change',['seed','test','license','count','suite'])
def test_balanced_source_rejects_wrong_contract(change):
    d=deepcopy(document())
    if change=='seed':d['runs'][1]['seed']=d['runs'][0]['seed']
    if change=='test':d['runs'][0]['partition']='test'
    if change=='license':d['data_license']='unknown'
    if change=='count':d['runs'].pop()
    if change=='suite':d['suite_sha256']='changed'
    with pytest.raises(ValueError):verify_manifest(d,930101,'suite','lock',set())
