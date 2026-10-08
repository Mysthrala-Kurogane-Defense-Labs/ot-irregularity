import json
import numpy as np
import pytest
from ot_irregularity.context_calibration import ContextMagnitudeCalibration


def fitted():
    return ContextMagnitudeCalibration(.9,4).fit([1,2,3,4,5],['P']*5,['mixed']*4+['IDLE'],np.zeros(5,dtype=bool))


def test_context_reference_fallback_and_roundtrip(tmp_path):
    m=fitted(); values=[4,4,4,4,np.nan]; regimes=['mixed','IDLE','unknown','NEW','mixed']
    score,route=m.score(values,['P']*5,regimes)
    assert score[0]<1 and np.array_equal(score[1:4],[4,4,4]) and np.isnan(score[4])
    assert route.tolist()==['context_reference','class_fallback_insufficient_reference','class_fallback_unknown_context','class_fallback_unseen_context','relational_unavailable']
    p=tmp_path/'model.json';m.save(p);loaded=ContextMagnitudeCalibration.load(p)
    assert np.array_equal(loaded.score(values,['P']*5,regimes)[0],score,equal_nan=True)
    assert m.score([4],['P'],[None])[1][0]=='class_fallback_unknown_context'


def test_normal_guards_and_untrained_class():
    with pytest.raises(ValueError,match='normal'): ContextMagnitudeCalibration().fit([1],['P'],['mixed'],[True])
    with pytest.raises(ValueError,match='Untrained'): fitted().score([1],['Q'],['mixed'])
    with pytest.raises(ValueError,match='Aligned'): fitted().score([1],['P'],[])
    with pytest.raises(ValueError,match='finite'): fitted().score([np.inf],['P'],['mixed'])


def test_corrupt_serialized_threshold_rejected(tmp_path):
    p=tmp_path/'model.json'; fitted().save(p);d=json.loads(p.read_text());d['groups'][1]['threshold']=-1;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='threshold'): ContextMagnitudeCalibration.load(p)


def test_zero_reference_uses_class_fallback():
    m=ContextMagnitudeCalibration(.9,2).fit([0,0],['P','P'],['mixed']*2,np.zeros(2,dtype=bool))
    assert m.score([1],['P'],['mixed'])[0][0]==1
