import numpy as np
import pytest
from ot_irregularity.class_ensemble import ClassMagnitudeEnsemble


def test_class_calibration_scales_each_member_and_class_and_roundtrips(tmp_path):
    raw=np.arange(1,31,dtype=float).reshape(10,3)
    classes=np.array(['PUMP']*10+['CNC']*10)
    values=np.vstack([raw,raw*100]); labels=np.zeros(20,dtype=bool)
    model=ClassMagnitudeEnsemble('mean',.9,10).fit(values,classes,labels)
    margin,available=model.score(values,classes)
    np.testing.assert_allclose(margin[:10],margin[10:])
    assert available.all() and np.all(margin < 1) # strict nextafter threshold
    path=tmp_path/'model.json';model.save(path)
    loaded=ClassMagnitudeEnsemble.load(path)
    np.testing.assert_array_equal(loaded.score(values,classes)[0],margin)
    probe=values.copy();probe[0,0]=np.nan
    scored,availability=loaded.score(probe,classes)
    assert np.isnan(scored[0]) and not availability[0]
    with pytest.raises(ValueError,match='Untrained'):loaded.score(raw,['UNKNOWN']*10)


def test_class_calibration_requires_normal_sufficient_informative_data():
    values=np.ones((3,3));classes=['PUMP']*3
    with pytest.raises(ValueError,match='normal'):ClassMagnitudeEnsemble(minimum_windows=2).fit(values,classes,[False,True,False])
    with pytest.raises(ValueError,match='Insufficient'):ClassMagnitudeEnsemble().fit(values,classes,[False]*3)
    with pytest.raises(ValueError,match='Uninformative'):ClassMagnitudeEnsemble(minimum_windows=2).fit(values*0,classes,[False]*3)
    with pytest.raises(ValueError,match='nonnegative'):ClassMagnitudeEnsemble(minimum_windows=2).fit(-values,classes,[False]*3)


def test_mean_keeps_extreme_member_magnitude_that_median_discards():
    values=np.ones((10,3)); labels=np.zeros(10,dtype=bool); classes=['PUMP']*10
    mean=ClassMagnitudeEnsemble('mean',.9,10).fit(values,classes,labels)
    median=ClassMagnitudeEnsemble('median',.9,10).fit(values,classes,labels)
    assert mean.score([[9,0,0]],['PUMP'])[0][0] > 1
    assert median.score([[9,0,0]],['PUMP'])[0][0] == 0
