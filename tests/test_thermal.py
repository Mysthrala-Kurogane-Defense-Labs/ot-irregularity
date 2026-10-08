import joblib
import numpy as np
import polars as pl
import pytest

from ot_irregularity.thermal import ThermalDynamics


def frame():
    load = np.sin(np.arange(60)/8)+2
    temperature = [20.]
    for value in load[1:]:
        temperature.append(temperature[-1]+.2*(20+value*10-temperature[-1]))
    f = pl.DataFrame({'run_id':['r']*60,'asset_id':['a']*60,'window_start':np.arange(60),
                      'window_end':np.arange(60)+1,'is_anomaly':[False]*60,
                      'temp_last':temperature,'temp_mean':temperature,'load_mean':load})
    return f.with_columns([pl.lit(value).alias(role+'_'+name) for role in ('temp','load')
                           for name,value in [('good_ratio',1.),('coverage_ratio',1.),('missing',0.)]])


def test_thermal_target_exclusion_future_safety_and_serialization(tmp_path):
    f=frame();model=ThermalDynamics('temp',['load']).fit(f)
    residual,valid=model.transform(f)
    changed=f.with_columns(pl.when(pl.col('window_start')==30).then(pl.col('temp_last')+5).otherwise(pl.col('temp_last')).alias('temp_last'))
    other,_=model.transform(changed)
    np.testing.assert_array_equal(other[:30],residual[:30])
    assert other[30]-residual[30]==pytest.approx(5)
    assert not valid[0] and valid[1:].all()
    joblib.dump(model,tmp_path/'thermal.joblib')
    np.testing.assert_array_equal(joblib.load(tmp_path/'thermal.joblib').transform(f)[0],residual)


def test_thermal_quality_gaps_and_normal_fit():
    f=frame();model=ThermalDynamics('temp',['load']).fit(f)
    bad=f.with_columns(pl.when(pl.col('window_start')==10).then(0.).otherwise(1.).alias('temp_good_ratio'))
    _,valid=model.transform(bad)
    assert not valid[10] and not valid[11] and valid[12]
    _,valid=model.transform(f.filter(pl.col('window_start')!=10))
    assert not valid[10]  # First surviving window after the gap.
    with pytest.raises(ValueError,match='normal'):
        ThermalDynamics('temp',['load']).fit(f.with_columns(pl.lit(True).alias('is_anomaly')))
    with pytest.raises(ValueError,match='duration'):
        model.transform(f.with_columns((pl.col('window_end')+1).alias('window_end')))
