import datetime as dt
import joblib
import numpy as np
import polars as pl
import pytest
from ot_irregularity.sample_thermal import SampleThermal,align_samples,aggregate_residuals


def raw():
    rows=[];temperature=20.
    for i in range(180):
        driver=2+np.sin(i/20);temperature+=.005*(20+driver*10-temperature)
        for role,value in [('temp',temperature),('load',driver)]:
            rows.append({'run_id':'r','asset_id':'a','timestamp':dt.datetime(2026,1,1,tzinfo=dt.timezone.utc)+dt.timedelta(seconds=i*.5),
                         'measurement_role':role,'value':value,'quality':'good','is_anomaly':False})
    return pl.DataFrame(rows)


def test_sample_thermal_causal_context_gap_and_roundtrip(tmp_path):
    f=raw();m=SampleThermal('temp',['load'],initial_context=True).fit(f)
    before=m.transform(f)
    stamp=f['timestamp'][120]
    after=m.transform(f.with_columns(pl.when((pl.col('timestamp')==stamp)&(pl.col('measurement_role')=='temp'))
                                    .then(pl.col('value')+3).otherwise(pl.col('value')).alias('value')))
    np.testing.assert_array_equal(before['residual_rate'][:60],after['residual_rate'][:60])
    assert after['residual_rate'][60]-before['residual_rate'][60]==pytest.approx(6.)
    missing=m.transform(f.filter(pl.col('timestamp')!=stamp))
    assert not missing['available'][60] and missing['available'][61]
    joblib.dump(m,tmp_path/'m.joblib')
    assert joblib.load(tmp_path/'m.joblib').transform(f).equals(before)


def test_sample_thermal_quality_normal_and_duplicate_guards():
    f=raw();m=SampleThermal('temp',['load']).fit(f)
    with pytest.raises(ValueError,match='normal'):m.fit(f.with_columns(pl.lit(True).alias('is_anomaly')))
    with pytest.raises(ValueError,match='Duplicate'):align_samples(pl.concat([f,f.head(1)]),['temp','load'])
    bad=f.with_columns(pl.when(pl.col('timestamp')==f['timestamp'][120]).then(pl.lit('bad')).otherwise(pl.col('quality')).alias('quality'))
    result=m.transform(bad)
    assert not result['available'][60] and not result['available'][61]


def test_residual_aggregation_uses_time_weights_and_requires_coverage():
    samples=pl.DataFrame({'run_id':['r']*120,'asset_id':['a']*120,'time_us':np.arange(120)*500000,
                         'residual_rate':[2.]*120,'available':[True]*120,'dt_seconds':[.5]*120})
    windows=pl.DataFrame({'run_id':['r'],'asset_id':['a'],'window_start':[0],'window_end':[60000000]})
    values,valid=aggregate_residuals(samples,windows,.5)
    assert valid[0] and values[0].tolist()==[2.,2.]
    assert not aggregate_residuals(samples.head(100),windows,.5)[1][0]
