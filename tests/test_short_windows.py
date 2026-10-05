import datetime as dt

import polars as pl
import pytest

from ot_irregularity.features import make_windows
from ot_irregularity.pipeline import _duration_minutes


@pytest.mark.parametrize('text,expected',[('30s',.5),('15s',.25),('1m',1),('1.5min',1.5),('1h',60)])
def test_duration_preserves_fractional_minutes(text,expected):
    assert _duration_minutes(text)==expected


@pytest.mark.parametrize('text',['0s','-1m','nans','infs','0.0000001s','garbage'])
def test_invalid_duration_rejected(text):
    with pytest.raises(ValueError): _duration_minutes(text)


def test_subminute_windows_have_correct_boundaries_counts_and_slopes():
    origin=dt.datetime(2025,1,1,tzinfo=dt.timezone.utc)
    data=pl.DataFrame({'run_id':['r']*60,'asset_id':['a']*60,'signal_class':['temperature']*60,
        'timestamp':[origin+dt.timedelta(seconds=s) for s in range(60)],'value':list(range(60)),
        'quality':['good']*60,'sampling_interval_ms':[1000.]*60})
    short=make_windows(data,.5,.5)
    assert len(short)==2
    assert short['window_end'][0]-short['window_start'][0]==30_000_000
    assert short['temperature_sample_count'].to_list()==[30.,30.]
    assert short['temperature_expected_sample_count'].to_list()==[30.,30.]
    assert short['temperature_slope'].to_list()==pytest.approx([1.,1.])
    assert len(make_windows(data,1,1))==1
    with pytest.raises(ValueError,match='No windows'):
        make_windows(data,1,1,tail_policy='complete')
    last=data.tail(1).with_columns(pl.lit(origin+dt.timedelta(seconds=60)).alias('timestamp'))
    complete=make_windows(pl.concat([data,last]),1,1,tail_policy='complete')
    assert len(complete)==1
    assert complete['temperature_sample_count'][0]==60
    with pytest.raises(ValueError,match='tail_policy'):
        make_windows(data,1,1,tail_policy='unknown')


def test_comparison_grid_rejects_missing_duplicate_and_overlapping_subwindows():
    from scripts.research_temporal import complete_minutes, minute_scores
    w=pl.DataFrame({'run_id':['r']*5,'asset_id':['a']*5,
        'window_start':[0,30_000_000,60_000_000,120_000_000,135_000_000],
        'window_end':[30_000_000,60_000_000,90_000_000,150_000_000,165_000_000]})
    grid=complete_minutes(w,30)
    assert grid['window_start'].to_list()==[0]
    assert minute_scores(w,[.1,.8,.2,.3,.4],grid).tolist()==[.8]
    duplicate=pl.concat([w,w.head(1)])
    assert complete_minutes(duplicate,30).is_empty()
    with pytest.raises(ValueError,match='Missing'):
        minute_scores(w,[.1,.8,.2,.3,.4],pl.DataFrame({'run_id':['missing'],'asset_id':['a'],'window_start':[0]}))


@pytest.mark.parametrize('size,stride',[(0,1),(1,0),(-1,1),(float('nan'),1),(1,float('inf'))])
def test_window_generator_rejects_invalid_durations(size,stride):
    with pytest.raises(ValueError,match='Window size'):
        make_windows(pl.DataFrame(),size,stride)


def test_old_artifacts_keep_historical_duration_interpretation(tmp_path):
    import yaml
    from ot_irregularity.pipeline import _model_config
    path=tmp_path/'training_config.yaml'
    cfg={'window':{'size':'30s','stride':'1.5m'}}
    path.write_text(yaml.safe_dump(cfg))
    assert _model_config(tmp_path)['window']=={'size':'1m','stride':'1m'}
    cfg['window_duration_version']=2
    path.write_text(yaml.safe_dump(cfg))
    assert _model_config(tmp_path)['window']==cfg['window']
    cfg['window_duration_version']=999
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='Unsupported window duration version'):_model_config(tmp_path)
