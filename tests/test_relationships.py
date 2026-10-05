import joblib
import numpy as np
import polars as pl
import pytest

from ot_irregularity.relationships import RelationshipResiduals,previous_indices


def sample():
    values=np.arange(50,dtype=float)
    frame=pl.DataFrame({'run_id':['r']*50,'asset_id':['a']*50,'window_start':list(range(50)),
        'window_end':list(range(1,51)),'is_anomaly':[False]*50,'x_mean':values,'y_mean':values*2})
    return frame.with_columns([pl.lit(v).alias(r+'_'+name) for r in ('x','y') for name,v in [('good_ratio',1.),('coverage_ratio',1.),('missing',0.)]])


def test_history_cannot_cross_runs_assets_or_gaps_and_is_order_independent():
    frame=pl.DataFrame({'run_id':['r','r','r','s','r'],'asset_id':['a','a','a','a','b'],
        'window_start':[1,0,4,2,2],'window_end':[2,1,5,3,3]})
    assert previous_indices(frame).tolist()==[1,-1,-1,-1,-1]
    with pytest.raises(ValueError,match='Duplicate'):previous_indices(pl.concat([frame,frame.head(1)]))


def test_target_is_not_a_current_predictor_and_no_future_input(tmp_path):
    frame=sample();model=RelationshipResiduals(['x','y'],temporal=True).fit(frame)
    before,valid,history=model.transform(frame)
    changed=frame.with_columns(pl.when(pl.col('window_start')==25).then(pl.col('x_mean')+100).otherwise(pl.col('x_mean')).alias('x_mean'))
    after,_,_=model.transform(changed)
    # A current target change cannot change its own predicted current value.
    assert after[25,0]-before[25,0]==pytest.approx(100/model.scaler_.scale_[0])
    np.testing.assert_array_equal(after[:25],before[:25])
    assert valid.all() and not history[0] and history[1:].all()
    joblib.dump(model,tmp_path/'m.joblib');loaded=joblib.load(tmp_path/'m.joblib')
    np.testing.assert_array_equal(loaded.transform(frame)[0],before)


def test_quality_and_normal_fit_guards():
    frame=sample();model=RelationshipResiduals(['x','y']).fit(frame)
    bad=frame.with_columns(pl.when(pl.col('window_start')==10).then(0.).otherwise(1.).alias('x_good_ratio'))
    residual,valid,_=model.transform(bad)
    assert not valid[10] and (residual[10]==0).all()
    with pytest.raises(ValueError,match='normal'):RelationshipResiduals(['x','y']).fit(frame.with_columns(pl.lit(True).alias('is_anomaly')))
    nonfinite=frame.with_columns(pl.when(pl.col('window_start')==10).then(float('inf')).otherwise(pl.col('x_mean')).alias('x_mean'))
    fitted=RelationshipResiduals(['x','y']).fit(nonfinite)
    residual,valid,_=fitted.transform(nonfinite)
    assert np.isfinite(residual).all() and not valid[10]


def test_static_fallback_does_not_bridge_bad_history():
    frame=sample();model=RelationshipResiduals(['x','y'],temporal=True).fit(frame)
    bad=frame.with_columns(pl.when(pl.col('window_start')==10).then(0.).otherwise(1.).alias('x_good_ratio'))
    _,valid,history=model.transform(bad)
    assert not valid[10] and valid[11] and not history[11] and history[12]


def test_candidate_integrity_fails_before_loading_dataset(tmp_path):
    import json
    from scripts.relationship_candidate import infer_records
    (tmp_path/'baseline').mkdir();(tmp_path/'relationships').mkdir()
    (tmp_path/'baseline'/'unexpected').write_text('changed')
    (tmp_path/'candidate.json').write_text(json.dumps({'format_version':1,'baseline_files':{},'relationship_files':{}}))
    with pytest.raises(ValueError,match='integrity'):infer_records(tmp_path,tmp_path/'not-read')
