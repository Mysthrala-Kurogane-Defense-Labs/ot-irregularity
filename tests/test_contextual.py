import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from ot_irregularity.contextual import run_bucket, tail_errors
from ot_irregularity.pipeline import infer, train, _scores, _windows
from ot_irregularity.features import encode_context


def test_tail_errors_preserves_local_deviation_and_additive_explanation():
    e = np.array([[4.,100.,0.,0.],[0.,0.,0.,36.]])
    raw, contribution = tail_errors(e,np.array([1.,100.,1.,1.]),2)
    assert np.allclose(raw,[2.5,18.])
    assert np.allclose(contribution.sum(axis=1),raw)
    assert contribution[0].tolist()==[2.,.5,0.,0.]
    with pytest.raises(ValueError): tail_errors(e,np.zeros(4),2)


def _dataset(tmp_path):
    data=tmp_path/'dataset'
    for partition,prefix in [('train','train'),('validation','val'),('test','test')]:
        rows=[]
        for idx in range(30):
            for asset_class,level in [('pump',10.),('compressor',100.)]:
                for minute in range(5):
                    anomaly=partition!='train' and idx%3==0 and minute==3
                    rows.append({'run_id':f'{prefix}-{idx}','asset_id':asset_class+'-1','asset_class':asset_class,
                        'timestamp':dt.datetime(2026,1,1,tzinfo=dt.timezone.utc)+dt.timedelta(minutes=minute),
                        'tag_id':'t','signal_class':'temperature','value':level+np.sin(idx+minute)+(40 if anomaly else 0),
                        'sampling_interval_ms':60000.,'operating_regime':'steady','quality':'good',
                        'is_anomaly':anomaly,'event_id':f'{prefix}-{idx}-{asset_class}' if anomaly else None})
        path=data/partition
        path.mkdir(parents=True)
        pl.DataFrame(rows).write_parquet(path/'telemetry.parquet')
    return data


def _config(tmp_path):
    cfg=yaml.safe_load((Path(__file__).parents[1]/'configs/model-v0.4-contextual.yaml').read_text())
    cfg['device']='cpu'
    cfg['autoencoder'].update(epochs=3,steps=6,patience=2,log_interval=0)
    cfg['isolation_forest']['n_estimators']=10
    cfg['contextual']['min_partition_windows']=3
    cfg['contextual']['calibration_quantile']=.9
    path=tmp_path/'cfg.yaml'
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_contextual_train_save_load_infer_and_disjoint_runs(tmp_path,monkeypatch,capsys):
    dataset=_dataset(tmp_path);config=_config(tmp_path);out=tmp_path/'model'
    result=train(dataset,config,out)
    assert result['supervised_validation']['irregularity']['pr_auc'] is not None
    metadata=json.loads((out/'training_metadata.json').read_text())
    groups=list(metadata['split_run_ids'].values())
    for i,a in enumerate(groups):
        for b in groups[i+1:]: assert not set(a)&set(b)
    manifest=json.loads((out/'contextual_models.json').read_text())
    assert set(manifest['groups'])=={'pump','compressor'}
    first=infer(out,dataset/'test',tmp_path/'first.jsonl')
    second=infer(out,dataset/'test',tmp_path/'second.jsonl')
    assert first==second
    assert (tmp_path/'first.jsonl').read_bytes()==(tmp_path/'second.jsonl').read_bytes()
    assert all(r['scores']['irregularity']==r['scores']['autoencoder'] for r in first)
    assert all(0<=v<=1 for r in first for v in r['scores'].values())
    assert all(sum(r['feature_contributions'].values())==pytest.approx(1) for r in first)
    frame=pl.read_parquet(dataset/'test'/'telemetry.parquet').with_columns(pl.lit('unknown').alias('asset_class'))
    frame.write_parquet(tmp_path/'unknown.parquet')
    with pytest.raises(ValueError,match='Untrained asset classes'):infer(out,tmp_path/'unknown.parquet',None)
    # Legacy output contract remains usable in evaluate's scoring path.
    w=_windows(dataset/'test',yaml.safe_load(config.read_text()),json.loads((out/'feature_schema.json').read_text()))
    w,_=encode_context(w,json.loads((out/'feature_schema.json').read_text())['context_vocabulary'])
    assert len(_scores(out,w)[4])==len(w)
    from ot_irregularity.cli import main
    unlabeled=pl.read_parquet(dataset/'test'/'telemetry.parquet').drop('is_anomaly','event_id')
    unlabeled.write_parquet(tmp_path/'unlabeled.parquet')
    onset=int(dt.datetime(2026,1,1,0,3,tzinfo=dt.timezone.utc).timestamp()*1e6)
    eventfile=tmp_path/'events.json'
    eventfile.write_text(json.dumps({'events':[{'run_id':'test-0','asset_id':'pump-1','event_id':'overlap-a','start_us':onset,'end_us':onset+1},
                                              {'run_id':'test-0','asset_id':'pump-1','event_id':'overlap-b','start_us':onset,'end_us':onset+1}]}))
    monkeypatch.setattr('sys.argv',['ot-irregularity','evaluate','--model',str(out),'--dataset',str(tmp_path/'unlabeled.parquet'),'--events',str(eventfile)])
    main()
    evaluation=json.loads(capsys.readouterr().out)
    assert evaluation['event_intervals']['irregularity']['event_count']==2
    assert evaluation['irregularity']['positive_samples']==1
    with pytest.raises(FileExistsError):train(dataset,config,out)


def test_contextual_refuses_insufficient_run_split_and_leakage(tmp_path):
    dataset=_dataset(tmp_path);config=_config(tmp_path)
    frame=pl.read_parquet(dataset/'train'/'telemetry.parquet')
    frame.with_columns(pl.lit('only-one-run').alias('run_id')).unique(['run_id','asset_id','timestamp','tag_id']).write_parquet(dataset/'train'/'telemetry.parquet')
    with pytest.raises(ValueError,match='Insufficient'):train(dataset,config,tmp_path/'model')
    frame.write_parquet(dataset/'train'/'telemetry.parquet')
    frame.write_parquet(dataset/'validation'/'telemetry.parquet')
    with pytest.raises(ValueError,match='Data leakage'):train(dataset,config,tmp_path/'leak')


def test_run_partition_is_stable_and_never_row_random():
    assert run_bucket('same-run',5)==run_bucket('same-run',5)
    with pytest.raises(ValueError):run_bucket('r',1)


def test_subminute_train_infer_persists_stride_and_correct_exposure(tmp_path):
    dataset=_dataset(tmp_path);config=_config(tmp_path);out=tmp_path/'short-model'
    origin=dt.datetime(2026,1,1,tzinfo=dt.timezone.utc)
    for path in dataset.glob('*/telemetry.parquet'):
        frame=pl.read_parquet(path)
        frame.with_columns(pl.Series('timestamp',[origin+(t-origin)/2 for t in frame['timestamp']]),
            pl.lit(30000.).alias('sampling_interval_ms')).write_parquet(path)
    cfg=yaml.safe_load(config.read_text());cfg['window']={'size':'30s','stride':'30s'}
    config.write_text(yaml.safe_dump(cfg))
    result=train(dataset,config,out)
    metadata=json.loads((out/'training_metadata.json').read_text())
    assert metadata['stride_seconds']==30
    scores=result['supervised_validation']['irregularity']
    assert scores['false_positives_per_asset_day']==pytest.approx(
        scores['false_positive_windows']/(scores['samples']*30/86400))
    windows=_windows(dataset/'test',cfg,json.loads((out/'feature_schema.json').read_text()))
    assert set((windows['window_end']-windows['window_start']).to_list())=={30_000_000}
    assert len(infer(out,dataset/'test',None))==len(windows)


def test_applicability_uses_exact_roles_not_prefixes():
    from ot_irregularity.contextual import select_columns
    frame=pl.DataFrame({'motor_mean':[1.],'motor_current_mean':[2.],'motor_bad_ratio':[0.]})
    assert select_columns(frame,{'signals_by_asset_class':{'pump':['motor']}},'pump')==['motor_mean','motor_bad_ratio']


def test_quality_error_does_not_claim_a_temperature_value_deviation():
    from ot_irregularity.pipeline import _prediction_records
    w=pl.DataFrame({'run_id':['r'],'asset_id':['a'],'window_start':[0],'window_end':[60_000_000],
                    'temperature_bad_ratio':[.5],'temperature_mean':[20.]})
    result=_prediction_records(w,['temperature_bad_ratio','temperature_mean'],np.array([[4.,0.]]),
                               np.array([1.]),np.array([1.]),np.array([1.]),.95)
    assert result[0]['observations']==['quality_degradation']


def test_contextual_rejects_an_unresolvable_calibration_quantile(tmp_path):
    dataset=_dataset(tmp_path);config=_config(tmp_path)
    cfg=yaml.safe_load(config.read_text());cfg['contextual']['calibration_quantile']=.999999
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError,match='cannot resolve this quantile'):
        train(dataset,config,tmp_path/'model')
