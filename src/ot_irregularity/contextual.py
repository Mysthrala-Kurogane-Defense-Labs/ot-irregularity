"""Opt-in class-specific AE/IF models with independently calibrated residual tails."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler

from .features import feature_columns, feature_signal
from .metrics import evaluate_scores
from .models import ae_errors, load_ae, train_ae


def run_bucket(value, modulo):
    """Stable whole-run allocation, independent of row order and anomaly labels."""
    if modulo < 2:
        raise ValueError('Split modulo must be at least 2')
    return int(hashlib.sha256(str(value).encode()).hexdigest()[:8], 16) % modulo


def tail_errors(errors, scale, top_k):
    """Return score and additive contributions in normalized squared-error units."""
    if top_k < 1 or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('top_k and residual scales must be positive and finite')
    values = np.asarray(errors, dtype=float) / scale
    k = min(top_k, values.shape[1])
    # Stable ordering makes ties reproducible, including explanations.
    indices = np.argsort(values, axis=1, kind='stable')[:, -k:]
    contributions = np.zeros_like(values)
    np.put_along_axis(contributions, indices, np.take_along_axis(values, indices, axis=1) / k, axis=1)
    return contributions.sum(axis=1), contributions


def _weights(cfg):
    ens = cfg.get('ensemble', {})
    a, b = float(ens.get('autoencoder_weight', .5)), float(ens.get('isolation_weight', .5))
    if not np.isfinite([a,b]).all() or min(a,b) < 0 or a+b <= 0:
        raise ValueError('Ensemble weights must be finite, nonnegative and sum to a positive value')
    return a/(a+b), b/(a+b)


def _dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def select_columns(frame, schema, group, compact=False):
    roles = schema['signals_by_asset_class'][group]
    cols = [c for c in feature_columns(frame) if feature_signal(c) in roles]
    if compact:
        suffixes = ('_mean','_std','_range','_delta','_slope','_coverage_ratio','_uncertain_ratio','_bad_ratio','_missing')
        cols = [c for c in cols if c.endswith(suffixes)]
    if not cols:
        raise ValueError(f'No applicable continuous features for asset_class={group!r}')
    return cols


def score_contextual(modeldir, windows):
    from .pipeline import cdf_calibrate
    root = Path(modeldir)
    manifest = json.loads((root/'contextual_models.json').read_text(encoding='utf-8'))
    schema = json.loads((root/'feature_schema.json').read_text(encoding='utf-8'))
    cfg = yaml.safe_load((root/'training_config.yaml').read_text(encoding='utf-8'))
    if 'asset_class' not in windows.columns or windows['asset_class'].null_count():
        raise ValueError('Contextual inference requires a non-null asset_class')
    known = set(manifest['groups'])
    unknown = set(windows['asset_class'].to_list())-known
    if unknown:
        raise ValueError(f'Untrained asset classes: {sorted(unknown)}; train a matching class model first')
    cols = schema['features']
    errors = np.zeros((len(windows),len(cols)))
    sa, si = np.zeros(len(windows)), np.zeros(len(windows))
    for group in sorted(set(windows['asset_class'].to_list())):
        info = manifest['groups'][group]
        # Group directory names are content-derived, never raw external identifiers.
        path = root/'groups'/hashlib.sha256(group.encode()).hexdigest()[:16]
        scaler = joblib.load(path/'scaler.joblib')
        iso = joblib.load(path/'isolation_forest.joblib')
        ae = load_ae(path/'autoencoder.pt',cfg.get('device','auto'))
        mask = windows['asset_class'].to_numpy()==group
        x = scaler.transform(windows.select(info['features']).to_numpy().astype(float)[mask])
        raw, contribution = tail_errors(ae_errors(ae,x),np.array(info['residual_scale']),manifest['top_k'])
        for j,name in enumerate(info['features']):
            errors[mask,cols.index(name)] = contribution[:,j]
        sa[mask] = cdf_calibrate(info['autoencoder_reference'],raw)
        si[mask] = cdf_calibrate(info['isolation_reference'],-iso.score_samples(x))
    a,b = _weights(cfg)
    return cols,errors,sa,si,a*sa+b*si,float(manifest['decision_threshold'])


def train_contextual_windows(tr, va, cfg, out, schema, metadata):
    from .pipeline import _record_progress, cdf_calibrate
    out = Path(out)
    options = cfg.get('contextual',{})
    if cfg.get('split_group','run_id') != 'run_id':
        raise ValueError('Contextual training currently requires split_group: run_id')
    if 'asset_class' not in tr.columns or 'asset_class' not in va.columns or tr['asset_class'].null_count() or va['asset_class'].null_count():
        raise ValueError('Contextual training requires non-null asset_class in both partitions')
    if 'is_anomaly' not in va.columns or va['is_anomaly'].null_count():
        raise ValueError('Contextual calibration requires explicit, non-null normal/anomaly validation labels')
    em,cm = int(options.get('early_stop_modulo',5)),int(options.get('calibration_modulo',2))
    fit_mask = np.array([run_bucket(r,em)!=0 for r in tr['run_id']])
    cal_mask = np.array([run_bucket(r,cm)==0 for r in va['run_id']])
    fit,stop = tr.filter(pl.Series(fit_mask)),tr.filter(pl.Series(~fit_mask))
    cal = va.filter(pl.Series(cal_mask)).filter(~pl.col('is_anomaly'))
    dev = va.filter(pl.Series(~cal_mask))
    floor = float(options.get('residual_floor',1e-8))
    q, rq = float(options.get('calibration_quantile',.99)),float(options.get('residual_quantile',.95))
    top_k = int(options.get('top_k',4))
    if not 0 < q < 1 or not 0 < rq < 1 or not np.isfinite(floor) or floor<=0 or top_k<1:
        raise ValueError('Invalid contextual quantiles, residual_floor or top_k')
    a,b = _weights(cfg)
    min_windows = int(options.get('min_partition_windows',20))
    if min_windows < 1:
        raise ValueError('min_partition_windows must be positive')
    groups = sorted(tr['asset_class'].unique().to_list())
    if set(va['asset_class'].unique().to_list())-set(groups):
        raise ValueError('Validation contains untrained asset classes')
    if dev.is_empty():
        raise ValueError('Whole-run split left no development windows')
    scaler_name = cfg.get('scaler',{}).get('type','robust')
    if scaler_name not in ('robust','standard'):
        raise ValueError('scaler.type must be robust or standard')
    manifest = {'version':1,'group_by':'asset_class','top_k':top_k,'groups':{},
                'calibration_quantile':q,'contributions':'top-k normalized squared reconstruction error / k'}
    calibration_scores=[]
    for group in groups:
        subsets = [f.filter(pl.col('asset_class')==group) for f in (fit,stop,cal)]
        if any(len(f)<min_windows for f in subsets):
            raise ValueError(f'Insufficient fit/early-stop/calibration windows for {group!r}: {[len(f) for f in subsets]}; need {min_windows} each')
        cols = select_columns(tr,schema,group,options.get('compact_features',False))
        arrays = [f.select(cols).to_numpy().astype(float) for f in subsets]
        if any(not np.isfinite(x).all() for x in arrays):
            raise ValueError('Training features must be finite')
        scaler = (RobustScaler() if scaler_name=='robust' else StandardScaler()).fit(arrays[0])
        x,xe,xc = [scaler.transform(arr) for arr in arrays]
        groupdir = out/'groups'/hashlib.sha256(group.encode()).hexdigest()[:16]
        groupdir.mkdir(parents=True,exist_ok=True)
        _record_progress(out/'training_progress.jsonl','contextual_group',asset_class=group,features=len(cols))
        ae = train_ae(x,xe,{**cfg.get('autoencoder',{}),'device':cfg.get('device','auto')},int(cfg.get('seed',42)),groupdir/'autoencoder.pt',out/'training_progress.jsonl')
        ic=cfg.get('isolation_forest',{})
        iso = IsolationForest(n_estimators=int(ic.get('n_estimators',300)),max_samples=ic.get('max_samples','auto'),
                              contamination=ic.get('contamination','auto'),max_features=ic.get('max_features',1.),
                              random_state=int(ic.get('random_state',cfg.get('seed',42))),n_jobs=2).fit(x)
        scale = np.maximum(np.quantile(ae_errors(ae,xe),rq,axis=0),floor)
        reference,_ = tail_errors(ae_errors(ae,xc),scale,top_k)
        isolation_reference = -iso.score_samples(xc)
        calibration_scores.extend(a*cdf_calibrate(reference,reference)+b*cdf_calibrate(isolation_reference,isolation_reference))
        joblib.dump(scaler,groupdir/'scaler.joblib')
        joblib.dump(iso,groupdir/'isolation_forest.joblib')
        manifest['groups'][group] = {'features':cols,'residual_scale':scale.tolist(),'autoencoder_reference':reference.tolist(),
            'isolation_reference':isolation_reference.tolist(),'training':ae.training_metadata,
            'fit_windows':len(x),'early_stop_windows':len(xe),'calibration_windows':len(xc)}
    threshold = float(np.nextafter(np.quantile(calibration_scores,q,method='higher'),np.inf))
    if threshold > 1:
        raise ValueError('Calibration cannot resolve this quantile without disabling every alert; add normal calibration runs or lower calibration_quantile')
    manifest['decision_threshold'] = threshold
    _dump(out/'contextual_models.json',manifest)
    schema['features'] = sorted({c for info in manifest['groups'].values() for c in info['features']})
    schema['version']='2-contextual'
    _dump(out/'feature_schema.json',schema)
    (out/'training_config.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
    _,_,sa,si,se,_ = score_contextual(out,dev)
    args={'asset_ids':dev['asset_id'].to_numpy(),'window_seconds':metadata['stride_seconds'],
          'event_ids':dev['event_id'].to_numpy() if 'event_id' in dev.columns else None,
          'event_start_us':dev['event_start_us'].to_list() if 'event_start_us' in dev.columns else None,
          'window_end_us':dev['window_end'].to_numpy()}
    result={'validation_windows':len(dev),'calibration_normal_windows':len(cal),'decision_threshold':threshold,
            'supervised_validation':{k:evaluate_scores(dev['is_anomaly'].to_numpy(),s,threshold,**args)
                                     for k,s in [('autoencoder',sa),('isolation_forest',si),('irregularity',se)]}}
    _dump(out/'metrics.json',result)
    _dump(out/'thresholds.json',{'calibration':'per-class empirical normal CDF, then pooled normal ensemble quantile',
        'decision_threshold':threshold,'quantile':q,'probability_of_fault':False})
    _dump(out/'training_metadata.json',{**metadata,'feature_schema_version':'2-contextual',
        'seed':cfg.get('seed',42),'scaler':scaler_name,'training_samples':len(fit),'early_stop_samples':len(stop),
        'validation_samples':len(dev),'calibration_samples':len(cal),'autoencoder_parameters':cfg.get('autoencoder',{}),
        'isolation_forest_parameters':cfg.get('isolation_forest',{}),
        'split_run_ids':{name:sorted(f['run_id'].unique().to_list()) for name,f in [('fit',fit),('early_stop',stop),('calibration',cal),('development',dev)]}})
    (out/'model_card.md').write_text(f"# Contextual model card\n\nVersion: {metadata['model_version']}\n\n"
        "Normal-only AE + Isolation Forest per asset class. Unknown classes are rejected. "
        "Scalers fit on fit runs; normal early-stop runs select checkpoints and residual scales; "
        "separate normal validation runs calibrate scores. Remaining validation runs report development metrics.\n\n"
        f"Top-{top_k} normalized reconstruction errors; ensemble weights AE={a}, IF={b}; threshold={threshold}. "
        "Scores are reference percentiles, not fault probabilities. Contributions explain normalized residuals, not causes.\n\n"
        "See metrics.json and training_metadata.json. Synthetic development evidence is not field validation.\n",encoding='utf-8')
    _record_progress(out/'training_progress.jsonl','completed',model_version=metadata['model_version'])
    return result
