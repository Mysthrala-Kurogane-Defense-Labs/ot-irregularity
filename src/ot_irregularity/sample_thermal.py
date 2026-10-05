"""Experimental sample-resolution thermal rate residuals for complete run batches."""
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler


def align_samples(raw, roles):
    """Exact timestamp join; no interpolation, backfill or future measurements."""
    keys=['run_id','asset_id','timestamp']
    selected=raw.filter(pl.col('measurement_role').is_in(roles))
    if selected.select([*keys,'measurement_role']).is_duplicated().any():
        raise ValueError('Duplicate role/timestamp observation')
    selected=selected.with_columns(pl.when((pl.col('quality')=='good')&pl.col('value').is_finite())
                                  .then(pl.col('value')).otherwise(None).alias('_value'))
    wide=selected.pivot(on='measurement_role',index=keys,values='_value').sort(keys)
    for role in roles:
        if role not in wide.columns:wide=wide.with_columns(pl.lit(None,dtype=pl.Float64).alias(role))
    return wide.with_columns(pl.col('timestamp').dt.timestamp('us').alias('time_us'))


@dataclass
class SampleThermal:
    target: str
    drivers: list[str]
    initial_context: bool = False
    max_gap_seconds: float = .8
    alpha: float = 1.

    def arrays(self, wide):
        if not self.drivers or self.target in self.drivers or len(set(self.drivers))!=len(self.drivers):
            raise ValueError('Unique operating drivers must exclude target')
        if not np.isfinite(self.max_gap_seconds) or self.max_gap_seconds<=0:
            raise ValueError('Positive finite sampling gap limit required')
        if wide.select(['run_id','asset_id','time_us']).is_duplicated().any():
            raise ValueError('Duplicate sample identity')
        # align_samples returns ordered rows; reject unsorted callers rather than bridge time.
        ordered=wide.sort(['run_id','asset_id','time_us'])
        if not wide.select(['run_id','asset_id','time_us']).equals(ordered.select(['run_id','asset_id','time_us'])):
            raise ValueError('Samples must be ordered by run, asset and time')
        n=len(wide);values=wide.select([self.target,*self.drivers]).to_numpy().astype(float)
        good=np.isfinite(values).all(axis=1)
        runs=wide['run_id'].to_numpy();assets=wide['asset_id'].to_numpy();time=wide['time_us'].to_numpy()
        same=np.zeros(n,dtype=bool)
        if n>1:same[1:]=(runs[1:]==runs[:-1])&(assets[1:]==assets[:-1])
        previous=np.maximum(np.arange(n)-1,0)
        dt=(time-time[previous])/1e6
        valid=good&good[previous]&same&(dt>0)&(dt<=self.max_gap_seconds)
        inputs=[values[previous,0],*[values[:,j] for j in range(1,values.shape[1])]]
        if self.initial_context:
            anchor=np.full(n,np.nan);current=np.nan
            for i in range(n):
                if not same[i]:current=np.nan
                if not np.isfinite(current) and good[i]:current=values[i,0]
                anchor[i]=current
            inputs.append(anchor)
        x=np.column_stack(inputs)
        y=np.divide(values[:,0]-values[previous,0],dt,out=np.zeros(n),where=dt>0)
        valid &= np.isfinite(x).all(axis=1)&np.isfinite(y)
        return x,y,valid,dt

    def fit(self, raw):
        if 'is_anomaly' not in raw.columns or raw['is_anomaly'].null_count() or raw['is_anomaly'].any():
            raise ValueError('Explicit normal-only fit rows required')
        if not np.isfinite(self.alpha) or self.alpha<=0:raise ValueError('Positive finite alpha required')
        wide=align_samples(raw,[self.target,*self.drivers]);x,y,valid,_=self.arrays(wide)
        if valid.sum()<100:raise ValueError('Insufficient valid normal sample pairs')
        self.scaler_=RobustScaler().fit(x[valid]);self.model_=Ridge(alpha=self.alpha).fit(self.scaler_.transform(x[valid]),y[valid])
        self.fit_samples_=int(valid.sum())
        return self

    def transform(self, raw):
        wide=align_samples(raw,[self.target,*self.drivers]);x,y,valid,dt=self.arrays(wide)
        residual=np.full(len(wide),np.nan)
        if valid.any():residual[valid]=y[valid]-self.model_.predict(self.scaler_.transform(x[valid]))
        return wide.select(['run_id','asset_id','time_us']).with_columns(
            pl.Series('residual_rate',residual),pl.Series('available',valid),pl.Series('dt_seconds',dt))


def aggregate_residuals(samples, windows, expected_interval_seconds, minimum_coverage=.9):
    """Time-weighted signed residual and absolute q95 in identical complete minute windows."""
    if expected_interval_seconds<=0 or not np.isfinite(expected_interval_seconds):raise ValueError('Invalid sampling interval')
    if not 0<minimum_coverage<=1:raise ValueError('Invalid coverage')
    result=np.zeros((len(windows),2));available=np.zeros(len(windows),dtype=bool)
    groups={key:part.sort('time_us') for key,part in samples.group_by(['run_id','asset_id'])}
    for i,row in enumerate(windows.iter_rows(named=True)):
        part=groups.get((row['run_id'],row['asset_id']))
        if part is None:continue
        times=part['time_us'].to_numpy();start,end=row['window_start'],row['window_end']
        if end<=start:raise ValueError('Invalid window bounds')
        left,right=np.searchsorted(times,[start,end],side='left')
        sub=part.slice(int(left),int(right-left)).filter(pl.col('available'))
        expected=(end-start)/1e6/expected_interval_seconds
        if len(sub)<max(20,minimum_coverage*expected):continue
        r=sub['residual_rate'].to_numpy();dt=sub['dt_seconds'].to_numpy()
        result[i]=[abs(np.average(r,weights=dt)),np.quantile(np.abs(r),.95)]
        available[i]=True
    return result,available
