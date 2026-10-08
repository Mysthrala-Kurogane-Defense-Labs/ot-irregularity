"""Experimental normal-only cross-signal residuals; no causal interpretation."""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler


def previous_indices(frame):
    """Previous contiguous window in the same run/asset, in original row order."""
    keys=list(zip(frame['run_id'],frame['asset_id'],frame['window_start']))
    if len(set(keys))!=len(keys):raise ValueError('Duplicate window identity')
    starts=frame['window_start'].to_numpy();ends=frame['window_end'].to_numpy()
    if np.any(ends<=starts):raise ValueError('Invalid window bounds')
    order=sorted(range(len(keys)),key=lambda i:keys[i]);previous=np.full(len(frame),-1,dtype=int)
    for before,now in zip(order,order[1:]):
        if keys[before][:2]==keys[now][:2] and ends[before]==starts[now]:previous[now]=before
    return previous


def measurement_arrays(frame,roles):
    x=frame.select([r+'_mean' for r in roles]).to_numpy().astype(float)
    valid=np.isfinite(x).all(axis=1)
    for role in roles:
        valid &= (frame[role+'_good_ratio'].to_numpy()>=.95)
        valid &= (frame[role+'_coverage_ratio'].to_numpy()>=.9)
        valid &= (frame[role+'_missing'].to_numpy()==0)
    return x,valid


@dataclass
class RelationshipResiduals:
    roles: list[str]
    temporal: bool=False
    alpha: float=10.

    def fit(self,frame):
        if len(self.roles)<2 or len(set(self.roles))!=len(self.roles):raise ValueError('At least two unique roles required')
        if self.alpha<=0 or not np.isfinite(self.alpha):raise ValueError('Positive finite ridge alpha required')
        if 'is_anomaly' not in frame.columns or frame['is_anomaly'].null_count() or frame['is_anomaly'].any():
            raise ValueError('Relationship fit requires explicitly normal windows')
        x,valid=measurement_arrays(frame,self.roles)
        if valid.sum()<20:raise ValueError('Insufficient good normal windows')
        self.scaler_=RobustScaler().fit(x[valid]);z=self.scaler_.transform(np.where(np.isfinite(x),x,self.scaler_.center_))
        previous=previous_indices(frame);history=(previous>=0)&valid&valid[np.maximum(previous,0)]
        lag=z[np.maximum(previous,0)]
        self.static_=[];self.temporal_=[]
        for target in range(len(self.roles)):
            other=[i for i in range(len(self.roles)) if i!=target]
            self.static_.append(Ridge(alpha=self.alpha).fit(z[valid][:,other],z[valid,target]))
            model=None
            if self.temporal:
                if history.sum()<20:raise ValueError('Insufficient contiguous normal history')
                design=np.column_stack([z[:,other],lag])
                model=Ridge(alpha=self.alpha).fit(design[history],z[history,target])
            self.temporal_.append(model)
        self.fit_counts_={'good_windows':int(valid.sum()),'contiguous_good_windows':int(history.sum())}
        return self

    def transform(self,frame):
        x,valid=measurement_arrays(frame,self.roles)
        # Invalid rows are unavailable, not evidence of a normal physical state.
        safe=np.where(np.isfinite(x),x,self.scaler_.center_)
        z=self.scaler_.transform(safe);previous=previous_indices(frame)
        history=(previous>=0)&valid&valid[np.maximum(previous,0)];lag=z[np.maximum(previous,0)]
        residual=np.zeros_like(z)
        for target in range(len(self.roles)):
            other=[i for i in range(len(self.roles)) if i!=target]
            prediction=self.static_[target].predict(z[:,other])
            if self.temporal:
                prediction[history]=self.temporal_[target].predict(np.column_stack([z[:,other],lag])[history]) if history.any() else prediction[history]
            residual[:,target]=z[:,target]-prediction
        residual[~valid]=0.
        return residual,valid,history
