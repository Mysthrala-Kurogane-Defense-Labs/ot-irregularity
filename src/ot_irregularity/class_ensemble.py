"""Normal-only class calibration of three raw reconstruction magnitudes."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def raw_matrix(raw, classes):
    raw = np.asarray(raw,dtype=float); classes = np.asarray(classes)
    if raw.ndim != 2 or raw.shape[1] != 3 or classes.shape != (len(raw),):
        raise ValueError('Three members and aligned class identities required')
    if np.isinf(raw).any() or np.any(raw[np.isfinite(raw)]<0):
        raise ValueError('Raw errors must be nonnegative finite or unavailable')
    return raw, classes, np.isfinite(raw).all(axis=1)


@dataclass
class ClassMagnitudeEnsemble:
    aggregation: str = 'median'
    quantile: float = .995
    minimum_windows: int = 400

    def fit(self, raw, classes, labels):
        raw, classes, available = raw_matrix(raw,classes)
        labels = np.asarray(labels)
        if labels.shape != (len(raw),) or labels.dtype != bool or labels.any():
            raise ValueError('Explicitly normal calibration labels required')
        if self.aggregation not in ('mean','median') or not 0 < self.quantile < 1 or self.minimum_windows < 1:
            raise ValueError('Invalid class calibration configuration')
        self.references_ = {}
        reduce = np.mean if self.aggregation=='mean' else np.median
        for group in sorted(set(classes)):
            values = raw[(classes==group)&available]
            if len(values)<self.minimum_windows:
                raise ValueError('Insufficient normal calibration for class '+str(group))
            anchors = np.maximum(np.quantile(values,.99,axis=0),1e-9)
            aggregated = reduce(values/anchors,axis=1)
            threshold = float(np.nextafter(np.quantile(aggregated,self.quantile,method='higher'),np.inf))
            if not np.isfinite(threshold) or threshold <= np.finfo(float).tiny:
                raise ValueError('Uninformative class calibration threshold')
            self.references_[str(group)] = {'anchors':anchors.tolist(),'threshold':threshold,'normal_windows':len(values)}
        if not self.references_:
            raise ValueError('No class references')
        return self

    def score(self, raw, classes):
        raw, classes, available = raw_matrix(raw,classes)
        margin = np.full(len(raw),np.nan)
        reduce = np.mean if self.aggregation=='mean' else np.median
        for group in sorted(set(classes)):
            if str(group) not in self.references_:
                raise ValueError('Untrained class')
            ref = self.references_[str(group)]; mask = (classes==group)&available
            margin[mask] = reduce(raw[mask]/np.asarray(ref['anchors']),axis=1)/ref['threshold']
        return margin, available

    def save(self,path):
        Path(path).write_text(json.dumps({'version':1,'aggregation':self.aggregation,'quantile':self.quantile,
            'minimum_windows':self.minimum_windows,'references':self.references_},indent=2,allow_nan=False)+'\n',encoding='utf-8')

    @classmethod
    def load(cls,path):
        doc=json.loads(Path(path).read_text())
        if doc['version']!=1: raise ValueError('Unsupported class ensemble version')
        model=cls(doc['aggregation'],doc['quantile'],doc['minimum_windows']);model.references_=doc['references']
        return model
