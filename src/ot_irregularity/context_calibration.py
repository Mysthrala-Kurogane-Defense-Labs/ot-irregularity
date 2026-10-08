"""Normal-only calibration of supported observed regimes with class fallback."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


def arrays(margins, classes, regimes):
    values=np.asarray(margins,dtype=float); classes=np.asarray(classes,dtype=str); regimes=np.asarray(regimes,dtype=object)
    if values.ndim!=1 or classes.shape!=values.shape or regimes.shape!=values.shape:
        raise ValueError('Aligned margin, class and regime arrays required')
    if np.isinf(values).any() or np.any(values[np.isfinite(values)]<0):
        raise ValueError('Nonnegative finite margins or unavailable NaN required')
    if any(not c for c in classes) or any(r is not None and not isinstance(r,str) for r in regimes):
        raise ValueError('Class identities and string or null regimes required')
    return values,classes,regimes


def known(regime):
    return regime is not None and regime.strip().lower() not in ('','unknown')


@dataclass
class ContextMagnitudeCalibration:
    quantile: float = .9975
    minimum_windows: int = 400

    def fit(self,margins,classes,regimes,labels):
        values,classes,regimes=arrays(margins,classes,regimes); labels=np.asarray(labels)
        if labels.shape!=values.shape or labels.dtype!=bool or labels.any():
            raise ValueError('Explicit normal labels required')
        if not 0<self.quantile<1 or self.minimum_windows<1 or not len(values):
            raise ValueError('Valid calibration settings and data required')
        self.classes_=sorted(set(classes)); self.groups_={}
        for group in self.classes_:
            for regime in sorted({r for r in regimes[classes==group] if known(r)}):
                v=values[(classes==group)&(regimes==regime)&np.isfinite(values)]
                threshold=None
                if len(v)>=self.minimum_windows:
                    candidate=float(np.nextafter(np.quantile(v,self.quantile,method='higher'),np.inf))
                    if np.isfinite(candidate) and candidate>np.finfo(float).tiny: threshold=candidate
                self.groups_[(group,regime)]={'normal_windows':len(v),'threshold':threshold}
        return self

    def score(self,margins,classes,regimes):
        values,classes,regimes=arrays(margins,classes,regimes)
        if set(classes)-set(self.classes_): raise ValueError('Untrained class')
        result=values.copy(); routes=[]
        for i,(value,group,regime) in enumerate(zip(values,classes,regimes)):
            ref=self.groups_.get((group,regime)) if known(regime) else None
            if not np.isfinite(value): routes.append('relational_unavailable')
            elif not known(regime): routes.append('class_fallback_unknown_context')
            elif ref is None: routes.append('class_fallback_unseen_context')
            elif ref['threshold'] is None: routes.append('class_fallback_insufficient_reference')
            else:
                result[i]=value/ref['threshold']; routes.append('context_reference')
        if np.isinf(result).any(): raise ValueError('Context margin overflow')
        return result,np.asarray(routes)

    def save(self,path):
        doc={'version':1,'quantile':self.quantile,'minimum_windows':self.minimum_windows,
             'classes':self.classes_,'groups':[{'asset_class':c,'regime':r,**v} for (c,r),v in sorted(self.groups_.items())]}
        Path(path).write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n',encoding='utf-8')

    @classmethod
    def load(cls,path):
        d=json.loads(Path(path).read_text())
        if d['version']!=1: raise ValueError('Unsupported context calibration version')
        model=cls(d['quantile'],d['minimum_windows'])
        if not 0<model.quantile<1 or model.minimum_windows<1: raise ValueError('Invalid calibration settings')
        model.classes_=d['classes']; model.groups_={}
        if not model.classes_ or len(set(model.classes_))!=len(model.classes_): raise ValueError('Invalid class inventory')
        for row in d['groups']:
            key=(row['asset_class'],row['regime']); threshold=row['threshold']; count=row['normal_windows']
            if key in model.groups_ or key[0] not in model.classes_ or not isinstance(key[1],str) or not known(key[1]) or not isinstance(count,int) or count<0:
                raise ValueError('Invalid context reference')
            if threshold is not None and (not np.isfinite(threshold) or threshold<=np.finfo(float).tiny or count<model.minimum_windows):
                raise ValueError('Invalid context threshold')
            model.groups_[key]={'normal_windows':count,'threshold':threshold}
        return model
