"""Experimental normal-only thermal dynamics; residuals describe deviations, not causes."""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from .relationships import measurement_arrays, previous_indices


@dataclass
class ThermalDynamics:
    target: str
    drivers: list[str]
    alpha: float = 1.

    def arrays(self, frame):
        widths=frame['window_end'].to_numpy()-frame['window_start'].to_numpy()
        if len(widths)==0 or np.any(widths!=self.window_duration_):
            raise ValueError('Thermal windows must match the fitted duration')
        _, good = measurement_arrays(frame, [self.target, *self.drivers])
        temperature = frame[self.target+'_last'].to_numpy().astype(float)
        previous = previous_indices(frame)
        prior = np.maximum(previous, 0)
        good &= np.isfinite(temperature)
        available = good & (previous >= 0) & good[prior]
        # No current temperature enters the predictors; target is endpoint change.
        inputs = frame.select([r+'_mean' for r in self.drivers]).to_numpy().astype(float)
        x = np.column_stack([temperature[prior], inputs])
        y = temperature-temperature[prior]
        available &= np.isfinite(x).all(axis=1) & np.isfinite(y)
        return x, y, available

    def fit(self, frame):
        if (not self.drivers or len(set(self.drivers)) != len(self.drivers)
                or self.target in self.drivers or not np.isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError('Unique drivers excluding target and positive finite alpha required')
        if 'is_anomaly' not in frame.columns or frame['is_anomaly'].null_count() or frame['is_anomaly'].any():
            raise ValueError('Thermal fit requires explicit normal-only windows')
        widths=frame['window_end'].to_numpy()-frame['window_start'].to_numpy()
        if len(widths)==0 or np.any(widths<=0) or np.any(widths!=widths[0]):
            raise ValueError('Thermal fit requires uniform positive window duration')
        self.window_duration_=int(widths[0])
        x, y, available = self.arrays(frame)
        if available.sum() < 20:
            raise ValueError('Insufficient contiguous good normal windows')
        self.scaler_ = RobustScaler().fit(x[available])
        self.model_ = Ridge(alpha=self.alpha).fit(self.scaler_.transform(x[available]), y[available])
        self.fit_windows_ = int(available.sum())
        return self

    def transform(self, frame):
        x, y, available = self.arrays(frame)
        residual = np.zeros(len(frame))
        if available.any():
            residual[available] = y[available]-self.model_.predict(self.scaler_.transform(x[available]))
        return residual, available
