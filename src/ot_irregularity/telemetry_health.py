"""Observed telemetry deviations with explicit availability and normal references.

This research component does not diagnose sensors or process faults. It consumes
complete batch windows; detecting silence beyond the file requires a watermark.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import polars as pl

KEYS = ['run_id', 'asset_id', 'asset_class', 'window_start', 'window_end']


def extract_health(raw, windows, roles_by_class):
    required = {'run_id', 'asset_id', 'timestamp', 'measurement_role', 'value'}
    if required - set(raw.columns):
        raise ValueError('Health extraction requires normalized roles, values and timestamps')
    if windows.select(KEYS).is_duplicated().any():
        raise ValueError('Duplicate window identity')
    if raw.select(['run_id', 'asset_id', 'timestamp', 'measurement_role']).is_duplicated().any():
        raise ValueError('Duplicate observation identity')
    data = raw.sort(['run_id', 'asset_id', 'measurement_role', 'timestamp']).with_columns(
        pl.col('timestamp').dt.timestamp('us').alias('_time'))
    groups = {}
    for key, part in data.partition_by(['run_id', 'asset_id', 'measurement_role'], as_dict=True).items():
        time = part['_time'].to_numpy()
        value = part['value'].cast(pl.Float64).to_numpy()
        quality = np.array([str(q).lower() if q is not None else None for q in part['quality']]) if 'quality' in part.columns else np.full(len(part), None)
        interval = part['sampling_interval_ms'].cast(pl.Float64).to_numpy() if 'sampling_interval_ms' in part.columns else np.full(len(part), np.nan)
        declared = np.flatnonzero(np.isfinite(interval) & (interval > 0))
        first = int(declared[0]) if len(declared) else None
        period = float(interval[first]) * 1000 if first is not None else None
        changed = np.flatnonzero(np.isfinite(interval) & (interval > 0) & (interval != interval[first])) if first is not None else []
        change_time = int(time[changed[0]]) if len(changed) else None
        units = set(part['unit'].drop_nulls().to_list()) if 'unit' in part.columns else set()
        if len(units) > 1:
            raise ValueError('Normalize units before telemetry health extraction')
        groups[key] = (time, value, quality, period, int(time[first]) if first is not None else None,
                       change_time, next(iter(units)) if units else None)
    rows = []
    for window in windows.iter_rows(named=True):
        start, end = window['window_start'], window['window_end']
        if end <= start:
            raise ValueError('Positive window duration required')
        roles = roles_by_class.get(window['asset_class'])
        if not roles or len(roles) != len(set(roles)):
            raise ValueError('Known class with unique applicable roles required')
        for role in roles:
            source = groups.get((window['run_id'], window['asset_id'], role))
            row = {k: window[k] for k in KEYS}
            row.update(measurement_role=role, is_anomaly=window.get('is_anomaly'), unit=None,
                       received_count=0, finite_value_count=0, expected_sample_count=None,
                       coverage_ratio=None, bad_ratio=None, uncertain_ratio=None, mean_value=None,
                       repetition_ratio=None, usable_pairs=0, cadence_reason='no_prior_declared_cadence',
                       quality_reason='no_observations')
            if source is None:
                rows.append(row); continue
            time, values, quality, period, first_time, change_time, unit = source
            lo, hi = np.searchsorted(time, [start, end], side='left')
            t, v, q = time[lo:hi], values[lo:hi], quality[lo:hi]
            finite = np.isfinite(v); count = int(finite.sum()); n = len(v)
            cadence = period is not None and first_time < end and first_time <= start + 1.5 * period
            reason = None if cadence else 'no_prior_declared_cadence'
            if change_time is not None and change_time < end:
                cadence = False; reason = 'declared_cadence_changed'
            row.update(unit=unit if time[0] < end else None, received_count=n, finite_value_count=count, cadence_reason=reason,
                       mean_value=float(np.mean(v[finite])) if count else None)
            if cadence:
                expected = (end - start) / period
                row.update(expected_sample_count=expected, coverage_ratio=count / expected)
            if n:
                known = np.isin(q, ['good', 'uncertain', 'bad']).all()
                row.update(quality_reason=None if known else 'quality_unobserved',
                           bad_ratio=float(np.mean(q == 'bad')) if known else None,
                           uncertain_ratio=float(np.mean(q == 'uncertain')) if known else None)
                if cadence and n > 1:
                    good = finite & (q == 'good')
                    dt = np.diff(t)
                    pairs = good[:-1] & good[1:] & (dt > 0) & (dt <= 1.5 * period)
                    row['usable_pairs'] = int(pairs.sum())
                    if pairs.sum() >= 20:
                        row['repetition_ratio'] = float(np.mean((v[:-1] == v[1:])[pairs]))
            rows.append(row)
    if not rows:
        raise ValueError('No health windows')
    return pl.DataFrame(rows, infer_schema_length=None)


def upper_excess(value, upper):
    if value is None or upper is None:
        return None
    return float(np.clip((value - upper) / (1 - upper), 0, 1)) if upper < 1 else 0.


@dataclass
class TelemetryHealthReference:
    quantile: float = .995
    minimum_windows: int = 20

    def fit(self, fit, calibration):
        if not 0 < self.quantile < 1 or self.minimum_windows < 1:
            raise ValueError('Invalid reference configuration')
        for frame in (fit, calibration):
            if 'is_anomaly' not in frame.columns or frame['is_anomaly'].null_count() or frame['is_anomaly'].any():
                raise ValueError('Health references require explicitly normal windows')
        if set(fit['run_id']) & set(calibration['run_id']):
            raise ValueError('Fit/calibration run leakage')
        self.groups_ = {}
        for (group, role), f in fit.partition_by(['asset_class', 'measurement_role'], as_dict=True).items():
            c = calibration.filter((pl.col('asset_class') == group) & (pl.col('measurement_role') == role))
            units = set(f['unit'].drop_nulls()) | set(c['unit'].drop_nulls())
            if len(units) > 1:
                raise ValueError('Role units differ between fit and calibration')
            ref = {'unit': next(iter(units)) if units else None, 'counts': {}}
            for column, q in [('bad_ratio', self.quantile), ('uncertain_ratio', self.quantile),
                              ('coverage_ratio', 1-self.quantile), ('repetition_ratio', self.quantile)]:
                values = c[column].drop_nulls().to_numpy().astype(float)
                if not np.isfinite(values).all() or np.any(values < 0) or (column != 'coverage_ratio' and np.any(values > 1)):
                    raise ValueError('Invalid observed health ratio')
                ref['counts'][column] = len(values)
                ref[column] = float(np.quantile(values, q)) if len(values) >= self.minimum_windows else None
            means = f['mean_value'].drop_nulls().to_numpy().astype(float)
            if not np.isfinite(means).all():
                raise ValueError('Nonfinite mean value')
            cuts = np.unique(np.quantile(means, [.25, .5, .75])).tolist() if len(means) >= self.minimum_windows else []
            ref['mean_cuts'] = cuts; ref['repetition_bins'] = []
            available = c.filter(pl.col('mean_value').is_not_null() & pl.col('repetition_ratio').is_not_null())
            bins = np.searchsorted(cuts, available['mean_value'].to_numpy(), side='right')
            for i in range(len(cuts)+1):
                values = available['repetition_ratio'].to_numpy()[bins == i]
                fallback = len(values) < self.minimum_windows
                ref['repetition_bins'].append({'upper': ref['repetition_ratio'] if fallback else float(np.quantile(values, self.quantile)),
                                              'count': len(values), 'global_fallback': fallback})
            self.groups_.setdefault(group, {})[role] = ref
        if not self.groups_:
            raise ValueError('No applicable normal references')
        return self

    def score(self, features, *, repetition=True):
        details = []
        for row in features.iter_rows(named=True):
            group, role = row['asset_class'], row['measurement_role']
            if group not in self.groups_ or role not in self.groups_[group]:
                raise ValueError('Untrained class or role')
            ref = self.groups_[group][role]
            if row['unit'] is not None and row['unit'] != ref['unit']:
                raise ValueError('Role unit mismatch')
            bad = upper_excess(row['bad_ratio'], ref['bad_ratio'])
            uncertain = upper_excess(row['uncertain_ratio'], ref['uncertain_ratio'])
            quality = max(x for x in (bad, uncertain) if x is not None) if any(x is not None for x in (bad, uncertain)) else None
            lower = ref['coverage_ratio']; coverage = row['coverage_ratio']
            sampling = float(np.clip((lower-coverage)/lower, 0, 1)) if lower is not None and lower > 0 and coverage is not None else None
            repeat = None; context = None
            if repetition and row['mean_value'] is not None and ref['unit'] is not None:
                context = ref['repetition_bins'][int(np.searchsorted(ref['mean_cuts'], row['mean_value'], side='right'))]
                repeat = upper_excess(row['repetition_ratio'], context['upper'])
            observations = []
            if quality is not None and quality > 0:
                observations.append('quality_deviation')
            if sampling is not None and sampling > 0:
                observations.append('signal_loss' if row['finite_value_count'] == 0 else 'sampling_degradation')
            if repeat is not None and repeat > 0:
                observations.append('signal_repetition')
            details.append({**row, 'observations': observations, 'quality_score': quality, 'sampling_score': sampling, 'repetition_score': repeat,
                            'quality_score_reason': None if quality is not None else (row['quality_reason'] or 'normal_quality_reference_unavailable'),
                            'sampling_score_reason': None if sampling is not None else (row['cadence_reason'] or 'normal_coverage_reference_unavailable'),
                            'repetition_score_reason': None if repeat is not None else ('disabled' if not repetition else 'insufficient_good_pairs_or_unit_or_normal_reference'),
                            'normal_bad_upper': ref['bad_ratio'], 'normal_uncertain_upper': ref['uncertain_ratio'],
                            'normal_coverage_lower': lower, 'normal_repetition_upper': context['upper'] if context else None,
                            'repetition_global_fallback': context['global_fallback'] if context else None})
        long = pl.DataFrame(details, infer_schema_length=None).with_columns(
            pl.col('quality_score', 'sampling_score', 'repetition_score').cast(pl.Float64),
            pl.col('observations').cast(pl.List(pl.String)))
        aggregated = long.group_by(KEYS, maintain_order=True).agg(
            pl.col('quality_score').max(), pl.col('sampling_score').max(), pl.col('repetition_score').max(),
            pl.col('observations').list.explode(keep_nulls=False, empty_as_null=False).unique().sort())
        aggregated = aggregated.with_columns(pl.max_horizontal('quality_score', 'sampling_score', 'repetition_score').alias('health_score'))
        return aggregated, long

    def save(self, path):
        Path(path).write_text(json.dumps({'version': 1, 'quantile': self.quantile, 'minimum_windows': self.minimum_windows,
                                         'groups': self.groups_}, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        doc = json.loads(Path(path).read_text())
        if doc['version'] != 1:
            raise ValueError('Unsupported telemetry health artifact version')
        model = cls(doc['quantile'], doc['minimum_windows']); model.groups_ = doc['groups']
        return model
