"""Normal-only exposure and run-cluster uncertainty; no supervised metrics."""
import numpy as np
import polars as pl


def alarm_counts(windows, alerts):
    alerts = np.asarray(alerts)
    if alerts.dtype != bool or alerts.shape != (len(windows),):
        raise ValueError('One boolean decision per window required')
    data = windows.with_columns(pl.Series('_alert', alerts)).sort(['run_id', 'asset_id', 'window_start'])
    count = episodes = 0
    total_us = alert_us = 0
    previous = None
    previous_alert = False
    for row in data.iter_rows(named=True):
        identity = (row['run_id'], row['asset_id'])
        start, end = row['window_start'], row['window_end']
        if end <= start:
            raise ValueError('Positive window exposure required')
        if previous and previous[0] == identity and start < previous[1]:
            raise ValueError('Overlapping windows would double-count exposure')
        duration = end - start
        total_us += duration
        if row['_alert']:
            count += 1
            alert_us += duration
            if not (previous and previous[0] == identity and previous[1] == start and previous_alert):
                episodes += 1
        previous = (identity, end)
        previous_alert = row['_alert']
    return {'windows': len(data), 'false_windows': count, 'alarm_episodes': episodes,
            'alarm_seconds': alert_us / 1e6, 'asset_hours': total_us / 3.6e9}


def summarize(records, models, *, repetitions=2000, seed=20261005):
    """Aggregate class records into run clusters before sampling (also across assets)."""
    if not records:
        raise ValueError('No evaluated runs')
    runs = sorted({r['run_id'] for r in records})
    index = {run: i for i, run in enumerate(runs)}
    values = {name: np.zeros((len(runs), 5)) for name in models}
    for record in records:
        for name in models:
            m = record['models'][name]
            values[name][index[record['run_id']]] += [m['asset_hours'], m['false_windows'],
                m['alarm_episodes'], m['alarm_seconds'], m['windows']]
    if any(np.any(v[:, 0] <= 0) for v in values.values()):
        raise ValueError('Every run requires positive exposure')
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(runs), size=(repetitions, len(runs)))
    result, rates = {}, {}
    for name, v in values.items():
        total = v.sum(axis=0)
        boot = v[picks].sum(axis=1)
        rates[name] = boot[:, 1] * 24 / boot[:, 0]
        result[name] = {
            'runs': len(runs), 'asset_hours': float(total[0]), 'windows': int(total[4]),
            'false_windows': int(total[1]), 'alarm_episodes': int(total[2]),
            'alarm_seconds': float(total[3]),
            'false_windows_per_asset_day': float(total[1] * 24 / total[0]),
            'alarm_episodes_per_asset_day': float(total[2] * 24 / total[0]),
            'fraction_runs_with_alarm': float(np.mean(v[:, 1] > 0)),
            'false_windows_per_asset_day_ci95': np.quantile(rates[name], [.025, .975]).tolist(),
            'alarm_episodes_per_asset_day_ci95': np.quantile(boot[:, 2] * 24 / boot[:, 0], [.025, .975]).tolist(),
            'zero_count_caution': bool(total[1] == 0),
        }
    baseline = models[0]
    for name in models[1:]:
        result[name]['paired_false_windows_per_asset_day_delta_ci95'] = np.quantile(
            rates[name] - rates[baseline], [.025, .975]).tolist()
    return result
