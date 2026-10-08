"""Fixed three-member decision consensus; margins are not probabilities."""
import numpy as np


def consensus_scores(scores, thresholds, health, *, health_threshold=.05, mode='majority'):
    scores = np.asarray(scores, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    health = np.asarray(health, dtype=float)
    if scores.ndim != 2 or scores.shape[1] != 3 or thresholds.shape != (3,) or health.shape != (scores.shape[0],):
        raise ValueError('Three complete members and aligned health vector required')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Finite member scores in [0,1] required')
    if not np.isfinite(thresholds).all() or np.any(thresholds <= 0):
        raise ValueError('Positive finite member thresholds required')
    if not np.isfinite(health_threshold) or not 0 < health_threshold <= 1:
        raise ValueError('Health threshold must be in (0,1]')
    available = ~np.isnan(health)
    if not np.isfinite(health[available]).all() or np.any((health[available] < 0) | (health[available] > 1)):
        raise ValueError('Health must be unavailable or finite in [0,1]')
    margins = scores / thresholds
    reduce = {'majority': np.median, 'unanimity': np.min, 'union': np.max}.get(mode)
    if reduce is None:
        raise ValueError('Unknown consensus mode')
    model_margin = reduce(margins, axis=1)
    combined = np.maximum(model_margin, np.where(available, health, 0) / health_threshold)
    return {'rank': combined / (1 + combined), 'alert': combined >= 1,
            'member_alerts': margins >= 1, 'consensus_margin': model_margin,
            'health_available': available}
