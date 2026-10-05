import numpy as np
import pytest

from ot_irregularity.consensus import consensus_scores


def test_consensus_uses_each_threshold_and_preserves_health_availability():
    scores = [[.8, .7, .3], [.8, .2, .3], [.1, .2, .1], [.1, .1, .1]]
    out = consensus_scores(scores, [.8, .7, .6], [np.nan, 0, .05, np.nan])
    assert out['alert'].tolist() == [True, False, True, False]
    assert out['health_available'].tolist() == [False, True, True, False]
    assert np.array_equal(out['rank'] >= .5, out['alert'])
    assert out['member_alerts'].sum(axis=1).tolist() == [2, 1, 0, 0]
    assert consensus_scores(scores, [.8, .7, .6], [np.nan]*4, mode='unanimity')['alert'].tolist() == [False]*4
    assert consensus_scores(scores, [.8, .7, .6], [np.nan]*4, mode='union')['alert'].tolist() == [True, True, False, False]


@pytest.mark.parametrize('scores,thresholds,health', [
    ([[.2,.3]], [.5,.5,.5], [0]), ([[.2,.3,np.nan]], [.5]*3,[0]),
    ([[.2,.3,.4]], [.5,0,.5],[0]), ([[.2,.3,.4]], [.5]*3,[np.inf]),
    ([[.2,.3,.4]], [.5]*3,[1.1]), ([[.2,.3,.4]], [.5]*3,[]),
])
def test_consensus_rejects_missing_members_and_invalid_contract(scores, thresholds, health):
    with pytest.raises(ValueError):
        consensus_scores(scores, thresholds, health)


def test_consensus_is_invariant_to_member_order():
    rng = np.random.default_rng(42)
    scores = rng.random((100,3)); thresholds = np.array([.7,.8,.9]); health = rng.random(100)
    first = consensus_scores(scores, thresholds, health)
    second = consensus_scores(scores[:,[2,0,1]], thresholds[[2,0,1]], health)
    np.testing.assert_array_equal(first['rank'], second['rank'])
