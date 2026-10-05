import numpy as np
import pytest

from scripts.research_alert_budgets import union_decisions


def test_union_preserves_every_baseline_alarm_and_unavailable_supplement():
    baseline = np.array([.991, .5, .1, .2])
    relational = np.array([0., .999, .9, 0.])
    decisions = union_decisions(baseline, relational, .99, .995)
    assert decisions.tolist() == [True, True, False, False]
    assert np.all(decisions[baseline >= .99])


def test_union_rejects_invalid_inputs():
    for a, b, threshold in [([.5], [np.nan], .99), ([.5], [], .99), ([.5], [.2], 1.01)]:
        with pytest.raises(ValueError):
            union_decisions(a, b, .99, threshold)
