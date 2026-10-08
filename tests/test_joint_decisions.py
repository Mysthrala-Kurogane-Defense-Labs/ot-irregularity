from copy import deepcopy

from scripts.research_joint_decisions import passes_gate


def test_joint_gate_keeps_each_predeclared_floor():
    baseline = {'false_positive_windows': 4}
    relational = {'detected_events': 71, 'physical_events': {'detected': 15}}
    thermal = {'by_family': {'cooling_degradation': {'detected': 5}}}
    candidate = dict(detected_events=87, physical_events={'detected': 18},
                     by_family={'cooling_degradation': {'detected': 5}},
                     false_positive_windows=4, precision=.9)
    assert passes_gate(candidate, 0, baseline, relational, thermal)
    assert not passes_gate(candidate, 1, baseline, relational, thermal)
    for key, value in [('false_positive_windows', 6), ('detected_events', 70),
                       ('physical_events', {'detected': 14}), ('precision', .49),
                       ('by_family', {'cooling_degradation': {'detected': 4}})]:
        failed = deepcopy(candidate)
        failed[key] = value
        assert not passes_gate(failed, 0, baseline, relational, thermal)
