from copy import deepcopy

from scripts.research_normal_coverage import eligible_component, integrity_losses


def test_integrity_gate_checks_event_identity_not_just_counts():
    def event(name, detected=True):
        return {'run_id': 'r', 'asset_id': 'a', 'event_id': name, 'detected': detected}
    baseline = [event('lost-signal'), event('other-signal', False), event('physical')]
    candidate = [event('lost-signal', False), event('other-signal'), event('physical', False)]
    families = {('r','a','lost-signal'): 'single_signal_loss',
                ('r','a','other-signal'): 'quality_degradation', ('r','a','physical'): 'cavitation'}
    assert integrity_losses(baseline, candidate, families) == [('r','a','lost-signal')]
    assert integrity_losses(baseline, baseline, families) == []


def test_coverage_gate_rejects_each_regression_and_class_budget_failure():
    mixed = {'detected_events': 53, 'physical_events': {'detected': 4},
             'false_positive_windows': 4, 'precision': .9}
    normal = {'overall': {'false_windows': 20},
              'by_class': {'PUMP': {'false_windows_per_asset_day': 5}}}
    assert eligible_component(mixed, normal, [], mixed, normal)
    assert not eligible_component(mixed, normal, ['lost'], mixed, normal)
    for key, value in [('detected_events', 52), ('physical_events', {'detected': 3}),
                       ('false_positive_windows', 5), ('precision', .49)]:
        failed = deepcopy(mixed); failed[key] = value
        assert not eligible_component(failed, normal, [], mixed, normal)
    failed = deepcopy(normal); failed['by_class']['PUMP']['false_windows_per_asset_day'] = 10.1
    assert not eligible_component(mixed, failed, [], mixed, normal)
    failed = deepcopy(normal); failed['overall']['false_windows'] = 21
    assert not eligible_component(mixed, failed, [], mixed, normal)
