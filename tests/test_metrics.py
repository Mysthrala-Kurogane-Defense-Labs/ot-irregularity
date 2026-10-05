import numpy as np

from ot_irregularity.metrics import evaluate_scores


def test_null_event_ids_do_not_count_as_events():
    result = evaluate_scores(
        [0, 0, 0], [0.1, 0.2, 0.3], event_ids=np.asarray([None, np.nan, ""], dtype=object)
    )

    assert result["event_count"] == 0
    assert result["event_detection_rate"] is None
def test_interval_metrics_count_overlaps_and_unobservable_events():
    from ot_irregularity.metrics import evaluate_event_intervals
    events=[{'run_id':'r','asset_id':'a','event_id':str(i),'start_us':start,'end_us':start+10} for i,start in enumerate([0,5,1000])]
    result=evaluate_event_intervals(events,[.99,.99],.95,['r','other'],['a','a'],[0,1000],[60,1060])
    assert result['event_count']==3 and result['detected_events']==2
    assert result['event_detection_rate']==2/3 and result['events_without_windows']==1
    assert result['events'][2]['detected'] is False
