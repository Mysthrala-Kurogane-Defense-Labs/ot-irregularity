import numpy as np

from ot_irregularity.metrics import evaluate_scores


def test_null_event_ids_do_not_count_as_events():
    result = evaluate_scores(
        [0, 0, 0], [0.1, 0.2, 0.3], event_ids=np.asarray([None, np.nan, ""], dtype=object)
    )

    assert result["event_count"] == 0
    assert result["event_detection_rate"] is None
