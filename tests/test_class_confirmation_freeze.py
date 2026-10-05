import json

import numpy as np
import polars as pl
import pytest

from ot_irregularity.class_ensemble import ClassMagnitudeEnsemble
from scripts.freeze_class_ensemble import verified_reference


def test_reference_freeze_reconstructs_calibration_and_rejects_mutation(tmp_path):
    classes = np.repeat(['CNC', 'COMPRESSOR', 'CONVEYOR', 'PUMP'], 400)
    raw = np.random.default_rng(42).lognormal(size=(1600, 3))
    model = ClassMagnitudeEnsemble('median', .9975).fit(raw, classes, np.zeros(1600, dtype=bool))
    path = tmp_path/'median-0.9975.json'
    model.save(path)
    frame = pl.DataFrame({'asset_class': classes, 'calibration': [True]*1600,
        **{'raw_'+str(seed): raw[:, i] for i, seed in enumerate((20261005, 20261006, 20261007))}})
    frame.write_parquet(tmp_path/'historical-raw.parquet')
    frame.with_columns(pl.lit(False).alias('calibration')).write_parquet(tmp_path/'normal-raw.parquet')
    assert verified_reference(tmp_path) == path
    original = json.loads(path.read_text())
    for field in ('threshold', 'anchors'):
        changed = json.loads(json.dumps(original))
        if field == 'threshold': changed['references']['PUMP'][field] *= 1.01
        else: changed['references']['PUMP'][field][0] *= 1.01
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='differs'):
            verified_reference(tmp_path)
