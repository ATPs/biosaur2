import numpy as np

from biosaur2.peak_splitting import _remap_worker_ids


def test_split_artifact_remap_preserves_first_encounter_order():
    remapped, next_id = _remap_worker_ids(
        np.asarray([9, 4, 9, 12, 4, 7]), next_id=31
    )
    assert remapped.tolist() == [31, 32, 31, 33, 32, 34]
    assert next_id == 35
