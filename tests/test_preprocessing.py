import pytest
import numpy as np

from biosaur2 import preprocessing, utils


def test_no_ms1_input_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(utils, "iter_ms1_spectra", lambda _path: iter(()))
    with pytest.raises(ValueError, match="No usable MS1"):
        preprocessing.process_mzml(
            {
                "file": "empty.mzML",
                "combine_every": 1,
                "mini": 1,
                "minmz": 350,
                "maxmz": 1500,
            }
        )


def test_unknown_raw_mobility_is_not_relabelled_as_inverse_k0(monkeypatch):
    spectrum = {
        "id": "scan=9",
        "ms level": 1,
        "scanList": {"scan": [{"scan start time": 2.0}]},
        "m/z array": np.asarray([500.0]),
        "intensity array": np.asarray([10.0]),
        "raw ion mobility array": np.asarray([7.5]),
    }
    monkeypatch.setattr(utils, "iter_ms1_spectra", lambda _path: iter([spectrum]))
    result = preprocessing.process_mzml(
        {
            "file": "one.mzML",
            "combine_every": 1,
            "mini": 1,
            "minmz": 350,
            "maxmz": 1500,
        }
    )
    assert result[0]["ignore_ion_mobility"] is True
    np.testing.assert_array_equal(
        result[0]["mean inverse reduced ion mobility array"], [0.0]
    )
