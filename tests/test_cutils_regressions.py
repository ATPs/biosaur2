import copy

import numpy as np

from biosaur2.cutils import detect_hills, get_and_calc_apex_intensity_and_scan


def _spectrum(mz, intensity):
    return {
        "m/z array": np.asarray(mz, dtype=np.float64),
        "intensity array": np.asarray(intensity, dtype=np.float64),
        "mean inverse reduced ion mobility array": np.zeros(len(mz)),
    }


def test_predecessor_index_zero_is_a_valid_link_and_calibration_observation():
    spectra = [_spectrum([500.0], [100.0]), _spectrum([500.0001], [200.0])]
    hills, mass_differences = detect_hills(
        spectra,
        {"htol": 8.0},
        0.001,
        0.0,
        md_correction_int=2,
    )

    assert hills["hills_idx_array"] == [0, 0]
    assert len(mass_differences) == 1


def test_detect_hills_does_not_mutate_spectrum_arrays():
    spectra = [
        _spectrum([499.9999, 500.0, 500.0001], [30.0, 100.0, 20.0]),
        _spectrum([500.0], [200.0]),
    ]
    original = copy.deepcopy(spectra)

    detect_hills(spectra, {"htol": 8.0}, 0.001, 0.0, md_correction_int=2)

    for before, after in zip(original, spectra):
        np.testing.assert_array_equal(before["m/z array"], after["m/z array"])
        np.testing.assert_array_equal(
            before["intensity array"], after["intensity array"]
        )


def test_apex_is_initialized_for_an_all_negative_hill():
    hills = {
        "hills_intensity_apex": [None],
        "hills_scan_apex": [None],
        "hills_intensity_array": [[-2.0, -1.0]],
        "hills_scan_lists": [[4, 5]],
    }
    _, intensity, scan = get_and_calc_apex_intensity_and_scan(hills, 0)
    assert intensity == -1.0
    assert scan == 5
