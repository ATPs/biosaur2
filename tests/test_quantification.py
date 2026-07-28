import math

import numpy as np

from biosaur2.quantification import (
    normalize_trace,
    raw_area_sum,
    trapezoid_area,
)


def test_normalize_trace_repairs_order_duplicates_and_nonfinite_values():
    trace = normalize_trace(
        [2.0, 1.0, 1.0, math.nan, 3.0],
        [2.0, 1.0, 4.0, 9.0, -1.0],
    )
    np.testing.assert_array_equal(trace.rt, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(trace.intensity, [5.0, 2.0, -1.0])
    assert trace.flags == {
        "nonfinite_points_removed",
        "negative_intensity",
        "nonmonotonic_rt_sorted",
        "duplicate_rt_merged",
    }


def test_trapezoid_area_uses_irregular_real_rt():
    trace = normalize_trace([0.0, 1.0, 3.0], [0.0, 2.0, 0.0])
    assert trapezoid_area(trace) == 3.0


def _quantification_input(mobility=True):
    hills = {
        "hills_idx_array_unique": [10, 11, 12],
        "hills_mz_median": [500.0, 501.00335, 502.0067],
        "hills_scan_lists": [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 2, 4]],
        "hills_intensity_array": [
            [0.0, 10.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 10.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        "rtStart": [0.0, 0.0, 0.0],
        "rtApex": [1.0, 2.0, 2.0],
        "rtEnd": [4.0, 4.0, 4.0],
    }
    if mobility:
        hills["hills_im_median"] = [1.00, 1.02, 1.04]
    feature = {
        "monoisotope idx": 0,
        "hill_mz_1": 500.0,
        "charge": 1,
        "cos_cor_isotopes": 0.9,
        "isotopes": [
            {"isotope_number": 1, "isotope_idx": 1, "mass_diff_ppm": 7.0},
            {"isotope_number": 2, "isotope_idx": 2, "mass_diff_ppm": 8.0},
        ],
    }
    return hills, feature


def test_raw_area_sum_respects_iuse_and_exact_rt():
    hills, feature = _quantification_input()
    rt = {index: float(index) for index in range(5)}
    mono, mono_approximate = raw_area_sum(hills, feature, rt, 0)
    first, first_approximate = raw_area_sum(hills, feature, rt, 1)
    all_isotopes, all_approximate = raw_area_sum(hills, feature, rt, -1)
    assert mono == 11.0
    assert first == 22.0
    assert all_isotopes == 26.0
    assert not any((mono_approximate, first_approximate, all_approximate))


def test_raw_area_sum_uses_stored_rt_then_approximation():
    hills, feature = _quantification_input()
    hills["hills_point_rt_array"] = [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 2.0, 4.0],
    ]
    exact, approximate = raw_area_sum(hills, feature, None, -1)
    assert exact == 26.0
    assert approximate is False
    del hills["hills_point_rt_array"]
    approximated, approximate = raw_area_sum(hills, feature, None, -1)
    assert approximated == 26.0
    assert approximate is True


def test_raw_area_sum_is_null_when_a_selected_trace_cannot_be_integrated():
    hills, feature = _quantification_input()
    hills["hills_scan_lists"][1] = [99]
    hills["hills_intensity_array"][1] = [5.0]
    area, approximate = raw_area_sum(
        hills, feature, {index: float(index) for index in range(5)}, 1
    )
    assert area is None
    assert approximate is False
