import copy

import pandas as pd
import pytest

from biosaur2.hills import assign_deterministic_hill_ids, normalize_hills_dataframe
from biosaur2.utils import get_hills_dict_from_hills_features, iter_hills_extra


def _hills(order):
    mz = [500.0, 400.0]
    scans = [[2, 3], [0, 1]]
    intensity = [[10.0, 20.0], [30.0, 40.0]]
    points = [[500.0, 500.1], [400.0, 400.1]]
    return {
        "hills_idx_array_unique": [91, 22],
        "hills_mz_median": [mz[index] for index in order],
        "hills_scan_lists": [scans[index] for index in order],
        "hills_intensity_array": [intensity[index] for index in order],
        "tmp_mz_array": [points[index] for index in order],
        "hills_lengths": [2, 2],
    }


def test_hill_ids_follow_content_not_input_order():
    forward = _hills([0, 1])
    reverse = _hills([1, 0])
    assert assign_deterministic_hill_ids(forward, 7) == 9
    assert assign_deterministic_hill_ids(reverse, 7) == 9
    forward_mapping = dict(zip(forward["hills_mz_median"], forward["hills_idx_array_unique"]))
    reverse_mapping = dict(zip(reverse["hills_mz_median"], reverse["hills_idx_array_unique"]))
    assert forward_mapping == reverse_mapping == {400.0: 7, 500.0: 8}


def test_hill_id_assignment_does_not_reorder_scientific_arrays():
    hills = _hills([0, 1])
    original = copy.deepcopy(hills)
    assign_deterministic_hill_ids(hills)
    for key in (
        "hills_mz_median",
        "hills_scan_lists",
        "hills_intensity_array",
        "tmp_mz_array",
    ):
        assert hills[key] == original[key]


def test_modern_hills_preserve_exact_point_metadata():
    frame = pd.DataFrame(
        {
            "hill_id": [4],
            "mz": [500.0],
            "point_count": [2],
            "rt_start_sec": [10.0],
            "rt_apex_sec": [11.0],
            "rt_end_sec": [12.0],
            "point_scan_indexes": [[3, 5]],
            "point_scan_numbers": [[103, None]],
            "point_rt_sec": [[10.0, 12.0]],
            "point_intensities": [[20.0, 30.0]],
            "point_mz": [[499.9, 500.1]],
        }
    )
    normalized = normalize_hills_dataframe(frame, "seconds")
    hills, _ = get_hills_dict_from_hills_features(normalized, 8.0, 0.0)
    assert hills["hills_point_rt_array"] == [[10.0, 12.0]]
    assert hills["tmp_mz_array"] == [[499.9, 500.1]]
    assert hills["hills_scan_number_array"] == [[103, None]]


def test_modern_hills_reject_mismatched_point_arrays():
    frame = pd.DataFrame(
        {
            "hill_idx": [1],
            "mz": [500.0],
            "nScans": [2],
            "rtStart": [1.0],
            "rtApex": [1.0],
            "rtEnd": [2.0],
            "FAIMS": [None],
            "im": [None],
            "hills_scan_lists": [[0, 1]],
            "hills_intensity_list": [[10.0]],
        }
    )
    with pytest.raises(ValueError, match="scan indexes"):
        get_hills_dict_from_hills_features(frame, 8.0, 0.0)


def test_no_hill_list_skips_materializing_large_point_payloads():
    hills = {
        "hills_idx_array_unique": [10],
        "hills_mz_median": [500.0],
        "hills_lengths": [2],
        "hills_scan_lists": [[0, 1]],
        "hills_intensity_array": [[2.0, 4.0]],
        "tmp_mz_array": [[499.99, 500.01]],
        "hills_intensity_apex": [None],
        "hills_scan_apex": [None],
    }
    spectra = [
        {"scan_index": 0, "scan_number": 101},
        {"scan_index": 1, "scan_number": 102},
    ]
    rows = list(
        iter_hills_extra(
            hills,
            {0: 60.0, 1: 120.0},
            None,
            0,
            0.01,
            0.0,
            data_for_analyse_tmp=spectra,
            include_point_lists=False,
        )
    )
    row = rows[0]
    assert row["hill_idx"] == 10
    assert row["scanApex"] == 102
    assert "hills_scan_lists" not in row
    assert "hills_intensity_list" not in row
    assert "hills_mz_array" not in row
    assert "hills_rt_list" not in row
    assert "_hill_points" not in row

    rows_with_points = list(
        iter_hills_extra(
            hills,
            {0: 60.0, 1: 120.0},
            None,
            0,
            0.01,
            0.0,
            data_for_analyse_tmp=spectra,
            include_point_lists=True,
        )
    )
    assert rows_with_points[0]["hills_rt_list"] == [60.0, 120.0]
    assert "_hill_points" not in rows_with_points[0]
