"""Hills interchange normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd


MODERN_TO_INTERNAL = {
    "hill_id": "hill_idx",
    "point_count": "nScans",
    "rt_start_sec": "rtStart",
    "rt_apex_sec": "rtApex",
    "rt_end_sec": "rtEnd",
    "faims_cv": "FAIMS",
    "ion_mobility_1_over_k0": "im",
    "point_scan_indexes": "hills_scan_lists",
    "point_scan_numbers": "hills_scan_number_list",
    "point_intensities": "hills_intensity_list",
    "point_mz": "hills_mz_array",
    "point_rt_sec": "hills_rt_list",
}


def normalize_hills_dataframe(frame: pd.DataFrame, input_rt_unit: str):
    """Return a copy with the legacy internal column vocabulary and RT seconds."""

    result = frame.copy()
    modern = "rt_apex_sec" in result.columns
    for modern_name, internal_name in MODERN_TO_INTERNAL.items():
        if modern_name in result.columns and internal_name not in result.columns:
            result[internal_name] = result[modern_name]
    if "FAIMS" not in result.columns:
        result["FAIMS"] = None
    if "im" not in result.columns:
        result["im"] = None
    if not modern and input_rt_unit == "minutes":
        for column in ("rtStart", "rtApex", "rtEnd"):
            result[column] = pd.to_numeric(result[column], errors="raise") * 60.0
    return result


def assign_deterministic_hill_ids(hills, first_id=1):
    """Assign content-ordered IDs without changing aligned hill array positions."""
    count = len(hills.get("hills_idx_array_unique", ()))
    if not count:
        return int(first_id)

    mz = np.asarray(hills["hills_mz_median"], dtype=np.float64)
    starts = np.fromiter(
        (int(scans[0]) for scans in hills["hills_scan_lists"]),
        dtype=np.int64,
        count=count,
    )
    ends = np.fromiter(
        (int(scans[-1]) for scans in hills["hills_scan_lists"]),
        dtype=np.int64,
        count=count,
    )
    lengths = np.asarray(hills["hills_lengths"], dtype=np.int64)
    ordered_indexes = np.lexsort((lengths, ends, starts, mz))
    assigned = np.empty(count, dtype=np.int64)
    assigned[ordered_indexes] = np.arange(
        int(first_id), int(first_id) + count, dtype=np.int64
    )
    hills["hills_idx_array_unique"] = assigned
    return int(first_id) + count
