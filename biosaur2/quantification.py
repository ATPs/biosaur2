"""Raw feature-area quantification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class NormalizedTrace:
    rt: np.ndarray
    intensity: np.ndarray
    flags: set


def normalize_trace(rt: Iterable[float], intensity: Iterable[float]) -> NormalizedTrace:
    rt_values = np.asarray(list(rt), dtype=np.float64)
    intensity_values = np.asarray(list(intensity), dtype=np.float64)
    if rt_values.size != intensity_values.size:
        raise ValueError("RT and intensity arrays must have equal length")

    flags = set()
    finite = np.isfinite(rt_values) & np.isfinite(intensity_values)
    if not np.all(finite):
        flags.add("nonfinite_points_removed")
        rt_values = rt_values[finite]
        intensity_values = intensity_values[finite]
    if np.any(intensity_values < 0):
        flags.add("negative_intensity")
    if rt_values.size and np.any(np.diff(rt_values) < 0):
        flags.add("nonmonotonic_rt_sorted")

    order = np.argsort(rt_values, kind="stable")
    rt_values = rt_values[order]
    intensity_values = intensity_values[order]
    if rt_values.size:
        unique_rt, inverse = np.unique(rt_values, return_inverse=True)
        if unique_rt.size != rt_values.size:
            flags.add("duplicate_rt_merged")
            summed = np.zeros(unique_rt.size, dtype=np.float64)
            np.add.at(summed, inverse, intensity_values)
            rt_values = unique_rt
            intensity_values = summed
    return NormalizedTrace(rt_values, intensity_values, flags)


def trapezoid_area(trace: NormalizedTrace, intensity: Optional[np.ndarray] = None):
    if trace.rt.size < 2:
        return None
    values = trace.intensity if intensity is None else intensity
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(values, trace.rt))


def _approximate_rt(
    scans: Sequence[int],
    scan_apex: int,
    rt_start: float,
    rt_apex: float,
    rt_end: float,
) -> List[float]:
    if not scans:
        return []
    scan_start = scans[0]
    scan_end = scans[-1]
    result = []
    for scan in scans:
        if scan <= scan_apex:
            denominator = scan_apex - scan_start
            fraction = 0.0 if denominator == 0 else (scan - scan_start) / denominator
            result.append(rt_start + fraction * (rt_apex - rt_start))
        else:
            denominator = scan_end - scan_apex
            fraction = 0.0 if denominator == 0 else (scan - scan_apex) / denominator
            result.append(rt_apex + fraction * (rt_end - rt_apex))
    return result


def _selected_isotopes(feature: Mapping[str, Any], isotope_count: int):
    candidates = list(feature.get("isotopes", ()))
    if isotope_count == 0:
        candidates = []
    elif isotope_count > 0:
        candidates = candidates[:isotope_count]
    return [(0, int(feature["monoisotope idx"]), None)] + [
        (int(candidate["isotope_number"]), int(candidate["isotope_idx"]), candidate)
        for candidate in candidates
    ]


def raw_area_sum(
    hills: Mapping[str, Any],
    feature: Mapping[str, Any],
    rt_by_local_scan: Optional[Mapping[int, float]],
    isotope_count: int,
) -> Tuple[Optional[float], bool]:
    """Return the selected-isotope raw area sum without materializing rows."""

    total = 0.0
    approximate = False
    stored_rt = hills.get("hills_point_rt_array")
    for _ordinal, hill_index, _candidate in _selected_isotopes(
        feature, isotope_count
    ):
        try:
            scans = [int(value) for value in hills["hills_scan_lists"][hill_index]]
            intensities = hills["hills_intensity_array"][hill_index]
        except (IndexError, KeyError, TypeError):
            return None, approximate
        if stored_rt is not None:
            try:
                rt_values = stored_rt[hill_index]
            except (IndexError, TypeError):
                return None, approximate
        elif rt_by_local_scan is not None:
            try:
                rt_values = [rt_by_local_scan[scan] for scan in scans]
            except KeyError:
                return None, approximate
        else:
            approximate = True
            apex_position = int(np.argmax(intensities)) if len(intensities) else 0
            rt_values = _approximate_rt(
                scans,
                scans[apex_position] if scans else 0,
                float(hills["rtStart"][hill_index]),
                float(hills["rtApex"][hill_index]),
                float(hills["rtEnd"][hill_index]),
            )
        if len(rt_values) != len(intensities):
            return None, approximate
        trace = normalize_trace(rt_values, intensities)
        area = trapezoid_area(trace)
        if area is None:
            return None, approximate
        total += area
    return total, approximate
