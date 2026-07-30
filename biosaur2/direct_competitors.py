"""Bounded exact-assay capture of valid processed-hill competitors.

The strict detector destructively resolves hill ownership after isotope/mass
and averagine filtering.  This module captures only q-filtered direct-assay
relevant candidates immediately before that conflict pass, so a later local
optimizer can compare accepted and losing strict representations without
persisting the complete candidate population.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DirectProcessedHillCompetitor:
    ms2_event_id: int
    psm_id: str
    candidate_key: tuple
    candidate: Mapping
    mono_mz_error_ppm: float
    rt_error_sec: float
    precursor_scan_distance: int | None
    selected_isotope_index: int | None
    selected_isotope_observed: bool | None
    isolation_supported: bool | None
    evidence_score: float


def _faims_equal(left, right):
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= 1e-6


def _candidate_key(candidate):
    return (
        int(candidate["monoisotope hill idx"]),
        int(candidate["charge"]),
        tuple(
            (
                int(value["isotope_number"]),
                int(value["isotope_hill_idx"]),
            )
            for value in candidate["isotopes"]
        ),
    )


def _hill_index_by_isotope(candidate):
    result = {0: int(candidate["monoisotope idx"])}
    result.update(
        {
            int(value["isotope_number"]): int(value["isotope_idx"])
            for value in candidate["isotopes"]
        }
    )
    return result


def _rt_distance(rt, start, end):
    if start <= rt <= end:
        return 0.0
    return min(abs(rt - start), abs(rt - end))


def _isolation_window(event):
    target = event.get("isolation_target_mz")
    lower = event.get("isolation_lower_offset")
    upper = event.get("isolation_upper_offset")
    if target is None or lower is None or upper is None:
        return None
    return float(target) - float(lower), float(target) + float(upper)


def _observed_at_precursor(
    hill_index,
    hills,
    precursor_source_scan,
    source_by_local,
    isolation_window,
):
    if precursor_source_scan is None:
        return None, None
    best_distance = None
    isolated = False if isolation_window is not None else None
    scans = hills["hills_scan_lists"][hill_index]
    mz_values = hills["tmp_mz_array"][hill_index]
    for local_scan, observed_mz in zip(scans, mz_values):
        source_scan = source_by_local[int(local_scan)]
        distance = abs(int(source_scan) - int(precursor_source_scan))
        if distance > 1:
            continue
        if best_distance is None or distance < best_distance:
            best_distance = distance
        if isolation_window is not None and (
            isolation_window[0]
            <= float(observed_mz)
            <= isolation_window[1]
        ):
            isolated = True
    return best_distance, isolated


def capture_direct_processed_hill_competitors(
    assays: Sequence,
    candidates: Sequence[Mapping],
    hills: Mapping,
    rt_by_local: Mapping,
    spectra: Sequence[Mapping],
    events_by_id: Mapping[int, Mapping],
    *,
    ppm: float,
    rt_tolerance_sec: float,
    top_k: int = 3,
):
    """Capture a deterministic bounded set of exact-assay strict candidates.

    ``candidates`` must already have passed ordinary strict isotope mass and
    cosine filters.  PSM evidence narrows and ranks this valid population; it
    never turns an invalid candidate into a feature.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if ppm <= 0 or rt_tolerance_sec < 0:
        raise ValueError("ppm must be positive and RT tolerance nonnegative")
    source_by_local = {
        local: int(spectrum["scan_index"])
        for local, spectrum in enumerate(spectra)
    }
    by_charge_candidates = {}
    for candidate in candidates:
        by_charge_candidates.setdefault(
            int(candidate["charge"]), []
        ).append(candidate)
    by_charge = {}
    for charge, values in by_charge_candidates.items():
        values.sort(
            key=lambda value: (
                float(value["hill_mz_1"]),
                _candidate_key(value),
            )
        )
        by_charge[charge] = (
            tuple(float(value["hill_mz_1"]) for value in values),
            tuple(values),
        )

    captured = []
    for assay in sorted(
        assays,
        key=lambda value: (int(value.ms2_event_id), str(value.psm_id)),
    ):
        event = events_by_id.get(int(assay.ms2_event_id), {})
        theoretical_mono = float(assay.isotope_peaks[0].mz)
        precursor_scan = event.get(
            "precursor_ms1_index", assay.precursor_ms1_index
        )
        window = _isolation_window(event)
        matches = []
        mz_values, charge_candidates = by_charge.get(
            int(assay.charge), ((), ())
        )
        mz_tolerance = theoretical_mono * float(ppm) * 1e-6
        start = bisect_left(mz_values, theoretical_mono - mz_tolerance)
        end = bisect_right(mz_values, theoretical_mono + mz_tolerance)
        for candidate in charge_candidates[start:end]:
            if not _faims_equal(candidate.get("FAIMS"), assay.faims_cv):
                continue
            mono_mz = float(candidate["hill_mz_1"])
            mz_error = (mono_mz - theoretical_mono) * 1e6 / theoretical_mono
            if abs(mz_error) > float(ppm):
                continue
            mono_hill = int(candidate["monoisotope idx"])
            mono_scans = hills["hills_scan_lists"][mono_hill]
            rt_start = float(rt_by_local[int(mono_scans[0])])
            rt_end = float(rt_by_local[int(mono_scans[-1])])
            rt_error = _rt_distance(float(assay.rt_sec), rt_start, rt_end)
            if rt_error > float(rt_tolerance_sec):
                continue

            hill_by_isotope = _hill_index_by_isotope(candidate)
            scan_distances = []
            any_isolated = False if window is not None else None
            for hill_index in hill_by_isotope.values():
                distance, isolated = _observed_at_precursor(
                    hill_index,
                    hills,
                    precursor_scan,
                    source_by_local,
                    window,
                )
                if distance is not None:
                    scan_distances.append(distance)
                if isolated:
                    any_isolated = True
            if precursor_scan is not None and not scan_distances:
                continue
            if window is not None and not any_isolated:
                continue

            selected_index = assay.selected_isotope_index
            selected_observed = None
            if selected_index is not None:
                selected_hill = hill_by_isotope.get(int(selected_index))
                if selected_hill is None:
                    selected_observed = False
                else:
                    selected_distance, selected_isolated = (
                        _observed_at_precursor(
                            selected_hill,
                            hills,
                            precursor_scan,
                            source_by_local,
                            window,
                        )
                    )
                    selected_observed = (
                        selected_distance is not None
                        and (window is None or bool(selected_isolated))
                    )
                if precursor_scan is not None and not selected_observed:
                    continue

            cosine = float(candidate.get("cos_cor_isotopes", 0.0))
            isotope_count = int(candidate.get("nIsotopes", 0))
            mass_support = max(0.0, 1.0 - abs(mz_error) / float(ppm))
            rt_support = max(
                0.0,
                1.0 - rt_error / max(float(rt_tolerance_sec), 1.0),
            )
            scan_distance = min(scan_distances) if scan_distances else None
            scan_support = (
                0.75
                if scan_distance is None
                else 1.0
                if scan_distance == 0
                else 0.8
            )
            selected_support = (
                0.75 if selected_observed is None else float(selected_observed)
            )
            score = (
                0.35 * cosine
                + 0.20 * mass_support
                + 0.15 * rt_support
                + 0.15 * scan_support
                + 0.10 * selected_support
                + 0.05 * min(isotope_count / 5.0, 1.0)
            )
            matches.append(
                DirectProcessedHillCompetitor(
                    int(assay.ms2_event_id),
                    str(assay.psm_id),
                    _candidate_key(candidate),
                    deepcopy(candidate),
                    float(mz_error),
                    float(rt_error),
                    scan_distance,
                    None if selected_index is None else int(selected_index),
                    selected_observed,
                    any_isolated,
                    float(score),
                )
            )
        matches.sort(
            key=lambda value: (
                -value.evidence_score,
                abs(value.mono_mz_error_ppm),
                value.rt_error_sec,
                value.candidate_key,
            )
        )
        captured.extend(matches[:top_k])
    return tuple(captured)
