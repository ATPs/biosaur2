"""Identification-aware direct assays and bounded local feature extraction."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Optional

import numpy as np

from .quantification import quantify_feature_traces
from .raw_ms1 import RawMS1Store, event_position_in_trace
from .optimization import ConflictCandidate, select_conflict_candidates
from .local_refinement import refine_local_isotope_components
from .hybrid_assays import DirectAssay, LocalFeatureCandidate


def _positive_segments(present: np.ndarray, max_gap_scans: int):
    positive = np.flatnonzero(present)
    if positive.size == 0:
        return []
    segments = []
    start = previous = int(positive[0])
    for value in positive[1:]:
        value = int(value)
        if value - previous - 1 > max_gap_scans:
            segments.append((start, previous + 1))
            start = value
        previous = value
    segments.append((start, previous + 1))
    return segments


def extract_local_feature(
    store: RawMS1Store,
    assay: DirectAssay,
    *,
    ppm: float = 8.0,
    rt_tolerance_sec: float = 120.0,
    min_mono_points: int = 3,
    max_gap_scans: int = 1,
    quant_method: str = "envelope_area",
    baseline: str = "none",
    allow_two_point_exception: bool = False,
    allow_partial_envelope: bool = False,
    mz_shift_ppm: float = 0.0,
    rt_center_sec: Optional[float] = None,
    require_event_scan: bool = True,
    _preextracted_traces=None,
) -> LocalFeatureCandidate:
    """Extract, repair and jointly segment a bounded exact-assay region.

    ``_preextracted_traces`` is an internal reuse hook for project external
    strict/weak gates.  It avoids repeating the expensive sparse MS1 lookup;
    all candidate construction and quality semantics remain unchanged.
    """

    selected_peaks = tuple(
        peak
        for peak in assay.isotope_peaks
        if peak.isotope_index == 0 or peak.relative_abundance >= 0.01
    )
    extraction_rt_center = (
        float(assay.rt_sec)
        if rt_center_sec is None
        else float(rt_center_sec)
    )
    traces = _preextracted_traces
    if traces is None:
        traces = store.extract_traces(
            tuple(
                peak.mz * (1.0 + float(mz_shift_ppm) * 1e-6)
                for peak in selected_peaks
            ),
            ppm,
            extraction_rt_center - rt_tolerance_sec,
            extraction_rt_center + rt_tolerance_sec,
            faims_cv=assay.faims_cv,
        )
    if not traces or traces[0].rt_sec.size == 0:
        return LocalFeatureCandidate(
            assay,
            "no_ms1_scans_in_window",
            False,
            None,
            None,
            None,
            None,
            0,
            0,
            None,
            None,
            None,
            traces,
            None,
        )
    matrix = np.stack([trace.intensity for trace in traces])
    combined = np.sum(matrix, axis=0, dtype=np.float64)
    if not np.any(combined > 0):
        return LocalFeatureCandidate(
            assay,
            "no_signal",
            False,
            None,
            None,
            None,
            None,
            0,
            0,
            None,
            None,
            None,
            traces,
            None,
        )
    theoretical = np.asarray(
        [peak.probability for peak in selected_peaks], dtype=np.float64
    )
    refinement = refine_local_isotope_components(
        matrix,
        theoretical,
        max_gap_scans=max_gap_scans,
    )
    if not refinement.components:
        return LocalFeatureCandidate(
            assay,
            "no_signal",
            False,
            None,
            None,
            None,
            None,
            0,
            0,
            None,
            None,
            None,
            traces,
            None,
            edit_history=refinement.edits,
        )

    event_position = event_position_in_trace(
        traces[0], assay.rt_sec, assay.precursor_ms1_index
    )
    eligible = []
    for index, component in enumerate(refinement.components):
        allocated = component.allocated_matrix
        envelope = np.sum(allocated, axis=0, dtype=np.float64)
        mono_points = int(np.count_nonzero(allocated[0] > 0))
        all_points = int(np.count_nonzero(envelope > 0))
        interval_distance = (
            0
            if component.start <= event_position < component.end
            else min(
                abs(event_position - component.start),
                abs(event_position - (component.end - 1)),
            )
        )
        eligible.append(
            (
                mono_points >= min_mono_points,
                -interval_distance,
                -abs(event_position - component.apex),
                float(np.sum(envelope)),
                -component.start,
                index,
                mono_points,
                all_points,
            )
        )
    eligible.sort(reverse=True)
    (
        _valid,
        _interval_distance,
        _apex_distance,
        _intensity,
        _negative_start,
        component_index,
        mono_points,
        all_points,
    ) = eligible[0]
    component = refinement.components[component_index]
    return _finalize_local_component(
        assay, selected_peaks, traces, theoretical, refinement, component,
        event_position, mono_points, all_points, min_mono_points, quant_method,
        baseline, allow_two_point_exception, allow_partial_envelope,
        require_event_scan,
    )


def _finalize_local_component(
    assay, selected_peaks, traces, theoretical, refinement, component,
    event_position, mono_points, all_points, min_mono_points, quant_method,
    baseline, allow_two_point_exception, allow_partial_envelope,
    require_event_scan,
):
    start, end = component.start, component.end
    segment_rt = traces[0].rt_sec[start:end]
    segment_values = tuple(
        np.asarray(values, dtype=np.float64)
        for values in component.allocated_matrix
    )
    selected_trace_position = next(
        (
            index
            for index, peak in enumerate(selected_peaks)
            if peak.isotope_index == assay.selected_isotope_index
        ),
        0,
    )
    event_inside_component = start <= event_position < end
    selected_allocated_at_event = (
        event_inside_component
        and segment_values[selected_trace_position][event_position - start] > 0
    )
    if require_event_scan and (
        not event_inside_component or not selected_allocated_at_event
    ):
        return LocalFeatureCandidate(
            assay=assay,
            status=(
                "component_not_at_event_scan"
                if not event_inside_component
                else "selected_isotope_absent_at_event_scan"
            ),
            quantitative=False,
            rt_start_sec=None,
            rt_apex_sec=None,
            rt_end_sec=None,
            scan_apex=None,
            point_count=all_points,
            mono_point_count=mono_points,
            isotope_cosine=None,
            mono_mz_error_ppm=None,
            quantification=None,
            traces=traces,
            segment_slice=(start, end),
            allocated_trace_values=segment_values,
            edit_history=refinement.edits,
            component_count=len(refinement.components),
            allocation_component_index=component.allocation_index,
            deconvolution_status=component.deconvolution_status,
            intensity_conserved=component.intensity_conserved,
        )
    envelope = np.sum(np.stack(segment_values), axis=0, dtype=np.float64)
    boundary_truncated = bool(
        (start == 0 and envelope.size and envelope[0] > 0)
        or (
            end == traces[0].rt_sec.size
            and envelope.size
            and envelope[-1] > 0
        )
    )
    apex_local = int(np.argmax(envelope))
    apex_global = start + apex_local
    integrated = np.asarray(
        [np.trapezoid(values, segment_rt) for values in segment_values],
        dtype=np.float64,
    )
    denominator = float(
        np.linalg.norm(integrated) * np.linalg.norm(theoretical)
    )
    cosine = (
        None
        if denominator == 0
        else float(np.dot(integrated, theoretical) / denominator)
    )
    channel_point_counts = [
        int(np.count_nonzero(values)) for values in segment_values
    ]
    two_point_exception = (
        allow_two_point_exception
        and mono_points == 2
        and sum(count >= 2 for count in channel_point_counts) >= 2
        and start <= event_position < end
        and bool(traces[selected_trace_position].point_present[event_position])
        and cosine is not None
        and cosine >= 0.9
    )
    partial_envelope_exception = (
        allow_partial_envelope
        and mono_points < min_mono_points
        and assay.selected_isotope_index is not None
        and assay.selected_isotope_index > 0
        and selected_trace_position > 0
        and channel_point_counts[selected_trace_position] >= 3
        and sum(count >= 3 for count in channel_point_counts[1:]) >= 2
        and start <= event_position < end
        and bool(traces[selected_trace_position].point_present[event_position])
        and cosine is not None
        and cosine >= 0.9
    )
    if (
        mono_points < min_mono_points
        and not two_point_exception
        and not partial_envelope_exception
    ):
        status = "precursor_signal_only" if all_points else "no_signal"
        return LocalFeatureCandidate(
            assay,
            status,
            False,
            None,
            None,
            None,
            None,
            all_points,
            mono_points,
            cosine,
            None,
            None,
            traces,
            (start, end),
            segment_values,
            refinement.edits,
            len(refinement.components),
            None,
            component.allocation_index,
            component.deconvolution_status,
            component.intensity_conserved,
        )

    observed_mz = traces[0].observed_mz[apex_global]
    mono_error = (
        None
        if not np.isfinite(observed_mz)
        else float(
            (observed_mz - selected_peaks[0].mz)
            * 1e6
            / selected_peaks[0].mz
        )
    )
    quantification = quantify_feature_traces(
        segment_rt,
        segment_values,
        method=quant_method,
        baseline=baseline,
    )
    quantitative = (
        quantification.value is not None and quantification.value > 0
    )
    allocation_group_key = None
    if component.source == "identifiable_nnls":
        related = [
            value
            for value in refinement.components
            if value.allocation_group == component.allocation_group
        ]
        apex_scans = sorted(
            int(traces[0].scan_index[value.apex]) for value in related
        )
        allocation_group_key = "%d:%.8f:%d:%d:%s" % (
            assay.charge,
            selected_peaks[0].mz,
            int(traces[0].scan_index[start]),
            int(traces[0].scan_index[end - 1]),
            ",".join(str(value) for value in apex_scans),
        )
    if quantitative and partial_envelope_exception:
        status = "accepted_local_feature_partial_envelope"
    elif quantitative and two_point_exception:
        status = "accepted_local_feature_two_point"
    elif quantitative and component.source == "identifiable_nnls":
        status = "accepted_local_feature_deconvolved"
    elif quantitative:
        status = "accepted_local_feature"
    else:
        status = "quantification_failed"
    return LocalFeatureCandidate(
        assay=assay,
        status=status,
        quantitative=quantitative,
        rt_start_sec=float(segment_rt[0]),
        rt_apex_sec=float(segment_rt[apex_local]),
        rt_end_sec=float(segment_rt[-1]),
        scan_apex=int(traces[0].scan_number[apex_global]),
        point_count=all_points,
        mono_point_count=mono_points,
        isotope_cosine=cosine,
        mono_mz_error_ppm=mono_error,
        quantification=quantification,
        traces=traces,
        segment_slice=(start, end),
        allocated_trace_values=segment_values,
        edit_history=refinement.edits,
        component_count=len(refinement.components),
        allocation_group_key=allocation_group_key,
        allocation_component_index=component.allocation_index,
        deconvolution_status=component.deconvolution_status,
        intensity_conserved=component.intensity_conserved,
        boundary_truncated=boundary_truncated,
    )


def _faims_equal(left, right):
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), abs_tol=1e-6)


def _local_feature_equivalent(left, right, ppm):
    """Return whether two exact-assay paths address one raw MS1 component."""

    if left.assay.charge != right.assay.charge:
        return False
    if not _faims_equal(left.assay.faims_cv, right.assay.faims_cv):
        return False
    left_mz = left.assay.isotope_peaks[0].mz
    right_mz = right.assay.isotope_peaks[0].mz
    if abs(left_mz - right_mz) * 1e6 / left_mz > ppm:
        return False
    if max(left.rt_start_sec, right.rt_start_sec) > min(
        left.rt_end_sec, right.rt_end_sec
    ):
        return False
    left_group = getattr(left, "allocation_group_key", None)
    right_group = getattr(right, "allocation_group_key", None)
    if left_group is not None and left_group == right_group:
        return (
            left.allocation_component_index
            == right.allocation_component_index
        )
    left_width = left.rt_end_sec - left.rt_start_sec
    right_width = right.rt_end_sec - right.rt_start_sec
    apex_tolerance = max(3.0, 0.25 * min(left_width, right_width))
    return abs(left.rt_apex_sec - right.rt_apex_sec) <= apex_tolerance


def _candidate_segment_values(candidate):
    allocated = getattr(candidate, "allocated_trace_values", None)
    if allocated is not None:
        return tuple(np.asarray(value, dtype=np.float64) for value in allocated)
    start, end = candidate.segment_slice
    return tuple(trace.intensity[start:end] for trace in candidate.traces)


def _allocate_candidate_component(ledger, allocation_id, candidate):
    """Atomically claim the RT/intensity slice quantified for one feature."""

    start, _end = candidate.segment_slice
    return ledger.allocate_component(
        allocation_id,
        candidate.traces,
        start,
        _candidate_segment_values(candidate),
    )


def _local_candidate_raw_points(candidate):
    """Return stable scan/mz identities for raw points integrated by a candidate."""

    if candidate.segment_slice is None:
        return frozenset()
    start, end = candidate.segment_slice
    allocated = getattr(candidate, "allocated_trace_values", None)
    points = set()
    for trace_index, trace in enumerate(candidate.traces):
        present = (
            trace.point_present[start:end]
            if allocated is None
            else np.asarray(allocated[trace_index]) > 0
        )
        for position in np.flatnonzero(present):
            absolute = start + int(position)
            observed_mz = trace.observed_mz[absolute]
            if np.isfinite(observed_mz):
                points.add(
                    (
                        int(trace.scan_index[absolute]),
                        round(float(observed_mz), 6),
                    )
                )
    return frozenset(points)


def _protected_local_conflict(protected, challenger):
    protected_group = getattr(protected, "allocation_group_key", None)
    challenger_group = getattr(challenger, "allocation_group_key", None)
    if (
        protected_group is not None
        and protected_group == challenger_group
        and protected.intensity_conserved
        and challenger.intensity_conserved
        and protected.allocation_component_index
        != challenger.allocation_component_index
    ):
        return False
    protected_points = _local_candidate_raw_points(protected)
    challenger_points = _local_candidate_raw_points(challenger)
    if not protected_points or not (protected_points & challenger_points):
        return False
    protected_score = float(protected.isotope_cosine or 0.0)
    challenger_score = float(challenger.isotope_cosine or 0.0)
    selection = select_conflict_candidates(
        [
            ConflictCandidate(
                "protected", protected_score, protected_points, protected=True
            ),
            ConflictCandidate(
                "challenger", challenger_score, challenger_points
            ),
        ]
    )
    return "challenger" not in selection.selected_ids


def _strict_record_raw_points(record):
    """Return stable raw-point identities owned by one strict candidate."""

    # This lives in the strict-record module, while local candidates also use
    # the point index below.  Import lazily to keep those modules acyclic.
    from .hybrid_strict import _strict_feature_observed_contributions

    return frozenset(
        (int(scan), round(float(mz), 6))
        for scan, mz, _intensity in _strict_feature_observed_contributions(
            record
        )
    )


def _build_final_strict_raw_point_index(records):
    """Index residual strict competitors by original centroid identity."""

    index = defaultdict(list)
    for record in records:
        for point in _strict_record_raw_points(record):
            index[point].append(record)
    return {
        point: tuple(sorted(values, key=lambda row: int(row["feature_id"])))
        for point, values in index.items()
    }


def _final_strict_protection_reason(challenger, raw_point_index, ppm):
    """Protect a final-strict competitor unless an equivalent MS2 model wins.

    Both candidate families are generated from the same residual state.  An
    equivalent MS2-guided candidate is preferred when strict MS1 evidence is
    not clearly better.  Raw-point overlap between non-equivalent envelopes is
    not identifiable with the current same-envelope NNLS model, so the weaker
    relaxed round must leave that signal for strict untargeted detection.
    """

    challenger_points = _local_candidate_raw_points(challenger)
    competitors = {}
    for point in challenger_points:
        for record in raw_point_index.get(point, ()):
            competitors[int(record["feature_id"])] = record
    if not competitors:
        return None

    challenger_mz = float(challenger.mono_mz)
    challenger_charge = int(challenger.event["charge"])
    challenger_faims = challenger.event.get("faims_cv")
    challenger_cosine = float(challenger.isotope_cosine or 0.0)
    challenger_points_count = int(
        getattr(challenger, "mono_points", 0)
        or getattr(challenger, "mono_point_count", 0)
    )
    for record in competitors.values():
        equivalent = (
            int(record["charge"]) == challenger_charge
            and _faims_equal(record["faims_cv"], challenger_faims)
            and abs(float(record["mz"]) - challenger_mz)
            * 1e6
            / challenger_mz
            <= float(ppm)
            and max(float(record["rt_start"]), challenger.rt_start_sec)
            <= min(float(record["rt_end"]), challenger.rt_end_sec)
        )
        if not equivalent:
            return "unidentifiable_cross_candidate_overlap"

        strict_candidate = record["candidate"]
        strict_cosine = float(strict_candidate.get("cos_cor_isotopes", 0.0))
        mono_hill = int(strict_candidate["monoisotope idx"])
        strict_points_count = len(
            record["hills"]["hills_scan_lists"][mono_hill]
        )
        if strict_cosine >= challenger_cosine + 0.01:
            return "superior_equivalent_strict_isotope_fit"
        if (
            strict_points_count >= challenger_points_count + 2
            and strict_cosine >= challenger_cosine - 0.02
        ):
            return "superior_equivalent_strict_chromatography"
    return None



__all__ = [name for name in globals() if not name.startswith("__")]
