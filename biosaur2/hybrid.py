"""Identification-aware direct assays and bounded local feature extraction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from dataclasses import replace
from bisect import bisect_left, bisect_right
import logging
import math
from typing import Mapping, Optional, Sequence

import numpy as np

from .chemistry import IsotopePeak, Peptidoform, isotope_library, parse_peptidoform
from .confidence import (
    TargetDecoyCompetition,
    deterministic_decoy_shift,
    target_decoy_q_values,
)
from .identifications import IdentificationMappingResult
from .generic_local import (
    cluster_compatible_generic_candidates,
    compete_generic_local_candidates,
    evaluate_generic_local_candidate_pairs,
    generic_local_width_limit,
)
from .generic_association import (
    GENERIC_ASSOCIATION_SCORE_WEIGHT_ITEMS,
    GENERIC_ASSOCIATION_SCORE_WEIGHTS,
    annotate_candidate_association,
    build_association_rows,
    composite_association_support,
    prepare_association_context,
    precursor_joint_support,
)
from .quantification import FeatureQuantification, quantify_feature_traces
from .raw_ms1 import ExtractedTrace, RawMS1Store, event_position_in_trace
from .residual import ResidualMS1Ledger
from .optimization import ConflictCandidate, select_conflict_candidates
from .local_refinement import SegmentEdit, refine_local_isotope_components
from .postprocess_cache import (
    load_local_candidate_pairs,
    local_candidate_fingerprint,
    save_local_candidate_pairs,
)


logger = logging.getLogger(__name__)


FEATURE_ORIGIN_DIRECT_IDENTIFIED = "direct_identified"
FEATURE_ORIGIN_ALIGNED_EXTERNAL = "aligned_external"
FEATURE_ORIGIN_STRICT_UNTARGETED = "strict_untargeted"
FEATURE_ORIGIN_MS2_GUIDED_FULL = "ms2_guided_full"
FEATURE_ORIGIN_MS2_GUIDED_PARTIAL = "ms2_guided_partial"
FEATURE_ORIGIN_MS2_GUIDED_MONO_ONLY = "ms2_guided_mono_only"


RELAXED_DIRECT_Q_VALUE_MAX = 0.01
QUALITY_FLAG_RELAXED_MS2_FEATURE = 0x0001
QUALITY_FLAG_BOUNDARY_TRUNCATED = 0x0002
QUALITY_FLAG_TWO_POINT_QUANT = 0x0004
QUALITY_FLAG_RAW_BASELINE_FALLBACK = 0x0008
GENERIC_SCORE_CALIBRATION_MIN_PAIRED_ANCHORS = 40
GENERIC_SCORE_CALIBRATION_PRIOR_FRACTIONS = (0.95, 0.90, 0.80, 0.70, 0.60, 0.50)
GENERIC_LOCAL_REFINEMENT_INPUT_STATUSES = frozenset(
    {
        "generic_no_standard_candidate",
        "generic_q_value_rejected",
        "generic_decoy_won",
        "generic_decoy_only",
    }
)


def _append_final_strict_features(manager, strict_contexts, args):
    """Emit strict rows only after hybrid targeted/conflict decisions finish."""

    from . import utils

    for context in strict_contexts:
        rows = utils.calc_peptide_features(
            context["hills"],
            context["candidates"],
            args["nm"],
            context["faims_cv"],
            context["rt_by_local"],
            0,
            args["iuse"],
            include_mono_hills=not args.get("no_mono_hills", False),
            quantification_args=args,
            spectra=context["spectra"],
        )
        manager.append_features(rows)


@dataclass(frozen=True)
class DirectAssay:
    run_id: str
    ms2_event_id: int
    psm_id: str
    canonical_peptidoform: str
    charge: int
    rt_sec: float
    faims_cv: Optional[float]
    selected_ion_mz: Optional[float]
    selected_isotope_index: Optional[int]
    selected_mz_error_ppm: Optional[float]
    peptidoform: Peptidoform
    isotope_peaks: tuple[IsotopePeak, ...]
    q_value: float
    pep: Optional[float]
    score: Optional[float]
    rank: Optional[int]
    precursor_ms1_index: Optional[int] = None
    conflict_status: str = "unique"


@dataclass(frozen=True)
class AssayBuildResult:
    assays: tuple[DirectAssay, ...]
    audit: tuple[Mapping, ...]
    status_counts: Mapping[str, int]


@dataclass(frozen=True)
class LocalFeatureCandidate:
    assay: DirectAssay
    status: str
    quantitative: bool
    rt_start_sec: Optional[float]
    rt_apex_sec: Optional[float]
    rt_end_sec: Optional[float]
    scan_apex: Optional[int]
    point_count: int
    mono_point_count: int
    isotope_cosine: Optional[float]
    mono_mz_error_ppm: Optional[float]
    quantification: Optional[FeatureQuantification]
    traces: tuple[ExtractedTrace, ...]
    segment_slice: Optional[tuple[int, int]]
    allocated_trace_values: Optional[tuple[np.ndarray, ...]] = None
    edit_history: tuple[SegmentEdit, ...] = ()
    component_count: int = 0
    allocation_group_key: Optional[str] = None
    allocation_component_index: Optional[int] = None
    deconvolution_status: Optional[str] = None
    intensity_conserved: bool = True
    refinement_round: int = 0
    boundary_truncated: bool = False


@dataclass(frozen=True)
class DirectRunCalibration:
    status: str
    anchor_count: int
    mass_error_center_ppm: float
    mass_error_mad_ppm: float
    rt_apex_offset_sec: float
    rt_apex_offset_mad_sec: float
    width_median_sec: Optional[float]
    width_p95_sec: Optional[float]
    retry_ppm: float
    retry_rt_tolerance_sec: float

    def as_dict(self):
        return {
            "status": self.status,
            "anchor_count": self.anchor_count,
            "mass_error_center_ppm": self.mass_error_center_ppm,
            "mass_error_mad_ppm": self.mass_error_mad_ppm,
            "rt_apex_offset_sec": self.rt_apex_offset_sec,
            "rt_apex_offset_mad_sec": self.rt_apex_offset_mad_sec,
            "width_median_sec": self.width_median_sec,
            "width_p95_sec": self.width_p95_sec,
            "retry_ppm": self.retry_ppm,
            "retry_rt_tolerance_sec": self.retry_rt_tolerance_sec,
        }


def _assay_sort_key(assay: DirectAssay):
    return (
        assay.ms2_event_id,
        assay.q_value,
        math.inf if assay.pep is None else assay.pep,
        -(assay.score if assay.score is not None else -math.inf),
        math.inf if assay.rank is None else assay.rank,
        assay.canonical_peptidoform,
        assay.psm_id,
    )


def _direct_relaxed_retry_enabled(assay: DirectAssay, args: Mapping):
    return bool(args.get("relaxed_ms2_feature", False)) and (
        float(assay.q_value) < RELAXED_DIRECT_Q_VALUE_MAX
    )


def build_direct_assays(
    mapping: IdentificationMappingResult,
    *,
    run_id: str,
    fixed_modifications: Sequence[str] = (),
    precursor_ppm: float = 5.0,
    max_isotopes: int = 6,
) -> AssayBuildResult:
    """Validate mapped PSM chemistry and deterministically resolve duplicates."""

    from collections import Counter, defaultdict

    candidates = []
    audit = []
    status_counts = Counter()
    for mapped in mapping.rows:
        record = mapped.identification
        base_audit = {
            "run_id": run_id,
            "psm_id": record.psm_id_raw,
            "ms2_event_id": mapped.ms2_event_id,
            "mapping_status": mapped.mapping_status,
            "formula_status": None,
            "assay_status": None,
            "q_value": record.q_value,
            "pep": record.pep,
            "score": record.score,
            "rank": record.parsed_rank,
            "peptide_raw": record.peptide_raw,
        }
        if mapped.ms2_event_id is None or mapped.event is None:
            base_audit["assay_status"] = "unmapped_psm"
            status_counts["unmapped_psm"] += 1
            audit.append(base_audit)
            continue
        if mapped.charge_agreement is False:
            base_audit["assay_status"] = "charge_mismatch"
            status_counts["charge_mismatch"] += 1
            audit.append(base_audit)
            continue
        peptidoform = parse_peptidoform(
            record.peptide_raw, fixed_modifications=fixed_modifications
        )
        base_audit["formula_status"] = peptidoform.formula_status
        if peptidoform.formula_status != "exact" or peptidoform.formula is None:
            base_audit["assay_status"] = "non_exact_formula"
            status_counts["non_exact_formula"] += 1
            audit.append(base_audit)
            continue
        charge = record.parsed_charge
        event = mapped.event
        if charge is None or event.get("rt_sec") is None:
            base_audit["assay_status"] = "missing_charge_or_rt"
            status_counts["missing_charge_or_rt"] += 1
            audit.append(base_audit)
            continue
        peaks = isotope_library(peptidoform.formula, int(charge), max_isotopes=max_isotopes)
        selected = event.get("selected_ion_mz")
        isotope_index = None
        mz_error = None
        if selected is not None:
            mz_error, isotope_index = min(
                (
                    abs(float(selected) - peak.mz) * 1e6 / peak.mz,
                    peak.isotope_index,
                )
                for peak in peaks
            )
        if mz_error is None or mz_error > precursor_ppm:
            base_audit["assay_status"] = "precursor_formula_mismatch"
            base_audit["selected_mz_error_ppm"] = mz_error
            status_counts["precursor_formula_mismatch"] += 1
            audit.append(base_audit)
            continue
        assay = DirectAssay(
            run_id=run_id,
            ms2_event_id=int(mapped.ms2_event_id),
            psm_id=record.psm_id_raw,
            canonical_peptidoform=peptidoform.canonical,
            charge=int(charge),
            rt_sec=float(event["rt_sec"]),
            faims_cv=event.get("faims_cv"),
            selected_ion_mz=float(selected),
            selected_isotope_index=isotope_index,
            selected_mz_error_ppm=float(mz_error),
            peptidoform=peptidoform,
            isotope_peaks=peaks,
            q_value=record.q_value,
            pep=record.pep,
            score=record.score,
            rank=record.parsed_rank,
            precursor_ms1_index=event.get("precursor_ms1_index"),
        )
        candidates.append((assay, base_audit))

    grouped = defaultdict(list)
    for assay, row in candidates:
        grouped[assay.ms2_event_id].append((assay, row))
    assays = []
    for event_id in sorted(grouped):
        values = grouped[event_id]
        by_identity = defaultdict(list)
        for assay, row in values:
            key = (assay.canonical_peptidoform, assay.charge)
            by_identity[key].append((assay, row))
        unique = {}
        for key, duplicates in by_identity.items():
            duplicates.sort(key=lambda value: _assay_sort_key(value[0]))
            unique[key] = duplicates[0]
            for duplicate_assay, duplicate_row in duplicates[1:]:
                duplicate_row["assay_status"] = "duplicate_identification_collapsed"
                duplicate_row["canonical_peptidoform"] = duplicate_assay.canonical_peptidoform
                duplicate_row["selected_isotope_index"] = duplicate_assay.selected_isotope_index
                duplicate_row["selected_mz_error_ppm"] = duplicate_assay.selected_mz_error_ppm
                status_counts["duplicate_identification_collapsed"] += 1
                audit.append(duplicate_row)
        conflict = len(unique) > 1
        for assay, row in sorted(unique.values(), key=lambda value: _assay_sort_key(value[0])):
            if conflict:
                assay = DirectAssay(**{**assay.__dict__, "conflict_status": "conflicting_identifications"})
                row["assay_status"] = "conflicting_identifications"
                status_counts["conflicting_identifications"] += 1
            else:
                row["assay_status"] = "accepted_direct_assay"
                status_counts["accepted_direct_assay"] += 1
            row["canonical_peptidoform"] = assay.canonical_peptidoform
            row["selected_isotope_index"] = assay.selected_isotope_index
            row["selected_mz_error_ppm"] = assay.selected_mz_error_ppm
            audit.append(row)
            assays.append(assay)
    assays.sort(key=_assay_sort_key)
    audit.sort(key=lambda row: (row.get("ms2_event_id") is None, row.get("ms2_event_id") or -1, row["psm_id"]))
    return AssayBuildResult(tuple(assays), tuple(audit), dict(status_counts))


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
) -> LocalFeatureCandidate:
    """Extract, repair and jointly segment a bounded exact-assay region."""

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
    traces = tuple(
        store.extract_trace(
            peak.mz * (1.0 + float(mz_shift_ppm) * 1e-6),
            ppm,
            extraction_rt_center - rt_tolerance_sec,
            extraction_rt_center + rt_tolerance_sec,
            faims_cv=assay.faims_cv,
        )
        for peak in selected_peaks
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
    if not event_inside_component or not selected_allocated_at_event:
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


def _rt_distance(rt, start, end):
    if start <= rt <= end:
        return 0.0
    return min(abs(rt - start), abs(rt - end))


def _strict_feature_records(strict_contexts):
    records = []
    for context in strict_contexts:
        hills = context["hills"]
        rt = context["rt_by_local"]
        for candidate in context["candidates"]:
            mono = int(candidate["monoisotope idx"])
            scans = hills["hills_scan_lists"][mono]
            intensities = np.asarray(
                hills["hills_intensity_array"][mono], dtype=np.float64
            )
            apex_position = (
                int(np.argmax(intensities)) if intensities.size else 0
            )
            records.append(
                {
                    "feature_id": int(candidate["feature_idx"]),
                    "candidate": candidate,
                    "hills": hills,
                    "rt_by_local": rt,
                    "spectra": context["spectra"],
                    "faims_cv": context["faims_cv"],
                    "mz": float(candidate["hill_mz_1"]),
                    "charge": int(candidate["charge"]),
                    "rt_start": float(rt[int(scans[0])]),
                    "rt_apex": float(rt[int(scans[apex_position])]),
                    "rt_end": float(rt[int(scans[-1])]),
                }
            )
    records.sort(key=lambda row: row["feature_id"])
    return records


def _strict_feature_observed_contributions(record):
    """Return original scan/mz/intensity triples owned by one strict feature."""

    candidate = record["candidate"]
    hills = record["hills"]
    spectra = record["spectra"]
    hill_indices = [int(candidate["monoisotope idx"])] + [
        int(value["isotope_idx"]) for value in candidate["isotopes"]
    ]
    contributions = []
    for hill_index in hill_indices:
        scans = hills["hills_scan_lists"][hill_index]
        mz_values = hills["tmp_mz_array"][hill_index]
        intensities = hills["hills_intensity_array"][hill_index]
        if not (len(scans) == len(mz_values) == len(intensities)):
            raise ValueError("strict hill point arrays have inconsistent lengths")
        for local_scan, observed_mz, intensity in zip(
            scans, mz_values, intensities
        ):
            source_scan = spectra[int(local_scan)].get("scan_index")
            if source_scan is None:
                raise ValueError("strict context lacks source scan provenance")
            contributions.append(
                (int(source_scan), float(observed_mz), float(intensity))
            )
    return tuple(contributions)


def _allocate_strict_feature_population(ledger, strict_records):
    """Register accepted strict ownership before targeted residual searches."""

    statuses = Counter()
    failed_feature_ids = []
    for record in strict_records:
        feature_id = int(record["feature_id"])
        try:
            contributions = _strict_feature_observed_contributions(record)
            result = ledger.allocate_observed_points(
                ("strict", feature_id), contributions
            )
            status = result.status
        except (IndexError, KeyError, TypeError, ValueError):
            status = "invalid_strict_provenance"
        statuses[status] += 1
        if status != "accepted":
            failed_feature_ids.append(feature_id)
    return {
        "status_counts": dict(sorted(statuses.items())),
        "accepted_feature_count": statuses["accepted"],
        "failed_feature_count": len(failed_feature_ids),
        "failed_feature_ids": tuple(failed_feature_ids),
    }


def _strict_record_existing_equivalents(record, strict_index, ppm):
    """Return accepted features equivalent to one residual strict record."""

    result = []
    mz_values, records = strict_index.get(int(record["charge"]), ((), ()))
    if not mz_values:
        return result
    mz = float(record["mz"])
    tolerance = mz * float(ppm) * 1e-6
    start = bisect_left(mz_values, mz - tolerance)
    end = bisect_right(mz_values, mz + tolerance)
    for existing in records[start:end]:
        if not _faims_equal(record["faims_cv"], existing["faims_cv"]):
            continue
        if max(record["rt_start"], existing["rt_start"]) <= min(
            record["rt_end"], existing["rt_end"]
        ):
            result.append(existing)
    return result


def _feature_row_as_strict_record(row, origin):
    """Adapt an accepted local feature row for final residual de-duplication."""

    return {
        "feature_id": int(row["feature_idx"]),
        "mz": float(row["mz"]),
        "charge": int(row["charge"]),
        "rt_start": float(row["rtStart"]),
        "rt_apex": float(row["rtApex"]),
        "rt_end": float(row["rtEnd"]),
        "faims_cv": row.get("FAIMS"),
        "feature_origin": origin,
    }


def _filter_context_feature_ids(contexts, rejected_ids):
    rejected = {int(value) for value in rejected_ids}
    if not rejected:
        return list(contexts)
    filtered = []
    for context in contexts:
        candidates = [
            candidate
            for candidate in context["candidates"]
            if int(candidate["feature_idx"]) not in rejected
        ]
        if candidates:
            filtered.append({**context, "candidates": candidates})
    return filtered


def _strict_hill_claim_indexes(strict_contexts):
    """Index only hills already assigned to accepted strict features.

    The index remains hill-sized rather than raw-point-sized.  A local
    candidate is checked by source scan and the actual centroid m/z, so a
    nearby but unassigned hill is not accidentally protected.
    """

    indexes = []
    for context in strict_contexts:
        assigned = set()
        for candidate in context["candidates"]:
            assigned.add(int(candidate["monoisotope idx"]))
            assigned.update(
                int(value["isotope_idx"])
                for value in candidate["isotopes"]
            )
        hills = context["hills"]
        ordered = sorted(
            assigned,
            key=lambda index: (
                float(hills["hills_mz_median"][index]), index
            ),
        )
        indexes.append(
            {
                "faims_cv": context["faims_cv"],
                "hills": hills,
                "source_to_local": {
                    int(spectrum["scan_index"]): local
                    for local, spectrum in enumerate(context["spectra"])
                },
                "mz": tuple(
                    float(hills["hills_mz_median"][index])
                    for index in ordered
                ),
                "hill": tuple(ordered),
            }
        )
    return tuple(indexes)


def _candidate_faims(candidate):
    if hasattr(candidate, "assay"):
        return candidate.assay.faims_cv
    return candidate.event.get("faims_cv")


def _candidate_uses_assigned_strict_hill(candidate, indexes, ppm):
    """Return true only when a candidate reuses a strict assigned centroid."""

    points = _local_candidate_raw_points(candidate)
    if not points:
        return False
    candidate_faims = _candidate_faims(candidate)
    # Median m/z can differ from an individual hill point by the hill-linking
    # tolerance.  The broad lookup is followed by exact source-scan/centroid
    # validation and therefore does not relax the conflict definition.
    lookup_ppm = max(50.0, 4.0 * float(ppm))
    for index in indexes:
        if not _faims_equal(index["faims_cv"], candidate_faims):
            continue
        hills = index["hills"]
        for source_scan, observed_mz in points:
            local_scan = index["source_to_local"].get(int(source_scan))
            if local_scan is None:
                continue
            delta = float(observed_mz) * lookup_ppm * 1e-6
            start = bisect_left(index["mz"], observed_mz - delta)
            end = bisect_right(index["mz"], observed_mz + delta)
            for hill_index in index["hill"][start:end]:
                scans = hills["hills_scan_lists"][hill_index]
                position = bisect_left(scans, local_scan)
                if position >= len(scans) or scans[position] != local_scan:
                    continue
                point_mz = float(hills["tmp_mz_array"][hill_index][position])
                if round(point_mz, 6) == round(float(observed_mz), 6):
                    return True
    return False


def build_strict_feature_index(strict_records):
    grouped = {}
    for record in strict_records:
        # Group only by the exact discrete charge.  FAIMS compatibility uses
        # an absolute tolerance in _faims_equal, so rounding FAIMS values into
        # index buckets could discard a compatible value at a bucket boundary.
        key = record["charge"]
        grouped.setdefault(key, []).append(record)
    result = {}
    for key, records in grouped.items():
        records.sort(key=lambda row: (row["mz"], row["feature_id"]))
        result[key] = (
            tuple(row["mz"] for row in records),
            tuple(records),
        )
    return result


def match_assay_to_strict_feature(
    assay: DirectAssay,
    strict_records,
    *,
    ppm: float = 8.0,
    rt_tolerance_sec: float = 120.0,
):
    target = assay.isotope_peaks[0].mz
    candidates = []
    if isinstance(strict_records, Mapping):
        mz_values, records = strict_records.get(
            assay.charge, ((), ())
        )
        delta = target * ppm * 1e-6
        start = bisect_left(mz_values, target - delta)
        end = bisect_right(mz_values, target + delta)
        records = records[start:end]
    else:
        records = strict_records
    for record in records:
        if record["charge"] != assay.charge or not _faims_equal(record["faims_cv"], assay.faims_cv):
            continue
        mz_error = abs(record["mz"] - target) * 1e6 / target
        if mz_error > ppm:
            continue
        rt_error = _rt_distance(assay.rt_sec, record["rt_start"], record["rt_end"])
        if rt_error > rt_tolerance_sec:
            continue
        candidates.append((rt_error, mz_error, record["feature_id"], record))
    candidates.sort(key=lambda value: value[:3])
    if not candidates:
        return None, "no_strict_match", 0
    if len(candidates) > 1:
        first, second = candidates[:2]
        if abs(first[0] - second[0]) <= 1e-9 and abs(first[1] - second[1]) <= 0.25:
            return None, "ambiguous_strict_match", len(candidates)
    return candidates[0][3], "matched_strict_feature", len(candidates)


def calibrate_direct_run(
    assays,
    match_results,
    *,
    base_ppm: float,
    base_rt_tolerance_sec: float,
    min_anchors: int = 5,
):
    """Estimate transparent robust retry windows from strict direct anchors."""

    mass_errors = []
    rt_offsets = []
    widths = []
    for assay, (record, status, _alternatives) in zip(
        assays, match_results
    ):
        if (
            record is None
            or status != "matched_strict_feature"
            or assay.conflict_status != "unique"
            or float(assay.q_value) >= RELAXED_DIRECT_Q_VALUE_MAX
        ):
            continue
        theoretical = float(assay.isotope_peaks[0].mz)
        mass_errors.append(
            (float(record["mz"]) - theoretical) * 1e6 / theoretical
        )
        rt_offsets.append(float(record["rt_apex"]) - float(assay.rt_sec))
        widths.append(float(record["rt_end"]) - float(record["rt_start"]))

    anchor_count = len(mass_errors)
    if anchor_count < int(min_anchors):
        return DirectRunCalibration(
            "insufficient_anchors",
            anchor_count,
            0.0,
            0.0,
            0.0,
            0.0,
            None,
            None,
            float(base_ppm),
            float(base_rt_tolerance_sec),
        )

    mass_errors = np.asarray(mass_errors, dtype=np.float64)
    rt_offsets = np.asarray(rt_offsets, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    mass_center = float(np.median(mass_errors))
    mass_mad = float(
        1.4826 * np.median(np.abs(mass_errors - mass_center))
    )
    rt_center = float(np.median(rt_offsets))
    rt_mad = float(1.4826 * np.median(np.abs(rt_offsets - rt_center)))
    width_median = float(np.median(widths))
    width_p95 = float(np.quantile(widths, 0.95))
    retry_ppm = min(
        2.0 * float(base_ppm),
        max(float(base_ppm), abs(mass_center) + 4.0 * max(mass_mad, 0.25)),
    )
    calibrated_rt = max(
        15.0,
        abs(rt_center) + 4.0 * max(rt_mad, 1.0) + width_p95,
    )
    retry_rt = min(float(base_rt_tolerance_sec), calibrated_rt)
    return DirectRunCalibration(
        "applied",
        anchor_count,
        mass_center,
        mass_mad,
        rt_center,
        rt_mad,
        width_median,
        width_p95,
        retry_ppm,
        retry_rt,
    )


def _local_candidate_evidence_key(candidate):
    return (
        bool(candidate.quantitative),
        int(candidate.mono_point_count),
        int(candidate.point_count),
        -math.inf
        if candidate.isotope_cosine is None
        else float(candidate.isotope_cosine),
        -int(candidate.refinement_round),
    )


def _processed_hill_retry_parameters(competitor, context, base_rt_tolerance):
    """Derive a bounded exact-assay retry window from a losing strict hill."""

    candidate = competitor.candidate
    mono = int(candidate["monoisotope idx"])
    hills = context["hills"]
    scans = hills["hills_scan_lists"][mono]
    if not scans:
        raise ValueError("processed-hill competitor has no mono scans")
    apex_scan = hills.get("hills_scan_apex", [None] * (mono + 1))[mono]
    if apex_scan is None:
        intensities = np.asarray(
            hills["hills_intensity_array"][mono], dtype=np.float64
        )
        apex_scan = scans[int(np.argmax(intensities))]
    rt_by_local = context["rt_by_local"]
    rt_start = float(rt_by_local[int(scans[0])])
    rt_apex = float(rt_by_local[int(apex_scan)])
    rt_end = float(rt_by_local[int(scans[-1])])
    retry_rt_tolerance = min(
        float(base_rt_tolerance),
        max(15.0, rt_end - rt_start + 5.0),
    )
    return {
        "rt_center_sec": rt_apex,
        "rt_tolerance_sec": retry_rt_tolerance,
        "mz_shift_ppm": float(competitor.mono_mz_error_ppm),
        "rt_start_sec": rt_start,
        "rt_end_sec": rt_end,
    }


def _strict_trace_grid(record):
    candidate = record["candidate"]
    hills = record["hills"]
    mono = int(candidate["monoisotope idx"])
    start = int(hills["hills_scan_lists"][mono][0])
    end = int(hills["hills_scan_lists"][mono][-1]) + 1
    local_scans = np.arange(start, end, dtype=np.int32)
    rt = np.asarray([record["rt_by_local"][int(scan)] for scan in local_scans], dtype=np.float64)
    hill_indices = [mono] + [int(value["isotope_idx"]) for value in candidate["isotopes"]]
    traces = []
    for hill_index in hill_indices:
        values = np.zeros(local_scans.size, dtype=np.float64)
        positions = np.asarray(
            hills["hills_scan_lists"][hill_index], dtype=np.int64
        ) - start
        intensities = np.asarray(
            hills["hills_intensity_array"][hill_index], dtype=np.float64
        )
        valid = (positions >= 0) & (positions < values.size)
        values[positions[valid]] = intensities[valid]
        traces.append(values)
    return local_scans, rt, traces


def _quant_row(
    run_id,
    feature_id,
    origin,
    confidence_tier,
    rt,
    traces,
    *,
    method,
    baseline,
    quality_score,
    isotope_cosine,
    mass_error,
    supporting_psm_count,
    supporting_ms2_count,
    extraction_q_value=None,
    quality_flags=0,
):
    rt = np.asarray(rt, dtype=np.float64)
    matrix = np.asarray(traces, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != rt.size or rt.size < 2:
        raw_areas = corrected_areas = np.asarray([], dtype=np.float64)
        selected_value = apex_value = None
        apex_index = None
        quant_status = "insufficient_points"
    else:
        raw_areas = np.trapezoid(matrix, rt, axis=1)
        edge_count = min(3, max(1, rt.size // 5))
        left = np.median(matrix[:, :edge_count], axis=1)[:, None]
        right = np.median(matrix[:, -edge_count:], axis=1)[:, None]
        fraction = ((rt - rt[0]) / (rt[-1] - rt[0]))[None, :]
        edges = left + fraction * (right - left)
        corrected_matrix = np.clip(matrix - edges, 0.0, None)
        corrected_areas = np.trapezoid(corrected_matrix, rt, axis=1)
        baseline_available = baseline != "edge_linear" or rt.size >= 5
        selected_matrix = corrected_matrix if baseline == "edge_linear" and baseline_available else matrix
        envelope = np.sum(selected_matrix, axis=0, dtype=np.float64)
        apex_index = int(np.argmax(envelope))
        apex_value = float(envelope[apex_index])
        if method in {"all", "envelope_area"}:
            selected_value = float(np.sum(corrected_areas if baseline == "edge_linear" else raw_areas))
        elif method == "mono_area":
            selected_value = float((corrected_areas if baseline == "edge_linear" else raw_areas)[0])
        else:
            selected_value = apex_value
        if baseline == "edge_linear" and (
            not baseline_available or selected_value is None or selected_value <= 0
        ):
            envelope = np.sum(matrix, axis=0, dtype=np.float64)
            apex_index = int(np.argmax(envelope))
            apex_value = float(envelope[apex_index])
            if method in {"all", "envelope_area"}:
                selected_value = float(np.sum(raw_areas))
            elif method == "mono_area":
                selected_value = float(raw_areas[0])
            else:
                selected_value = apex_value
            quant_status = "raw_fallback"
        elif baseline == "edge_linear":
            quant_status = "baseline_corrected"
        else:
            quant_status = "quantified"
    observed_point_count = (
        int(np.count_nonzero(np.any(matrix > 0, axis=0)))
        if matrix.ndim == 2 and matrix.size
        else 0
    )
    if observed_point_count == 2:
        quality_flags |= QUALITY_FLAG_TWO_POINT_QUANT
    if quant_status == "raw_fallback":
        quality_flags |= QUALITY_FLAG_RAW_BASELINE_FALLBACK
    return {
        "run_id": run_id,
        "feature_id": feature_id,
        "feature_origin": origin,
        "confidence_tier": confidence_tier,
        "quant_value": selected_value,
        "quant_method": method,
        "quant_status": quant_status,
        "area_envelope_raw": float(np.sum(raw_areas)) if raw_areas.size else None,
        "area_envelope_corrected": float(np.sum(corrected_areas)) if corrected_areas.size else None,
        "area_mono_raw": float(raw_areas[0]) if raw_areas.size else None,
        "area_mono_corrected": float(corrected_areas[0]) if corrected_areas.size else None,
        "envelope_apex": apex_value,
        "quant_envelope_area": (
            float(np.sum(corrected_areas if baseline == "edge_linear" and quant_status != "raw_fallback" else raw_areas))
            if raw_areas.size else None
        ),
        "quant_mono_area": (
            float((corrected_areas if baseline == "edge_linear" and quant_status != "raw_fallback" else raw_areas)[0])
            if raw_areas.size else None
        ),
        "quant_envelope_apex": apex_value,
        "feature_quality_score": quality_score,
        "quality_flags": int(quality_flags),
        "extraction_q_value": extraction_q_value,
        "supporting_psm_count": supporting_psm_count,
        "supporting_ms2_count": supporting_ms2_count,
        "points_across_peak": observed_point_count,
        "rt_start_sec": float(rt[0]) if len(rt) else None,
        "rt_apex_sec": float(rt[apex_index]) if apex_index is not None else None,
        "rt_end_sec": float(rt[-1]) if len(rt) else None,
        "isotope_cosine": isotope_cosine,
        "mass_error_ppm_median": mass_error,
    }


def _recovered_feature_row(candidate: LocalFeatureCandidate, feature_id: int):
    start, end = candidate.segment_slice
    traces = _candidate_segment_values(candidate)
    envelope = np.sum(np.stack(traces), axis=0, dtype=np.float64)
    apex = int(np.argmax(envelope))
    mono_trace = candidate.traces[0]
    mono_values = traces[0]
    positive = np.flatnonzero(mono_values > 0)
    peaks = candidate.assay.isotope_peaks[: len(traces)]
    return {
        "massCalib": candidate.assay.peptidoform.monoisotopic_mass,
        "rtApex": candidate.rt_apex_sec,
        "intensityApex": float(envelope[apex]),
        "intensitySum": float(sum(np.sum(values, dtype=np.float64) for values in traces)),
        "charge": candidate.assay.charge,
        "nIsotopes": len(traces),
        "nScans": (
            max(int(np.count_nonzero(values)) for values in traces)
            if candidate.status == "accepted_local_feature_partial_envelope"
            else candidate.mono_point_count
        ),
        "mz": peaks[0].mz,
        "rtStart": candidate.rt_start_sec,
        "rtEnd": candidate.rt_end_sec,
        "FAIMS": candidate.assay.faims_cv,
        "im": None,
        "mono_hills_scan_lists": [int(mono_trace.scan_index[start + value]) for value in positive],
        "mono_hills_intensity_list": [float(mono_values[value]) for value in positive],
        "scanApex": candidate.scan_apex,
        "isoerror": candidate.mono_mz_error_ppm,
        "isoerror2": None,
        "feature_idx": feature_id,
        "area_sum": None,
    }


def _generic_recovered_feature_row(candidate, feature_id):
    start, end = candidate.segment_slice
    traces = list(_candidate_segment_values(candidate))
    envelope = np.sum(np.stack(traces), axis=0, dtype=np.float64)
    apex = int(np.argmax(envelope))
    mono_trace = candidate.traces[0]
    mono_values = traces[0]
    positive = np.flatnonzero(mono_values > 0)
    return {
        "massCalib": candidate.neutral_mass,
        "rtApex": candidate.rt_apex_sec,
        "intensityApex": float(envelope[apex]),
        "intensitySum": float(
            sum(np.sum(values, dtype=np.float64) for values in traces)
        ),
        "charge": int(candidate.event["charge"]),
        "nIsotopes": len(traces),
        "nScans": candidate.mono_points,
        "mz": candidate.mono_mz,
        "rtStart": candidate.rt_start_sec,
        "rtEnd": candidate.rt_end_sec,
        "FAIMS": candidate.event.get("faims_cv"),
        "im": candidate.event.get("ion_mobility"),
        "mono_hills_scan_lists": [
            int(mono_trace.scan_index[start + value]) for value in positive
        ],
        "mono_hills_intensity_list": [
            float(mono_values[value]) for value in positive
        ],
        "scanApex": candidate.scan_apex,
        "isoerror": candidate.selected_event_mz_error_ppm,
        "isoerror2": None,
        "feature_idx": feature_id,
        "area_sum": None,
    }


def _generic_local_equivalent(left, right, ppm):
    if int(left.event["charge"]) != int(right.event["charge"]):
        return False
    if not _faims_equal(left.event.get("faims_cv"), right.event.get("faims_cv")):
        return False
    if abs(left.mono_mz - right.mono_mz) * 1e6 / left.mono_mz > ppm:
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


def _generic_local_strict_equivalents(candidate, strict_index, ppm):
    mz_values, records = strict_index.get(int(candidate.event["charge"]), ((), ()))
    delta = candidate.mono_mz * ppm * 1e-6
    start = bisect_left(mz_values, candidate.mono_mz - delta)
    end = bisect_right(mz_values, candidate.mono_mz + delta)
    result = []
    for record in records[start:end]:
        if not _faims_equal(record["faims_cv"], candidate.event.get("faims_cv")):
            continue
        if max(record["rt_start"], candidate.rt_start_sec) > min(
            record["rt_end"], candidate.rt_end_sec
        ):
            continue
        result.append(record)
    return result


def _generic_standard_links(ms2_rows, ingestion, strict_contexts, args):
    """Match generic precursor hypotheses to final strict features.

    This deliberately invokes the same bounded scan/isolation-aware matching
    path for targets and decoys.  It annotates only the already selected strict
    candidates, so generic evidence cannot alter or fabricate the feature
    population at this stage.
    """

    generic_args = dict(args)
    generic_args["generic_ms2_isotope_errors"] = tuple(range(-3, 4))
    # Hybrid generic association combines precursor localization with
    # chromatographic/isotope feature quality. Targets and paired decoys
    # traverse this identical path.
    generic_args["generic_ms2_composite_score"] = True
    contexts = []
    final_candidates = []
    next_candidate_id = 0
    for strict in strict_contexts:
        context = prepare_association_context(
            strict["hills"],
            strict["spectra"],
            ms2_rows,
            ingestion.ms1_metadata,
            strict["faims_cv"],
            strict["rt_by_local"],
            generic_args,
            len(strict_contexts),
        )
        context["next_candidate_id"] = next_candidate_id
        for candidate in strict["candidates"]:
            # Target and decoy passes reuse the final candidate objects. Clear
            # only transient association annotations before rebuilding edges.
            for key in (
                "_generic_association_id",
                "_generic_association_edges",
            ):
                candidate.pop(key, None)
            annotate_candidate_association(
                candidate, strict["hills"], context, generic_args
            )
        next_candidate_id = context["next_candidate_id"]
        contexts.append(context)
        final_candidates.extend(strict["candidates"])

    aggregate = {
        "events": {},
        "event_edges": {},
        "summary": {
            "eligible_event_count": 0,
            "association_local_hill_count": 0,
            "local_candidate_counts": [],
        },
    }
    for context in contexts:
        for event_id, event in context["events"].items():
            if event.get("eligible") or event_id not in aggregate["events"]:
                aggregate["events"][event_id] = event
        for event_id, edges in context["event_edges"].items():
            aggregate["event_edges"].setdefault(event_id, []).extend(edges)
        for key in ("eligible_event_count", "association_local_hill_count"):
            aggregate["summary"][key] += context["summary"][key]
        aggregate["summary"]["local_candidate_counts"].extend(
            context["summary"]["local_candidate_counts"]
        )
    component_values = defaultdict(list)
    for edges in aggregate["event_edges"].values():
        if not edges:
            continue
        best = max(
            edges,
            key=lambda edge: (
                float(edge["support"]),
                -abs(int(edge["offset"])),
                -int(edge["offset"]),
            ),
        )
        for name, value in (best.get("score_components") or {}).items():
            if value is not None and math.isfinite(float(value)):
                component_values[name].append(float(value))
    aggregate["summary"]["best_edge_score_components"] = {
        name: {
            "count": len(values),
            "min": float(np.min(values)),
            "p10": float(np.quantile(values, 0.10)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
            "max": float(np.max(values)),
        }
        for name, values in sorted(component_values.items())
        if values
    }
    return build_association_rows(ms2_rows, aggregate, final_candidates), aggregate["summary"]


def _compact_generic_association_summary(summary):
    counts = [int(value) for value in summary.get("local_candidate_counts", ())]
    histogram = Counter(counts)
    return {
        "eligible_event_count": int(summary.get("eligible_event_count", 0)),
        "association_local_hill_count": int(summary.get("association_local_hill_count", 0)),
        "status_counts": dict(sorted(summary.get("status_counts", {}).items())),
        "local_candidate_event_count": len(counts),
        "local_candidate_count_min": min(counts) if counts else None,
        "local_candidate_count_max": max(counts) if counts else None,
        "local_candidate_count_mean": (
            float(np.mean(counts, dtype=np.float64)) if counts else None
        ),
        "local_candidate_count_histogram": {
            str(key): value for key, value in sorted(histogram.items())
        },
        "best_edge_score_components": summary.get(
            "best_edge_score_components", {}
        ),
    }


def _generic_decoy_rows(run_id, ms2_rows):
    rows = []
    for source in ms2_rows:
        row = dict(source)
        selected = row.get("selected_ion_mz")
        charge = row.get("charge")
        if selected is not None and charge is not None and int(charge) > 0:
            shift = deterministic_decoy_shift(run_id, str(row["ms2_event_id"]))
            mz_shift = shift / int(charge)
            row["selected_ion_mz"] = float(selected) + mz_shift
            if row.get("isolation_target_mz") is not None:
                row["isolation_target_mz"] = (
                    float(row["isolation_target_mz"]) + mz_shift
                )
        rows.append(row)
    return rows


def _complete_score_components(row):
    components = dict(row.get("_score_components") or {})
    if "precursor_joint_support" not in components:
        components["precursor_joint_support"] = precursor_joint_support(
            components
        )
    result = {}
    for name, _weight in GENERIC_ASSOCIATION_SCORE_WEIGHT_ITEMS:
        value = components.get(name)
        if value is None or not math.isfinite(float(value)):
            return None
        result[name] = min(1.0, max(0.0, float(value)))
    return result


def _paired_score_metrics(pairs, weights):
    margins = np.asarray(
        [
            composite_association_support(target, weights)
            - composite_association_support(decoy, weights)
            for target, decoy in pairs
        ],
        dtype=np.float64,
    )
    if margins.size == 0:
        return {
            "pair_count": 0,
            "target_win_count": 0,
            "target_win_fraction": None,
            "median_margin": None,
            "mean_margin": None,
        }
    return {
        "pair_count": int(margins.size),
        "target_win_count": int(np.sum(margins > 0.0)),
        "target_win_fraction": float(np.mean(margins > 0.0)),
        "median_margin": float(np.median(margins)),
        "mean_margin": float(np.mean(margins)),
    }


def _generic_q_metrics(
    audit_by_event, target_rows, decoy_rows, weights, q_value_max
):
    matched = {"matched_existing_feature"}
    unresolved_ids = {
        int(event_id)
        for event_id, audit in audit_by_event.items()
        if audit.get("status") == "unresolved_no_direct_identification"
    }
    decoy_by_event = {
        int(row["ms2_event_id"]): row for row in decoy_rows
    }
    competitions = []
    for target in target_rows:
        event_id = int(target["ms2_event_id"])
        if event_id not in unresolved_ids:
            continue
        decoy = decoy_by_event[event_id]
        target_components = _complete_score_components(target)
        decoy_components = _complete_score_components(decoy)
        target_score = (
            composite_association_support(target_components, weights)
            if target.get("status") in matched and target_components is not None
            else None
        )
        decoy_score = (
            composite_association_support(decoy_components, weights)
            if decoy.get("status") in matched and decoy_components is not None
            else None
        )
        competitions.append(
            TargetDecoyCompetition(str(event_id), target_score, decoy_score)
        )
    results = target_decoy_q_values(competitions)
    return {
        "competition_count": len(results),
        "accepted_target_count": sum(
            result.winner == "target" and result.q_value <= q_value_max
            for result in results
        ),
        "target_winner_count": sum(
            result.winner == "target" for result in results
        ),
        "decoy_winner_count": sum(
            result.winner == "decoy" for result in results
        ),
    }


def _calibrate_generic_score_weights(
    audit_by_event, target_rows, decoy_rows, *, q_value_max=0.01
):
    """Learn run-specific generic weights from held-out direct PSM anchors.

    Only exact direct events already linked to the same strict feature selected
    by generic matching are positive anchors.  A paired decoy candidate must be
    present, so training and validation compare identical evidence paths.
    """

    base_weights = dict(GENERIC_ASSOCIATION_SCORE_WEIGHTS)
    matched = {"matched_existing_feature"}
    decoy_by_event = {
        int(row["ms2_event_id"]): row for row in decoy_rows
    }
    pairs = []
    for target in sorted(target_rows, key=lambda row: int(row["ms2_event_id"])):
        event_id = int(target["ms2_event_id"])
        audit = audit_by_event.get(event_id, {})
        decoy = decoy_by_event.get(event_id)
        if (
            audit.get("status") != "matched_strict_feature"
            or target.get("status") not in matched
            or decoy is None
            or decoy.get("status") not in matched
            or target.get("feature_id") != audit.get("feature_id")
        ):
            continue
        target_components = _complete_score_components(target)
        decoy_components = _complete_score_components(decoy)
        if target_components is None or decoy_components is None:
            continue
        pairs.append((target_components, decoy_components))

    report = {
        "status": "base_weights_insufficient_paired_anchors",
        "paired_anchor_count": len(pairs),
        "minimum_paired_anchor_count": (
            GENERIC_SCORE_CALIBRATION_MIN_PAIRED_ANCHORS
        ),
        "prior_fractions_evaluated": list(
            GENERIC_SCORE_CALIBRATION_PRIOR_FRACTIONS
        ),
        "base_weights": base_weights,
        "selected_weights": base_weights,
    }
    if len(pairs) < GENERIC_SCORE_CALIBRATION_MIN_PAIRED_ANCHORS:
        return base_weights, report

    # Alternating sorted event IDs gives deterministic, similarly distributed
    # train and held-out sets without using generic competition outcomes.
    training = pairs[::2]
    validation = pairs[1::2]
    if not validation:
        return base_weights, report

    component_statistics = {}
    discriminative_signal = {}
    for name, _base_weight in GENERIC_ASSOCIATION_SCORE_WEIGHT_ITEMS:
        target_values = np.asarray(
            [target[name] for target, _decoy in training], dtype=np.float64
        )
        decoy_values = np.asarray(
            [decoy[name] for _target, decoy in training], dtype=np.float64
        )
        target_median = float(np.median(target_values))
        decoy_median = float(np.median(decoy_values))
        delta = max(0.0, target_median - decoy_median)
        discriminative_signal[name] = delta
        component_statistics[name] = {
            "target_median": target_median,
            "decoy_median": decoy_median,
            "positive_median_difference": delta,
        }
    signal_total = sum(discriminative_signal.values())
    report.update(
        {
            "training_pair_count": len(training),
            "validation_pair_count": len(validation),
            "training_component_statistics": component_statistics,
        }
    )
    if signal_total <= 0.0:
        report["status"] = "base_weights_no_positive_training_signal"
        return base_weights, report

    base_validation = _paired_score_metrics(validation, base_weights)
    base_q_metrics = _generic_q_metrics(
        audit_by_event,
        target_rows,
        decoy_rows,
        base_weights,
        q_value_max,
    )
    report.update(
        {
            "base_validation": base_validation,
            "base_generic_q_metrics": base_q_metrics,
            "generic_q_value_max": q_value_max,
        }
    )

    eligible = []
    evaluations = []
    for prior_fraction in GENERIC_SCORE_CALIBRATION_PRIOR_FRACTIONS:
        learned_fraction = 1.0 - prior_fraction
        candidate_weights = {
            name: (
                prior_fraction * base_weights[name]
                + learned_fraction
                * discriminative_signal[name]
                / signal_total
            )
            for name, _base_weight in GENERIC_ASSOCIATION_SCORE_WEIGHT_ITEMS
        }
        candidate_validation = _paired_score_metrics(
            validation, candidate_weights
        )
        candidate_q_metrics = _generic_q_metrics(
            audit_by_event,
            target_rows,
            decoy_rows,
            candidate_weights,
            q_value_max,
        )
        margin_improved = (
            candidate_validation["median_margin"]
            > base_validation["median_margin"] + 1e-12
        )
        wins_preserved = (
            candidate_validation["target_win_count"]
            >= base_validation["target_win_count"]
        )
        q_acceptance_preserved = (
            candidate_q_metrics["accepted_target_count"]
            >= base_q_metrics["accepted_target_count"]
        )
        evaluation = {
            "prior_fraction": prior_fraction,
            "weights": candidate_weights,
            "direct_validation": candidate_validation,
            "generic_q_metrics": candidate_q_metrics,
            "direct_margin_improved": margin_improved,
            "direct_wins_preserved": wins_preserved,
            "generic_q_acceptance_preserved": q_acceptance_preserved,
        }
        evaluations.append(evaluation)
        if margin_improved and wins_preserved and q_acceptance_preserved:
            eligible.append(evaluation)
    report["candidate_evaluations"] = evaluations
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                item["generic_q_metrics"]["accepted_target_count"],
                item["direct_validation"]["median_margin"],
                item["prior_fraction"],
            ),
        )
        report["status"] = "applied"
        report["selected_prior_fraction"] = selected["prior_fraction"]
        report["selected_weights"] = selected["weights"]
        report["selected_validation"] = selected["direct_validation"]
        report["selected_generic_q_metrics"] = selected[
            "generic_q_metrics"
        ]
        return selected["weights"], report

    report["status"] = "base_weights_retained_by_dual_validation"
    return base_weights, report


def _rescore_generic_link_rows(rows, weights):
    rescored = 0
    for row in rows:
        components = _complete_score_components(row)
        if components is None:
            continue
        row["association_support"] = composite_association_support(components, weights)
        rescored += 1
    return rescored


def _generic_local_refinement_events(ms2_rows, audit_by_event):
    """Retry every unresolved generic competition with local raw evidence."""

    return [
        event
        for event in ms2_rows
        if audit_by_event[int(event["ms2_event_id"])]["status"]
        in GENERIC_LOCAL_REFINEMENT_INPUT_STATUSES
    ]


def _compete_generic_local_by_input_family(
    targets, decoys, input_status_by_event
):
    """Keep established no-candidate q-values separate from strict rechecks."""

    family_by_event = {
        int(event_id): (
            "no_standard_candidate"
            if status == "generic_no_standard_candidate"
            else "strict_recheck"
        )
        for event_id, status in input_status_by_event.items()
    }
    decoy_by_event = {
        int(candidate.event["ms2_event_id"]): candidate for candidate in decoys
    }
    competitions_by_event = {}
    family_counts = {}
    for family in ("no_standard_candidate", "strict_recheck"):
        family_targets = [
            candidate
            for candidate in targets
            if family_by_event[int(candidate.event["ms2_event_id"])] == family
        ]
        family_decoys = [
            decoy_by_event[int(candidate.event["ms2_event_id"])]
            for candidate in family_targets
        ]
        family_competitions = compete_generic_local_candidates(
            family_targets, family_decoys
        )
        competitions_by_event.update(
            (value.event_id, value) for value in family_competitions
        )
        family_counts[family] = len(family_competitions)
    return (
        tuple(
            competitions_by_event[int(candidate.event["ms2_event_id"])]
            for candidate in targets
        ),
        family_counts,
    )


def _generic_competitions(target_rows, decoy_rows):
    matched = {"matched_existing_feature"}
    decoy_by_event = {row["ms2_event_id"]: row for row in decoy_rows}
    competitions = []
    for target in target_rows:
        decoy = decoy_by_event[target["ms2_event_id"]]
        competitions.append(
            TargetDecoyCompetition(
                str(target["ms2_event_id"]),
                target.get("association_support") if target["status"] in matched else None,
                decoy.get("association_support") if decoy["status"] in matched else None,
            )
        )
    return {
        int(result.seed_id): result
        for result in target_decoy_q_values(competitions)
    }


def _apply_generic_strict_associations(
    audit_by_event,
    target_rows,
    decoy_rows,
    *,
    q_value_max,
    eligible_event_ids=None,
    preserve_failed_audit=False,
):
    """Apply q-filtered generic links only to events without direct IDs."""

    unresolved_ids = (
        {
            event_id
            for event_id, audit in audit_by_event.items()
            if audit["status"] == "unresolved_no_direct_identification"
        }
        if eligible_event_ids is None
        else {int(value) for value in eligible_event_ids}
    )
    filtered_targets = [
        row for row in target_rows if int(row["ms2_event_id"]) in unresolved_ids
    ]
    filtered_decoys = [
        row for row in decoy_rows if int(row["ms2_event_id"]) in unresolved_ids
    ]
    competitions = _generic_competitions(filtered_targets, filtered_decoys)
    matched = {"matched_existing_feature"}
    status_counts = {}
    component_values_by_status = defaultdict(lambda: defaultdict(list))
    for target in filtered_targets:
        event_id = int(target["ms2_event_id"])
        audit = audit_by_event[event_id]
        competition = competitions[event_id]
        if (
            target["status"] in matched
            and competition.winner == "target"
            and competition.q_value <= q_value_max
        ):
            audit.update(
                {
                    "feature_id": target["feature_id"],
                    "association_tier": "generic_ms2",
                    "status": "generic_matched_strict_feature",
                    "generic_isotope_error": target[
                        "selected_ion_isotope_offset"
                    ],
                    "mz_error_ppm": target["mz_error_ppm"],
                    "rt_error_sec": target["rt_distance_sec"],
                    "score": target["association_support"],
                    "extraction_q_value": competition.q_value,
                    "reason_flags": target["reason_flags"],
                }
            )
        elif preserve_failed_audit:
            # A final residual recheck is additive evidence.  Its failed or
            # sparse competition must not erase the earlier, more specific
            # chromatographic failure outcome.
            pass
        elif target["status"] in matched:
            audit.update(
                {
                    "association_tier": "generic_ms2",
                    "status": (
                        "generic_decoy_won"
                        if competition.winner == "decoy"
                        else "generic_q_value_rejected"
                    ),
                    "generic_isotope_error": target[
                        "selected_ion_isotope_offset"
                    ],
                    "score": target["association_support"],
                    "extraction_q_value": competition.q_value,
                    "reason_flags": target["reason_flags"],
                }
            )
        elif competition.winner == "decoy":
            audit.update(
                {
                    "association_tier": "generic_ms2",
                    "status": "generic_decoy_only",
                    "reason_flags": target["reason_flags"],
                }
            )
        else:
            audit.update(
                {
                    "association_tier": "generic_ms2",
                    "status": "generic_" + target["status"],
                    "reason_flags": target["reason_flags"],
                }
            )
        status_counts[audit["status"]] = status_counts.get(audit["status"], 0) + 1
        for name, value in (target.get("_score_components") or {}).items():
            if value is not None and math.isfinite(float(value)):
                component_values_by_status[audit["status"]][name].append(
                    float(value)
                )
    competition_counts = {
        "competition_count": len(competitions),
        "target_candidate_count": sum(
            result.target_score is not None for result in competitions.values()
        ),
        "decoy_candidate_count": sum(
            result.decoy_score is not None for result in competitions.values()
        ),
        "both_candidate_count": sum(
            result.target_score is not None and result.decoy_score is not None
            for result in competitions.values()
        ),
        "target_only_candidate_count": sum(
            result.target_score is not None and result.decoy_score is None
            for result in competitions.values()
        ),
        "decoy_only_candidate_count": sum(
            result.target_score is None and result.decoy_score is not None
            for result in competitions.values()
        ),
        "target_winner_count": sum(
            result.winner == "target" for result in competitions.values()
        ),
        "decoy_winner_count": sum(
            result.winner == "decoy" for result in competitions.values()
        ),
        "no_winner_count": sum(
            result.winner == "none" for result in competitions.values()
        ),
    }
    component_summary = {
        status: {
            name: {
                "count": len(values),
                "p10": float(np.quantile(values, 0.10)),
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.90)),
            }
            for name, values in sorted(components.items())
            if values
        }
        for status, components in sorted(component_values_by_status.items())
    }
    if any(component_summary.values()):
        competition_counts["target_score_components_by_status"] = (
            component_summary
        )
    return status_counts, competition_counts


def _update_generic_quant_support(strict_quant_rows, audit_by_event):
    """Attach accepted generic event counts without duplicating abundance."""

    from collections import Counter

    accepted_statuses = {
        "generic_matched_strict_feature",
        "generic_local_matched_strict_feature",
        "generic_local_matched_direct_feature",
        "generic_recovered_local_feature",
        "generic_matched_recovered_local_feature",
        "generic_relaxed_recovered_local_feature",
        "generic_relaxed_matched_recovered_local_feature",
    }
    generic_support = Counter(
        row["feature_id"]
        for row in audit_by_event.values()
        if row["status"] in accepted_statuses
    )
    generic_q_values = {}
    for row in audit_by_event.values():
        if row["status"] not in accepted_statuses:
            continue
        q_value = row.get("extraction_q_value")
        if q_value is None:
            continue
        feature_id = row["feature_id"]
        generic_q_values[feature_id] = min(
            float(q_value), generic_q_values.get(feature_id, math.inf)
        )
    quant_by_feature = {row["feature_id"]: row for row in strict_quant_rows}
    for feature_id, event_count in generic_support.items():
        quant_row = quant_by_feature[feature_id]
        quant_row["supporting_ms2_count"] += event_count
        if quant_row["confidence_tier"] == "strict":
            quant_row["confidence_tier"] = "generic_ms2"
        if "extraction_q_value" in quant_row:
            quant_row["extraction_q_value"] = generic_q_values.get(feature_id)
    return dict(generic_support)


def _feature_population_summary(quant_rows, audit_by_event):
    """Summarize features first, separately from MS2 event coverage."""

    from collections import Counter

    feature_ids = {
        int(row["feature_id"])
        for row in quant_rows
    }
    linked_events = [
        row
        for row in audit_by_event.values()
        if row.get("feature_id") is not None
    ]
    linked_feature_ids = {
        int(row["feature_id"])
        for row in linked_events
    }
    return {
        "feature_count": len(quant_rows),
        "quantified_feature_count": sum(
            row.get("quant_value") is not None
            and float(row["quant_value"]) > 0
            for row in quant_rows
        ),
        "null_or_nonpositive_quant_count": sum(
            row.get("quant_value") is None
            or float(row["quant_value"]) <= 0
            for row in quant_rows
        ),
        "feature_origin_counts": dict(
            sorted(Counter(row.get("feature_origin") for row in quant_rows).items())
        ),
        "quant_status_counts": dict(
            sorted(Counter(row.get("quant_status") for row in quant_rows).items())
        ),
        "features_with_psm_support": sum(
            int(row.get("supporting_psm_count") or 0) > 0
            for row in quant_rows
        ),
        "features_with_ms2_support": sum(
            int(row.get("supporting_ms2_count") or 0) > 0
            for row in quant_rows
        ),
        "features_linked_from_ms2_audit": len(linked_feature_ids),
        "features_without_ms2_audit_link": len(feature_ids - linked_feature_ids),
        "linked_ms2_event_count": len(linked_events),
        "unlinked_ms2_event_count": len(audit_by_event) - len(linked_events),
    }


def _ms2_audit_summary(quant_rows, audit_by_event):
    """Return explicit, non-overlapping MS2 outcome and coverage metrics."""

    from collections import Counter

    quantified_feature_ids = {
        int(row["feature_id"])
        for row in quant_rows
        if row.get("quant_value") is not None
        and float(row["quant_value"]) > 0
    }
    outcomes = Counter()
    direct_quantitative = 0
    generic_quantitative = 0
    any_signal = 0
    quantitative = 0

    for row in audit_by_event.values():
        status = str(row.get("status") or "")
        feature_id = row.get("feature_id")
        linked_quantitative = (
            feature_id is not None
            and int(feature_id) in quantified_feature_ids
        )
        if linked_quantitative:
            outcome = "quantitative_feature"
            quantitative += 1
            tier = str(row.get("association_tier") or "")
            if tier == "direct_id":
                direct_quantitative += 1
            elif tier.startswith("generic_ms2"):
                generic_quantitative += 1
        elif "precursor_signal_only" in status:
            outcome = "precursor_signal_only"
        elif any(
            marker in status
            for marker in (
                "ambiguous",
                "conflict",
                "conflicting_identifications",
            )
        ):
            outcome = "ambiguous"
        elif any(
            marker in status
            for marker in (
                "no_signal",
                "no_ms1_scans_in_window",
            )
        ):
            outcome = "no_ms1_signal"
        elif any(
            marker in status
            for marker in (
                "q_value_rejected",
                "q_value_above_limit",
                "decoy_only",
                "decoy_won",
                "decoy_winner",
            )
        ):
            outcome = "statistical_rejection"
        elif any(
            marker in status
            for marker in (
                "component",
                "isotope",
                "mono_points",
                "channel",
                "cosine",
                "apex_spread",
                "too_wide",
                "quantification_failed",
                "boundary",
            )
        ):
            outcome = "insufficient_chromatographic_evidence"
        else:
            outcome = "metadata_or_assay_unavailable"
        outcomes[outcome] += 1

        # This metric intentionally means observed local MS1 evidence, not a
        # quantitative association. Statistical rejection and chromatographic
        # failures arise only after signal-bearing candidates or traces exist.
        if linked_quantitative or outcome in {
            "precursor_signal_only",
            "ambiguous",
            "statistical_rejection",
            "insufficient_chromatographic_evidence",
        }:
            any_signal += 1

    total = len(audit_by_event)
    def fraction(count):
        return count / total if total else None
    return {
        "total_ms2_event_count": total,
        "audit_row_count": total,
        "audit_coverage_fraction": 1.0 if total else None,
        "any_ms1_signal_association_count": any_signal,
        "any_ms1_signal_association_fraction": fraction(any_signal),
        "quantitative_feature_count": quantitative,
        "quantitative_feature_fraction": fraction(quantitative),
        "direct_psm_quantitative_feature_count": direct_quantitative,
        "direct_psm_quantitative_feature_fraction": fraction(
            direct_quantitative
        ),
        "generic_ms2_quantitative_feature_count": generic_quantitative,
        "generic_ms2_quantitative_feature_fraction": fraction(
            generic_quantitative
        ),
        "outcome_counts": dict(sorted(outcomes.items())),
        "outcomes_cover_all_ms2": sum(outcomes.values()) == total,
    }


def _evaluate_cached_generic_pair_stage(
    *,
    source_path,
    cache_root,
    stage,
    residual_ledger,
    target_events,
    decoy_events,
    workers,
    options,
    telemetry,
):
    fingerprint = local_candidate_fingerprint(
        source_path,
        stage=stage,
        target_events=target_events,
        decoy_events=decoy_events,
        options=options,
        residual_state=residual_ledger.state_fingerprint(),
        raw_scan_count=residual_ledger.store.scan_count,
        raw_point_count=residual_ledger.store.point_count,
    )
    cached, cache_path = load_local_candidate_pairs(cache_root, fingerprint)
    if cached is not None:
        targets, decoys = cached
        telemetry.append(
            {
                "stage": stage,
                "status": "reused",
                "path": str(cache_path),
                "event_count": len(targets),
                "residual_state": fingerprint["residual_state"],
            }
        )
        logger.info(
            "Reused hybrid local-candidate cache %s: %d event pairs",
            cache_path,
            len(targets),
        )
        return targets, decoys

    targets, decoys = evaluate_generic_local_candidate_pairs(
        residual_ledger,
        target_events,
        decoy_events,
        workers=workers,
        **options,
    )
    status = "disabled"
    path_value = None
    payload_bytes = None
    if cache_path is not None:
        saved = save_local_candidate_pairs(
            cache_path, fingerprint, targets, decoys
        )
        status = "created"
        path_value = str(saved)
        payload_bytes = (saved / "candidate_pairs.pkl").stat().st_size
        logger.info(
            "Published hybrid local-candidate cache %s: %d event pairs, %d bytes",
            saved,
            len(targets),
            payload_bytes,
        )
    telemetry.append(
        {
            "stage": stage,
            "status": status,
            "path": path_value,
            "event_count": len(targets),
            "payload_bytes": payload_bytes,
            "residual_state": fingerprint["residual_state"],
        }
    )
    return targets, decoys


def run_hybrid_postprocessing(
    *,
    run_id: str,
    ingestion,
    assay_result: AssayBuildResult,
    strict_contexts,
    manager,
    next_feature_id: int,
    args: Mapping,
    final_strict_detector=None,
):
    """Match direct assays, recover bounded local features and write audit rows."""

    from collections import Counter, defaultdict

    strict_records = _strict_feature_records(strict_contexts)
    strict_index = build_strict_feature_index(strict_records)
    strict_hill_claims = _strict_hill_claim_indexes(strict_contexts)
    residual_ledger = ResidualMS1Ledger(ingestion.raw_ms1_store)
    residual_allocation_status_counts = Counter()
    strict_ownership = _allocate_strict_feature_population(
        residual_ledger, strict_records
    )
    for status, count in strict_ownership["status_counts"].items():
        residual_allocation_status_counts["strict_" + status] += count
    logger.info(
        "Hybrid strict residual ownership: %d accepted, %d failed",
        strict_ownership["accepted_feature_count"],
        strict_ownership["failed_feature_count"],
    )
    local_candidate_cache_telemetry = []
    direct_processed_competitors = []
    direct_processed_by_psm = defaultdict(list)
    for context in strict_contexts:
        for competitor in context.get("direct_competitors", ()):
            direct_processed_competitors.append(competitor)
            direct_processed_by_psm[str(competitor.psm_id)].append(
                (competitor, context)
            )
    for values in direct_processed_by_psm.values():
        values.sort(
            key=lambda value: (
                -float(value[0].evidence_score),
                abs(float(value[0].mono_mz_error_ppm)),
                value[0].candidate_key,
            )
        )
    base_ppm = float(args.get("itol", 8.0))
    base_rt_tolerance = float(
        args.get("ms2_rt_tolerance_sec", 120.0)
    )
    direct_match_results = tuple(
        match_assay_to_strict_feature(
            assay,
            strict_index,
            ppm=base_ppm,
            rt_tolerance_sec=base_rt_tolerance,
        )
        for assay in assay_result.assays
    )
    direct_calibration = calibrate_direct_run(
        assay_result.assays,
        direct_match_results,
        base_ppm=base_ppm,
        base_rt_tolerance_sec=base_rt_tolerance,
    )
    logger.info(
        "Hybrid direct stage: %d strict features, %d exact direct assays, %d MS2 events",
        len(strict_records),
        len(assay_result.assays),
        len(ingestion.ms2_rows),
    )
    support = defaultdict(list)
    audit_by_event = {
        int(event["ms2_event_id"]): {
            "run_id": run_id,
            "ms2_event_id": int(event["ms2_event_id"]),
            "feature_id": None,
            "association_tier": "none",
            "status": "unresolved_no_direct_identification",
            "primary_identification_id": None,
            "assay_id": None,
            "charge_used": event.get("charge"),
            "charge_source": "mzml" if event.get("charge") is not None else None,
            "selected_isotope_index": None,
            "generic_isotope_error": None,
            "mz_error_ppm": None,
            "rt_error_sec": None,
            "score": None,
            "extraction_q_value": None,
            "alternative_count": 0,
            "reason_flags": 0,
        }
        for event in ingestion.ms2_rows
    }
    assay_rows = []
    recovered = []
    recovered_feature_rows = []
    recovered_quant_rows = []
    direct_status_counts = Counter()
    direct_retry_counts = Counter()
    for assay_id, assay in enumerate(assay_result.assays, start=1):
        assay_rows.append(
            {
                "run_id": run_id,
                "assay_id": assay_id,
                "ms2_event_id": assay.ms2_event_id,
                "psm_id": assay.psm_id,
                "canonical_peptidoform": assay.canonical_peptidoform,
                "charge": assay.charge,
                "rt_sec": assay.rt_sec,
                "faims_cv": assay.faims_cv,
                "monoisotopic_mz": assay.isotope_peaks[0].mz,
                "selected_isotope_index": assay.selected_isotope_index,
                "selected_mz_error_ppm": assay.selected_mz_error_ppm,
                "q_value": assay.q_value,
                "pep": assay.pep,
                "conflict_status": assay.conflict_status,
            }
        )
        audit = audit_by_event[assay.ms2_event_id]
        audit["alternative_count"] += 1
        if audit["primary_identification_id"] is None:
            audit["primary_identification_id"] = assay.psm_id
            audit["assay_id"] = assay_id
            audit["selected_isotope_index"] = assay.selected_isotope_index
            audit["mz_error_ppm"] = assay.selected_mz_error_ppm
        if assay.conflict_status != "unique":
            audit["association_tier"] = "direct_id"
            audit["status"] = "conflicting_identifications"
            direct_status_counts["conflicting_identifications"] += 1
            continue
        matched, status, alternatives = direct_match_results[assay_id - 1]
        audit["alternative_count"] = max(audit["alternative_count"], alternatives)
        if matched is not None:
            feature_id = matched["feature_id"]
            support[feature_id].append(assay)
            audit.update(
                {
                    "feature_id": feature_id,
                    "association_tier": "direct_id",
                    "status": status,
                    "charge_used": assay.charge,
                    "charge_source": "psm",
                    "rt_error_sec": _rt_distance(assay.rt_sec, matched["rt_start"], matched["rt_end"]),
                }
            )
            direct_status_counts[status] += 1
            continue

        local = extract_local_feature(
            ingestion.raw_ms1_store,
            assay,
            ppm=base_ppm,
            rt_tolerance_sec=base_rt_tolerance,
            quant_method=args.get("quant_method", "envelope_area"),
            baseline=args.get("feature_baseline", "edge_linear"),
            allow_two_point_exception=False,
            allow_partial_envelope=False,
        )
        relaxed_retry = False
        retry_attempted = False
        retry_selected = False
        processed_retry_selected = False
        processed_matches = direct_processed_by_psm.get(str(assay.psm_id), ())
        processed_retry = processed_matches[0] if processed_matches else None
        if not local.quantitative and (
            direct_calibration.status == "applied"
            or _direct_relaxed_retry_enabled(assay, args)
            or processed_retry is not None
        ):
            retry_attempted = True
            if processed_retry is not None:
                processed_parameters = _processed_hill_retry_parameters(
                    processed_retry[0], processed_retry[1], base_rt_tolerance
                )
                retry_ppm = base_ppm
                retry_rt_tolerance = processed_parameters[
                    "rt_tolerance_sec"
                ]
                retry_mz_shift = processed_parameters["mz_shift_ppm"]
                retry_rt_center = processed_parameters["rt_center_sec"]
                direct_retry_counts["processed_hill_attempted"] += 1
            else:
                retry_ppm = direct_calibration.retry_ppm
                retry_rt_tolerance = (
                    direct_calibration.retry_rt_tolerance_sec
                )
                retry_mz_shift = direct_calibration.mass_error_center_ppm
                retry_rt_center = (
                    assay.rt_sec + direct_calibration.rt_apex_offset_sec
                )
            retry = replace(
                extract_local_feature(
                    ingestion.raw_ms1_store,
                    assay,
                    ppm=retry_ppm,
                    rt_tolerance_sec=retry_rt_tolerance,
                    quant_method=args.get("quant_method", "envelope_area"),
                    baseline=args.get("feature_baseline", "edge_linear"),
                    allow_two_point_exception=(
                        _direct_relaxed_retry_enabled(assay, args)
                    ),
                    allow_partial_envelope=(
                        _direct_relaxed_retry_enabled(assay, args)
                    ),
                    mz_shift_ppm=retry_mz_shift,
                    rt_center_sec=retry_rt_center,
                ),
                refinement_round=1,
            )
            if _local_candidate_evidence_key(retry) > (
                _local_candidate_evidence_key(local)
            ):
                local = retry
                retry_selected = True
                if processed_retry is not None:
                    direct_retry_counts["processed_hill_selected"] += 1
                    processed_retry_selected = True
            relaxed_retry = local.quantitative and local.status in {
                "accepted_local_feature_two_point",
                "accepted_local_feature_partial_envelope",
            }
        if retry_attempted:
            direct_retry_counts["attempted"] += 1
            direct_retry_counts[
                "selected" if retry_selected else "no_monotonic_improvement"
            ] += 1
            if relaxed_retry:
                direct_retry_counts["selected_relaxed"] += 1
            elif retry_selected and local.quantitative:
                direct_retry_counts["selected_calibrated_strict"] += 1
        if local.quantitative and _candidate_uses_assigned_strict_hill(
            local, strict_hill_claims, float(args.get("itol", 8.0))
        ):
            local = replace(
                local,
                status="local_raw_point_conflict",
                quantitative=False,
            )
            relaxed_retry = False
        existing = None
        recovered_conflict = False
        if local.quantitative:
            for previous, feature_id in recovered:
                if _local_feature_equivalent(
                    previous, local, float(args.get("itol", 8.0))
                ):
                    existing = (previous, feature_id)
                    break
                if _protected_local_conflict(previous, local):
                    recovered_conflict = True
                    break
        if recovered_conflict:
            local = replace(
                local,
                status="local_raw_point_conflict",
                quantitative=False,
            )
            relaxed_retry = False
        if existing is None and local.quantitative:
            allocation = _allocate_candidate_component(
                residual_ledger,
                ("direct", next_feature_id),
                local,
            )
            residual_allocation_status_counts[
                "direct_" + allocation.status
            ] += 1
            if not allocation.accepted:
                local = replace(
                    local,
                    status="local_residual_intensity_conflict",
                    quantitative=False,
                )
                relaxed_retry = False
        if existing is not None:
            feature_id = existing[1]
            support[feature_id].append(assay)
            status = (
                "matched_recovered_feature"
                if existing[0].assay.canonical_peptidoform
                == assay.canonical_peptidoform
                else "matched_recovered_feature_ambiguous_identity"
            )
            audit.update(
                {
                    "feature_id": feature_id,
                    "association_tier": "direct_id",
                    "status": status,
                }
            )
            direct_status_counts[status] += 1
            if processed_retry_selected:
                direct_retry_counts["processed_hill_reused"] += 1
        elif local.quantitative:
            feature_id = next_feature_id
            next_feature_id += 1
            recovered.append((local, feature_id))
            support[feature_id].append(assay)
            recovered_feature_rows.append(_recovered_feature_row(local, feature_id))
            start, end = local.segment_slice
            rt = local.traces[0].rt_sec[start:end]
            traces = _candidate_segment_values(local)
            recovered_quant_rows.append(
                _quant_row(
                    run_id,
                    feature_id,
                    (
                        FEATURE_ORIGIN_MS2_GUIDED_PARTIAL
                        if local.status == "accepted_local_feature_partial_envelope"
                        else FEATURE_ORIGIN_MS2_GUIDED_PARTIAL
                        if local.status == "accepted_local_feature_two_point"
                        else FEATURE_ORIGIN_DIRECT_IDENTIFIED
                    ),
                    "direct_id_relaxed" if relaxed_retry else "direct_id",
                    rt,
                    traces,
                    method=args.get("quant_method", "envelope_area"),
                    baseline=args.get("feature_baseline", "edge_linear"),
                    quality_score=local.isotope_cosine,
                    isotope_cosine=local.isotope_cosine,
                    mass_error=local.mono_mz_error_ppm,
                    supporting_psm_count=1,
                    supporting_ms2_count=1,
                    quality_flags=(
                        QUALITY_FLAG_RELAXED_MS2_FEATURE
                        if relaxed_retry else 0
                    )
                    | (
                        QUALITY_FLAG_BOUNDARY_TRUNCATED
                        if local.boundary_truncated else 0
                    ),
                )
            )
            recovered_status = (
                "recovered_direct_relaxed_partial_envelope"
                if local.status == "accepted_local_feature_partial_envelope"
                else "recovered_direct_relaxed_two_point"
                if local.status == "accepted_local_feature_two_point"
                else "recovered_direct_feature"
            )
            audit.update(
                {
                    "feature_id": feature_id,
                    "association_tier": "direct_id",
                    "status": recovered_status,
                }
            )
            direct_status_counts[recovered_status] += 1
            if processed_retry_selected:
                direct_retry_counts["processed_hill_accepted"] += 1
        else:
            audit.update({"association_tier": "direct_id", "status": local.status})
            direct_status_counts[local.status] += 1

    logger.info(
        "Hybrid direct association stage complete: %s; %d de-duplicated recovered features",
        dict(sorted(direct_status_counts.items())),
        len(recovered),
    )

    strict_quant_rows = []
    logger.info("Hybrid quantifying %d strict features", len(strict_records))
    for record in strict_records:
        _scans, rt, traces = _strict_trace_grid(record)
        candidate = record["candidate"]
        direct_support = support.get(record["feature_id"], ())
        strict_quant_rows.append(
            _quant_row(
                run_id,
                record["feature_id"],
                FEATURE_ORIGIN_STRICT_UNTARGETED,
                "direct_id" if direct_support else "strict",
                rt,
                traces,
                method=args.get("quant_method", "envelope_area"),
                baseline=args.get("feature_baseline", "edge_linear"),
                quality_score=float(candidate.get("cos_cor_isotopes", 0.0)),
                isotope_cosine=float(candidate.get("cos_cor_isotopes", 0.0)),
                mass_error=float(np.median([value["mass_diff_ppm"] for value in candidate["isotopes"]])),
                supporting_psm_count=len(direct_support),
                supporting_ms2_count=len({assay.ms2_event_id for assay in direct_support}),
            )
        )
    logger.info("Hybrid strict-feature quantification complete")
    # Update deduplicated recovered support counts after every assay is linked.
    by_id = {row["feature_id"]: row for row in recovered_quant_rows}
    for feature_id, assays in support.items():
        if feature_id in by_id:
            by_id[feature_id]["supporting_psm_count"] = len(assays)
            by_id[feature_id]["supporting_ms2_count"] = len({assay.ms2_event_id for assay in assays})

    generic_summary = None
    generic_recovered_feature_rows = []
    generic_recovered_quant_rows = []
    generic_recovered = []
    generic_score_weights = dict(GENERIC_ASSOCIATION_SCORE_WEIGHTS)
    if args.get("generic_ms2_refine", True):
        logger.info("Hybrid generic stage: matching target precursor hypotheses")
        target_links, target_summary = _generic_standard_links(
            ingestion.ms2_rows, ingestion, strict_contexts, args
        )
        logger.info("Hybrid generic stage: matching paired decoy hypotheses")
        decoy_links, decoy_summary = _generic_standard_links(
            _generic_decoy_rows(run_id, ingestion.ms2_rows),
            ingestion,
            strict_contexts,
            args,
        )
        generic_score_weights, generic_score_calibration = (
            _calibrate_generic_score_weights(
                audit_by_event,
                target_links,
                decoy_links,
                q_value_max=float(args.get("generic_q_value_max", 0.01)),
            )
        )
        generic_score_calibration["rescored_target_count"] = (
            _rescore_generic_link_rows(target_links, generic_score_weights)
        )
        generic_score_calibration["rescored_decoy_count"] = (
            _rescore_generic_link_rows(decoy_links, generic_score_weights)
        )
        logger.info(
            "Hybrid generic score calibration: %s; %d paired direct anchors",
            generic_score_calibration["status"],
            generic_score_calibration["paired_anchor_count"],
        )
        generic_status_counts, generic_competition_counts = _apply_generic_strict_associations(
            audit_by_event,
            target_links,
            decoy_links,
            q_value_max=float(args.get("generic_q_value_max", 0.01)),
        )
        generic_summary = {
            "target": _compact_generic_association_summary(target_summary),
            "decoy": _compact_generic_association_summary(decoy_summary),
            "audit_status_counts": generic_status_counts,
            "competition_counts": generic_competition_counts,
            "score_calibration": generic_score_calibration,
        }
        logger.info(
            "Hybrid generic strict-feature association complete: %s",
            {
                "audit": dict(sorted(generic_status_counts.items())),
                "competition": generic_competition_counts,
            },
        )

        local_events = _generic_local_refinement_events(
            ingestion.ms2_rows, audit_by_event
        )
        local_input_status_counts = Counter(
            audit_by_event[int(event["ms2_event_id"])]["status"]
            for event in local_events
        )
        local_input_status_by_event = {
            int(event["ms2_event_id"]): audit_by_event[
                int(event["ms2_event_id"])
            ]["status"]
            for event in local_events
        }
        width_limit = generic_local_width_limit(strict_quant_rows)
        local_ppm = float(args.get("generic_ms2_ppm", 10.0))
        local_rt_tolerance = float(
            args.get("ms2_rt_tolerance_sec", 120.0)
        )
        logger.info(
            "Hybrid generic local stage: evaluating %d unresolved events; width limit %.3f sec",
            len(local_events),
            width_limit,
        )
        decoy_events = {
            int(event["ms2_event_id"]): event
            for event in _generic_decoy_rows(run_id, local_events)
        }
        local_workers = max(1, int(args.get("nprocs", 1)))
        logger.info(
            "Hybrid generic local paired extraction: %d events, %d workers",
            len(local_events),
            local_workers,
        )
        standard_local_options = {
            "width_limit_sec": width_limit,
            "ppm": local_ppm,
            "rt_tolerance_sec": local_rt_tolerance,
        }
        target_local, decoy_local = _evaluate_cached_generic_pair_stage(
            source_path=args["file"],
            cache_root=args.get("hybrid_candidate_cache_dir"),
            stage="generic_standard",
            residual_ledger=residual_ledger,
            target_events=local_events,
            decoy_events=[
                decoy_events[int(event["ms2_event_id"])]
                for event in local_events
            ],
            workers=local_workers,
            options=standard_local_options,
            telemetry=local_candidate_cache_telemetry,
        )
        local_competitions, local_q_family_counts = (
            _compete_generic_local_by_input_family(
                target_local, decoy_local, local_input_status_by_event
            )
        )
        local_status_counts = Counter()
        q_value_max = float(args.get("generic_q_value_max", 0.01))
        for competition in local_competitions:
            event_id = competition.event_id
            target = competition.target
            decoy = competition.decoy
            audit = audit_by_event[event_id]
            accepted = (
                target.quantitative_candidate
                and competition.winner == "target"
                and competition.q_value <= q_value_max
            )
            if accepted:
                feature_id = None
                status = None
                strict_equivalents = _generic_local_strict_equivalents(
                    target, strict_index, local_ppm
                )
                if len(strict_equivalents) == 1:
                    feature_id = strict_equivalents[0]["feature_id"]
                    status = "generic_local_matched_strict_feature"
                elif len(strict_equivalents) > 1:
                    status = "generic_local_ambiguous_strict_equivalent"
                if status is None:
                    for direct_candidate, direct_feature_id in recovered:
                        direct_mz = direct_candidate.assay.isotope_peaks[0].mz
                        if (
                            int(direct_candidate.assay.charge)
                            == int(target.event["charge"])
                            and _faims_equal(
                                direct_candidate.assay.faims_cv,
                                target.event.get("faims_cv"),
                            )
                            and abs(direct_mz - target.mono_mz)
                            * 1e6
                            / target.mono_mz
                            <= local_ppm
                            and max(
                                direct_candidate.rt_start_sec,
                                target.rt_start_sec,
                            )
                            <= min(
                                direct_candidate.rt_end_sec,
                                target.rt_end_sec,
                            )
                        ):
                            feature_id = direct_feature_id
                            status = "generic_local_matched_direct_feature"
                            break
                if status is None:
                    for direct_candidate, _direct_feature_id in recovered:
                        if _protected_local_conflict(direct_candidate, target):
                            status = "generic_local_raw_point_conflict"
                            break
                if status is None:
                    for previous, previous_feature_id in generic_recovered:
                        if _generic_local_equivalent(
                            previous, target, local_ppm
                        ):
                            feature_id = previous_feature_id
                            status = "generic_matched_recovered_local_feature"
                            break
                        if _protected_local_conflict(previous, target):
                            status = "generic_local_raw_point_conflict"
                            break
                if status is None and _candidate_uses_assigned_strict_hill(
                    target, strict_hill_claims, local_ppm
                ):
                    status = "generic_local_assigned_strict_hill_conflict"
                if status is None:
                    allocation = _allocate_candidate_component(
                        residual_ledger,
                        ("generic", next_feature_id),
                        target,
                    )
                    residual_allocation_status_counts[
                        "generic_" + allocation.status
                    ] += 1
                    if not allocation.accepted:
                        status = "generic_local_residual_intensity_conflict"
                if status is None:
                    feature_id = next_feature_id
                    next_feature_id += 1
                    status = "generic_recovered_local_feature"
                    generic_recovered.append((target, feature_id))
                    generic_recovered_feature_rows.append(
                        _generic_recovered_feature_row(target, feature_id)
                    )
                    start, end = target.segment_slice
                    rt = target.traces[0].rt_sec[start:end]
                    traces = list(_candidate_segment_values(target))
                    generic_recovered_quant_rows.append(
                        _quant_row(
                            run_id,
                            feature_id,
                            FEATURE_ORIGIN_MS2_GUIDED_FULL,
                            "generic_ms2",
                            rt,
                            traces,
                            method=args.get("quant_method", "envelope_area"),
                            baseline=args.get(
                                "feature_baseline", "edge_linear"
                            ),
                            quality_score=target.isotope_cosine,
                            isotope_cosine=target.isotope_cosine,
                            mass_error=target.selected_event_mz_error_ppm,
                            supporting_psm_count=0,
                            supporting_ms2_count=0,
                            extraction_q_value=competition.q_value,
                            quality_flags=(
                                QUALITY_FLAG_BOUNDARY_TRUNCATED
                                if target.boundary_truncated else 0
                            ),
                        )
                    )
                audit.update(
                    {
                        "feature_id": feature_id,
                        "association_tier": "generic_ms2",
                        "status": status,
                        "charge_used": int(target.event["charge"]),
                        "charge_source": "mzml",
                        "generic_isotope_error": target.isotope_error,
                        "mz_error_ppm": target.selected_event_mz_error_ppm,
                        "rt_error_sec": _rt_distance(
                            float(target.event["rt_sec"]),
                            target.rt_start_sec,
                            target.rt_end_sec,
                        ),
                        "score": target.score,
                        "extraction_q_value": competition.q_value,
                    }
                )
            elif target.quantitative_candidate:
                status = (
                    "generic_local_decoy_won"
                    if competition.winner == "decoy"
                    else "generic_local_q_value_rejected"
                )
                audit.update(
                    {
                        "association_tier": "generic_ms2",
                        "status": status,
                        "generic_isotope_error": target.isotope_error,
                        "mz_error_ppm": target.selected_event_mz_error_ppm,
                        "score": target.score,
                        "extraction_q_value": competition.q_value,
                    }
                )
            elif decoy.quantitative_candidate:
                status = "generic_local_decoy_only"
                audit.update(
                    {"association_tier": "generic_ms2", "status": status}
                )
            else:
                status = "generic_local_" + target.status
                audit.update(
                    {"association_tier": "generic_ms2", "status": status}
                )
            local_status_counts[status] += 1

        relaxed_target_local = []
        relaxed_decoy_local = []
        relaxed_competitions = ()
        relaxed_recovered = []
        relaxed_strict_competition = {
            "status": "not_run",
            "reason": "relaxed_retry_disabled",
            "strict_candidate_count": 0,
            "target_protection_reason_counts": {},
            "decoy_protection_reason_counts": {},
        }
        if bool(args.get("relaxed_ms2_feature", False)):
            final_strict_raw_point_index = {}
            target_strict_protection_counts = Counter()
            decoy_strict_protection_counts = Counter()
            retry_ids = {
                value.event_id
                for value in local_competitions
                if not value.target.quantitative_candidate
                and not value.decoy.quantitative_candidate
                and value.target.status in {
                    "insufficient_mono_points",
                    "insufficient_isotope_channel_support",
                }
            }
            retry_events = [
                event
                for event in local_events
                if int(event["ms2_event_id"]) in retry_ids
            ]
            if not retry_events:
                relaxed_strict_competition.update(
                    {
                        "status": "not_run",
                        "reason": "no_relaxed_retry_events",
                    }
                )
            elif final_strict_detector is None:
                relaxed_strict_competition.update(
                    {"status": "not_run", "reason": "detector_not_provided"}
                )
            elif strict_ownership["failed_feature_count"]:
                relaxed_strict_competition.update(
                    {
                        "status": "not_run",
                        "reason": "incomplete_input_strict_ownership",
                    }
                )
            else:
                strict_competitor_result = final_strict_detector(
                    residual_ledger.materialize(),
                    strict_contexts=strict_contexts,
                    next_feature_id=next_feature_id,
                    args=args,
                )
                strict_competitor_records = _strict_feature_records(
                    strict_competitor_result.get("contexts", ())
                )
                final_strict_raw_point_index = (
                    _build_final_strict_raw_point_index(
                        strict_competitor_records
                    )
                )
                relaxed_strict_competition.update(
                    {
                        "status": strict_competitor_result["status"],
                        "reason": strict_competitor_result["reason"],
                        "strict_candidate_count": len(
                            strict_competitor_records
                        ),
                        "indexed_raw_point_count": len(
                            final_strict_raw_point_index
                        ),
                    }
                )
            relaxed_local_options = {
                "width_limit_sec": width_limit,
                "ppm": local_ppm,
                "rt_tolerance_sec": local_rt_tolerance,
                "min_mono_points": 2,
                "min_channel_points": 2,
                "min_supported_channels": 2,
                "min_cosine": 0.95,
                "relaxed": True,
            }
            raw_relaxed_target, raw_relaxed_decoy = (
                _evaluate_cached_generic_pair_stage(
                    source_path=args["file"],
                    cache_root=args.get("hybrid_candidate_cache_dir"),
                    stage="generic_relaxed",
                    residual_ledger=residual_ledger,
                    target_events=retry_events,
                    decoy_events=[
                        decoy_events[int(event["ms2_event_id"])]
                        for event in retry_events
                    ],
                    workers=local_workers,
                    options=relaxed_local_options,
                    telemetry=local_candidate_cache_telemetry,
                )
            )
            for candidate, decoy_candidate in zip(
                raw_relaxed_target, raw_relaxed_decoy
            ):
                target_strict_reason = (
                    _final_strict_protection_reason(
                        candidate, final_strict_raw_point_index, local_ppm
                    )
                    if candidate.quantitative_candidate
                    else None
                )
                if target_strict_reason is not None:
                    target_strict_protection_counts[target_strict_reason] += 1
                    candidate = replace(
                        candidate,
                        status="final_strict_competitor_"
                        + target_strict_reason,
                        score=None,
                    )
                if candidate.quantitative_candidate and (
                    _candidate_uses_assigned_strict_hill(
                        candidate, strict_hill_claims, local_ppm
                    )
                    or any(
                        _protected_local_conflict(previous, candidate)
                        for previous, _feature_id in recovered
                    )
                    or any(
                        _protected_local_conflict(previous, candidate)
                        for previous, _feature_id in generic_recovered
                    )
                ):
                    candidate = replace(
                        candidate,
                        status="assigned_raw_point_conflict",
                        score=None,
                    )
                relaxed_target_local.append(candidate)
                decoy_strict_reason = (
                    _final_strict_protection_reason(
                        decoy_candidate,
                        final_strict_raw_point_index,
                        local_ppm,
                    )
                    if decoy_candidate.quantitative_candidate
                    else None
                )
                if decoy_strict_reason is not None:
                    decoy_strict_protection_counts[decoy_strict_reason] += 1
                    decoy_candidate = replace(
                        decoy_candidate,
                        status="final_strict_competitor_"
                        + decoy_strict_reason,
                        score=None,
                    )
                if decoy_candidate.quantitative_candidate and (
                    _candidate_uses_assigned_strict_hill(
                        decoy_candidate, strict_hill_claims, local_ppm
                    )
                    or any(
                        _protected_local_conflict(previous, decoy_candidate)
                        for previous, _feature_id in recovered
                    )
                    or any(
                        _protected_local_conflict(previous, decoy_candidate)
                        for previous, _feature_id in generic_recovered
                    )
                ):
                    decoy_candidate = replace(
                        decoy_candidate,
                        status="assigned_raw_point_conflict",
                        score=None,
                    )
                relaxed_decoy_local.append(decoy_candidate)

            relaxed_strict_competition.update(
                {
                    "target_protection_reason_counts": dict(
                        sorted(target_strict_protection_counts.items())
                    ),
                    "decoy_protection_reason_counts": dict(
                        sorted(decoy_strict_protection_counts.items())
                    ),
                }
            )

            relaxed_competitions = compete_generic_local_candidates(
                relaxed_target_local, relaxed_decoy_local
            )
            for competition in relaxed_competitions:
                event_id = competition.event_id
                target = competition.target
                audit = audit_by_event[event_id]
                old_status = audit["status"]
                accepted = (
                    target.quantitative_candidate
                    and competition.winner == "target"
                    and competition.q_value <= q_value_max
                )
                if accepted:
                    feature_id = None
                    status = None
                    for previous, previous_feature_id in relaxed_recovered:
                        if _generic_local_equivalent(
                            previous, target, local_ppm
                        ):
                            feature_id = previous_feature_id
                            status = (
                                "generic_relaxed_matched_recovered_local_feature"
                            )
                            break
                        if _protected_local_conflict(previous, target):
                            status = "generic_relaxed_raw_point_conflict"
                            break
                    if status is None:
                        allocation = _allocate_candidate_component(
                            residual_ledger,
                            ("generic_relaxed", next_feature_id),
                            target,
                        )
                        residual_allocation_status_counts[
                            "generic_relaxed_" + allocation.status
                        ] += 1
                        if not allocation.accepted:
                            status = (
                                "generic_relaxed_residual_intensity_conflict"
                            )
                    if status is None:
                        feature_id = next_feature_id
                        next_feature_id += 1
                        status = "generic_relaxed_recovered_local_feature"
                        relaxed_recovered.append((target, feature_id))
                        generic_recovered.append((target, feature_id))
                        generic_recovered_feature_rows.append(
                            _generic_recovered_feature_row(target, feature_id)
                        )
                        start, end = target.segment_slice
                        rt = target.traces[0].rt_sec[start:end]
                        traces = list(_candidate_segment_values(target))
                        generic_recovered_quant_rows.append(
                            _quant_row(
                                run_id,
                                feature_id,
                                FEATURE_ORIGIN_MS2_GUIDED_PARTIAL,
                                "generic_ms2_relaxed",
                                rt,
                                traces,
                                method=args.get(
                                    "quant_method", "envelope_area"
                                ),
                                baseline=args.get(
                                    "feature_baseline", "edge_linear"
                                ),
                                quality_score=target.isotope_cosine,
                                isotope_cosine=target.isotope_cosine,
                                mass_error=target.selected_event_mz_error_ppm,
                                supporting_psm_count=0,
                                supporting_ms2_count=0,
                                extraction_q_value=competition.q_value,
                                quality_flags=(
                                    QUALITY_FLAG_RELAXED_MS2_FEATURE
                                    | (
                                        QUALITY_FLAG_BOUNDARY_TRUNCATED
                                        if target.boundary_truncated else 0
                                    )
                                ),
                            )
                        )
                    audit.update(
                        {
                            "feature_id": feature_id,
                            "association_tier": "generic_ms2_relaxed",
                            "status": status,
                            "charge_used": int(target.event["charge"]),
                            "charge_source": "mzml",
                            "generic_isotope_error": target.isotope_error,
                            "mz_error_ppm": target.selected_event_mz_error_ppm,
                            "rt_error_sec": _rt_distance(
                                float(target.event["rt_sec"]),
                                target.rt_start_sec,
                                target.rt_end_sec,
                            ),
                            "score": target.score,
                            "extraction_q_value": competition.q_value,
                        }
                    )
                elif target.quantitative_candidate:
                    status = (
                        "generic_relaxed_decoy_won"
                        if competition.winner == "decoy"
                        else "generic_relaxed_q_value_rejected"
                    )
                    audit.update(
                        {
                            "association_tier": "generic_ms2_relaxed",
                            "status": status,
                            "generic_isotope_error": target.isotope_error,
                            "mz_error_ppm": target.selected_event_mz_error_ppm,
                            "score": target.score,
                            "extraction_q_value": competition.q_value,
                        }
                    )
                elif competition.decoy.quantitative_candidate:
                    status = "generic_relaxed_decoy_only"
                    audit.update(
                        {
                            "association_tier": "generic_ms2_relaxed",
                            "status": status,
                        }
                    )
                else:
                    status = "generic_relaxed_" + target.status
                    audit.update(
                        {
                            "association_tier": "generic_ms2_relaxed",
                            "status": status,
                        }
                    )
                local_status_counts[old_status] -= 1
                if local_status_counts[old_status] <= 0:
                    del local_status_counts[old_status]
                local_status_counts[status] += 1

        target_local_clusters = cluster_compatible_generic_candidates(
            target_local, ppm=local_ppm
        )
        generic_summary["local"] = {
            "width_limit_sec": width_limit,
            "isotope_errors": [0, 1, 2, 3],
            "input_strict_status_counts": dict(
                sorted(local_input_status_counts.items())
            ),
            "q_value_family_counts": dict(sorted(local_q_family_counts.items())),
            "target_status_counts": dict(
                sorted(Counter(value.status for value in target_local).items())
            ),
            "decoy_status_counts": dict(
                sorted(Counter(value.status for value in decoy_local).items())
            ),
            "audit_status_counts": dict(sorted(local_status_counts.items())),
            "competition_counts": {
                "competition_count": len(local_competitions),
                "target_candidate_count": sum(
                    value.target.quantitative_candidate
                    for value in local_competitions
                ),
                "decoy_candidate_count": sum(
                    value.decoy.quantitative_candidate
                    for value in local_competitions
                ),
                "target_winner_count": sum(
                    value.winner == "target" for value in local_competitions
                ),
                "decoy_winner_count": sum(
                    value.winner == "decoy" for value in local_competitions
                ),
                "no_winner_count": sum(
                    value.winner == "none" for value in local_competitions
                ),
            },
            "target_refinement": {
                "candidate_count": sum(
                    value.quantitative_candidate for value in target_local
                ),
                "accepted_edit_count": sum(
                    edit.accepted
                    for value in target_local
                    for edit in value.edit_history
                ),
                "accepted_edit_action_counts": dict(
                    sorted(
                        Counter(
                            edit.action
                            for value in target_local
                            for edit in value.edit_history
                            if edit.accepted
                        ).items()
                    )
                ),
                "component_count_histogram": {
                    str(key): count
                    for key, count in sorted(
                        Counter(
                            value.component_count
                            for value in target_local
                            if value.quantitative_candidate
                        ).items()
                    )
                },
                "deconvolution_status_counts": dict(
                    sorted(
                        Counter(
                            value.deconvolution_status
                            for value in target_local
                            if value.quantitative_candidate
                        ).items()
                    )
                ),
                "compatible_ms2_cluster_count": len(
                    target_local_clusters
                ),
                "multi_ms2_cluster_count": sum(
                    len(group) > 1
                    for group in target_local_clusters
                ),
                "events_in_multi_ms2_clusters": sum(
                    len(group)
                    for group in target_local_clusters
                    if len(group) > 1
                ),
            },
            "new_feature_count": len(generic_recovered),
            "relaxed_retry": {
                "enabled": bool(args.get("relaxed_ms2_feature", False)),
                "q_value_family": "generic_ms2_relaxed",
                "min_mono_points": 2,
                "min_channel_points": 2,
                "min_supported_channels": 2,
                "min_cosine": 0.95,
                "retry_event_count": len(relaxed_target_local),
                "target_status_counts": dict(
                    sorted(
                        Counter(
                            value.status for value in relaxed_target_local
                        ).items()
                    )
                ),
                "decoy_status_counts": dict(
                    sorted(
                        Counter(
                            value.status for value in relaxed_decoy_local
                        ).items()
                    )
                ),
                "competition_count": len(relaxed_competitions),
                "target_candidate_count": sum(
                    value.target.quantitative_candidate
                    for value in relaxed_competitions
                ),
                "decoy_candidate_count": sum(
                    value.decoy.quantitative_candidate
                    for value in relaxed_competitions
                ),
                "accepted_event_count": sum(
                    value.winner == "target"
                    and value.target.quantitative_candidate
                    and value.q_value <= q_value_max
                    for value in relaxed_competitions
                ),
                "new_feature_count": len(relaxed_recovered),
                "final_strict_competition": relaxed_strict_competition,
            },
        }
        all_quant_rows = (
            strict_quant_rows
            + recovered_quant_rows
            + generic_recovered_quant_rows
        )
        feature_support_counts = _update_generic_quant_support(
            all_quant_rows, audit_by_event
        )
        generic_summary["feature_support_summary"] = {
            "feature_count": len(feature_support_counts),
            "event_count": sum(feature_support_counts.values()),
            "max_events_per_feature": max(
                feature_support_counts.values(), default=0
            ),
            "multi_ms2_feature_count": sum(
                count > 1 for count in feature_support_counts.values()
            ),
            "events_linked_to_multi_ms2_features": sum(
                count
                for count in feature_support_counts.values()
                if count > 1
            ),
            "events_per_feature_histogram": {
                str(key): count
                for key, count in sorted(
                    Counter(feature_support_counts.values()).items()
                )
            },
        }
        logger.info(
            "Hybrid generic local stage complete: %s; %d new features",
            dict(sorted(local_status_counts.items())),
            len(generic_recovered),
        )

    final_residual_contexts = []
    final_residual_records = []
    final_residual_quant_rows = []
    final_residual_summary = {
        "status": "not_run",
        "reason": "detector_not_provided",
        "detected_candidate_count": 0,
        "duplicate_existing_strict_count": 0,
        "accepted_feature_count": 0,
        "allocation_status_counts": {},
    }
    if final_strict_detector is not None:
        if strict_ownership["failed_feature_count"]:
            final_residual_summary["reason"] = (
                "incomplete_input_strict_ownership"
            )
        else:
            detector_result = final_strict_detector(
                residual_ledger.materialize(),
                strict_contexts=strict_contexts,
                next_feature_id=next_feature_id,
                args=args,
            )
            next_feature_id = int(detector_result["next_feature_id"])
            final_residual_summary.update(
                {
                    "status": detector_result["status"],
                    "reason": detector_result["reason"],
                    "isotope_calibration_reference": detector_result.get(
                        "isotope_calibration_reference", {}
                    ),
                    "calibration_boundary_guard": detector_result.get(
                        "calibration_boundary_guard", {}
                    ),
                }
            )
            detected_contexts = list(detector_result.get("contexts", ()))
            detected_records = _strict_feature_records(detected_contexts)
            final_residual_summary["detected_candidate_count"] = len(
                detected_records
            )
            accepted_population_records = list(strict_records)
            accepted_population_records.extend(
                _feature_row_as_strict_record(
                    row, FEATURE_ORIGIN_DIRECT_IDENTIFIED
                )
                for row in recovered_feature_rows
            )
            accepted_population_records.extend(
                _feature_row_as_strict_record(
                    row, FEATURE_ORIGIN_MS2_GUIDED_FULL
                )
                for row in generic_recovered_feature_rows
            )
            accepted_population_index = build_strict_feature_index(
                accepted_population_records
            )
            duplicate_matches = {
                int(record["feature_id"]): tuple(
                    _strict_record_existing_equivalents(
                        record,
                        accepted_population_index,
                        float(args.get("itol", 8.0)),
                    )
                )
                for record in detected_records
            }
            duplicate_matches = {
                feature_id: matches
                for feature_id, matches in duplicate_matches.items()
                if matches
            }
            duplicate_ids = set(duplicate_matches)
            duplicate_origin_counts = Counter()
            for matches in duplicate_matches.values():
                duplicate_origin_counts.update(
                    match.get(
                        "feature_origin", FEATURE_ORIGIN_STRICT_UNTARGETED
                    )
                    for match in matches
                )
            final_residual_summary[
                "duplicate_existing_strict_count"
            ] = sum(
                any(
                    match.get("feature_origin") is None
                    for match in matches
                )
                for matches in duplicate_matches.values()
            )
            final_residual_summary[
                "duplicate_existing_feature_count"
            ] = len(duplicate_ids)
            final_residual_summary[
                "duplicate_existing_origin_pair_counts"
            ] = dict(sorted(duplicate_origin_counts.items()))
            detected_contexts = _filter_context_feature_ids(
                detected_contexts, duplicate_ids
            )
            detected_records = _strict_feature_records(detected_contexts)
            final_ownership = _allocate_strict_feature_population(
                residual_ledger, detected_records
            )
            for status, count in final_ownership["status_counts"].items():
                residual_allocation_status_counts[
                    "final_strict_" + status
                ] += count
            detected_contexts = _filter_context_feature_ids(
                detected_contexts, final_ownership["failed_feature_ids"]
            )
            final_residual_contexts = detected_contexts
            final_residual_records = _strict_feature_records(
                final_residual_contexts
            )
            final_residual_summary.update(
                {
                    "accepted_feature_count": len(final_residual_records),
                    "allocation_status_counts": final_ownership[
                        "status_counts"
                    ],
                }
            )
            for record in final_residual_records:
                _scans, rt, traces = _strict_trace_grid(record)
                candidate = record["candidate"]
                final_residual_quant_rows.append(
                    _quant_row(
                        run_id,
                        record["feature_id"],
                        FEATURE_ORIGIN_STRICT_UNTARGETED,
                        "strict",
                        rt,
                        traces,
                        method=args.get("quant_method", "envelope_area"),
                        baseline=args.get(
                            "feature_baseline", "edge_linear"
                        ),
                        quality_score=float(
                            candidate.get("cos_cor_isotopes", 0.0)
                        ),
                        isotope_cosine=float(
                            candidate.get("cos_cor_isotopes", 0.0)
                        ),
                        mass_error=float(
                            np.median(
                                [
                                    value["mass_diff_ppm"]
                                    for value in candidate["isotopes"]
                                ]
                            )
                        ),
                        supporting_psm_count=0,
                        supporting_ms2_count=0,
                    )
                )
            logger.info(
                "Hybrid final residual strict stage: %s; %d accepted features",
                final_residual_summary["reason"],
                len(final_residual_records),
            )

    final_quant_rows = (
        strict_quant_rows
        + recovered_quant_rows
        + generic_recovered_quant_rows
        + final_residual_quant_rows
    )
    final_residual_direct_recheck = {
        "status": "not_run",
        "eligible_unlinked_assay_count": 0,
        "matched_assay_count": 0,
        "matched_event_count": 0,
        "status_counts": {},
    }
    if final_residual_records:
        final_strict_index = build_strict_feature_index(
            final_residual_records
        )
        direct_recheck_counts = Counter()
        eligible_assays = 0
        matched_events = set()
        for assay in assay_result.assays:
            audit = audit_by_event[assay.ms2_event_id]
            if (
                audit.get("feature_id") is not None
                or assay.conflict_status != "unique"
            ):
                continue
            eligible_assays += 1
            matched, status, alternatives = match_assay_to_strict_feature(
                assay,
                final_strict_index,
                ppm=base_ppm,
                rt_tolerance_sec=base_rt_tolerance,
            )
            direct_recheck_counts[status] += 1
            audit["alternative_count"] = max(
                int(audit.get("alternative_count") or 0), alternatives
            )
            if matched is None:
                continue
            feature_id = int(matched["feature_id"])
            support[feature_id].append(assay)
            matched_events.add(int(assay.ms2_event_id))
            audit.update(
                {
                    "feature_id": feature_id,
                    "association_tier": "direct_id",
                    "status": "matched_final_residual_strict_feature",
                    "charge_used": assay.charge,
                    "charge_source": "psm",
                    "rt_error_sec": _rt_distance(
                        assay.rt_sec,
                        matched["rt_start"],
                        matched["rt_end"],
                    ),
                }
            )
        final_quant_by_id = {
            int(row["feature_id"]): row
            for row in final_residual_quant_rows
        }
        for feature_id, assays in support.items():
            row = final_quant_by_id.get(int(feature_id))
            if row is None:
                continue
            row["supporting_psm_count"] = len(assays)
            row["supporting_ms2_count"] = len(
                {assay.ms2_event_id for assay in assays}
            )
            row["confidence_tier"] = "direct_id"
        final_residual_direct_recheck = {
            "status": "completed",
            "eligible_unlinked_assay_count": eligible_assays,
            "matched_assay_count": direct_recheck_counts[
                "matched_strict_feature"
            ],
            "matched_event_count": len(matched_events),
            "status_counts": dict(sorted(direct_recheck_counts.items())),
        }
        logger.info(
            "Hybrid final residual strict direct-ID recheck complete: %s",
            final_residual_direct_recheck,
        )
    final_residual_summary[
        "direct_ms2_recheck"
    ] = final_residual_direct_recheck
    final_residual_recheck = {
        "status": "not_run",
        "eligible_unlinked_event_count": 0,
        "rescored_target_count": 0,
        "rescored_decoy_count": 0,
        "audit_status_counts": {},
        "competition_counts": {},
    }
    if final_residual_contexts and generic_summary is not None:
        unlinked_ids = {
            event_id
            for event_id, audit in audit_by_event.items()
            if audit.get("feature_id") is None
            and audit.get("primary_identification_id") is None
        }
        unlinked_events = [
            event
            for event in ingestion.ms2_rows
            if int(event["ms2_event_id"]) in unlinked_ids
        ]
        final_target_links, final_target_summary = _generic_standard_links(
            unlinked_events, ingestion, final_residual_contexts, args
        )
        final_decoy_links, final_decoy_summary = _generic_standard_links(
            _generic_decoy_rows(run_id, unlinked_events),
            ingestion,
            final_residual_contexts,
            args,
        )
        rescored_target = _rescore_generic_link_rows(
            final_target_links, generic_score_weights
        )
        rescored_decoy = _rescore_generic_link_rows(
            final_decoy_links, generic_score_weights
        )
        recheck_status, recheck_competition = (
            _apply_generic_strict_associations(
                audit_by_event,
                final_target_links,
                final_decoy_links,
                q_value_max=float(
                    args.get("generic_q_value_max", 0.01)
                ),
                eligible_event_ids=unlinked_ids,
                preserve_failed_audit=True,
            )
        )
        final_residual_recheck = {
            "status": "completed",
            "q_value_family": "final_residual_strict_recheck",
            "eligible_unlinked_event_count": len(unlinked_events),
            "rescored_target_count": rescored_target,
            "rescored_decoy_count": rescored_decoy,
            "target": _compact_generic_association_summary(final_target_summary),
            "decoy": _compact_generic_association_summary(final_decoy_summary),
            "audit_status_counts": recheck_status,
            "competition_counts": recheck_competition,
        }
        logger.info(
            "Hybrid final residual strict MS2 recheck complete: %s",
            {
                "audit": dict(sorted(recheck_status.items())),
                "competition": recheck_competition,
            },
        )
    final_residual_summary["ms2_recheck"] = final_residual_recheck
    _update_generic_quant_support(final_quant_rows, audit_by_event)
    _append_final_strict_features(manager, strict_contexts, args)
    if final_residual_contexts:
        _append_final_strict_features(
            manager, final_residual_contexts, args
        )
    if recovered_feature_rows or generic_recovered_feature_rows:
        manager.append_features(
            recovered_feature_rows + generic_recovered_feature_rows
        )
    manager.append_hybrid_feature_quant(
        final_quant_rows
    )
    manager.append_hybrid_ms2_audit(list(audit_by_event.values()))
    manager.append_identifications(assay_result.audit)
    manager.append_id_assays(assay_rows)
    args["_hybrid_summary"] = {
        "relaxed_ms2_feature_enabled": bool(
            args.get("relaxed_ms2_feature", False)
        ),
        "relaxed_direct_q_value_exclusive_max": RELAXED_DIRECT_Q_VALUE_MAX,
        "direct_calibration": direct_calibration.as_dict(),
        "direct_retry_counts": dict(sorted(direct_retry_counts.items())),
        "direct_processed_hill_competitors": {
            "match_count": len(direct_processed_competitors),
            "assay_count": len(direct_processed_by_psm),
            "unique_candidate_count": len(
                {
                    competitor.candidate_key
                    for competitor in direct_processed_competitors
                }
            ),
            "top_k_per_assay": 3,
            "status": (
                "captured_preconflict_losing_candidates"
                if direct_processed_competitors
                else "none_captured_or_legacy_cache"
            ),
        },
        "strict_feature_count": len(strict_records) + len(final_residual_records),
        "input_strict_feature_count": len(strict_records),
        "final_residual_strict_feature_count": len(final_residual_records),
        "direct_assay_count": len(assay_result.assays),
        "recovered_feature_count": len(recovered),
        "generic_recovered_feature_count": len(generic_recovered),
        "audit_row_count": len(audit_by_event),
        "audit_status_counts": dict(Counter(row["status"] for row in audit_by_event.values())),
        "feature_population_summary": _feature_population_summary(
            final_quant_rows, audit_by_event
        ),
        "ms2_audit_summary": _ms2_audit_summary(
            final_quant_rows, audit_by_event
        ),
        "identification_parser_qc": args.get(
            "_identification_parser_qc"
        ),
        "generic_summary": generic_summary,
        "input_strict_ownership": strict_ownership,
        "final_residual_strict": final_residual_summary,
        "targeted_residual_allocation": {
            "allocation_status_counts": dict(
                sorted(residual_allocation_status_counts.items())
            ),
            "accepted_allocation_count": residual_ledger.allocation_count,
            "claimed_raw_point_count": residual_ledger.claimed_point_count,
            "claimed_intensity": residual_ledger.claimed_intensity,
            "original_raw_intensity": residual_ledger.original_intensity,
            "residual_raw_intensity": residual_ledger.residual_intensity,
            "intensity_conserved": math.isclose(
                residual_ledger.claimed_intensity
                + residual_ledger.residual_intensity,
                residual_ledger.original_intensity,
                rel_tol=1e-12,
                abs_tol=1e-8,
            ),
        },
        "local_candidate_cache": local_candidate_cache_telemetry,
    }
    return next_feature_id
