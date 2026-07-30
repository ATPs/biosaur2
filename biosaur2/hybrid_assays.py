"""Identification-aware direct assays and bounded local feature extraction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from dataclasses import replace
from bisect import bisect_left, bisect_right
import logging
import math
import time
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


from .hybrid_constants import RELAXED_DIRECT_Q_VALUE_MAX

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



__all__ = [name for name in globals() if not name.startswith("__")]
