"""Identification-aware direct assays and bounded local feature extraction."""

from __future__ import annotations

from collections import Counter
from bisect import bisect_left, bisect_right
import logging

import numpy as np

from .generic_local import (
    evaluate_generic_local_candidate_pairs,
)
from .postprocess_cache import (
    load_local_candidate_pairs,
    local_candidate_fingerprint,
    save_local_candidate_pairs,
)


logger = logging.getLogger(__name__)

from .hybrid_assays import *
from .hybrid_constants import *
from .hybrid_local import *
from .hybrid_strict import *


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


def _feature_population_summary(quant_rows, audit_by_event):
    """Summarize features first, separately from MS2 event coverage."""

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
        options={**options, "trace_extractor": "cython"},
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

    # Candidate evaluation is read-only.  Materializing once lets workers use
    # RawMS1Store's indexed and native batch extraction while later accepted
    # candidates still allocate against the authoritative sparse ledger.
    residual_store = residual_ledger.materialize()
    targets, decoys = evaluate_generic_local_candidate_pairs(
        residual_store,
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



__all__ = [name for name in globals() if not name.startswith("__")]
