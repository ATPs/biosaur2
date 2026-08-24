"""Hybrid workflow orchestration and compatibility entry point implementation."""

from collections import Counter
import gc
import logging
import math
import time
from typing import Mapping

from .hybrid_runtime import *
from .hybrid_direct_stage import run_direct_stage
from .hybrid_generic_stage import run_generic_stage
from .hybrid_residual_stage import run_final_residual_stage
from .external_mbr import (
    sidecar_rows,
    write_feature_sidecars,
)
from .external_weak import weak_feature_rows_from_contexts


logger = logging.getLogger(__name__)


def _finalize_hybrid_results(*, run_id, ingestion, assay_result, strict_contexts, manager, args, direct, generic, residual):
    hybrid_started = direct["hybrid_started"]
    strict_records = direct["strict_records"]
    strict_ownership = direct["strict_ownership"]
    residual_ledger = direct["residual_ledger"]
    residual_allocation_status_counts = direct["residual_allocation_status_counts"]
    local_candidate_cache_telemetry = direct["local_candidate_cache_telemetry"]
    direct_processed_competitors = direct["direct_processed_competitors"]
    direct_processed_by_psm = direct["direct_processed_by_psm"]
    direct_calibration = direct["direct_calibration"]
    direct_retry_counts = direct["direct_retry_counts"]
    recovered = direct["recovered"]
    recovered_feature_rows = direct["recovered_feature_rows"]
    assay_rows = direct["assay_rows"]
    audit_by_event = direct["audit_by_event"]
    generic_summary = generic["generic_summary"]
    generic_recovered_feature_rows = generic["generic_recovered_feature_rows"]
    generic_recovered = generic["generic_recovered"]
    final_residual_contexts = residual["final_residual_contexts"]
    final_residual_records = residual["final_residual_records"]
    final_residual_summary = residual["final_residual_summary"]
    final_quant_rows = residual["final_quant_rows"]
    next_feature_id = residual["next_feature_id"]
    output_assembly_started = time.monotonic()
    _update_generic_quant_support(final_quant_rows, audit_by_event)
    final_feature_rows = _final_strict_feature_rows(strict_contexts, args)
    if final_residual_contexts:
        final_feature_rows.extend(
            _final_strict_feature_rows(final_residual_contexts, args)
        )
    if recovered_feature_rows or generic_recovered_feature_rows:
        final_feature_rows.extend(
            recovered_feature_rows + generic_recovered_feature_rows
        )
    # Weak candidates are private Project sidecar rows.  They are never
    # appended to the ordinary run output until cross-run FDR accepts them.
    weak_feature_rows, weak_candidate_audit = (
        weak_feature_rows_from_contexts(
            run_id, strict_contexts, final_feature_rows, args,
            residual_ledger,
        )
        if args.get("external_id") and args.get("external_weak_candidates_cache_path")
        else ([], {})
    )
    if weak_candidate_audit:
        logger.info(
            "External weak-candidate local funnel: %s",
            weak_candidate_audit,
        )
    logger.debug(
        'Hybrid final output assembly complete: runtime_sec=%.3f feature_rows=%d quant_rows=%d',
        time.monotonic() - output_assembly_started,
        len(final_feature_rows),
        len(final_quant_rows),
    )
    args["_hybrid_summary"] = {
        "trace_extractor": "cython",
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
    if args.get("external_id") and args.get("external_weak_candidates_cache_path"):
        strong_rows, weak_rows = sidecar_rows(
            run_id, final_feature_rows, final_quant_rows, weak_feature_rows
        )
        write_feature_sidecars(
            args["file"],
            {
                "external_strong_features": args.get("external_strong_features_cache_path"),
                "external_weak_candidates": args.get("external_weak_candidates_cache_path"),
            },
            strong_rows,
            weak_rows,
            args,
        )
        args["_hybrid_summary"]["external_feature_mbr"] = {
            "strong_feature_count": len(strong_rows),
            "weak_candidate_count": len(weak_rows),
            "weak_candidate_audit": weak_candidate_audit,
        }
    output_started = time.monotonic()
    manager.append_hybrid_results(
        final_feature_rows,
        final_quant_rows,
        list(audit_by_event.values()),
        ingestion.ms2_rows,
        assay_result.audit,
        assay_rows,
    )
    logger.debug(
        'Hybrid output append complete: runtime_sec=%.3f feature_rows=%d quant_rows=%d audit_rows=%d',
        time.monotonic() - output_started,
        len(final_feature_rows),
        len(final_quant_rows),
        len(audit_by_event),
    )
    logger.debug(
        'Hybrid postprocessing complete: runtime_sec=%.3f next_feature_id=%d',
        time.monotonic() - hybrid_started,
        next_feature_id,
    )
    return next_feature_id


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

    direct = run_direct_stage(
        run_id=run_id,
        ingestion=ingestion,
        assay_result=assay_result,
        strict_contexts=strict_contexts,
        next_feature_id=next_feature_id,
        args=args,
    )
    generic = run_generic_stage(
        run_id=run_id,
        ingestion=ingestion,
        strict_contexts=strict_contexts,
        args=args,
        audit_by_event=direct["audit_by_event"],
        strict_index=direct["strict_index"],
        strict_hill_claims=direct["strict_hill_claims"],
        residual_ledger=direct["residual_ledger"],
        residual_allocation_status_counts=direct["residual_allocation_status_counts"],
        strict_ownership=direct["strict_ownership"],
        strict_quant_rows=direct["strict_quant_rows"],
        recovered=direct["recovered"],
        recovered_quant_rows=direct["recovered_quant_rows"],
        local_candidate_cache_telemetry=direct["local_candidate_cache_telemetry"],
        next_feature_id=direct["next_feature_id"],
        final_strict_detector=final_strict_detector,
    )
    direct.pop("strict_index", None)
    direct.pop("strict_hill_claims", None)
    gc.collect()
    residual = run_final_residual_stage(
        run_id=run_id,
        ingestion=ingestion,
        assay_result=assay_result,
        strict_contexts=strict_contexts,
        args=args,
        final_strict_detector=final_strict_detector,
        strict_records=direct["strict_records"],
        strict_ownership=direct["strict_ownership"],
        residual_ledger=direct["residual_ledger"],
        residual_allocation_status_counts=direct["residual_allocation_status_counts"],
        audit_by_event=direct["audit_by_event"],
        support=direct["support"],
        base_ppm=direct["base_ppm"],
        base_rt_tolerance=direct["base_rt_tolerance"],
        recovered_feature_rows=direct["recovered_feature_rows"],
        recovered_quant_rows=direct["recovered_quant_rows"],
        generic_recovered_feature_rows=generic["generic_recovered_feature_rows"],
        generic_recovered_quant_rows=generic["generic_recovered_quant_rows"],
        generic_summary=generic["generic_summary"],
        generic_score_weights=generic["generic_score_weights"],
        strict_quant_rows=direct["strict_quant_rows"],
        next_feature_id=generic["next_feature_id"],
    )
    return _finalize_hybrid_results(
        run_id=run_id,
        ingestion=ingestion,
        assay_result=assay_result,
        strict_contexts=strict_contexts,
        manager=manager,
        args=args,
        direct=direct,
        generic=generic,
        residual=residual,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
