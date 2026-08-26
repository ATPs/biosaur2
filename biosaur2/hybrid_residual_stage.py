"""Final residual strict detection and association recheck stage."""

from collections import Counter
import logging
import time

import numpy as np

from .hybrid_runtime import *


logger = logging.getLogger(__name__)


def _run_final_residual_direct_recheck(
    *,
    run_id,
    ingestion,
    assay_result,
    strict_contexts,
    args,
    final_strict_detector,
    strict_records,
    strict_ownership,
    residual_ledger,
    residual_allocation_status_counts,
    audit_by_event,
    support,
    base_ppm,
    base_rt_tolerance,
    recovered_feature_rows,
    recovered_quant_rows,
    generic_recovered_feature_rows,
    generic_recovered_quant_rows,
    generic_summary,
    generic_score_weights,
    strict_quant_rows,
    next_feature_id,
):
    residual_started = time.monotonic()
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
            detector_started = time.monotonic()
            detector_result = final_strict_detector(
                residual_ledger.materialize(),
                strict_contexts=strict_contexts,
                next_feature_id=next_feature_id,
                args=args,
            )
            logger.debug(
                'Hybrid final residual strict detection complete: runtime_sec=%.3f',
                time.monotonic() - detector_started,
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
            ownership_started = time.monotonic()
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
            logger.debug(
                'Hybrid final residual ownership complete: runtime_sec=%.3f detected=%d accepted=%d',
                time.monotonic() - ownership_started,
                final_residual_summary['detected_candidate_count'],
                len(final_residual_records),
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
                    "charge_source": assay.charge_source,
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
    logger.debug(
        'Hybrid final residual direct recheck complete: runtime_sec=%.3f accepted=%d',
        time.monotonic() - residual_started,
        len(final_residual_records),
    )
    return _run_final_residual_generic_recheck(
        run_id=run_id,
        ingestion=ingestion,
        args=args,
        final_residual_contexts=final_residual_contexts,
        final_residual_records=final_residual_records,
        final_residual_quant_rows=final_residual_quant_rows,
        final_residual_summary=final_residual_summary,
        audit_by_event=audit_by_event,
        generic_summary=generic_summary,
        generic_score_weights=generic_score_weights,
        strict_quant_rows=strict_quant_rows,
        recovered_quant_rows=recovered_quant_rows,
        generic_recovered_quant_rows=generic_recovered_quant_rows,
        next_feature_id=next_feature_id,
        final_quant_rows=final_quant_rows,
    )


def _run_final_residual_generic_recheck(
    *,
    run_id,
    ingestion,
    args,
    final_residual_contexts,
    final_residual_records,
    final_residual_quant_rows,
    final_residual_summary,
    audit_by_event,
    generic_summary,
    generic_score_weights,
    strict_quant_rows,
    recovered_quant_rows,
    generic_recovered_quant_rows,
    next_feature_id,
    final_quant_rows,
):
    generic_recheck_started = time.monotonic()
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
        final_target_result, final_decoy_result = generic_standard_link_pair(
            run_id,
            unlinked_events,
            ingestion,
            final_residual_contexts,
            args,
        )
        final_target_links, final_target_summary, _target_runtime = final_target_result
        final_decoy_links, final_decoy_summary, _decoy_runtime = final_decoy_result
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
                    args.get("generic_q_value_max", 0.05)
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
    logger.debug(
        'Hybrid final residual generic recheck complete: runtime_sec=%.3f eligible_events=%d',
        time.monotonic() - generic_recheck_started,
        final_residual_recheck['eligible_unlinked_event_count'],
    )

    return {
        "final_residual_contexts": final_residual_contexts,
        "final_residual_records": final_residual_records,
        "final_residual_quant_rows": final_residual_quant_rows,
        "final_residual_summary": final_residual_summary,
        "final_quant_rows": final_quant_rows,
        "next_feature_id": next_feature_id,
    }


def run_final_residual_stage(
    *,
    run_id,
    ingestion,
    assay_result,
    strict_contexts,
    args,
    final_strict_detector,
    strict_records,
    strict_ownership,
    residual_ledger,
    residual_allocation_status_counts,
    audit_by_event,
    support,
    base_ppm,
    base_rt_tolerance,
    recovered_feature_rows,
    recovered_quant_rows,
    generic_recovered_feature_rows,
    generic_recovered_quant_rows,
    generic_summary,
    generic_score_weights,
    strict_quant_rows,
    next_feature_id,
):
    return _run_final_residual_direct_recheck(
        run_id=run_id,
        ingestion=ingestion,
        assay_result=assay_result,
        strict_contexts=strict_contexts,
        args=args,
        final_strict_detector=final_strict_detector,
        strict_records=strict_records,
        strict_ownership=strict_ownership,
        residual_ledger=residual_ledger,
        residual_allocation_status_counts=residual_allocation_status_counts,
        audit_by_event=audit_by_event,
        support=support,
        base_ppm=base_ppm,
        base_rt_tolerance=base_rt_tolerance,
        recovered_feature_rows=recovered_feature_rows,
        recovered_quant_rows=recovered_quant_rows,
        generic_recovered_feature_rows=generic_recovered_feature_rows,
        generic_recovered_quant_rows=generic_recovered_quant_rows,
        generic_summary=generic_summary,
        generic_score_weights=generic_score_weights,
        strict_quant_rows=strict_quant_rows,
        next_feature_id=next_feature_id,
    )
