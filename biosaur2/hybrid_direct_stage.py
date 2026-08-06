"""Direct identification matching and bounded local recovery stage."""

from collections import Counter, defaultdict
from dataclasses import replace
import logging
import time

import numpy as np

from .hybrid_runtime import *
from .residual import ResidualMS1Ledger


logger = logging.getLogger(__name__)


def _prepare_and_run_direct_stage(*, run_id, ingestion, assay_result, strict_contexts, next_feature_id, args):
    hybrid_started = time.monotonic()
    logger.debug(
        'Hybrid postprocessing start: run_id=%s contexts=%d direct_assays=%d ms2_events=%d',
        run_id,
        len(strict_contexts),
        len(assay_result.assays),
        len(ingestion.ms2_rows),
    )
    strict_ownership_started = time.monotonic()
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
    logger.debug(
        'Hybrid strict ownership complete: runtime_sec=%.3f strict_features=%d statuses=%s',
        time.monotonic() - strict_ownership_started,
        len(strict_records),
        strict_ownership['status_counts'],
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
    direct_matching_started = time.monotonic()
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
    logger.debug(
        'Hybrid direct matching/calibration complete: runtime_sec=%.3f calibration=%s',
        time.monotonic() - direct_matching_started,
        direct_calibration,
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
    direct_retry_counts = Counter()
    state = {
        "hybrid_started": hybrid_started,
        "strict_records": strict_records,
        "strict_index": strict_index,
        "strict_hill_claims": strict_hill_claims,
        "residual_ledger": residual_ledger,
        "residual_allocation_status_counts": residual_allocation_status_counts,
        "strict_ownership": strict_ownership,
        "local_candidate_cache_telemetry": local_candidate_cache_telemetry,
        "direct_processed_competitors": direct_processed_competitors,
        "direct_processed_by_psm": direct_processed_by_psm,
        "base_ppm": base_ppm,
        "base_rt_tolerance": base_rt_tolerance,
        "direct_calibration": direct_calibration,
        "direct_match_results": direct_match_results,
        "audit_by_event": audit_by_event,
        "assay_rows": assay_rows,
        "recovered": recovered,
        "recovered_feature_rows": recovered_feature_rows,
        "recovered_quant_rows": recovered_quant_rows,
        "direct_retry_counts": direct_retry_counts,
        "support": support,
        "next_feature_id": next_feature_id,
    }
    return _recover_direct_assays(run_id, ingestion, assay_result, args, state)


def _recover_direct_assays(run_id, ingestion, assay_result, args, state):
    strict_hill_claims = state["strict_hill_claims"]
    residual_ledger = state["residual_ledger"]
    residual_allocation_status_counts = state["residual_allocation_status_counts"]
    direct_processed_by_psm = state["direct_processed_by_psm"]
    base_ppm = state["base_ppm"]
    base_rt_tolerance = state["base_rt_tolerance"]
    direct_calibration = state["direct_calibration"]
    direct_match_results = state["direct_match_results"]
    audit_by_event = state["audit_by_event"]
    assay_rows = state["assay_rows"]
    recovered = state["recovered"]
    recovered_feature_rows = state["recovered_feature_rows"]
    recovered_quant_rows = state["recovered_quant_rows"]
    direct_retry_counts = state["direct_retry_counts"]
    support = state["support"]
    next_feature_id = state["next_feature_id"]
    direct_status_counts = Counter()
    direct_local_started = time.monotonic()
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
    logger.debug(
        'Hybrid direct local recovery complete: runtime_sec=%.3f statuses=%s recovered=%d retries=%s',
        time.monotonic() - direct_local_started,
        dict(sorted(direct_status_counts.items())),
        len(recovered),
        dict(sorted(direct_retry_counts.items())),
    )
    state["next_feature_id"] = next_feature_id
    return _quantify_direct_stage(run_id, args, state)


def _quantify_direct_stage(run_id, args, state):
    strict_records = state["strict_records"]
    support = state["support"]
    recovered_quant_rows = state["recovered_quant_rows"]
    strict_quant_rows = []
    logger.info("Hybrid quantifying %d strict features", len(strict_records))
    strict_quantification_started = time.monotonic()
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
    logger.debug(
        'Hybrid strict quantification complete: runtime_sec=%.3f rows=%d',
        time.monotonic() - strict_quantification_started,
        len(strict_quant_rows),
    )
    # Update deduplicated recovered support counts after every assay is linked.
    by_id = {row["feature_id"]: row for row in recovered_quant_rows}
    for feature_id, assays in support.items():
        if feature_id in by_id:
            by_id[feature_id]["supporting_psm_count"] = len(assays)
            by_id[feature_id]["supporting_ms2_count"] = len({assay.ms2_event_id for assay in assays})

    state.pop("direct_match_results", None)
    return {**state, "strict_quant_rows": strict_quant_rows}


def run_direct_stage(*, run_id, ingestion, assay_result, strict_contexts, next_feature_id, args):
    return _prepare_and_run_direct_stage(
        run_id=run_id,
        ingestion=ingestion,
        assay_result=assay_result,
        strict_contexts=strict_contexts,
        next_feature_id=next_feature_id,
        args=args,
    )
