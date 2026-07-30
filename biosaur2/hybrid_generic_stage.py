"""Generic MS2 association and local recovery stage."""

from .hybrid_runtime import *


def run_generic_stage(
    *,
    run_id,
    ingestion,
    strict_contexts,
    args,
    audit_by_event,
    strict_index,
    strict_hill_claims,
    residual_ledger,
    residual_allocation_status_counts,
    strict_ownership,
    strict_quant_rows,
    recovered,
    recovered_quant_rows,
    local_candidate_cache_telemetry,
    next_feature_id,
    final_strict_detector,
):
    generic_summary = None
    generic_recovered_feature_rows = []
    generic_recovered_quant_rows = []
    generic_recovered = []
    generic_score_weights = dict(GENERIC_ASSOCIATION_SCORE_WEIGHTS)
    if args.get("generic_ms2_refine", True):
        generic_started = time.monotonic()
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
        configured_width = args.get("generic_local_max_width_sec", "auto")
        width_limit = (
            generic_local_width_limit(strict_quant_rows)
            if configured_width == "auto"
            else float(configured_width)
        )
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
            "isotope_count": int(args.get("generic_local_isotope_count", 5)),
            "isotope_errors": tuple(
                value
                for value in args.get("generic_ms2_isotope_errors", (0, 1, 2, 3))
                if int(value) >= 0
            ),
            "min_mono_points": int(args.get("generic_local_min_mono_points", 3)),
            "min_channel_points": int(args.get("generic_local_min_channel_points", 3)),
            "min_supported_channels": int(args.get("generic_local_min_supported_channels", 2)),
            "min_cosine": float(args.get("generic_local_min_isotope_cosine", 0.90)),
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
                "isotope_count": int(args.get("generic_local_isotope_count", 5)),
                "isotope_errors": tuple(
                    value
                    for value in args.get("generic_ms2_isotope_errors", (0, 1, 2, 3))
                    if int(value) >= 0
                ),
                "min_mono_points": int(args.get("generic_relaxed_min_mono_points", 2)),
                "min_channel_points": int(args.get("generic_relaxed_min_channel_points", 2)),
                "min_supported_channels": int(args.get("generic_relaxed_min_supported_channels", 2)),
                "min_cosine": float(args.get("generic_relaxed_min_isotope_cosine", 0.95)),
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
            "isotope_errors": list(
                args.get("generic_ms2_isotope_errors", (0, 1, 2, 3))
            ),
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
        logger.debug(
            'Hybrid generic stage complete: runtime_sec=%.3f local_events=%d recovered=%d cache=%s',
            time.monotonic() - generic_started,
            len(local_events),
            len(generic_recovered),
            local_candidate_cache_telemetry,
        )


    return {
        "generic_summary": generic_summary,
        "generic_recovered_feature_rows": generic_recovered_feature_rows,
        "generic_recovered_quant_rows": generic_recovered_quant_rows,
        "generic_recovered": generic_recovered,
        "generic_score_weights": generic_score_weights,
        "next_feature_id": next_feature_id,
    }
