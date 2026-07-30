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


from .hybrid_constants import *

def _generic_standard_links(ms2_rows, ingestion, strict_contexts, args):
    """Match generic precursor hypotheses to final strict features.

    This deliberately invokes the same bounded scan/isolation-aware matching
    path for targets and decoys.  It annotates only the already selected strict
    candidates, so generic evidence cannot alter or fabricate the feature
    population at this stage.
    """

    generic_args = dict(args)
    generic_args["generic_ms2_isotope_errors"] = tuple(
        args.get("generic_ms2_isotope_errors", (0, 1, 2, 3))
    )
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



__all__ = [name for name in globals() if not name.startswith("__")]
