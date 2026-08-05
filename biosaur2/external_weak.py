"""Local weak-candidate materialization for feature-only Project MBR."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from copy import deepcopy
import math

import numpy as np


DEFAULT_MAX_STRONG_OVERLAP = 0.30


def remember_reject_snapshot(snapshots, candidate):
    snapshots.setdefault(id(candidate), deepcopy(candidate))


def append_rejected_candidate(sink, snapshots, candidate):
    if sink is not None:
        sink.append(snapshots.pop(id(candidate), candidate))


def publish_detector_outcomes(
    hills, smart_rejects, greedy_rejects, *, initial_count,
    smart_accepted_count, strict_selected_count,
):
    for candidate in smart_rejects:
        candidate["_external_reject_source"] = "smart_filter_reject"
    for candidate in greedy_rejects:
        candidate["_external_reject_source"] = "greedy_conflict_reject"
    hills["_external_weak_candidates"] = tuple(
        smart_rejects + greedy_rejects
    )
    hills["_external_weak_detector_audit"] = {
        "initial_candidates": int(initial_count),
        "smart_filter_accepted": int(smart_accepted_count),
        "smart_filter_rejected": len(smart_rejects),
        "strict_selected": int(strict_selected_count),
        "greedy_rejected": len(greedy_rejects),
    }


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _faims_key(value):
    result = _finite(value)
    return "none" if result is None else format(result, ".9g")


def _candidate_key(candidate, faims_cv):
    return (
        _faims_key(faims_cv),
        int(candidate["charge"]),
        int(candidate["monoisotope hill idx"]),
    )


def _quality_key(candidate):
    isotope_count = int(candidate.get("nIsotopes", 0))
    cosine = float(candidate.get("cos_cor_isotopes", 0.0) or 0.0)
    errors = [
        abs(float(value.get("mass_diff_ppm", float("inf"))))
        for value in candidate.get("isotopes", ())
    ]
    return (
        -(isotope_count + cosine),
        -isotope_count,
        -cosine,
        float(np.mean(errors)) if errors else float("inf"),
        float(candidate.get("hill_mz_1", float("inf"))),
        tuple(
            [int(candidate["monoisotope hill idx"])]
            + [
                int(value["isotope_hill_idx"])
                for value in candidate.get("isotopes", ())
            ]
        ),
    )


def _candidate_interval_sec(context, candidate):
    scans = context["hills"]["hills_scan_lists"][
        int(candidate["monoisotope idx"])
    ]
    rt = context["rt_by_local"]
    return float(rt[int(scans[0])]), float(rt[int(scans[-1])])


def _strong_interval_sec(row):
    start = _finite(row.get("rt_start_sec"))
    end = _finite(row.get("rt_end_sec"))
    if start is None:
        legacy = _finite(row.get("rtStart"))
        start = legacy
    if end is None:
        legacy = _finite(row.get("rtEnd"))
        end = legacy
    return start, end


def _strong_index(final_feature_rows):
    result = defaultdict(list)
    for row in final_feature_rows:
        mz = _finite(row.get("mz"))
        start, end = _strong_interval_sec(row)
        try:
            charge = int(row.get("charge"))
        except (TypeError, ValueError):
            continue
        if mz is None or start is None or end is None:
            continue
        result[(charge, _faims_key(row.get("FAIMS", row.get("faims_cv"))))].append(
            (mz, start, end)
        )
    for rows in result.values():
        rows.sort()
    return {
        key: (tuple(row[0] for row in rows), tuple(rows))
        for key, rows in result.items()
    }


def _strong_equivalent(index, context, candidate):
    indexed = index.get((
        int(candidate["charge"]), _faims_key(context["faims_cv"])
    ))
    if indexed is None:
        return False
    masses, rows = indexed
    mz = float(candidate["hill_mz_1"])
    start, end = _candidate_interval_sec(context, candidate)
    tolerance = mz * 8e-6
    left = bisect_left(masses, mz - tolerance)
    right = bisect_right(masses, mz + tolerance)
    return any(
        max(start, strong_start) <= min(end, strong_end)
        for _strong_mz, strong_start, strong_end in rows[left:right]
    )


def _candidate_contributions(context, candidate):
    hills = context["hills"]
    spectra = context["spectra"]
    indices = [int(candidate["monoisotope idx"])] + [
        int(value["isotope_idx"])
        for value in candidate.get("isotopes", ())
    ]
    contributions = []
    for index in indices:
        scans = hills["hills_scan_lists"][index]
        mz_values = hills["tmp_mz_array"][index]
        intensities = hills["hills_intensity_array"][index]
        if not (len(scans) == len(mz_values) == len(intensities)):
            raise ValueError("weak candidate hill point arrays are inconsistent")
        for local_scan, mz, intensity in zip(scans, mz_values, intensities):
            source_scan = spectra[int(local_scan)].get("scan_index")
            if source_scan is None:
                raise ValueError("weak candidate context lacks scan provenance")
            contributions.append(
                (int(source_scan), float(mz), float(intensity))
            )
    return tuple(contributions)


def _candidate_basic_status(context, candidate, args):
    hills = context["hills"]
    mono_points = len(hills["hills_scan_lists"][
        int(candidate["monoisotope idx"])
    ])
    secondary_points = max((
        len(hills["hills_scan_lists"][int(value["isotope_idx"])])
        for value in candidate.get("isotopes", ())
    ), default=0)
    cosine = _finite(candidate.get("cos_cor_isotopes"))
    if mono_points < int(args.get("external_weak_min_mono_points", 2)):
        return "mono_points_below_minimum", mono_points, secondary_points, cosine
    if secondary_points < int(
        args.get("external_weak_min_secondary_points", 2)
    ):
        return "secondary_points_below_minimum", mono_points, secondary_points, cosine
    if cosine is None or cosine < float(
        args.get("external_weak_min_isotope_cosine", 0.6)
    ):
        return "isotope_cosine_below_minimum", mono_points, secondary_points, cosine
    return "accepted", mono_points, secondary_points, cosine


def _weak_row(base, candidate, quant, mono_points, secondary_points, cosine, overlap):
    return {
        **base,
        "run_id": base.get("run_id"),
        "feature_id": int(base["feature_idx"]),
        "feature_origin": "aligned_external_weak",
        "confidence_tier": "external_id_weak",
        "quant_value": quant,
        "quant_status": "quantified",
        "area_envelope_raw": quant,
        "area_envelope_corrected": quant,
        "area_mono_raw": quant,
        "area_mono_corrected": quant,
        "envelope_apex": base.get("intensityApex"),
        "quant_envelope_area": quant,
        "quant_mono_area": quant,
        "quant_envelope_apex": base.get("intensityApex"),
        "feature_quality_score": cosine,
        "quality_flags": 0,
        "extraction_q_value": None,
        "supporting_psm_count": 0,
        "supporting_ms2_count": 0,
        "points_across_peak": mono_points,
        "rt_start_sec": float(base["rtStart"]),
        "rt_apex_sec": float(base["rtApex"]),
        "rt_end_sec": float(base["rtEnd"]),
        "isotope_cosine": cosine,
        "mass_error_ppm_median": base.get("isoerror"),
        "ms2_events": [],
        "external_secondary_points": secondary_points,
        "external_reject_source": candidate["_external_reject_source"],
        "external_strong_overlap_fraction": overlap,
        "external_local_gate_status": "accepted",
    }


def weak_feature_rows_from_contexts(
    run_id, contexts, final_feature_rows, args, residual_ledger
):
    """Filter detector rejects against final ownership and quantify survivors."""

    from . import utils

    max_strong_overlap = float(args.get(
        "external_weak_max_strong_overlap",
        DEFAULT_MAX_STRONG_OVERLAP,
    ))
    if not math.isfinite(max_strong_overlap) or not 0 <= max_strong_overlap <= 1:
        raise ValueError(
            "external weak maximum strong overlap must be finite and in [0, 1]"
        )
    audit = Counter()
    source_counts = Counter()
    for context in contexts:
        for key, value in context["hills"].get(
            "_external_weak_detector_audit", {}
        ).items():
            audit[key] += int(value)
    strong_index = _strong_index(final_feature_rows)
    survivors = {}
    for context_index, context in enumerate(contexts):
        for candidate in context["hills"].get(
            "_external_weak_candidates", ()
        ):
            audit["weak_pool"] += 1
            source = candidate.get(
                "_external_reject_source", "unknown_reject"
            )
            source_counts[source] += 1
            status, mono_points, secondary_points, cosine = (
                _candidate_basic_status(context, candidate, args)
            )
            if status != "accepted":
                audit[status] += 1
                continue
            contributions = _candidate_contributions(context, candidate)
            candidate_intensity = float(sum(value[2] for value in contributions))
            if not math.isfinite(candidate_intensity) or candidate_intensity <= 0:
                audit["nonpositive_candidate_intensity"] += 1
                continue
            audit["basic_gates_accepted"] += 1
            key = _candidate_key(candidate, context["faims_cv"])
            current = survivors.get(key)
            item = (
                _quality_key(candidate), context_index, candidate,
                mono_points, secondary_points, cosine, contributions,
            )
            if current is None or item[0] < current[0]:
                if current is not None:
                    audit["weak_deduplicated"] += 1
                survivors[key] = item
            else:
                audit["weak_deduplicated"] += 1

    accepted_by_context = defaultdict(list)
    for item in sorted(survivors.values(), key=lambda value: (
        value[1], value[0]
    )):
        _quality, context_index, candidate, mono_points, secondary_points, cosine, contributions = item
        context = contexts[context_index]
        if _strong_equivalent(strong_index, context, candidate):
            audit["strong_equivalent_rejected"] += 1
            continue
        footprint = residual_ledger.observed_point_footprint(contributions)
        if footprint.status != "accepted":
            audit["footprint_" + footprint.status] += 1
            continue
        overlap = residual_ledger.footprint_overlap(footprint)
        if overlap.fraction > max_strong_overlap + 1e-12:
            audit["strong_overlap_rejected"] += 1
            continue
        audit["ownership_gate_accepted"] += 1
        accepted_by_context[context_index].append((
            candidate, mono_points, secondary_points, cosine,
            float(overlap.fraction),
        ))

    rows = []
    temporary_id = -1
    for context_index, values in sorted(accepted_by_context.items()):
        context = contexts[context_index]
        candidates = []
        for candidate, _mono, _secondary, _cosine, _overlap in values:
            candidate["feature_idx"] = temporary_id
            temporary_id -= 1
            candidates.append(candidate)
        base_rows = utils.calc_peptide_features(
            context["hills"], candidates, args["nm"], context["faims_cv"],
            context["rt_by_local"], 0, args["iuse"],
            include_mono_hills=not args.get("no_mono_hills", False),
            quantification_args=args, spectra=context["spectra"],
        )
        for values_for_candidate, base in zip(values, base_rows):
            candidate, mono_points, secondary_points, cosine, overlap = values_for_candidate
            quant = _finite(base.get("area_sum"))
            if quant is None or quant <= 0:
                quant = _finite(base.get("intensitySum"))
            if quant is None or quant <= 0:
                audit["nonpositive_quantification"] += 1
                continue
            base["run_id"] = run_id
            base["quant_method"] = args.get("quant_method", "all")
            rows.append(_weak_row(
                base, candidate, quant, mono_points, secondary_points,
                cosine, overlap,
            ))
    audit["persisted_weak_candidates"] = len(rows)
    return rows, {
        **dict(sorted(audit.items())),
        "reject_source_counts": dict(sorted(source_counts.items())),
        "max_strong_overlap_fraction": max_strong_overlap,
    }


__all__ = [
    "DEFAULT_MAX_STRONG_OVERLAP",
    "append_rejected_candidate",
    "publish_detector_outcomes",
    "remember_reject_snapshot",
    "weak_feature_rows_from_contexts",
]
