"""Identification-aware direct assays and bounded local feature extraction."""

from __future__ import annotations

from collections import Counter
from bisect import bisect_left, bisect_right
import math
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from .hybrid_assays import (
    DirectAssay,
    DirectRunCalibration,
    LocalFeatureCandidate,
)
from .hybrid_constants import *
from .hybrid_local import (
    _candidate_segment_values,
    _faims_equal,
    _local_candidate_raw_points,
)
from .parallel import balanced_ranges, run_process_tasks
from .residual import ResidualMS1Ledger


def _final_strict_feature_row_task(context_segments, args):
    """Build one ordered group of independent strict output rows."""

    from . import utils

    result = []
    for context in context_segments:
        result.extend(utils.calc_peptide_features(
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
        ))
    return result


def _final_strict_feature_rows(strict_contexts, args):
    """Build strict rows only after hybrid targeted/conflict decisions finish."""

    contexts = tuple(context for context in strict_contexts if context["candidates"])
    total_candidates = sum(len(context["candidates"]) for context in contexts)
    workers = min(int(args.get("nprocs", 1)), total_candidates)
    if workers <= 1:
        return _final_strict_feature_row_task(contexts, args)

    # Divide the global context/candidate order into at most nprocs tasks.  A
    # task may contain pieces of neighbouring contexts, but its row order is
    # exactly the legacy context-major order and is merged in task order.
    offsets = []
    offset = 0
    for context in contexts:
        offsets.append((offset, offset + len(context["candidates"]), context))
        offset += len(context["candidates"])
    tasks = []
    for start, end in balanced_ranges(total_candidates, workers):
        segments = []
        for context_start, context_end, context in offsets:
            overlap_start = max(start, context_start)
            overlap_end = min(end, context_end)
            if overlap_start >= overlap_end:
                continue
            local_start = overlap_start - context_start
            local_end = overlap_end - context_start
            segments.append({
                **context,
                "candidates": context["candidates"][local_start:local_end],
            })
        tasks.append((tuple(segments), args))
    rows = []
    for batch in run_process_tasks(_final_strict_feature_row_task, tasks):
        rows.extend(batch)
    return rows


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


_STRICT_FOOTPRINT_META_DTYPE = np.dtype([
    ("feature_id", np.int64),
    ("requested", np.float64),
    ("start", np.int64),
    ("end", np.int64),
    ("valid", np.bool_),
    ("status", "S48"),
])


def _strict_footprint_artifact_paths(directory, worker_id):
    prefix = Path(directory) / ("worker-%03d" % int(worker_id))
    return {
        "meta": str(prefix.with_suffix(".meta.npy")),
        "indices": str(prefix.with_suffix(".indices.npy")),
        "intensities": str(prefix.with_suffix(".intensities.npy")),
    }


def _build_strict_observed_footprint_artifact(store, records, directory, worker_id):
    """Map immutable strict points and persist a compact worker artifact."""

    mapper = ResidualMS1Ledger(store)
    metadata = np.empty(len(records), dtype=_STRICT_FOOTPRINT_META_DTYPE)
    point_indices = []
    intensities = []
    for position, record in enumerate(records):
        feature_id = int(record["feature_id"])
        try:
            footprint = mapper.observed_point_footprint(
                _strict_feature_observed_contributions(record)
            )
        except (IndexError, KeyError, TypeError, ValueError):
            footprint = None
        start = len(point_indices)
        if footprint is None:
            valid = False
            status = b""
            requested = 0.0
        else:
            valid = True
            status = footprint.status.encode("ascii")
            requested = float(footprint.requested_intensity)
            if footprint.status == "accepted":
                point_indices.extend(
                    int(value.point_index) for value in footprint.allocations
                )
                intensities.extend(
                    float(value.intensity) for value in footprint.allocations
                )
        metadata[position] = (
            feature_id,
            requested,
            start,
            len(point_indices),
            valid,
            status,
        )
    paths = _strict_footprint_artifact_paths(directory, worker_id)
    np.save(paths["meta"], metadata, allow_pickle=False)
    np.save(
        paths["indices"], np.asarray(point_indices, dtype=np.int64),
        allow_pickle=False,
    )
    np.save(
        paths["intensities"], np.asarray(intensities, dtype=np.float64),
        allow_pickle=False,
    )
    return paths


def _remove_strict_footprint_artifacts(directory):
    path = Path(directory)
    for artifact in path.iterdir():
        artifact.unlink()
    path.rmdir()


def _allocate_strict_feature_population(ledger, strict_records, workers=1):
    """Register accepted strict ownership before targeted residual searches."""

    statuses = Counter()
    failed_feature_ids = []
    records = tuple(sorted(strict_records, key=lambda record: int(record["feature_id"])))
    # Initial strict features originate from a conflict-free detector population,
    # so their raw-point mappings can be prepared concurrently. Residual strict
    # candidates may overlap already claimed points and retain the serial path.
    if int(workers) > 1 and len(records) > 1 and not ledger.allocation_count:
        ranges = balanced_ranges(len(records), int(workers))
        artifact_directory = tempfile.mkdtemp(prefix=".strict-footprints-")
        try:
            artifacts = run_process_tasks(
                _build_strict_observed_footprint_artifact,
                [
                    (ledger.store, records[start:end], artifact_directory, worker_id)
                    for worker_id, (start, end) in enumerate(ranges)
                ],
            )
            for (start, end), paths in zip(ranges, artifacts):
                metadata = np.load(paths["meta"], mmap_mode="r")
                point_indices = np.load(paths["indices"], mmap_mode="r")
                intensities = np.load(paths["intensities"], mmap_mode="r")
                for record, entry in zip(records[start:end], metadata):
                    feature_id = int(record["feature_id"])
                    if feature_id != int(entry["feature_id"]):
                        raise ValueError("strict footprint feature order mismatch")
                    try:
                        if bool(entry["valid"]):
                            status = entry["status"].decode("ascii")
                            result = ledger.commit_observed_point_arrays(
                                ("strict", feature_id),
                                status,
                                float(entry["requested"]),
                                point_indices[int(entry["start"]):int(entry["end"])],
                                intensities[int(entry["start"]):int(entry["end"])],
                            )
                            if result.status == "raw_point_overallocation":
                                result = ledger.allocate_observed_points(
                                    ("strict", feature_id),
                                    _strict_feature_observed_contributions(record),
                                )
                        else:
                            result = ledger.allocate_observed_points(
                                ("strict", feature_id),
                                _strict_feature_observed_contributions(record),
                            )
                        status = result.status
                        if status == "accepted":
                            ledger.seal_allocation(("strict", feature_id))
                    except (IndexError, KeyError, TypeError, ValueError):
                        status = "invalid_strict_provenance"
                    statuses[status] += 1
                    if status != "accepted":
                        failed_feature_ids.append(feature_id)
        finally:
            _remove_strict_footprint_artifacts(artifact_directory)
    else:
        for record in records:
            feature_id = int(record["feature_id"])
            try:
                result = ledger.allocate_observed_points(
                    ("strict", feature_id),
                    _strict_feature_observed_contributions(record),
                )
                status = result.status
                if status == "accepted":
                    ledger.seal_allocation(("strict", feature_id))
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
        "external_support_count": 0,
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
        "scanStart": int(mono_trace.scan_number[start]),
        "scanApex": candidate.scan_apex,
        "scanEnd": int(mono_trace.scan_number[end - 1]),
        "isoerror": candidate.mono_mz_error_ppm,
        "isoerror2": None,
        "feature_idx": feature_id,
        "area_sum": None,
    }



__all__ = [name for name in globals() if not name.startswith("__")]
