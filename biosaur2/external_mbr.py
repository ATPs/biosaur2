"""Feature-only match-between-runs for Project external weak candidates.

This module deliberately has no RawMS1Store dependency.  Local detection
publishes compact strong/weak sidecars; the Project stage only aligns and
matches those measured features.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
import csv
import json
import logging
import math
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from .alignment import AlignmentAnchor, RTAlignmentModel, choose_reference_run, fit_rt_alignment
from .confidence import TargetDecoyCompetition, deterministic_decoy_shift, target_decoy_q_values
from .external_alignment import AlignmentForest, MAX_REFERENCE_CANDIDATES, ReferenceStarAlignment, alignment_group_for_run, faims_key
from .output import _temporary_neighbor, publish_staged_files
from .raw_ms1 import source_fingerprint
from .schema import compact_schemas


SIDECAR_VERSION = "feature-mbr-v2"
MAX_SUPPORTS = 4
logger = logging.getLogger(__name__)
EVIDENCE_FIELDS = (
    "target_run", "weak_candidate_id", "feature_id", "source_run",
    "source_feature_id", "support_rank", "support_score", "mz_error_ppm",
    "rt_error_sec", "predicted_rt_sec", "target_score", "decoy_score",
    "competition_winner", "acceptance_q_value", "status",
    "alignment_method", "alignment_anchor_count", "alignment_residual_mad_sec",
)


def _read_table(path, table_name):
    source = Path(path)
    if source.suffix.lower() == ".duckdb":
        import duckdb
        with duckdb.connect(str(source), read_only=True) as connection:
            return connection.execute('SELECT * FROM "%s"' % table_name).fetch_arrow_table()
    if source.suffix.lower() == ".tsv":
        schema = compact_schemas()["hybrid_features"]
        with source.open(encoding="utf-8", newline="") as handle:
            return pa.Table.from_pylist(list(csv.DictReader(handle, delimiter="\t")), schema=schema)
    return pq.read_table(source)


def _write_like_existing(table, final_path):
    destination = Path(final_path)
    temporary = _temporary_neighbor(destination)
    if destination.suffix.lower() == ".tsv":
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=table.schema.names, delimiter="\t")
            writer.writeheader()
            writer.writerows(table.to_pylist())
    else:
        pq.write_table(table, temporary, compression="zstd")
    return temporary


def _alignment_model_rows(models):
    rows = []
    for (source, target), (declared, model) in sorted(models.items()):
        component = models.component_by_run.get(source)
        rows.append({
            "alignment_group": component or declared,
            "reference_run": models.reference_runs.get(component),
            "source_run": source, "target_run": target, "method": model.method,
            "anchor_count": model.anchor_count, "inlier_count": model.inlier_count,
            "slope": model.slope, "intercept": model.intercept,
            "residual_mad_sec": model.residual_mad_sec, "status": model.status,
            "validation_anchor_count": model.validation_anchor_count,
            "validation_median_bias_sec": model.validation_median_bias_sec,
            "validation_mad_sec": model.validation_mad_sec,
            "validation_q90_abs_error_sec": model.validation_q90_abs_error_sec,
            "x_knots_json": json.dumps(model.x_knots), "y_knots_json": json.dumps(model.y_knots),
        })
    return rows


@dataclass(frozen=True)
class FeatureRecord:
    run_id: str
    feature_id: int
    mz: float
    charge: int
    faims_cv: float | None
    rt_start_sec: float
    rt_apex_sec: float
    rt_end_sec: float
    quant_value: float
    quality: float


@dataclass(frozen=True)
class WeakRecord:
    candidate_id: int
    feature: FeatureRecord
    row_json: str
    mono_points: int
    secondary_points: int
    isotope_cosine: float
    reject_source: str
    strong_overlap_fraction: float
    local_gate_status: str


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _faims_equal(left, right):
    if left is None or right is None:
        return left is right
    left = _finite(left)
    right = _finite(right)
    return left is not None and right is not None and math.isclose(left, right, abs_tol=1e-6)


def _signature(mzml_path):
    return json.dumps(
        {"version": SIDECAR_VERSION, "mzml": source_fingerprint(mzml_path)},
        sort_keys=True, separators=(",", ":"),
    )


def _strong_schema():
    return pa.schema([
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("feature_id", pa.int64(), nullable=False),
        pa.field("mz", pa.float64(), nullable=False),
        pa.field("charge", pa.int16(), nullable=False),
        pa.field("faims_cv", pa.float64()),
        pa.field("rt_start_sec", pa.float64(), nullable=False),
        pa.field("rt_apex_sec", pa.float64(), nullable=False),
        pa.field("rt_end_sec", pa.float64(), nullable=False),
        pa.field("quant_value", pa.float64(), nullable=False),
        pa.field("quality", pa.float64(), nullable=False),
    ])


def _weak_schema():
    return _strong_schema().append(pa.field("candidate_id", pa.int64(), nullable=False)).append(
        pa.field("row_json", pa.string(), nullable=False)
    ).append(pa.field("mono_points", pa.int32(), nullable=False)).append(
        pa.field("secondary_points", pa.int32(), nullable=False)
    ).append(pa.field("isotope_cosine", pa.float64(), nullable=False)).append(
        pa.field("reject_source", pa.string(), nullable=False)
    ).append(
        pa.field("strong_overlap_fraction", pa.float64(), nullable=False)
    ).append(pa.field("local_gate_status", pa.string(), nullable=False))


def _record_from_row(run_id, row):
    mz = _finite(row.get("mz"))
    quant = _finite(row.get("quant_value"))
    apex = _finite(row.get("rt_apex_sec", row.get("rtApex")))
    start = _finite(row.get("rt_start_sec", row.get("rtStart")))
    end = _finite(row.get("rt_end_sec", row.get("rtEnd")))
    quality = _finite(row.get("quality", row.get("feature_quality_score", row.get("isotope_cosine"))))
    try:
        charge = int(row.get("charge"))
        feature_id = int(row.get("feature_id", row.get("feature_idx")))
    except (TypeError, ValueError):
        return None
    if (
        mz is None or quant is None or quant <= 0 or apex is None or start is None
        or end is None or quality is None or charge < 1 or feature_id < 1
    ):
        return None
    faims = _finite(row.get("faims_cv", row.get("FAIMS")))
    return FeatureRecord(run_id, feature_id, mz, charge, faims, start, apex, end, quant, quality)


def _atomic_parquet(path, table):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_neighbor(destination)
    try:
        pq.write_table(table, temporary, compression="zstd")
        publish_staged_files([(temporary, destination)])
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_feature_sidecars(mzml_path, paths, strong_rows, weak_rows):
    """Publish source-provenanced feature-only sidecars after local detection."""

    strong_path = paths.get("external_strong_features")
    weak_path = paths.get("external_weak_candidates")
    if not strong_path or not weak_path:
        return
    signature = _signature(mzml_path).encode()
    strong_table = pa.Table.from_pylist(strong_rows, schema=_strong_schema()).replace_schema_metadata(
        {b"biosaur2_external_feature_mbr_signature": signature}
    )
    normalized_weak = [
        {
            **row,
            "reject_source": row.get("reject_source", "unknown_reject"),
            "strong_overlap_fraction": float(
                row.get("strong_overlap_fraction", 0.0)
            ),
            "local_gate_status": row.get("local_gate_status", "accepted"),
        }
        for row in weak_rows
    ]
    weak_table = pa.Table.from_pylist(normalized_weak, schema=_weak_schema()).replace_schema_metadata(
        {b"biosaur2_external_feature_mbr_signature": signature}
    )
    _atomic_parquet(strong_path, strong_table)
    _atomic_parquet(weak_path, weak_table)


def read_feature_sidecars(run, paths):
    expected = _signature(run.mzml_path).encode()
    try:
        strong_table = pq.read_table(paths["external_strong_features"])
        weak_table = pq.read_table(paths["external_weak_candidates"])
    except (KeyError, OSError, pa.ArrowException):
        return None
    for table in (strong_table, weak_table):
        if (table.schema.metadata or {}).get(b"biosaur2_external_feature_mbr_signature") != expected:
            return None
    strong = tuple(
        record for row in strong_table.to_pylist()
        if (record := _record_from_row(run.run_id, row)) is not None
    )
    weak = []
    for row in weak_table.to_pylist():
        feature = _record_from_row(run.run_id, row)
        if feature is None:
            continue
        weak.append(WeakRecord(
            int(row["candidate_id"]), feature, row["row_json"], int(row["mono_points"]),
            int(row["secondary_points"]), float(row["isotope_cosine"]),
            row["reject_source"], float(row["strong_overlap_fraction"]),
            row["local_gate_status"],
        ))
    return strong, tuple(weak)


def sidecar_rows(run_id, feature_rows, quant_rows, weak_rows):
    """Build compact sidecar rows from local final and rejected candidates."""

    quant_by_id = {int(row["feature_id"]): row for row in quant_rows if row.get("feature_id") is not None}
    strong = []
    for row in feature_rows:
        feature_id = row.get("feature_idx")
        if feature_id is None:
            continue
        merged = {**row, **quant_by_id.get(int(feature_id), {})}
        record = _record_from_row(run_id, merged)
        if record is not None:
            strong.append(record.__dict__)
    weak = []
    for candidate_id, row in enumerate(sorted(weak_rows, key=lambda item: (
        item.get("charge", 0), faims_key(item.get("FAIMS")), item.get("mz", 0.0),
        item.get("rtApex", 0.0), item.get("scanApex", -1),
    )), start=1):
        record = _record_from_row(
            run_id, {**row, "feature_id": candidate_id}
        )
        if record is None:
            continue
        weak.append({
            **record.__dict__, "candidate_id": candidate_id,
            "row_json": json.dumps(
                row, sort_keys=True, separators=(",", ":"),
                default=_json_default,
            ),
            "mono_points": int(row.get("points_across_peak", row.get("nScans", 0))),
            "secondary_points": int(row.get("external_secondary_points", 2)),
            "isotope_cosine": float(row.get("isotope_cosine", row.get("feature_quality_score", 0.0))),
            "reject_source": row.get("external_reject_source", "unknown_reject"),
            "strong_overlap_fraction": float(row.get("external_strong_overlap_fraction", 0.0)),
            "local_gate_status": row.get("external_local_gate_status", "accepted"),
        })
    return strong, weak


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError("unsupported sidecar JSON value: %s" % type(value).__name__)


def _group_key(record):
    return record.charge, faims_key(record.faims_cv)


def _sorted_index(records):
    result = {}
    for key, values in defaultdict(list).items():
        result[key] = values
    grouped = defaultdict(list)
    for record in records:
        grouped[_group_key(record)].append(record)
    return {
        key: (tuple(item.mz for item in sorted(values, key=lambda item: (item.mz, item.feature_id))),
              tuple(sorted(values, key=lambda item: (item.mz, item.feature_id))))
        for key, values in grouped.items()
    }


def _nearest(index, record, ppm):
    values = index.get(_group_key(record))
    if not values:
        return None
    mzs, rows = values
    delta = record.mz * ppm * 1e-6
    start = bisect_left(mzs, record.mz - delta)
    end = bisect_right(mzs, record.mz + delta)
    if start == end:
        return None
    return min(rows[start:end], key=lambda item: (abs(item.mz - record.mz), item.feature_id))


def _mutual_pairs(source, target, ppm):
    target_index = _sorted_index(target)
    source_index = _sorted_index(source)
    pairs = []
    for item in source:
        other = _nearest(target_index, item, ppm)
        if other is None:
            continue
        if _nearest(source_index, other, ppm) != item:
            continue
        pairs.append((item, other))
    return pairs


def _longest_monotonic_anchors(pairs):
    ordered = sorted(
        pairs,
        key=lambda pair: (
            pair[0].rt_apex_sec, -pair[1].rt_apex_sec,
            pair[0].feature_id, pair[1].feature_id,
        ),
    )
    tails, tail_indices, predecessors = [], [], [-1] * len(ordered)
    for index, (left, right) in enumerate(ordered):
        position = bisect_left(tails, right.rt_apex_sec)
        if position:
            predecessors[index] = tail_indices[position - 1]
        if position == len(tails):
            tails.append(right.rt_apex_sec)
            tail_indices.append(index)
        else:
            current = ordered[tail_indices[position]]
            replacement_key = (
                right.rt_apex_sec, left.rt_apex_sec,
                left.feature_id, right.feature_id,
            )
            current_key = (
                current[1].rt_apex_sec, current[0].rt_apex_sec,
                current[0].feature_id, current[1].feature_id,
            )
            if replacement_key < current_key:
                tails[position] = right.rt_apex_sec
                tail_indices[position] = index
    if not tail_indices:
        return []
    chain = []
    cursor = tail_indices[-1]
    while cursor >= 0:
        chain.append(ordered[cursor])
        cursor = predecessors[cursor]
    chain.reverse()
    return [
        AlignmentAnchor(
            "%d:%d" % (left.feature_id, right.feature_id),
            left.rt_apex_sec, right.rt_apex_sec,
            max(1e-6, min(left.quality, right.quality)),
        )
        for left, right in chain
    ]


def _downsample_fit_anchors(anchors, max_anchors):
    if len(anchors) <= max_anchors:
        return list(anchors)
    selected = []
    for index in range(max_anchors):
        start = index * len(anchors) // max_anchors
        end = (index + 1) * len(anchors) // max_anchors
        selected.append(min(
            anchors[start:end], key=lambda anchor: (
                -anchor.quality, anchor.source_rt_sec,
                anchor.target_rt_sec, anchor.ion_key,
            )
        ))
    return sorted(selected, key=lambda anchor: (
        anchor.source_rt_sec, anchor.target_rt_sec, anchor.ion_key
    ))


def _validated_alignment(
    source_run, target_run, anchors, *, min_anchors, max_mad,
    max_anchors, q90_limit,
):
    validation = [anchor for index, anchor in enumerate(anchors) if index % 5 == 2]
    fit = [anchor for index, anchor in enumerate(anchors) if index % 5 != 2]
    minimum_validation = max(1, (int(min_anchors) + 3) // 4)
    if len(fit) < min_anchors:
        return _rejected(source_run, target_run, len(fit), "insufficient_fit_anchors")
    if len(validation) < minimum_validation:
        return _rejected(
            source_run, target_run, len(fit),
            "insufficient_validation_anchors",
        )
    model = fit_rt_alignment(
        source_run, target_run,
        _downsample_fit_anchors(fit, int(max_anchors)),
    )
    if model.status != "accepted":
        return model
    residuals = np.asarray([
        anchor.target_rt_sec - model.predict(anchor.source_rt_sec)
        for anchor in validation
    ], dtype=np.float64)
    bias = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - bias)))
    q90 = float(np.quantile(np.abs(residuals), 0.9))
    status = "accepted"
    if abs(bias) > max_mad:
        status = "validation_bias_exceeds_limit"
    elif mad > max_mad:
        status = "validation_mad_exceeds_limit"
    elif q90 > q90_limit:
        status = "validation_q90_exceeds_rt_window"
    return replace(
        model,
        residual_mad_sec=mad,
        status=status,
        validation_anchor_count=len(validation),
        validation_median_bias_sec=bias,
        validation_mad_sec=mad,
        validation_q90_abs_error_sec=q90,
    )


def _rejected(source, target, count, status):
    return RTAlignmentModel(source, target, "none", count, 0, (), (), 1.0, 0.0, None, status)


def build_feature_alignment_models(
    runs, strong_by_run, *, ppm, min_anchors, max_mad, max_anchors,
    validation_q90_limit=120.0,
):
    """Build a bounded reference forest from mutual strong-feature anchors."""

    grouped = defaultdict(list)
    for run in runs:
        grouped[alignment_group_for_run(run)].append(run.run_id)
    models, components, references, parents = {}, {}, {}, {}
    for declared, ids in sorted(grouped.items()):
        ids = sorted(ids)
        counts = {
            run_id: len(strong_by_run.get(run_id, ())) for run_id in ids
        }
        ranked = sorted(ids, key=lambda run_id: (-counts[run_id], run_id))
        disjoint = {run_id: run_id for run_id in ids}
        rank = {run_id: 0 for run_id in ids}
        adjacency = defaultdict(list)

        def find(run_id):
            while disjoint[run_id] != run_id:
                disjoint[run_id] = disjoint[disjoint[run_id]]
                run_id = disjoint[run_id]
            return run_id

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return
            if rank[left_root] < rank[right_root]:
                left_root, right_root = right_root, left_root
            disjoint[right_root] = left_root
            if rank[left_root] == rank[right_root]:
                rank[left_root] += 1

        for run_id in ranked[1:]:
            candidates = [
                value for value in ranked if value != run_id
            ][:MAX_REFERENCE_CANDIDATES]
            for candidate_reference in candidates:
                if find(run_id) == find(candidate_reference):
                    break
                existing = models.get((run_id, candidate_reference))
                if existing is None:
                    forward_anchors = _longest_monotonic_anchors(_mutual_pairs(
                        strong_by_run.get(run_id, ()),
                        strong_by_run.get(candidate_reference, ()), ppm,
                    ))
                    reverse_anchors = _longest_monotonic_anchors(_mutual_pairs(
                        strong_by_run.get(candidate_reference, ()),
                        strong_by_run.get(run_id, ()), ppm,
                    ))
                    forward = _validated_alignment(
                        run_id, candidate_reference, forward_anchors,
                        min_anchors=min_anchors, max_mad=max_mad,
                        max_anchors=max_anchors,
                        q90_limit=validation_q90_limit,
                    )
                    reverse = _validated_alignment(
                        candidate_reference, run_id, reverse_anchors,
                        min_anchors=min_anchors, max_mad=max_mad,
                        max_anchors=max_anchors,
                        q90_limit=validation_q90_limit,
                    )
                    models[(run_id, candidate_reference)] = (
                        declared, forward,
                    )
                    models[(candidate_reference, run_id)] = (
                        declared, reverse,
                    )
                else:
                    forward = existing[1]
                    reverse = models[(candidate_reference, run_id)][1]
                if (
                    forward.status == "accepted"
                    and reverse.status == "accepted"
                ):
                    union(run_id, candidate_reference)
                    adjacency[run_id].append(candidate_reference)
                    adjacency[candidate_reference].append(run_id)
                    break

        grouped_components = defaultdict(list)
        for run_id in ids:
            grouped_components[find(run_id)].append(run_id)
        for members in grouped_components.values():
            reference = choose_reference_run({
                run_id: counts[run_id] for run_id in members
            })
            component = "%s|component=%s" % (declared, reference)
            references[component] = reference
            queue = deque([reference])
            components[reference] = component
            parents[reference] = None
            while queue:
                parent = queue.popleft()
                for child in sorted(adjacency[parent]):
                    if child in components:
                        continue
                    components[child] = component
                    parents[child] = parent
                    queue.append(child)
    return AlignmentForest(models, components, references, parents)


def _support_score(mz_error_ppm, rt_error_sec, ppm, rt_tolerance):
    return 1.0 - math.sqrt((mz_error_ppm / ppm) ** 2 + (rt_error_sec / rt_tolerance) ** 2) / math.sqrt(2.0)


def _supports_by_run(
    candidate, index, alignments, *, ppm, rt_tolerance, shifted=False
):
    candidate_mz = candidate.feature.mz
    if shifted:
        candidate_mz += deterministic_decoy_shift(candidate.feature.run_id, str(candidate.candidate_id)) / candidate.feature.charge
    probe = FeatureRecord(candidate.feature.run_id, candidate.feature.feature_id, candidate_mz, candidate.feature.charge, candidate.feature.faims_cv, candidate.feature.rt_start_sec, candidate.feature.rt_apex_sec, candidate.feature.rt_end_sec, candidate.feature.quant_value, candidate.feature.quality)
    values = index.get(_group_key(probe))
    if not values:
        return []
    mzs, rows = values
    delta = probe.mz * ppm * 1e-6
    start = bisect_left(mzs, probe.mz - delta)
    end = bisect_right(mzs, probe.mz + delta)
    options = {}
    for source in rows[start:end]:
        alignment = alignments.get(source.run_id)
        if alignment is None or alignment.status != "accepted":
            continue
        predicted = alignment.predict(source.rt_apex_sec)
        rt_error = abs(candidate.feature.rt_apex_sec - predicted)
        ppm_error = abs((source.mz - probe.mz) / probe.mz * 1e6)
        if rt_error > rt_tolerance:
            continue
        score = _support_score(ppm_error, rt_error, ppm, rt_tolerance)
        support = (
            score, ppm_error, rt_error, -source.quality,
            source.feature_id, source, predicted,
        )
        current = options.get(source.run_id)
        if current is None or (
            -support[0], support[1], support[2], support[3], support[4]
        ) < (
            -current[0], current[1], current[2], current[3], current[4]
        ):
            options[source.run_id] = support
    return [
        (support, source_run, alignments[source_run])
        for source_run, support in options.items()
    ]


def _aggregate_support_score(supports):
    """Combine up to four distinct-run supports for symmetric competition."""

    if not supports:
        return None
    return float(sum(item[0][0] for item in supports))


def _candidate_outcomes(runs, strong_by_run, weak_by_run, models, options):
    by_component = defaultdict(list)
    for run in runs:
        by_component[models.component_by_run[run.run_id]].append(run.run_id)
    grouped = defaultdict(list)
    for component, run_ids in by_component.items():
        component_index = _sorted_index([
            record
            for run_id in run_ids
            for record in strong_by_run.get(run_id, ())
        ])
        for target_id in sorted(run_ids):
            alignments = {}
            for source_id in sorted(run_ids):
                if source_id == target_id:
                    continue
                alignment = ReferenceStarAlignment(
                    source_id, target_id,
                    models.path_to_reference(source_id),
                    models.reference_to_run_path(target_id),
                )
                if alignment.status == "accepted":
                    alignments[source_id] = alignment
            for candidate in weak_by_run.get(target_id, ()):
                target_supports = _supports_by_run(
                    candidate, component_index, alignments,
                    ppm=options["ppm"],
                    rt_tolerance=options["rt_tolerance_sec"],
                )
                decoy_supports = _supports_by_run(
                    candidate, component_index, alignments,
                    ppm=options["ppm"],
                    rt_tolerance=options["rt_tolerance_sec"], shifted=True,
                )
                target_supports.sort(key=lambda item: (-item[0][0], item[0][1], item[0][2], item[0][3], item[0][4], item[1]))
                decoy_supports.sort(key=lambda item: (-item[0][0], item[0][1], item[0][2], item[0][3], item[0][4], item[1]))
                grouped[component].append({
                    "target_run": target_id, "candidate": candidate, "component": component,
                    "targets": target_supports[:MAX_SUPPORTS], "decoys": decoy_supports[:MAX_SUPPORTS],
                    "accepted_alignment_count": len(alignments),
                })
    for component, outcomes in grouped.items():
        competitions = []
        for item in outcomes:
            seed = "%s:%d" % (item["target_run"], item["candidate"].candidate_id)
            item["seed"] = seed
            competitions.append(TargetDecoyCompetition(
                seed,
                _aggregate_support_score(item["targets"]),
                _aggregate_support_score(item["decoys"]),
            ))
        results = {item.seed_id: item for item in target_decoy_q_values(competitions)}
        for item in outcomes:
            item["competition"] = results[item["seed"]]
    return grouped


def _outcome_status(item, q_value_max):
    result = item["competition"]
    if not item.get("accepted_alignment_count"):
        return "no_accepted_alignment"
    if result.winner == "none":
        return "no_external_support"
    if result.winner == "decoy":
        return "decoy_winner"
    if not item["targets"]:
        return "no_external_support"
    if result.q_value > q_value_max:
        return "target_q_value_above_limit"
    return "accepted_matched_weak_feature"


def _evidence_rows(outcomes, q_value_max):
    rows = defaultdict(list)
    for values in outcomes.values():
        for item in values:
            result = item["competition"]
            status = _outcome_status(item, q_value_max)
            accepted = status == "accepted_matched_weak_feature"
            supports = item["targets"] if accepted else item["targets"][:1]
            if not supports:
                supports = [(None, None, None)]
            for rank, (support, source_id, alignment) in enumerate(supports, start=1):
                source = None if support is None else support[5]
                rows[item["target_run"]].append({
                    "target_run": item["target_run"], "weak_candidate_id": item["candidate"].candidate_id,
                    "feature_id": None, "source_run": source_id, "source_feature_id": None if source is None else source.feature_id,
                    "support_rank": rank if accepted else None, "support_score": None if support is None else support[0],
                    "mz_error_ppm": None if support is None else support[1], "rt_error_sec": None if support is None else support[2],
                    "predicted_rt_sec": None if support is None else support[6],
                    "target_score": result.target_score, "decoy_score": result.decoy_score,
                    "competition_winner": result.winner, "acceptance_q_value": result.q_value,
                    "status": status,
                    "alignment_method": None if alignment is None else alignment.method,
                    "alignment_anchor_count": None if alignment is None else alignment.anchor_count,
                    "alignment_residual_mad_sec": None if alignment is None else alignment.residual_mad_sec,
                })
    return rows


def _write_evidence(path, rows):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows) if rows else pa.table({name: pa.array([], type=pa.string()) for name in EVIDENCE_FIELDS})
    if destination.suffix == ".parquet":
        _atomic_parquet(destination, table)
        return
    if destination.suffix == ".tsv":
        temporary = _temporary_neighbor(destination)
        fields = tuple(table.schema.names)
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        publish_staged_files([(temporary, destination)])
        return
    import duckdb
    temporary = _temporary_neighbor(destination)
    shutil.copy2(destination, temporary)
    with duckdb.connect(str(temporary)) as connection:
        connection.register("_evidence", table)
        connection.execute("CREATE OR REPLACE TABLE external_id_evidence AS SELECT * FROM _evidence")
        connection.unregister("_evidence")
    publish_staged_files([(temporary, destination)])


def _publish_features(path, accepted):
    table = _read_table(path, "features")
    original = table.to_pylist()
    rows = list(original)
    rows = [row for row in rows if row.get("feature_origin") != "aligned_external_weak"]
    next_id = 1 + max((int(row["feature_idx"]) for row in rows if row.get("feature_idx") is not None), default=0)
    for item in sorted(accepted, key=lambda value: value["candidate"].candidate_id):
        row = json.loads(item["candidate"].row_json)
        row.update({
            "feature_idx": next_id, "feature_id": next_id, "feature_origin": "aligned_external_weak",
            "confidence_tier": "external_id_weak", "extraction_q_value": item["competition"].q_value,
            "supporting_psm_count": 0, "supporting_ms2_count": 0,
            "external_support_count": len(item["targets"]), "ms2_events": [],
        })
        item["feature_id"] = next_id
        next_id += 1
        rows.append(_coerce_feature_row(row, table.schema))
    if rows != original or accepted:
        schema = table.schema
        combined = pa.Table.from_pylist(rows, schema=schema)
        temporary = _write_like_existing(combined, path)
        publish_staged_files([(temporary, Path(path))])
    return len(accepted)


def _coerce_feature_row(row, schema):
    """Read numeric strings written by the initial feature-mbr-v2 encoder."""

    normalized = dict(row)
    for field in schema:
        value = normalized.get(field.name)
        if not isinstance(value, str):
            continue
        if pa.types.is_floating(field.type):
            normalized[field.name] = float(value)
        elif pa.types.is_integer(field.type):
            normalized[field.name] = int(value)
    return normalized


def run_feature_mbr_stage(runs, results, options):
    """Run complete Project matching in memory without raw MS1 access."""

    successful = [run for index, run in enumerate(runs) if results[index]["status"] in {"success", "skipped_resume"}]
    result_by_id = {runs[index].run_id: results[index] for index in range(len(runs))}
    sidecars = {run.run_id: read_feature_sidecars(run, result_by_id[run.run_id]["paths"]) for run in successful}
    missing = [run_id for run_id, value in sidecars.items() if value is None]
    if missing:
        raise RuntimeError("feature-MBR sidecars are missing or stale: " + ", ".join(sorted(missing)))
    strong = {run_id: value[0] for run_id, value in sidecars.items()}
    weak = {run_id: value[1] for run_id, value in sidecars.items()}
    models = build_feature_alignment_models(
        successful, strong, ppm=float(options.get("external_ppm", 8.0)),
        min_anchors=int(options.get("external_alignment_min_anchors", 20)),
        max_mad=float(options.get("external_alignment_max_mad_sec", 30.0)),
        max_anchors=int(options.get("external_alignment_max_anchors", 256)),
        validation_q90_limit=float(
            options.get("external_rt_tolerance_sec", 120.0)
        ),
    )
    outcomes = _candidate_outcomes(successful, strong, weak, models, {
        "ppm": float(options.get("external_ppm", 8.0)),
        "rt_tolerance_sec": float(options.get("external_rt_tolerance_sec", 120.0)),
    })
    evidence = _evidence_rows(outcomes, float(options.get("external_q_value_max", 0.05)))
    q_value_max = float(options.get("external_q_value_max", 0.05))
    summaries = {}
    for run in successful:
        all_outcomes = [item for values in outcomes.values() for item in values if item["target_run"] == run.run_id]
        accepted = [
            item for item in all_outcomes
            if _outcome_status(item, q_value_max)
            == "accepted_matched_weak_feature"
        ]
        count = _publish_features(result_by_id[run.run_id]["paths"]["features"], accepted)
        for row in evidence.get(run.run_id, ()):
            if row["status"] == "accepted_matched_weak_feature":
                match = next(item for item in accepted if item["candidate"].candidate_id == row["weak_candidate_id"])
                row["feature_id"] = match["feature_id"]
        _write_evidence(result_by_id[run.run_id]["paths"]["external_evidence"], evidence.get(run.run_id, ()))
        statuses = Counter(
            _outcome_status(item, q_value_max) for item in all_outcomes
        )
        summaries[run.run_id] = {
            "run_id": run.run_id, "planned_assay_count": len(weak.get(run.run_id, ())),
            "evaluated_assay_count": len(all_outcomes), "new_external_feature_count": count,
            "new_strict_external_feature_count": 0, "new_weak_external_feature_count": count,
            "status_counts": dict(sorted(statuses.items())),
        }
        logger.info(
            "Feature-MBR run %s: weak=%d evaluated=%d rescued=%d statuses=%s",
            run.run_id, len(weak.get(run.run_id, ())), len(all_outcomes),
            count, dict(sorted(statuses.items())),
        )
    alignment_statuses = Counter(
        model.status for _declared, model in models.values()
    )
    project_statuses = Counter()
    for summary in summaries.values():
        project_statuses.update(summary["status_counts"])
    return {
        "summaries": summaries,
        "alignment_models": _alignment_model_rows(models),
        "reference_runs": models.reference_runs,
        "scheduler_summary": {
            "mode": "feature_mbr_no_raw_ms1",
            "component_strong_index_build_count": len({
                models.component_by_run[run.run_id] for run in successful
            }),
            "alignment_status_counts": dict(sorted(alignment_statuses.items())),
            "candidate_status_counts": dict(sorted(project_statuses.items())),
            "planned_weak_candidate_count": sum(
                summary["planned_assay_count"] for summary in summaries.values()
            ),
            "rescued_weak_feature_count": sum(
                summary["new_weak_external_feature_count"]
                for summary in summaries.values()
            ),
        },
    }
