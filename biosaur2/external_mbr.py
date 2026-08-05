"""Feature-only match-between-runs for Project external weak candidates.

This module deliberately has no RawMS1Store dependency.  Local detection
publishes compact strong/weak sidecars; the Project stage only aligns and
matches those measured features.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

from .alignment import AlignmentAnchor, RTAlignmentModel, choose_reference_run, fit_rt_alignment
from .confidence import TargetDecoyCompetition, deterministic_decoy_shift, target_decoy_q_values
from .external_alignment import AlignmentForest, ReferenceStarAlignment, alignment_group_for_run, faims_key
from .output import _temporary_neighbor, publish_staged_files
from .raw_ms1 import source_fingerprint
from .schema import compact_schemas


SIDECAR_VERSION = "feature-mbr-v1"
MAX_SUPPORTS = 4
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
    ).append(pa.field("isotope_cosine", pa.float64(), nullable=False))


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
    weak_table = pa.Table.from_pylist(weak_rows, schema=_weak_schema()).replace_schema_metadata(
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
            "row_json": json.dumps(row, sort_keys=True, separators=(",", ":"), default=str),
            "mono_points": int(row.get("points_across_peak", row.get("nScans", 0))),
            "secondary_points": int(row.get("external_secondary_points", 1)),
            "isotope_cosine": float(row.get("isotope_cosine", row.get("feature_quality_score", 0.0))),
        })
    return strong, weak


def weak_feature_rows_from_contexts(run_id, contexts, final_feature_rows, args):
    """Materialize rejected local envelopes as private weak candidate rows.

    The ordinary feature tables are not touched.  These rows are deliberately
    complete enough to be promoted later without reopening mzML.
    """

    from . import utils

    min_mono = int(args.get("external_weak_min_mono_points", 2))
    min_secondary = int(args.get("external_weak_min_secondary_points", 1))
    min_cosine = float(args.get("external_weak_min_isotope_cosine", 0.6))
    existing = [
        (int(row.get("charge", 0)), row.get("FAIMS"), float(row.get("mz", 0.0)),
         float(row.get("rtStart", 0.0)), float(row.get("rtEnd", 0.0)))
        for row in final_feature_rows
        if row.get("mz") is not None and row.get("rtStart") is not None and row.get("rtEnd") is not None
    ]
    rows = []
    temporary_id = -1
    for context in contexts:
        candidates = list(context["hills"].get("_external_weak_candidates", ()))
        if not candidates:
            continue
        for candidate in candidates:
            candidate["feature_idx"] = temporary_id
            temporary_id -= 1
        base_rows = utils.calc_peptide_features(
            context["hills"], candidates, args["nm"], context["faims_cv"],
            context["rt_by_local"], 0, args["iuse"],
            include_mono_hills=not args.get("no_mono_hills", False),
            quantification_args=args, spectra=context["spectra"],
        )
        for candidate, base in zip(candidates, base_rows):
            mono_points = int(base.get("nScans", 0))
            secondary_points = max(
                (len(context["hills"]["hills_scan_lists"][int(value["isotope_idx"])]) for value in candidate["isotopes"]),
                default=0,
            )
            cosine = _finite(candidate.get("cos_cor_isotopes"))
            quant = _finite(base.get("area_sum"))
            if quant is None or quant <= 0:
                quant = _finite(base.get("intensitySum"))
            if (
                mono_points < min_mono or secondary_points < min_secondary
                or cosine is None or cosine < min_cosine or quant is None or quant <= 0
            ):
                continue
            duplicate = False
            for charge, faims, mz, start, end in existing:
                if charge != int(base["charge"]) or not _faims_equal(faims, base.get("FAIMS")):
                    continue
                if abs(mz - float(base["mz"])) > float(base["mz"]) * 8e-6:
                    continue
                if max(start, float(base["rtStart"])) <= min(end, float(base["rtEnd"])):
                    duplicate = True
                    break
            if duplicate:
                continue
            rows.append({
                **base,
                "run_id": run_id,
                "feature_id": int(base["feature_idx"]),
                "feature_origin": "aligned_external_weak",
                "confidence_tier": "external_id_weak",
                "quant_value": quant,
                "quant_method": args.get("quant_method", "all"),
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
                "rt_start_sec": base.get("rtStart"),
                "rt_apex_sec": base.get("rtApex"),
                "rt_end_sec": base.get("rtEnd"),
                "isotope_cosine": cosine,
                "mass_error_ppm_median": base.get("isoerror"),
                "ms2_events": [],
                "external_secondary_points": secondary_points,
            })
    return rows


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


def _mutual_anchors(source, target, ppm, max_anchors):
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
    anchors = [AlignmentAnchor(
        "%d:%d" % (left.feature_id, right.feature_id), left.rt_apex_sec,
        right.rt_apex_sec, max(1e-6, min(left.quality, right.quality)),
    ) for left, right in pairs]
    anchors.sort(key=lambda item: (item.source_rt_sec, item.target_rt_sec, item.ion_key))
    if len(anchors) > max_anchors:
        anchors = [anchors[index * len(anchors) // max_anchors] for index in range(max_anchors)]
    return anchors


def _rejected(source, target, count, status):
    return RTAlignmentModel(source, target, "none", count, 0, (), (), 1.0, 0.0, None, status)


def build_feature_alignment_models(runs, strong_by_run, *, ppm, min_anchors, max_mad, max_anchors):
    """Build a bounded reference-star forest from mutual strong feature anchors."""

    grouped = defaultdict(list)
    for run in runs:
        grouped[alignment_group_for_run(run)].append(run.run_id)
    models, components, references, parents = {}, {}, {}, {}
    for declared, ids in sorted(grouped.items()):
        ids = sorted(ids)
        reference = choose_reference_run({run_id: len(strong_by_run.get(run_id, ())) for run_id in ids})
        component = "%s|component=%s" % (declared, reference)
        references[component] = reference
        components[reference] = component
        parents[reference] = None
        for run_id in ids:
            if run_id == reference:
                continue
            forward_anchors = _mutual_anchors(strong_by_run.get(run_id, ()), strong_by_run.get(reference, ()), ppm, max_anchors)
            reverse_anchors = _mutual_anchors(strong_by_run.get(reference, ()), strong_by_run.get(run_id, ()), ppm, max_anchors)
            forward = fit_rt_alignment(run_id, reference, forward_anchors) if len(forward_anchors) >= min_anchors else _rejected(run_id, reference, len(forward_anchors), "insufficient_anchors")
            reverse = fit_rt_alignment(reference, run_id, reverse_anchors) if len(reverse_anchors) >= min_anchors else _rejected(reference, run_id, len(reverse_anchors), "insufficient_anchors")
            if forward.residual_mad_sec is not None and forward.residual_mad_sec > max_mad:
                forward = RTAlignmentModel(**{**forward.__dict__, "status": "residual_mad_exceeds_limit"})
            if reverse.residual_mad_sec is not None and reverse.residual_mad_sec > max_mad:
                reverse = RTAlignmentModel(**{**reverse.__dict__, "status": "residual_mad_exceeds_limit"})
            models[(run_id, reference)] = (declared, forward)
            models[(reference, run_id)] = (declared, reverse)
            if forward.status == "accepted" and reverse.status == "accepted":
                components[run_id] = component
                parents[run_id] = reference
            else:
                isolated = "%s|component=%s" % (declared, run_id)
                components[run_id] = isolated
                references[isolated] = run_id
                parents[run_id] = None
    return AlignmentForest(models, components, references, parents)


def _support_score(mz_error_ppm, rt_error_sec, ppm, rt_tolerance):
    return 1.0 - math.sqrt((mz_error_ppm / ppm) ** 2 + (rt_error_sec / rt_tolerance) ** 2) / math.sqrt(2.0)


def _supports(candidate, source_run, source_records, alignment, *, ppm, rt_tolerance, shifted=False):
    candidate_mz = candidate.feature.mz
    if shifted:
        candidate_mz += deterministic_decoy_shift(candidate.feature.run_id, str(candidate.candidate_id)) / candidate.feature.charge
    probe = FeatureRecord(candidate.feature.run_id, candidate.feature.feature_id, candidate_mz, candidate.feature.charge, candidate.feature.faims_cv, candidate.feature.rt_start_sec, candidate.feature.rt_apex_sec, candidate.feature.rt_end_sec, candidate.feature.quant_value, candidate.feature.quality)
    index = _sorted_index(source_records)
    values = index.get(_group_key(probe))
    if not values:
        return None
    mzs, rows = values
    delta = probe.mz * ppm * 1e-6
    start = bisect_left(mzs, probe.mz - delta)
    end = bisect_right(mzs, probe.mz + delta)
    options = []
    for source in rows[start:end]:
        predicted = alignment.predict(source.rt_apex_sec)
        rt_error = abs(candidate.feature.rt_apex_sec - predicted)
        ppm_error = abs((source.mz - probe.mz) / probe.mz * 1e6)
        if rt_error > rt_tolerance:
            continue
        score = _support_score(ppm_error, rt_error, ppm, rt_tolerance)
        options.append((score, ppm_error, rt_error, -source.quality, source.feature_id, source, predicted))
    return min(options, key=lambda item: (-item[0], item[1], item[2], item[3], item[4])) if options else None


def _candidate_outcomes(runs, strong_by_run, weak_by_run, models, options):
    by_component = defaultdict(list)
    for run in runs:
        by_component[models.component_by_run[run.run_id]].append(run.run_id)
    grouped = defaultdict(list)
    for component, run_ids in by_component.items():
        for target_id in sorted(run_ids):
            for candidate in weak_by_run.get(target_id, ()):
                target_supports, decoy_supports = [], []
                for source_id in sorted(run_ids):
                    if source_id == target_id:
                        continue
                    alignment = ReferenceStarAlignment(
                        source_id, target_id, models.path_to_reference(source_id), models.reference_to_run_path(target_id)
                    )
                    if alignment.status != "accepted":
                        continue
                    target = _supports(candidate, source_id, strong_by_run.get(source_id, ()), alignment, ppm=options["ppm"], rt_tolerance=options["rt_tolerance_sec"])
                    decoy = _supports(candidate, source_id, strong_by_run.get(source_id, ()), alignment, ppm=options["ppm"], rt_tolerance=options["rt_tolerance_sec"], shifted=True)
                    if target is not None:
                        target_supports.append((target, source_id, alignment))
                    if decoy is not None:
                        decoy_supports.append((decoy, source_id, alignment))
                target_supports.sort(key=lambda item: (-item[0][0], item[0][1], item[0][2], item[0][3], item[0][4], item[1]))
                decoy_supports.sort(key=lambda item: (-item[0][0], item[0][1], item[0][2], item[0][3], item[0][4], item[1]))
                grouped[component].append({
                    "target_run": target_id, "candidate": candidate, "component": component,
                    "targets": target_supports[:MAX_SUPPORTS], "decoys": decoy_supports[:MAX_SUPPORTS],
                })
    for component, outcomes in grouped.items():
        competitions = []
        for item in outcomes:
            seed = "%s:%d" % (item["target_run"], item["candidate"].candidate_id)
            item["seed"] = seed
            competitions.append(TargetDecoyCompetition(seed, item["targets"][0][0][0] if item["targets"] else None, item["decoys"][0][0][0] if item["decoys"] else None))
        results = {item.seed_id: item for item in target_decoy_q_values(competitions)}
        for item in outcomes:
            item["competition"] = results[item["seed"]]
    return grouped


def _evidence_rows(outcomes, q_value_max):
    rows = defaultdict(list)
    for values in outcomes.values():
        for item in values:
            result = item["competition"]
            accepted = result.winner == "target" and result.q_value <= q_value_max and item["targets"]
            supports = item["targets"] if accepted else item["targets"][:1]
            if not supports and result.winner == "none":
                continue
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
                    "status": "accepted_matched_weak_feature" if accepted else ("decoy_winner" if result.winner == "decoy" else "target_q_value_above_limit"),
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
        rows.append(row)
    if rows != original or accepted:
        schema = table.schema
        combined = pa.Table.from_pylist(rows, schema=schema)
        temporary = _write_like_existing(combined, path)
        publish_staged_files([(temporary, Path(path))])
    return len(accepted)


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
    )
    outcomes = _candidate_outcomes(successful, strong, weak, models, {
        "ppm": float(options.get("external_ppm", 8.0)),
        "rt_tolerance_sec": float(options.get("external_rt_tolerance_sec", 120.0)),
    })
    evidence = _evidence_rows(outcomes, float(options.get("external_q_value_max", 0.05)))
    summaries = {}
    for run in successful:
        all_outcomes = [item for values in outcomes.values() for item in values if item["target_run"] == run.run_id]
        accepted = [item for item in all_outcomes if item["competition"].winner == "target" and item["competition"].q_value <= float(options.get("external_q_value_max", 0.05)) and item["targets"]]
        count = _publish_features(result_by_id[run.run_id]["paths"]["features"], accepted)
        for row in evidence.get(run.run_id, ()):
            if row["status"] == "accepted_matched_weak_feature":
                match = next(item for item in accepted if item["candidate"].candidate_id == row["weak_candidate_id"])
                row["feature_id"] = match["feature_id"]
        _write_evidence(result_by_id[run.run_id]["paths"]["external_evidence"], evidence.get(run.run_id, ()))
        statuses = Counter(row["status"] for row in evidence.get(run.run_id, ()))
        summaries[run.run_id] = {
            "run_id": run.run_id, "planned_assay_count": len(weak.get(run.run_id, ())),
            "evaluated_assay_count": len(all_outcomes), "new_external_feature_count": count,
            "new_strict_external_feature_count": 0, "new_weak_external_feature_count": count,
            "status_counts": dict(sorted(statuses.items())),
        }
    return {"summaries": summaries, "alignment_models": _alignment_model_rows(models), "reference_runs": models.reference_runs, "scheduler_summary": {"mode": "feature_mbr_no_raw_ms1"}}
