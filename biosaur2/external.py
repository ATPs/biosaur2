"""Project-level RT alignment and calibrated external exact-assay extraction."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
from typing import Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .alignment import (
    AlignmentAnchor,
    RTAlignmentModel,
    choose_reference_run,
    fit_rt_alignment,
)
from .chemistry import IsotopePeak, isotope_library, parse_peptidoform
from .confidence import (
    TargetDecoyCompetition,
    deterministic_decoy_shift,
    target_decoy_q_values,
)
from .hybrid import (
    DirectAssay,
    FEATURE_ORIGIN_ALIGNED_EXTERNAL,
    QUALITY_FLAG_BOUNDARY_TRUNCATED,
    LocalFeatureCandidate,
    _candidate_segment_values,
    _quant_row,
    _recovered_feature_row,
    extract_local_feature,
)
from .output import _format_tsv, _temporary_neighbor, publish_staged_files
from .legacy_output import merge_hybrid_output_rows
from .raw_ms1 import load_raw_ms1_cache
from .schema import compact_schemas


EXTERNAL_EVIDENCE_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class ExternalObservation:
    run_id: str
    ion_key: str
    canonical_peptidoform: str
    charge: int
    faims_cv: Optional[float]
    rt_apex_sec: float
    q_value: float
    assay_id: int
    psm_id: str


@dataclass(frozen=True)
class ExternalPlan:
    target_run: str
    source_run: str
    alignment_group: str
    observation: ExternalObservation
    predicted_rt_sec: float
    alignment: RTAlignmentModel


def alignment_group_for_run(run) -> str:
    """Return an explicit group or a conservative fraction/batch-derived one."""

    explicit = (run.metadata.get("alignment_group") or "").strip()
    if explicit:
        return "explicit:" + explicit
    fraction = (run.metadata.get("fraction") or "").strip()
    batch = (run.metadata.get("batch") or "").strip()
    if fraction or batch:
        return "derived:fraction=%s|batch=%s" % (fraction, batch)
    return "derived:default"


def _faims_key(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "none"
    return format(float(value), ".9g")


def exact_ion_key(canonical_peptidoform, charge, faims_cv) -> str:
    return "%s\x1f%d\x1f%s" % (
        canonical_peptidoform,
        int(charge),
        _faims_key(faims_cv),
    )


def _read_rows(path, columns=None, table_name=None):
    source = Path(path)
    if not source.is_file():
        return []
    if source.suffix.lower() == ".duckdb":
        import duckdb

        selection = "*" if columns is None else ", ".join(
            '"%s"' % name.replace('"', '""') for name in columns
        )
        with duckdb.connect(str(source), read_only=True) as connection:
            return connection.execute(
                'SELECT %s FROM "%s"' % (selection, table_name)
            ).fetch_arrow_table().to_pylist()
    if source.suffix.lower() == ".tsv":
        schema_name = (
            "hybrid_features"
            if table_name == "features"
            else "merged_identifications"
        )
        schema = compact_schemas()[schema_name]
        fields = {field.name: field for field in schema}
        selected = set(columns) if columns is not None else None
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = []
            for raw in csv.DictReader(handle, delimiter="\t"):
                converted = {}
                for name, value in raw.items():
                    if selected is not None and name not in selected:
                        continue
                    converted[name] = _parse_tsv_value(value, fields[name].type)
                rows.append(converted)
            return rows
    return pq.read_table(
        source, columns=None if columns is None else list(columns)
    ).to_pylist()


def _parse_tsv_value(value, data_type):
    if value == "":
        return None
    if pa.types.is_dictionary(data_type):
        data_type = data_type.value_type
    if pa.types.is_list(data_type) or pa.types.is_struct(data_type):
        return json.loads(value)
    if pa.types.is_integer(data_type):
        return int(value)
    if pa.types.is_floating(data_type):
        return float(value)
    if pa.types.is_boolean(data_type):
        return value.lower() in {"1", "true", "yes"}
    return value


def _read_table(path, table_name):
    source = Path(path)
    if source.suffix.lower() == ".duckdb":
        import duckdb

        with duckdb.connect(str(source), read_only=True) as connection:
            return connection.execute(
                'SELECT * FROM "%s"' % table_name
            ).fetch_arrow_table()
    if source.suffix.lower() == ".tsv":
        schema_name = (
            "hybrid_features"
            if table_name == "features"
            else "merged_identifications"
        )
        return pa.Table.from_pylist(
            _read_rows(source, table_name=table_name),
            schema=compact_schemas()[schema_name],
        )
    return pq.read_table(source)


def read_external_observations(run, paths) -> tuple[ExternalObservation, ...]:
    """Read de-duplicated, directly observed quantitative peptide-ion anchors."""

    assays = _read_rows(
        paths["identifications"], table_name="identifications"
    )
    features = _read_rows(paths["features"], table_name="features")
    feature_rt = {
        int(row["feature_idx"]): float(row["rt_apex_sec"])
        for row in features
        if row.get("feature_idx") is not None
        and row.get("rt_apex_sec") is not None
        and row.get("quant_value") is not None
        and float(row["quant_value"]) > 0
    }
    feature_by_assay = {}
    for row in features:
        for event in row.get("ms2_events") or ():
            if (
                event.get("assay_id") is not None
                and event.get("association_tier") == "direct_id"
            ):
                feature_by_assay[int(event["assay_id"])] = int(
                    row["feature_idx"]
                )
    selected = {}
    for row in assays:
        if row.get("assay_id") is None:
            continue
        assay_id = int(row["assay_id"])
        feature_id = feature_by_assay.get(assay_id)
        if feature_id not in feature_rt or row.get("assay_conflict_status") != "unique":
            continue
        faims = row.get("assay_faims_cv")
        faims = None if faims is None else float(faims)
        key = exact_ion_key(row["canonical_peptidoform"], row["assay_charge"], faims)
        observation = ExternalObservation(
            run_id=run.run_id,
            ion_key=key,
            canonical_peptidoform=row["canonical_peptidoform"],
            charge=int(row["assay_charge"]),
            faims_cv=faims,
            rt_apex_sec=feature_rt[feature_id],
            q_value=float(row["q_value"]),
            assay_id=assay_id,
            psm_id=row["psm_id"],
        )
        rank = (observation.q_value, observation.assay_id, observation.psm_id)
        previous = selected.get(key)
        if previous is None or rank < previous[0]:
            selected[key] = (rank, observation)
    return tuple(selected[key][1] for key in sorted(selected))


def _rejected_alignment(source_run, target_run, count, status):
    return RTAlignmentModel(
        source_run,
        target_run,
        "none",
        count,
        0,
        (),
        (),
        1.0,
        0.0,
        None,
        status,
    )


def build_alignment_models(
    runs,
    observations_by_run: Mapping[str, Sequence[ExternalObservation]],
    *,
    min_anchors: int = 5,
    max_residual_mad_sec: float = 30.0,
):
    """Fit every useful ordered source-to-target model inside each group."""

    grouped = defaultdict(list)
    for run in runs:
        grouped[alignment_group_for_run(run)].append(run.run_id)
    models = {}
    for group, run_ids in sorted(grouped.items()):
        run_ids = sorted(run_ids)
        indexed = {
            run_id: {item.ion_key: item for item in observations_by_run.get(run_id, ())}
            for run_id in run_ids
        }
        for source_run in run_ids:
            for target_run in run_ids:
                if source_run == target_run:
                    continue
                common = sorted(set(indexed[source_run]) & set(indexed[target_run]))
                if len(common) < min_anchors:
                    model = _rejected_alignment(
                        source_run, target_run, len(common), "insufficient_anchors"
                    )
                else:
                    anchors = [
                        AlignmentAnchor(
                            key,
                            indexed[source_run][key].rt_apex_sec,
                            indexed[target_run][key].rt_apex_sec,
                            max(
                                1e-6,
                                1.0
                                - max(
                                    indexed[source_run][key].q_value,
                                    indexed[target_run][key].q_value,
                                ),
                            ),
                        )
                        for key in common
                    ]
                    model = fit_rt_alignment(source_run, target_run, anchors)
                    if (
                        model.residual_mad_sec is None
                        or model.residual_mad_sec > max_residual_mad_sec
                    ):
                        model = RTAlignmentModel(
                            **{
                                **model.__dict__,
                                "status": "residual_mad_exceeds_limit",
                            }
                        )
                models[(source_run, target_run)] = (group, model)
    return models


def choose_group_reference_runs(runs, observations_by_run):
    grouped = defaultdict(dict)
    for run in runs:
        grouped[alignment_group_for_run(run)][run.run_id] = len(
            observations_by_run.get(run.run_id, ())
        )
    return {
        group: choose_reference_run(counts)
        for group, counts in sorted(grouped.items())
    }


def plan_external_assays(
    runs,
    observations_by_run: Mapping[str, Sequence[ExternalObservation]],
    models,
):
    """Choose one deterministic quantitative donor for every recipient-only ion."""

    run_by_id = {run.run_id: run for run in runs}
    group_by_run = {
        run.run_id: alignment_group_for_run(run) for run in runs
    }
    sources_by_group = defaultdict(lambda: defaultdict(list))
    for source_run, observations in observations_by_run.items():
        group = group_by_run[source_run]
        for observation in observations:
            sources_by_group[group][observation.ion_key].append(observation)
    result = {run.run_id: [] for run in runs}
    for target_run in sorted(run_by_id):
        group = group_by_run[target_run]
        target_ions = {
            item.ion_key for item in observations_by_run.get(target_run, ())
        }
        for ion_key in sorted(sources_by_group[group]):
            if ion_key in target_ions:
                continue
            choices = []
            for observation in sources_by_group[group][ion_key]:
                if observation.run_id == target_run:
                    continue
                _model_group, model = models.get(
                    (observation.run_id, target_run),
                    (group, _rejected_alignment(observation.run_id, target_run, 0, "missing_model")),
                )
                if model.status != "accepted":
                    continue
                predicted = model.predict(observation.rt_apex_sec)
                if not math.isfinite(predicted):
                    continue
                choices.append(
                    (
                        observation.q_value,
                        math.inf
                        if model.residual_mad_sec is None
                        else model.residual_mad_sec,
                        observation.run_id,
                        observation.assay_id,
                        observation,
                        model,
                        predicted,
                    )
                )
            if choices:
                choice = min(choices, key=lambda value: value[:4])
                result[target_run].append(
                    ExternalPlan(
                        target_run,
                        choice[4].run_id,
                        group,
                        choice[4],
                        choice[6],
                        choice[5],
                    )
                )
    return {key: tuple(value) for key, value in result.items()}


def _shifted_assay(assay: DirectAssay, neutral_shift: float) -> DirectAssay:
    mz_shift = neutral_shift / assay.charge
    peaks = tuple(
        IsotopePeak(
            peak.isotope_index,
            peak.probability,
            peak.relative_abundance,
            peak.neutral_mass_shift,
            peak.centroid_mass_shift,
            peak.mz + mz_shift,
        )
        for peak in assay.isotope_peaks
    )
    return DirectAssay(
        **{
            **assay.__dict__,
            "canonical_peptidoform": assay.canonical_peptidoform
            + "|external_decoy:%+.6f" % neutral_shift,
            "selected_ion_mz": peaks[0].mz,
            "isotope_peaks": peaks,
        }
    )


def _candidate_score(
    candidate: LocalFeatureCandidate,
    predicted_rt_sec: float,
    *,
    ppm: float,
    rt_tolerance_sec: float,
    min_isotope_cosine: float,
):
    if not candidate.quantitative or candidate.mono_point_count < 3:
        return None
    if candidate.isotope_cosine is None or candidate.isotope_cosine < min_isotope_cosine:
        return None
    if (
        candidate.mono_mz_error_ppm is None
        or abs(candidate.mono_mz_error_ppm) > ppm
    ):
        return None
    rt_error = abs(candidate.rt_apex_sec - predicted_rt_sec)
    if rt_error > rt_tolerance_sec:
        return None
    value = candidate.quantification.value
    if value is None or value <= 0:
        return None
    return float(
        math.log1p(value)
        + 5.0 * candidate.isotope_cosine
        - abs(candidate.mono_mz_error_ppm) / ppm
        - rt_error / max(rt_tolerance_sec, 1e-12)
    )


def _candidate_gate_status(
    candidate: LocalFeatureCandidate,
    predicted_rt_sec: float,
    *,
    ppm: float,
    rt_tolerance_sec: float,
    min_isotope_cosine: float,
):
    if not candidate.quantitative:
        return candidate.status
    if candidate.mono_point_count < 3:
        return "insufficient_mono_points"
    if candidate.isotope_cosine is None:
        return "missing_isotope_cosine"
    if candidate.isotope_cosine < min_isotope_cosine:
        return "isotope_cosine_below_limit"
    if candidate.mono_mz_error_ppm is None:
        return "missing_mono_mass_error"
    if abs(candidate.mono_mz_error_ppm) > ppm:
        return "mono_mass_error_above_limit"
    if abs(candidate.rt_apex_sec - predicted_rt_sec) > rt_tolerance_sec:
        return "apex_rt_error_above_limit"
    if candidate.quantification.value is None or candidate.quantification.value <= 0:
        return "nonpositive_quantification"
    return "passed"


def _faims_equal(left, right):
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), abs_tol=1e-6)


def _equivalent_features(candidate, features, ppm):
    target_mz = candidate.assay.isotope_peaks[0].mz
    matches = []
    for row in features:
        if int(row["charge"]) != candidate.assay.charge:
            continue
        if not _faims_equal(row.get("FAIMS"), candidate.assay.faims_cv):
            continue
        if abs(float(row["mz"]) - target_mz) * 1e6 / target_mz > ppm:
            continue
        if max(float(row["rtStart"]) * 60.0, candidate.rt_start_sec) > min(
            float(row["rtEnd"]) * 60.0, candidate.rt_end_sec
        ):
            continue
        matches.append(row)
    return sorted(matches, key=lambda row: int(row["feature_idx"]))


def _evidence_schema():
    return pa.schema(
        [
            pa.field("target_run", pa.string(), nullable=False),
            pa.field("source_run", pa.string(), nullable=False),
            pa.field("alignment_group", pa.string(), nullable=False),
            pa.field("ion_key", pa.string(), nullable=False),
            pa.field("canonical_peptidoform", pa.string(), nullable=False),
            pa.field("charge", pa.int16(), nullable=False),
            pa.field("faims_cv", pa.float32()),
            pa.field("donor_assay_id", pa.int32(), nullable=False),
            pa.field("donor_psm_id", pa.string(), nullable=False),
            pa.field("donor_q_value", pa.float64(), nullable=False),
            pa.field("donor_rt_apex_sec", pa.float64(), nullable=False),
            pa.field("predicted_rt_sec", pa.float64(), nullable=False),
            pa.field("alignment_method", pa.string(), nullable=False),
            pa.field("alignment_anchor_count", pa.int32(), nullable=False),
            pa.field("alignment_inlier_count", pa.int32(), nullable=False),
            pa.field("alignment_residual_mad_sec", pa.float64()),
            pa.field("decoy_neutral_shift", pa.float64(), nullable=False),
            pa.field("target_score", pa.float64()),
            pa.field("decoy_score", pa.float64()),
            pa.field("target_extraction_status", pa.string(), nullable=False),
            pa.field("target_gate_status", pa.string(), nullable=False),
            pa.field("decoy_extraction_status", pa.string(), nullable=False),
            pa.field("decoy_gate_status", pa.string(), nullable=False),
            pa.field("competition_winner", pa.string(), nullable=False),
            pa.field("extraction_q_value", pa.float64(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("feature_id", pa.int64()),
            pa.field("target_mono_points", pa.int32(), nullable=False),
            pa.field("target_isotope_cosine", pa.float32()),
            pa.field("target_mass_error_ppm", pa.float32()),
            pa.field("target_rt_error_sec", pa.float64()),
        ],
        metadata={
            b"biosaur2_external_evidence_schema_version": EXTERNAL_EVIDENCE_SCHEMA_VERSION.encode()
        },
    )


def _write_like_existing(table, final_path, *, schema=None):
    temporary = _temporary_neighbor(Path(final_path))
    output = table if schema is None else table.cast(schema)
    if Path(final_path).suffix.lower() == ".tsv":
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=output.schema.names,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in output.to_pylist():
                writer.writerow(
                    {name: _format_tsv(row.get(name)) for name in output.schema.names}
                )
        return temporary
    pq.write_table(
        output,
        temporary,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        data_page_version="2.0",
        version="2.6",
        row_group_size=122880,
    )
    return temporary


def _assay_for_external_plan(run_id, plan):
    peptidoform = parse_peptidoform(plan.observation.canonical_peptidoform)
    if peptidoform.formula_status != "exact" or peptidoform.formula is None:
        return None
    peaks = isotope_library(
        peptidoform.formula, plan.observation.charge, max_isotopes=6
    )
    return DirectAssay(
        run_id=run_id,
        ms2_event_id=-1,
        psm_id="external:%s:%d"
        % (plan.source_run, plan.observation.assay_id),
        canonical_peptidoform=plan.observation.canonical_peptidoform,
        charge=plan.observation.charge,
        rt_sec=plan.predicted_rt_sec,
        faims_cv=plan.observation.faims_cv,
        selected_ion_mz=peaks[0].mz,
        selected_isotope_index=0,
        selected_mz_error_ppm=0.0,
        peptidoform=peptidoform,
        isotope_peaks=peaks,
        q_value=plan.observation.q_value,
        pep=None,
        score=None,
        rank=None,
    )


def _extract_external_candidate(store, assay, options):
    return extract_local_feature(
        store,
        assay,
        ppm=float(options["ppm"]),
        rt_tolerance_sec=float(options["rt_tolerance_sec"]),
        quant_method=options["quant_method"],
        baseline=options["baseline"],
        allow_two_point_exception=False,
        allow_partial_envelope=False,
    )


def run_external_recipient(task):
    """Extract, compete, de-duplicate and atomically publish one recipient."""

    run = task["run"]
    paths = task["paths"]
    plans = task["plans"]
    options = task["options"]
    ppm = float(options["ppm"])
    rt_tolerance = float(options["rt_tolerance_sec"])
    min_cosine = float(options["min_isotope_cosine"])
    store = load_raw_ms1_cache(paths["raw_ms1_cache"], run.mzml_path)
    evaluated = []
    competitions = []
    for plan in plans:
        assay = _assay_for_external_plan(run.run_id, plan)
        if assay is None:
            continue
        shift = deterministic_decoy_shift(run.run_id, plan.observation.ion_key)
        decoy_assay = _shifted_assay(assay, shift)
        target = _extract_external_candidate(store, assay, options)
        decoy = _extract_external_candidate(store, decoy_assay, options)
        target_score = _candidate_score(
            target,
            plan.predicted_rt_sec,
            ppm=ppm,
            rt_tolerance_sec=rt_tolerance,
            min_isotope_cosine=min_cosine,
        )
        decoy_score = _candidate_score(
            decoy,
            plan.predicted_rt_sec,
            ppm=ppm,
            rt_tolerance_sec=rt_tolerance,
            min_isotope_cosine=min_cosine,
        )
        target_gate = _candidate_gate_status(
            target,
            plan.predicted_rt_sec,
            ppm=ppm,
            rt_tolerance_sec=rt_tolerance,
            min_isotope_cosine=min_cosine,
        )
        decoy_gate = _candidate_gate_status(
            decoy,
            plan.predicted_rt_sec,
            ppm=ppm,
            rt_tolerance_sec=rt_tolerance,
            min_isotope_cosine=min_cosine,
        )
        seed_id = plan.observation.ion_key
        competitions.append(TargetDecoyCompetition(seed_id, target_score, decoy_score))
        evaluated.append(
            (
                plan,
                shift,
                target.status,
                target_gate,
                target.mono_point_count,
                target.isotope_cosine,
                target.mono_mz_error_ppm,
                (
                    None
                    if target.rt_apex_sec is None
                    else abs(target.rt_apex_sec - plan.predicted_rt_sec)
                ),
                decoy.status,
                decoy_gate,
            )
        )

    competition = {
        item.seed_id: item for item in target_decoy_q_values(competitions)
    }
    feature_table = _read_table(paths["features"], "features")
    feature_rows = feature_table.to_pylist()
    new_feature_rows = []
    new_quant_rows = []
    evidence_rows = []
    next_feature_id = 1 + max(
        (int(row["feature_idx"]) for row in feature_rows), default=0
    )
    accepted_q = float(options["q_value_max"])
    counts = Counter()
    for (
        plan,
        shift,
        target_extraction_status,
        target_gate,
        target_mono_points,
        target_isotope_cosine,
        target_mass_error_ppm,
        target_rt_error_sec,
        decoy_extraction_status,
        decoy_gate,
    ) in sorted(
        evaluated, key=lambda value: value[0].observation.ion_key
    ):
        result = competition[plan.observation.ion_key]
        feature_id = None
        target = None
        if result.winner != "target":
            status = (
                "decoy_winner"
                if result.winner == "decoy"
                else "target_" + target_gate
            )
        elif result.q_value > accepted_q:
            status = "target_q_value_above_limit"
        else:
            assay = _assay_for_external_plan(run.run_id, plan)
            target = _extract_external_candidate(store, assay, options)
            equivalents = _equivalent_features(target, feature_rows, ppm)
            if len(equivalents) == 1:
                feature_id = int(equivalents[0]["feature_idx"])
                status = "accepted_matched_existing_feature"
            elif len(equivalents) > 1:
                status = "ambiguous_existing_features"
            else:
                feature_id = next_feature_id
                next_feature_id += 1
                feature_row = _recovered_feature_row(target, feature_id)
                feature_rows.append(feature_row)
                new_feature_rows.append(feature_row)
                start, end = target.segment_slice
                rt = target.traces[0].rt_sec[start:end]
                traces = _candidate_segment_values(target)
                new_quant_rows.append(
                    _quant_row(
                        run.run_id,
                        feature_id,
                        FEATURE_ORIGIN_ALIGNED_EXTERNAL,
                        "external_id",
                        rt,
                        traces,
                        method=options["quant_method"],
                        baseline=options["baseline"],
                        quality_score=target.isotope_cosine,
                        isotope_cosine=target.isotope_cosine,
                        mass_error=target.mono_mz_error_ppm,
                        supporting_psm_count=0,
                        supporting_ms2_count=0,
                        extraction_q_value=result.q_value,
                        quality_flags=(
                            QUALITY_FLAG_BOUNDARY_TRUNCATED
                            if target.boundary_truncated else 0
                        ),
                    )
                )
                status = "accepted_new_external_feature"
        counts[status] += 1
        evidence_rows.append(
            {
                "target_run": run.run_id,
                "source_run": plan.source_run,
                "alignment_group": plan.alignment_group,
                "ion_key": plan.observation.ion_key,
                "canonical_peptidoform": plan.observation.canonical_peptidoform,
                "charge": plan.observation.charge,
                "faims_cv": plan.observation.faims_cv,
                "donor_assay_id": plan.observation.assay_id,
                "donor_psm_id": plan.observation.psm_id,
                "donor_q_value": plan.observation.q_value,
                "donor_rt_apex_sec": plan.observation.rt_apex_sec,
                "predicted_rt_sec": plan.predicted_rt_sec,
                "alignment_method": plan.alignment.method,
                "alignment_anchor_count": plan.alignment.anchor_count,
                "alignment_inlier_count": plan.alignment.inlier_count,
                "alignment_residual_mad_sec": plan.alignment.residual_mad_sec,
                "decoy_neutral_shift": shift,
                "target_score": result.target_score,
                "decoy_score": result.decoy_score,
                "target_extraction_status": target_extraction_status,
                "target_gate_status": target_gate,
                "decoy_extraction_status": decoy_extraction_status,
                "decoy_gate_status": decoy_gate,
                "competition_winner": result.winner,
                "extraction_q_value": result.q_value,
                "status": status,
                "feature_id": feature_id,
                "target_mono_points": target_mono_points,
                "target_isotope_cosine": target_isotope_cosine,
                "target_mass_error_ppm": target_mass_error_ppm,
                "target_rt_error_sec": target_rt_error_sec,
            }
        )

    summary = {
        "planned_assay_count": len(plans),
        "evaluated_assay_count": len(evaluated),
        "new_external_feature_count": len(new_feature_rows),
        "status_counts": dict(sorted(counts.items())),
    }
    evidence_schema = _evidence_schema().with_metadata(
        {
            **(_evidence_schema().metadata or {}),
            b"biosaur2_external_summary_json": json.dumps(
                summary, sort_keys=True, separators=(",", ":")
            ).encode(),
        }
    )
    staged = []
    evidence_table = pa.Table.from_pylist(evidence_rows, schema=evidence_schema)
    database_path = (
        paths.get("run_output") if paths.get("format") == "duckdb" else None
    )
    if not database_path:
        staged.append(
            (_write_like_existing(evidence_table, paths["external_evidence"]), Path(paths["external_evidence"]))
        )
    appended = None
    if new_feature_rows:
        merge_args = {
            "no_mono_hills": "mono_hills_scan_lists" not in feature_table.schema.names,
            "write_extra_details": "isotopes" in feature_table.schema.names,
        }
        merged_new, _ = merge_hybrid_output_rows(
            new_feature_rows,
            new_quant_rows,
            (),
            (),
            (),
            (),
            merge_args,
        )
        appended = pa.Table.from_pylist(merged_new, schema=feature_table.schema)
        combined = pa.concat_tables([feature_table, appended])
        if not database_path:
            staged.append(
                (_write_like_existing(combined, paths["features"]), Path(paths["features"]))
            )
    if database_path:
        import duckdb

        target = Path(database_path)
        temporary = _temporary_neighbor(target)
        shutil.copy2(target, temporary)
        try:
            with duckdb.connect(str(temporary)) as connection:
                connection.register("_external_evidence", evidence_table)
                connection.execute(
                    "CREATE OR REPLACE TABLE external_id_evidence AS "
                    "SELECT * FROM _external_evidence"
                )
                connection.unregister("_external_evidence")
                if appended is not None:
                    connection.register("_new_features", appended)
                    connection.execute(
                        "INSERT INTO features SELECT * FROM _new_features"
                    )
                    connection.unregister("_new_features")
            staged.append((temporary, target))
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    publish_staged_files(staged)
    return {"run_id": run.run_id, **summary}


def alignment_model_rows(models, reference_runs=None):
    reference_runs = reference_runs or {}
    rows = []
    for (source_run, target_run), (group, model) in sorted(models.items()):
        rows.append(
            {
                "alignment_group": group,
                "reference_run": reference_runs.get(group),
                "source_run": source_run,
                "target_run": target_run,
                "method": model.method,
                "anchor_count": model.anchor_count,
                "inlier_count": model.inlier_count,
                "slope": model.slope,
                "intercept": model.intercept,
                "residual_mad_sec": model.residual_mad_sec,
                "status": model.status,
                "x_knots_json": json.dumps(model.x_knots),
                "y_knots_json": json.dumps(model.y_knots),
            }
        )
    return rows
