"""Project-level RT alignment and calibrated external exact-assay extraction."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
import math
import os
from multiprocessing import get_context
from pathlib import Path
import shutil
import time
from typing import Mapping, Optional, Sequence

import numpy as np
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
from .external_evidence import (
    EXTERNAL_EVIDENCE_SCHEMA_VERSION,
    evidence_schema as _evidence_schema,
)
from .hybrid import (
    DirectAssay,
    FEATURE_ORIGIN_ALIGNED_EXTERNAL,
    FEATURE_ORIGIN_ALIGNED_EXTERNAL_WEAK,
    QUALITY_FLAG_BOUNDARY_TRUNCATED,
    QUALITY_FLAG_TWO_POINT_QUANT,
    QUALITY_FLAG_WEAK_EXTERNAL_FEATURE,
    LocalFeatureCandidate,
    _candidate_segment_values,
    _quant_row,
    _recovered_feature_row,
    extract_local_feature,
)
from .output import _format_tsv, _temporary_neighbor, publish_staged_files
from .legacy_output import merge_hybrid_output_rows
from .raw_ms1 import load_raw_ms1_cache
from .residual import (
    ResidualMS1Ledger,
    load_residual_ownership_cache,
)
from .schema import compact_schemas


logger = logging.getLogger(__name__)


# Spawned raw-extraction workers map the immutable cache files independently.
# The arrays are NumPy memmaps, so workers share the kernel page cache without
# pickling the 200+ MiB MS1 store for every assay.
_EXTERNAL_RAW_STORE = None
_EXTERNAL_RAW_OPTIONS = None

_NATIVE_THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


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


def _external_raw_summary(store, assay, decoy_assay, predicted_rt_sec, options):
    """Extract one raw pair and retain only data needed before re-extraction."""

    raw_ledger = ResidualMS1Ledger(store)
    strict_target = _extract_external_candidate(
        store, assay, options, min_mono_points=3
    )
    strict_decoy = _extract_external_candidate(
        store, decoy_assay, options, min_mono_points=3
    )
    weak_target = _extract_external_candidate(
        store, assay, options, min_mono_points=2
    )
    weak_decoy = _extract_external_candidate(
        store, decoy_assay, options, min_mono_points=2
    )
    ppm = float(options["ppm"])
    rt_tolerance = float(options["rt_tolerance_sec"])
    min_cosine = float(options["min_isotope_cosine"])

    def strict_summary(candidate):
        return {
            "score": _candidate_score(
                candidate, predicted_rt_sec, ppm=ppm,
                rt_tolerance_sec=rt_tolerance,
                min_isotope_cosine=min_cosine,
            ),
            "strict_gate": _candidate_gate_status(
                candidate, predicted_rt_sec, ppm=ppm,
                rt_tolerance_sec=rt_tolerance,
                min_isotope_cosine=min_cosine,
            ),
            "status": candidate.status,
            "mono_points": candidate.mono_point_count,
            "isotope_cosine": candidate.isotope_cosine,
            "mass_error_ppm": candidate.mono_mz_error_ppm,
            "rt_apex_sec": candidate.rt_apex_sec,
        }

    def weak_summary(candidate):
        weak_gate = _weak_candidate_gate_status(
            candidate, predicted_rt_sec, ppm=ppm,
            rt_tolerance_sec=rt_tolerance, min_isotope_cosine=min_cosine,
        )
        footprint = None
        if weak_gate == "passed":
            footprint = raw_ledger.component_footprint(
                candidate.traces, candidate.segment_slice[0],
                _candidate_segment_values(candidate),
            )
        return {
            "weak_gate": weak_gate,
            "footprint": footprint,
            "status": candidate.status,
            "mono_points": candidate.mono_point_count,
            "secondary_channels": _candidate_secondary_channel_count(candidate, 2),
            "isotope_cosine": candidate.isotope_cosine,
            "mass_error_ppm": candidate.mono_mz_error_ppm,
            "rt_apex_sec": candidate.rt_apex_sec,
        }

    return {
        "target": {
            "strict": strict_summary(strict_target),
            "weak": weak_summary(weak_target),
        },
        "decoy": {
            "strict": strict_summary(strict_decoy),
            "weak": weak_summary(weak_decoy),
        },
    }


def _initialize_external_raw_worker(raw_ms1_cache, mzml_path, options):
    """Open one recipient's immutable inputs in a spawned extraction worker."""

    global _EXTERNAL_RAW_STORE, _EXTERNAL_RAW_OPTIONS
    _EXTERNAL_RAW_STORE = load_raw_ms1_cache(raw_ms1_cache, mzml_path)
    _EXTERNAL_RAW_OPTIONS = options


def _extract_external_raw_summary_worker(item):
    """Extract one compact raw result after worker initialization."""

    assay, decoy_assay, predicted_rt_sec = item
    return _external_raw_summary(
        _EXTERNAL_RAW_STORE, assay, decoy_assay,
        predicted_rt_sec, _EXTERNAL_RAW_OPTIONS,
    )


def _parallel_raw_extract(
    items, workers, *, store, raw_ms1_cache, mzml_path, options,
):
    """Use spawned processes for Python-bound raw local-feature extraction."""

    if int(workers) <= 1 or len(items) < 2:
        return [
            _external_raw_summary(
                store, assay, decoy_assay, predicted_rt_sec, options
            )
            for assay, decoy_assay, predicted_rt_sec in items
        ]
    worker_count = min(int(workers), len(items))
    previous_environment = {
        name: os.environ.get(name) for name in _NATIVE_THREAD_ENVIRONMENT
    }
    try:
        for name in _NATIVE_THREAD_ENVIRONMENT:
            os.environ[name] = "1"
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=get_context("spawn"),
            initializer=_initialize_external_raw_worker,
            initargs=(str(raw_ms1_cache), str(mzml_path), options),
        ) as executor:
            return list(
                executor.map(
                    _extract_external_raw_summary_worker, items,
                    chunksize=max(1, len(items) // (worker_count * 8)),
                )
            )
    finally:
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _extract_external_residual_pair(item):
    """Extract only sides that survived raw weak gating and overlap pruning."""

    store, assay, decoy_assay, options, extract_target, extract_decoy = item
    target = (
        _extract_external_candidate(store, assay, options, min_mono_points=2)
        if extract_target
        else None
    )
    decoy = (
        _extract_external_candidate(
            store, decoy_assay, options, min_mono_points=2
        )
        if extract_decoy
        else None
    )
    return target, decoy


def _weak_raw_overlap(summary, ledger):
    """Evaluate a raw footprint against ownership after strict allocation."""

    footprint = summary.get("footprint")
    if summary["weak_gate"] != "passed" or footprint is None:
        return None
    preview = ledger.footprint_overlap(footprint)
    return preview.fraction if preview.status == "accepted" else 1.0


def _parallel_extract(function, items, workers):
    """Keep candidate objects in-process while Cython/NumPy releases the GIL."""

    if int(workers) <= 1 or len(items) < 2:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        return list(executor.map(function, items))


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


def _candidate_secondary_channel_count(candidate, minimum_points):
    if candidate.segment_slice is None:
        return 0
    values = _candidate_segment_values(candidate)
    if not values:
        return 0
    return sum(
        int(np.count_nonzero(np.asarray(value) > 0)) >= int(minimum_points)
        for value in values[1:]
    )


def _weak_candidate_gate_status(
    candidate: LocalFeatureCandidate,
    predicted_rt_sec: float,
    *,
    ppm: float,
    rt_tolerance_sec: float,
    min_isotope_cosine: float,
):
    """Conservative 2+2 donor-guided feature gate for Project recovery."""

    if not candidate.quantitative:
        return candidate.status
    if candidate.mono_point_count < 2:
        return "weak_insufficient_mono_points"
    if _candidate_secondary_channel_count(candidate, 2) < 1:
        return "weak_insufficient_secondary_isotope_points"
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


def _weak_candidate_score(
    candidate,
    predicted_rt_sec,
    *,
    ppm,
    rt_tolerance_sec,
    min_isotope_cosine,
):
    if _weak_candidate_gate_status(
        candidate,
        predicted_rt_sec,
        ppm=ppm,
        rt_tolerance_sec=rt_tolerance_sec,
        min_isotope_cosine=min_isotope_cosine,
    ) != "passed":
        return None
    value = candidate.quantification.value
    return float(
        math.log1p(value)
        + 5.0 * candidate.isotope_cosine
        - abs(candidate.mono_mz_error_ppm) / ppm
        - abs(candidate.rt_apex_sec - predicted_rt_sec)
        / max(rt_tolerance_sec, 1e-12)
    )


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


def _restore_initial_external_claims(ledger, store, feature_rows, prepared, options, ppm):
    """Rebuild external ownership omitted from the pre-external cache.

    Hybrid post-processing persists ownership before Project external recovery.
    A resumed Project run therefore starts with external rows in its feature
    output but without their claims in the ledger.  Reconstructing those
    claims before any new competition makes a resume equivalent to the first
    execution and fails safely if the output cannot be reconstructed.
    """

    origins = {
        FEATURE_ORIGIN_ALIGNED_EXTERNAL: ("strict", 3),
        FEATURE_ORIGIN_ALIGNED_EXTERNAL_WEAK: ("weak", 2),
    }
    restored = set()
    for row in sorted(feature_rows, key=lambda value: int(value["feature_idx"])):
        origin = row.get("feature_origin")
        if origin not in origins:
            continue
        family, min_mono_points = origins[origin]
        matching = []
        for plan, assay, _decoy_assay, _shift in prepared:
            target_mz = assay.isotope_peaks[0].mz
            if int(row["charge"]) != assay.charge:
                continue
            if not _faims_equal(row.get("FAIMS"), assay.faims_cv):
                continue
            if abs(float(row["mz"]) - target_mz) * 1e6 / target_mz > ppm:
                continue
            candidate = _extract_external_candidate(
                store, assay, options, min_mono_points=min_mono_points
            )
            if _equivalent_features(candidate, (row,), ppm):
                matching.append((plan, candidate))
        if len(matching) != 1:
            raise RuntimeError(
                "cannot deterministically restore %s external feature %s "
                "from current Project plans (matches=%d)"
                % (family, row["feature_idx"], len(matching))
            )
        _plan, raw_candidate = matching[0]
        residual_candidate = _extract_external_candidate(
            ledger.materialize(), raw_candidate.assay, options,
            min_mono_points=min_mono_points,
        )
        start, _end = residual_candidate.segment_slice
        allocation = ledger.allocate_component(
            ("resume_external_" + family, int(row["feature_idx"])),
            residual_candidate.traces,
            start,
            _candidate_segment_values(residual_candidate),
        )
        if not allocation.accepted:
            raise RuntimeError(
                "cannot restore %s external feature %s: %s"
                % (family, row["feature_idx"], allocation.status)
            )
        restored.add(int(row["feature_idx"]))
    return restored


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


def _extract_external_candidate(store, assay, options, *, min_mono_points=3):
    return extract_local_feature(
        store,
        assay,
        ppm=float(options["ppm"]),
        rt_tolerance_sec=float(options["rt_tolerance_sec"]),
        min_mono_points=int(min_mono_points),
        quant_method=options["quant_method"],
        baseline=options["baseline"],
        allow_two_point_exception=False,
        allow_partial_envelope=False,
        # Project RT is a prediction, not an observed MS2 event.  Quality is
        # assessed from apex distance below, not from one exact scan index.
        require_event_scan=False,
    )


def run_external_recipient(task, workers=1):
    """Extract, compete, de-duplicate and atomically publish one recipient."""

    run = task["run"]
    paths = task["paths"]
    plans = task["plans"]
    options = task["options"]
    ppm = float(options["ppm"])
    rt_tolerance = float(options["rt_tolerance_sec"])
    min_cosine = float(options["min_isotope_cosine"])
    store = load_raw_ms1_cache(paths["raw_ms1_cache"], run.mzml_path)
    ownership_path = paths.get("residual_ownership_cache")
    if ownership_path:
        if not Path(ownership_path).is_dir():
            raise FileNotFoundError(
                "external-ID requires final residual ownership cache: %s"
                % ownership_path
            )
        ledger = load_residual_ownership_cache(ownership_path, run.mzml_path, store)
    else:
        # Direct unit tests may deliberately exercise an otherwise empty run.
        logger.warning(
            "External-ID run %s has no residual ownership cache; using empty ownership",
            run.run_id,
        )
        ledger = ResidualMS1Ledger(store)
    evaluated = []
    strict_competitions = []
    prepared = []
    for plan in plans:
        assay = _assay_for_external_plan(run.run_id, plan)
        if assay is None:
            continue
        shift = deterministic_decoy_shift(run.run_id, plan.observation.ion_key)
        decoy_assay = _shifted_assay(assay, shift)
        prepared.append((plan, assay, decoy_assay, shift))
    raw_started = time.monotonic()
    logger.info(
        "External-ID run %s: extracting %d raw target/decoy pairs with %d mmap process workers",
        run.run_id,
        len(prepared),
        max(1, int(workers)),
    )
    raw_pairs = _parallel_raw_extract(
        [
            (assay, decoy_assay, plan.predicted_rt_sec)
            for plan, assay, decoy_assay, _shift in prepared
        ],
        workers,
        store=store,
        raw_ms1_cache=paths["raw_ms1_cache"],
        mzml_path=run.mzml_path,
        options=options,
    )
    raw_extract_sec = time.monotonic() - raw_started
    for (plan, assay, decoy_assay, shift), raw_pair in zip(
        prepared, raw_pairs
    ):
        target_summary = raw_pair["target"]["strict"]
        decoy_summary = raw_pair["decoy"]["strict"]
        seed_id = plan.observation.ion_key
        strict_competitions.append(
            TargetDecoyCompetition(
                seed_id, target_summary["score"], decoy_summary["score"]
            )
        )
        evaluated.append(
            {
                "plan": plan,
                "assay": assay,
                "decoy_assay": decoy_assay,
                "shift": shift,
                "raw_target": target_summary,
                "raw_decoy": decoy_summary,
                "raw_weak_target": raw_pair["target"]["weak"],
                "raw_weak_decoy": raw_pair["decoy"]["weak"],
                "target_gate": target_summary["strict_gate"],
                "decoy_gate": decoy_summary["strict_gate"],
            }
        )

    strict_competition = {
        item.seed_id: item for item in target_decoy_q_values(strict_competitions)
    }
    feature_table = _read_table(paths["features"], "features")
    feature_rows = feature_table.to_pylist()
    restored_external_feature_ids = _restore_initial_external_claims(
        ledger, store, feature_rows, prepared, options, ppm
    )
    if restored_external_feature_ids:
        logger.info(
            "External-ID run %s: restored ownership for %d existing external features",
            run.run_id, len(restored_external_feature_ids),
        )
    new_feature_rows = []
    new_quant_rows = []
    evidence_rows = []
    next_feature_id = 1 + max(
        (int(row["feature_idx"]) for row in feature_rows), default=0
    )
    accepted_q = float(options["q_value_max"])
    weak_enabled = bool(options.get("weak_feature", True))
    weak_q_max = float(options.get("weak_q_value_max", 0.05))
    weak_overlap_max = float(options.get("weak_overlap_max", 0.20))
    counts = Counter()
    for record in sorted(evaluated, key=lambda value: value["plan"].observation.ion_key):
        plan = record["plan"]
        result = strict_competition[plan.observation.ion_key]
        record["strict_result"] = result
        record["feature_id"] = None
        record["acceptance_family"] = None
        record["acceptance_q_value"] = None
        if result.winner != "target":
            record["status"] = "decoy_winner" if result.winner == "decoy" else "target_" + record["target_gate"]
            continue
        if result.q_value > accepted_q:
            record["status"] = "target_q_value_above_limit"
            continue
        # Candidate traces are deliberately not returned from the raw process
        # pool (one no-signal candidate can exceed 100 KiB).  Re-extract the
        # small number of strict target winners in the recipient process.
        target = _extract_external_candidate(
            store, record["assay"], options, min_mono_points=3
        )
        record["target"] = target
        equivalents = _equivalent_features(target, feature_rows, ppm)
        if len(equivalents) == 1:
            record["feature_id"] = int(equivalents[0]["feature_idx"])
            record["status"] = "accepted_matched_existing_feature"
            record["acceptance_family"] = "strict_external"
            record["acceptance_q_value"] = result.q_value
            continue
        if len(equivalents) > 1:
            record["status"] = "ambiguous_existing_features"
            continue
        residual_target = _extract_external_candidate(
            ledger.materialize(), target.assay, options
        )
        residual_gate = _candidate_gate_status(
            residual_target, plan.predicted_rt_sec, ppm=ppm,
            rt_tolerance_sec=rt_tolerance, min_isotope_cosine=min_cosine,
        )
        if residual_gate != "passed":
            record["status"] = "strict_residual_" + residual_gate
            continue
        start, _end = residual_target.segment_slice
        allocation = ledger.allocate_component(
            ("external_strict", plan.observation.ion_key),
            residual_target.traces,
            start,
            _candidate_segment_values(residual_target),
        )
        if not allocation.accepted:
            record["status"] = "strict_residual_" + allocation.status
            continue
        feature_id = next_feature_id
        next_feature_id += 1
        feature_row = _recovered_feature_row(residual_target, feature_id)
        feature_rows.append(feature_row)
        new_feature_rows.append(feature_row)
        rt = residual_target.traces[0].rt_sec[start:residual_target.segment_slice[1]]
        new_quant_rows.append(
            _quant_row(
                run.run_id, feature_id, FEATURE_ORIGIN_ALIGNED_EXTERNAL,
                "external_id", rt, _candidate_segment_values(residual_target),
                method=options["quant_method"], baseline=options["baseline"],
                quality_score=residual_target.isotope_cosine,
                isotope_cosine=residual_target.isotope_cosine,
                mass_error=residual_target.mono_mz_error_ppm,
                supporting_psm_count=0, supporting_ms2_count=0,
                extraction_q_value=result.q_value,
                quality_flags=(QUALITY_FLAG_BOUNDARY_TRUNCATED if residual_target.boundary_truncated else 0),
            )
        )
        record["feature_id"] = feature_id
        record["status"] = "accepted_new_external_feature"
        record["acceptance_family"] = "strict_external"
        record["acceptance_q_value"] = result.q_value

    # Weak recovery evaluates every not-yet-accepted assay on one frozen
    # residual snapshot, so q-values are not affected by processing order.
    weak_competitions = []
    residual_snapshot = ledger.materialize()
    weak_started = time.monotonic()
    weak_items = []
    if weak_enabled:
        for record in evaluated:
            if record["acceptance_family"] is not None:
                continue
            plan = record["plan"]
            raw_target = record["raw_weak_target"]
            raw_decoy = record["raw_weak_decoy"]
            raw_target_gate = raw_target["weak_gate"]
            raw_decoy_gate = raw_decoy["weak_gate"]
            # Strict external features have now been allocated, including
            # claims reconstructed for a resumed Project run.  Raw footprints
            # must be compared against this current ownership state rather
            # than the stale state seen by spawned extraction workers.
            target_overlap = _weak_raw_overlap(raw_target, ledger)
            decoy_overlap = _weak_raw_overlap(raw_decoy, ledger)
            record.update({
                "weak_raw_target": raw_target, "weak_raw_decoy": raw_decoy,
                "weak_target": None, "weak_decoy": None,
                "weak_raw_target_gate": raw_target_gate,
                "weak_raw_decoy_gate": raw_decoy_gate,
                "weak_target_gate": None,
                "weak_decoy_gate": None,
                "weak_target_overlap": target_overlap,
                "weak_decoy_overlap": decoy_overlap,
                "weak_residual_target_evaluated": False,
                "weak_residual_decoy_evaluated": False,
            })
            weak_items.append(
                (
                    record,
                    raw_target_gate == "passed"
                    and target_overlap is not None
                    and target_overlap <= weak_overlap_max,
                    raw_decoy_gate == "passed"
                    and decoy_overlap is not None
                    and decoy_overlap <= weak_overlap_max,
                )
            )

        active_weak_items = [
            item for item in weak_items if item[1] or item[2]
        ]
        residual_pairs = _parallel_extract(
            _extract_external_residual_pair,
            [
                (
                    residual_snapshot,
                    record["assay"],
                    record["decoy_assay"],
                    options,
                    extract_target,
                    extract_decoy,
                )
                for record, extract_target, extract_decoy in active_weak_items
            ],
            workers,
        )
        logger.info(
            "External-ID run %s: weak residual extraction target=%d decoy=%d workers=%d",
            run.run_id,
            sum(bool(value[1]) for value in weak_items),
            sum(bool(value[2]) for value in weak_items),
            max(1, int(workers)),
        )
        for (record, _extract_target, _extract_decoy), (residual_target, residual_decoy) in zip(
            active_weak_items, residual_pairs
        ):
            plan = record["plan"]
            if residual_target is not None:
                record["weak_target"] = residual_target
                record["weak_residual_target_evaluated"] = True
                record["weak_target_gate"] = _weak_candidate_gate_status(
                    residual_target, plan.predicted_rt_sec, ppm=ppm,
                    rt_tolerance_sec=rt_tolerance,
                    min_isotope_cosine=min_cosine,
                )
            if residual_decoy is not None:
                record["weak_decoy"] = residual_decoy
                record["weak_residual_decoy_evaluated"] = True
                record["weak_decoy_gate"] = _weak_candidate_gate_status(
                    residual_decoy, plan.predicted_rt_sec, ppm=ppm,
                    rt_tolerance_sec=rt_tolerance,
                    min_isotope_cosine=min_cosine,
                )
            target_score = (
                _weak_candidate_score(
                    residual_target, plan.predicted_rt_sec, ppm=ppm,
                    rt_tolerance_sec=rt_tolerance,
                    min_isotope_cosine=min_cosine,
                )
                if residual_target is not None else None
            )
            decoy_score = (
                _weak_candidate_score(
                    residual_decoy, plan.predicted_rt_sec, ppm=ppm,
                    rt_tolerance_sec=rt_tolerance,
                    min_isotope_cosine=min_cosine,
                )
                if residual_decoy is not None else None
            )
            weak_competitions.append(
                TargetDecoyCompetition(
                    plan.observation.ion_key, target_score, decoy_score
                )
            )
        active_records = {id(record) for record, _target, _decoy in active_weak_items}
        for record, _extract_target, _extract_decoy in weak_items:
            if id(record) not in active_records:
                weak_competitions.append(
                    TargetDecoyCompetition(record["plan"].observation.ion_key, None, None)
                )
    weak_extract_sec = time.monotonic() - weak_started

    weak_results = {item.seed_id: item for item in target_decoy_q_values(weak_competitions)}
    for record in sorted(evaluated, key=lambda value: value["plan"].observation.ion_key):
        plan = record["plan"]
        weak_result = weak_results.get(plan.observation.ion_key)
        record["weak_result"] = weak_result
        if weak_result is None:
            continue
        if record.get("weak_raw_target_gate") != "passed":
            record["status"] = "weak_target_raw_" + record.get(
                "weak_raw_target_gate", "not_evaluated"
            )
            continue
        if (
            record.get("weak_target_overlap") is not None
            and record["weak_target_overlap"] > weak_overlap_max
        ):
            record["status"] = "weak_target_overlap_above_limit"
            continue
        if weak_result.winner != "target":
            record["status"] = "weak_decoy_winner" if weak_result.winner == "decoy" else "weak_target_" + record["weak_target_gate"]
            continue
        if weak_result.q_value > weak_q_max:
            record["status"] = "weak_target_q_value_above_limit"
            continue
        target = record["weak_target"]
        if target is None:
            record["status"] = "weak_target_residual_not_evaluated"
            continue
        equivalents = _equivalent_features(target, feature_rows, ppm)
        if len(equivalents) == 1:
            record["feature_id"] = int(equivalents[0]["feature_idx"])
            record["status"] = "accepted_matched_existing_feature"
            record["acceptance_family"] = "weak_external"
            record["acceptance_q_value"] = weak_result.q_value
            continue
        if len(equivalents) > 1:
            record["status"] = "weak_ambiguous_existing_features"
            continue
        start, _end = target.segment_slice
        allocation = ledger.allocate_component(
            ("external_weak", plan.observation.ion_key), target.traces, start,
            _candidate_segment_values(target),
        )
        if not allocation.accepted:
            record["status"] = "weak_residual_" + allocation.status
            continue
        feature_id = next_feature_id
        next_feature_id += 1
        feature_row = _recovered_feature_row(target, feature_id)
        feature_rows.append(feature_row)
        new_feature_rows.append(feature_row)
        rt = target.traces[0].rt_sec[start:target.segment_slice[1]]
        flags = QUALITY_FLAG_WEAK_EXTERNAL_FEATURE | QUALITY_FLAG_TWO_POINT_QUANT
        if target.boundary_truncated:
            flags |= QUALITY_FLAG_BOUNDARY_TRUNCATED
        new_quant_rows.append(
            _quant_row(
                run.run_id, feature_id, FEATURE_ORIGIN_ALIGNED_EXTERNAL_WEAK,
                "external_id_weak", rt, _candidate_segment_values(target),
                method=options["quant_method"], baseline=options["baseline"],
                quality_score=target.isotope_cosine,
                isotope_cosine=target.isotope_cosine,
                mass_error=target.mono_mz_error_ppm,
                supporting_psm_count=0, supporting_ms2_count=0,
                extraction_q_value=weak_result.q_value, quality_flags=flags,
            )
        )
        record["feature_id"] = feature_id
        record["status"] = "accepted_new_weak_external_feature"
        record["acceptance_family"] = "weak_external"
        record["acceptance_q_value"] = weak_result.q_value

    for record in sorted(evaluated, key=lambda value: value["plan"].observation.ion_key):
        plan = record["plan"]
        target = record.get("target")
        raw_target = record["raw_target"]
        raw_decoy = record["raw_decoy"]
        result = record["strict_result"]
        raw_weak_target = record["raw_weak_target"]
        raw_weak_decoy = record["raw_weak_decoy"]
        weak_target = record.get("weak_target")
        weak_decoy = record.get("weak_decoy")
        weak_result = record.get("weak_result")
        status = record["status"]
        counts[status] += 1
        evidence_rows.append({
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
                "decoy_neutral_shift": record["shift"],
                "target_score": result.target_score,
                "decoy_score": result.decoy_score,
                "target_extraction_status": raw_target["status"],
                "target_gate_status": record["target_gate"],
                "decoy_extraction_status": raw_decoy["status"],
                "decoy_gate_status": record["decoy_gate"],
                "competition_winner": result.winner,
                "extraction_q_value": result.q_value,
                "status": status,
                "feature_id": record["feature_id"],
                "target_mono_points": raw_target["mono_points"],
                "target_isotope_cosine": raw_target["isotope_cosine"],
                "target_mass_error_ppm": raw_target["mass_error_ppm"],
                "target_rt_error_sec": None if raw_target["rt_apex_sec"] is None else abs(raw_target["rt_apex_sec"] - plan.predicted_rt_sec),
                "weak_raw_target_extraction_status": raw_weak_target["status"],
                "weak_raw_target_gate_status": raw_weak_target["weak_gate"],
                "weak_raw_decoy_extraction_status": raw_weak_decoy["status"],
                "weak_raw_decoy_gate_status": raw_weak_decoy["weak_gate"],
                "weak_raw_target_mono_points": raw_weak_target["mono_points"],
                "weak_raw_target_secondary_channels": raw_weak_target.get("secondary_channels"),
                "weak_raw_target_isotope_cosine": raw_weak_target["isotope_cosine"],
                "weak_raw_target_mass_error_ppm": raw_weak_target["mass_error_ppm"],
                "weak_raw_target_rt_error_sec": (
                    None if raw_weak_target["rt_apex_sec"] is None else
                    abs(raw_weak_target["rt_apex_sec"] - plan.predicted_rt_sec)
                ),
                "weak_raw_target_overlap_fraction": record.get("weak_target_overlap"),
                "weak_raw_decoy_overlap_fraction": record.get("weak_decoy_overlap"),
                "weak_residual_target_evaluated": bool(
                    record.get("weak_residual_target_evaluated", False)
                ),
                "weak_residual_decoy_evaluated": bool(
                    record.get("weak_residual_decoy_evaluated", False)
                ),
                "weak_target_extraction_status": (
                    None if weak_target is None else weak_target.status
                ),
                "weak_target_gate_status": record.get("weak_target_gate"),
                "weak_decoy_extraction_status": (
                    None if weak_decoy is None else weak_decoy.status
                ),
                "weak_decoy_gate_status": record.get("weak_decoy_gate"),
                "weak_target_score": None if weak_result is None else weak_result.target_score,
                "weak_decoy_score": None if weak_result is None else weak_result.decoy_score,
                "weak_competition_winner": None if weak_result is None else weak_result.winner,
                "weak_extraction_q_value": None if weak_result is None else weak_result.q_value,
                "weak_target_mono_points": (
                    None if weak_target is None else weak_target.mono_point_count
                ),
                "weak_target_secondary_channels": (
                    None if weak_target is None else _candidate_secondary_channel_count(weak_target, 2)
                ),
                "weak_target_isotope_cosine": (
                    None if weak_target is None else weak_target.isotope_cosine
                ),
                "weak_target_mass_error_ppm": (
                    None if weak_target is None else weak_target.mono_mz_error_ppm
                ),
                "weak_target_rt_error_sec": (
                    None if weak_target is None or weak_target.rt_apex_sec is None else
                    abs(weak_target.rt_apex_sec - plan.predicted_rt_sec)
                ),
                "weak_overlap_fraction": record.get("weak_target_overlap"),
                "acceptance_family": record["acceptance_family"],
                "acceptance_q_value": record["acceptance_q_value"],
            })

    summary = {
        "planned_assay_count": len(plans),
        "evaluated_assay_count": len(evaluated),
        "new_external_feature_count": len(new_feature_rows),
        "new_strict_external_feature_count": sum(
            row["status"] == "accepted_new_external_feature"
            for row in evaluated
        ),
        "new_weak_external_feature_count": sum(
            row["status"] == "accepted_new_weak_external_feature"
            for row in evaluated
        ),
        "status_counts": dict(sorted(counts.items())),
        "performance": {
            "workers": int(workers),
            "raw_pair_count": len(raw_pairs),
            "weak_residual_pair_count": len(weak_items),
            "weak_residual_target_count": sum(
                bool(value[1]) for value in weak_items
            ),
            "weak_residual_decoy_count": sum(
                bool(value[2]) for value in weak_items
            ),
            "raw_extract_sec": raw_extract_sec,
            "weak_extract_sec": weak_extract_sec,
        },
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
