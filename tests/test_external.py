import csv
import os
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biosaur2 import external
from biosaur2 import external_alignment
from biosaur2 import external_observations
from biosaur2.alignment import RTAlignmentModel
from biosaur2.chemistry import isotope_library, parse_peptidoform
from biosaur2.external import (
    ExternalObservation,
    ExternalPlan,
    alignment_group_for_run,
    build_alignment_models,
    choose_group_reference_runs,
    plan_external_assays,
    run_external_recipient,
)
from biosaur2.raw_ms1 import RawMS1StoreBuilder, save_raw_ms1_cache
from biosaur2.residual import ResidualMS1Ledger, save_residual_ownership_cache
from biosaur2.schema import compact_schemas
from biosaur2.project import _run_external_stage


def _run(run_id, group="g"):
    return SimpleNamespace(
        run_id=run_id,
        metadata={"alignment_group": group, "fraction": "", "batch": ""},
    )


def _observation(run_id, ion_key, rt, assay_id=1):
    return ExternalObservation(
        run_id=run_id,
        ion_key=ion_key,
        canonical_peptidoform="PEPTIDE",
        charge=2,
        faims_cv=None,
        rt_apex_sec=rt,
        q_value=0.001,
        assay_id=assay_id,
        psm_id="psm-%s-%d" % (run_id, assay_id),
    )


def test_alignment_and_external_planning_stay_inside_group():
    runs = [_run("a"), _run("b"), _run("outside", "other")]
    observations = {
        "a": tuple(_observation("a", "shared-%d" % i, 100 + i) for i in range(5))
        + (_observation("a", "donor-only", 150, 20),),
        "b": tuple(_observation("b", "shared-%d" % i, 110 + i) for i in range(5)),
        "outside": (_observation("outside", "outside-only", 200),),
    }
    models = build_alignment_models(runs, observations, min_anchors=5)
    assert models[("a", "b")][1].status == "accepted"
    assert ("outside", "b") not in models
    plans = plan_external_assays(runs, observations, models)
    assert [plan.observation.ion_key for plan in plans["b"]] == ["donor-only"]
    assert plans["b"][0].predicted_rt_sec == 160
    assert alignment_group_for_run(runs[0]) == "explicit:g"
    assert choose_group_reference_runs(runs, observations) == {
        "explicit:g": "a",
        "explicit:other": "outside",
    }


def test_reference_star_uses_two_edges_per_nonreference_run(monkeypatch):
    def fast_fit(source_run, target_run, anchors):
        return RTAlignmentModel(
            source_run,
            target_run,
            "median_shift",
            len(anchors),
            len(anchors),
            (),
            (),
            1.0,
            0.0,
            0.0,
            "accepted",
        )

    monkeypatch.setattr(external_alignment, "fit_rt_alignment", fast_fit)
    run_count = 1808
    runs = [_run("r%04d" % index) for index in range(run_count)]
    observations = {}
    for index, run in enumerate(runs):
        observations[run.run_id] = tuple(
            _observation(run.run_id, "shared-%d" % anchor, 100.0 + anchor + index)
            for anchor in range(5)
        )
    models = build_alignment_models(runs, observations, min_anchors=5)
    assert len(models) == 2 * (run_count - 1)
    assert len(models) <= 8 * run_count


def test_alignment_forest_retries_another_reference_after_failed_edge(monkeypatch):
    def fit(source_run, target_run, anchors):
        if {source_run, target_run} == {"a", "c"}:
            return RTAlignmentModel(
                source_run, target_run, "none", len(anchors), 0, (), (),
                1.0, 0.0, None, "residual_mad_exceeds_limit",
            )
        intercept = 10.0 if source_run < target_run else -10.0
        return RTAlignmentModel(
            source_run, target_run, "median_shift", len(anchors), len(anchors),
            (), (), 1.0, intercept, 0.0, "accepted",
        )

    monkeypatch.setattr(external_alignment, "fit_rt_alignment", fit)
    runs = [_run("a"), _run("b"), _run("c")]
    observations = {
        "a": tuple(_observation("a", "ab-%d" % index, 80 + index) for index in range(5))
        + tuple(_observation("a", "ac-%d" % index, 90 + index, 10 + index) for index in range(5))
        + (_observation("a", "donor-only", 150, 30),),
        "b": tuple(_observation("b", "ab-%d" % index, 90 + index) for index in range(5))
        + tuple(_observation("b", "bc-%d" % index, 100 + index, 10 + index) for index in range(5)),
        "c": tuple(_observation("c", "ac-%d" % index, 100 + index) for index in range(5))
        + tuple(_observation("c", "bc-%d" % index, 110 + index, 10 + index) for index in range(5)),
    }
    forest = build_alignment_models(runs, observations, min_anchors=5)
    assert forest[("a", "c")][1].status == "residual_mad_exceeds_limit"
    assert forest.component_by_run["a"] == forest.component_by_run["c"]
    donor = next(
        plan for plan in plan_external_assays(runs, observations, forest)["c"]
        if plan.observation.ion_key == "donor-only"
    )
    assert donor.alignment.method == "reference_forest"
    assert donor.predicted_rt_sec == pytest.approx(170.0)


def test_reference_star_downsamples_anchors_and_composes_prediction():
    runs = [_run("reference"), _run("source"), _run("recipient")]
    shared = tuple("shared-%03d" % index for index in range(32))
    observations = {
        "reference": tuple(
            _observation("reference", key, 100.0 + index)
            for index, key in enumerate(shared)
        ),
        "source": tuple(
            _observation("source", key, 110.0 + index)
            for index, key in enumerate(shared)
        ) + (_observation("source", "donor-only", 160.0, 99),),
        "recipient": tuple(
            _observation("recipient", key, 120.0 + index)
            for index, key in enumerate(shared)
        ),
    }
    models = build_alignment_models(
        runs, observations, min_anchors=5, max_anchors=7
    )
    assert len(models) == 4
    assert all(model.anchor_count <= 7 for _group, model in models.values())
    plans = plan_external_assays(runs, observations, models)
    donor = next(
        plan for plan in plans["recipient"]
        if plan.observation.ion_key == "donor-only"
    )
    assert donor.source_run == "source"
    assert donor.alignment.method == "reference_star"
    assert donor.predicted_rt_sec == pytest.approx(170.0)


def test_external_planning_prefers_zero_mad_donor_path():
    def model(source, target, mad):
        return RTAlignmentModel(
            source, target, "median_shift", 5, 5, (), (), 1.0, 0.0, mad,
            "accepted",
        )

    runs = [_run("a"), _run("reference"), _run("target"), _run("zero")]
    group = "explicit:g|component=reference"
    models = {
        ("zero", "reference"): ("explicit:g", model("zero", "reference", 0.0)),
        ("reference", "zero"): ("explicit:g", model("reference", "zero", 0.0)),
        ("a", "reference"): ("explicit:g", model("a", "reference", 1.0)),
        ("reference", "a"): ("explicit:g", model("reference", "a", 1.0)),
        ("reference", "target"): ("explicit:g", model("reference", "target", 0.0)),
        ("target", "reference"): ("explicit:g", model("target", "reference", 0.0)),
    }
    forest = external_alignment.AlignmentForest(
        models,
        {run.run_id: group for run in runs},
        {group: "reference"},
        {"reference": None, "zero": "reference", "a": "reference", "target": "reference"},
    )
    observations = {
        "zero": (_observation("zero", "donor-only", 100.0),),
        "a": (_observation("a", "donor-only", 100.0),),
        "reference": (),
        "target": (),
    }

    plans = plan_external_assays(runs, observations, forest)
    target_plan = next(
        plan for plan in plans["target"] if plan.observation.ion_key == "donor-only"
    )
    assert target_plan.source_run == "zero"


def test_external_observation_sidecar_reuses_published_rows(tmp_path, monkeypatch):
    features = tmp_path / "features.parquet"
    identifications = tmp_path / "identifications.parquet"
    mzml = tmp_path / "sidecar.mzML"
    features.write_bytes(b"feature-output")
    identifications.write_bytes(b"identification-output")
    mzml.write_bytes(b"raw-input")
    paths = {
        "features": str(features),
        "identifications": str(identifications),
        "external_observations": str(tmp_path / "external" / "observations-v2.parquet"),
    }
    assay_rows = [{
        "assay_id": 3,
        "assay_conflict_status": "unique",
        "assay_faims_cv": None,
        "canonical_peptidoform": "PEPTIDE",
        "assay_charge": 2,
        "q_value": 0.001,
        "psm_id": "psm-3",
    }]
    feature_rows = [{
        "feature_idx": 2,
        "rt_apex_sec": 123.0,
        "quant_value": 50.0,
        "ms2_events": [{"assay_id": 3, "association_tier": "direct_id"}],
    }]

    def read_rows(_path, columns=None, table_name=None):
        return assay_rows if table_name == "identifications" else feature_rows

    monkeypatch.setattr(external, "_read_rows", read_rows)
    run = _run("sidecar")
    run.mzml_path = mzml
    run.psm_path = None
    first = external.read_external_observations(run, paths)
    assert len(first) == 1
    assert os.path.isfile(paths["external_observations"])

    def must_not_read(*_args, **_kwargs):
        raise AssertionError("sidecar should be used")

    monkeypatch.setattr(external, "_read_rows", must_not_read)
    assert external.read_external_observations(run, paths) == first


def test_observation_sidecar_treats_empty_psm_path_as_missing(tmp_path):
    mzml = tmp_path / "run.mzML"
    mzml.write_bytes(b"mzml")
    paths = {"external_observations": str(tmp_path / "observations.parquet")}
    observation = _observation("run", "ion", 123.0)

    external_observations.write_observation_sidecar(
        mzml, "", paths["external_observations"], (observation,)
    )
    run = SimpleNamespace(run_id="run", mzml_path=mzml, psm_path=None)
    assert external_observations.read_observation_sidecar(run, paths) == (observation,)


def test_feature_equivalence_index_builds_once_and_stays_sorted():
    rows = [
        {"feature_idx": 2, "charge": 2, "FAIMS": None, "mz": 501.0, "rtStart": 1.0, "rtEnd": 2.0},
        {"feature_idx": 1, "charge": 2, "FAIMS": None, "mz": 500.0, "rtStart": 1.0, "rtEnd": 2.0},
        {"feature_idx": 3, "charge": 3, "FAIMS": None, "mz": 500.0, "rtStart": 1.0, "rtEnd": 2.0},
    ]
    index = external._FeatureEquivalenceIndex(rows)
    candidate = SimpleNamespace(
        assay=SimpleNamespace(
            isotope_peaks=(SimpleNamespace(mz=500.0),), charge=2, faims_cv=None
        ),
        rt_start_sec=60.0,
        rt_end_sec=120.0,
    )
    assert [row["feature_idx"] for row in external._equivalent_features(candidate, index, 10.0)] == [1]
    index.add({"feature_idx": 4, "charge": 2, "FAIMS": None, "mz": 500.001, "rtStart": 1.0, "rtEnd": 2.0})
    assert [row["feature_idx"] for row in external._equivalent_features(candidate, index, 10.0)] == [1, 4]


def test_feature_equivalence_index_preserves_faims_tolerance_and_nonfinite_rules():
    rows = [
        {"feature_idx": 1, "charge": 2, "FAIMS": -44.9999999, "mz": 500.0, "rtStart": 1.0, "rtEnd": 2.0},
        {"feature_idx": 2, "charge": 2, "FAIMS": None, "mz": 500.0, "rtStart": 1.0, "rtEnd": 2.0},
        {"feature_idx": 3, "charge": 2, "FAIMS": float("inf"), "mz": 500.0, "rtStart": 1.0, "rtEnd": 2.0},
        {"feature_idx": 4, "charge": 2, "FAIMS": float("nan"), "mz": 500.0, "rtStart": 1.0, "rtEnd": 2.0},
    ]
    index = external._FeatureEquivalenceIndex(rows)

    def matched(faims):
        candidate = SimpleNamespace(
            assay=SimpleNamespace(
                isotope_peaks=(SimpleNamespace(mz=500.0),), charge=2, faims_cv=faims
            ),
            rt_start_sec=60.0,
            rt_end_sec=120.0,
        )
        return [row["feature_idx"] for row in external._equivalent_features(candidate, index, 10.0)]

    assert matched(-45.0) == [1]
    assert matched(None) == [2]
    assert matched(float("inf")) == [3]
    assert matched(float("nan")) == []


def test_observation_sidecar_uses_upstream_provenance_not_output_mtime(tmp_path, monkeypatch):
    mzml = tmp_path / "run.mzML"
    psm = tmp_path / "run.psms.tsv"
    database = tmp_path / "run.duckdb"
    mzml.write_bytes(b"mzml")
    psm.write_text("psm", encoding="utf-8")
    database.write_bytes(b"single-run-output")
    paths = {
        "features": str(database),
        "identifications": str(database),
        "external_observations": str(tmp_path / "observations-v2.parquet"),
    }
    run = SimpleNamespace(run_id="run", mzml_path=mzml, psm_path=psm)
    observation = _observation("run", "ion", 123.0)
    external_observations.write_observation_sidecar(
        mzml, psm, paths["external_observations"], (observation,)
    )
    database.write_bytes(b"external stage changed this database")
    monkeypatch.setattr(external, "_read_rows", lambda *_args, **_kwargs: pytest.fail("must use sidecar"))
    assert external.read_external_observations(run, paths) == (observation,)
    mzml.write_bytes(b"changed mzml source")
    assert external_observations.read_observation_sidecar(run, paths) is None


def test_observations_from_hybrid_rows_are_direct_unique_positive_only():
    observations = external_observations.observations_from_hybrid_rows(
        "run",
        [
            {"feature_id": 1, "rt_apex_sec": 100.0, "quant_value": 10.0},
            {"feature_id": 2, "rt_apex_sec": 101.0, "quant_value": 0.0},
        ],
        [
            {"assay_id": 1, "feature_id": 1, "association_tier": "direct_id"},
            {"assay_id": 4, "feature_id": 1, "association_tier": "direct_id"},
            {"assay_id": 2, "feature_id": 2, "association_tier": "direct_id"},
            {"assay_id": 3, "feature_id": 1, "association_tier": "generic_ms2"},
        ],
        [
            {"assay_id": 1, "canonical_peptidoform": "PEPTIDE", "charge": 2, "faims_cv": None, "q_value": 0.01, "psm_id": "late", "conflict_status": "unique"},
            {"assay_id": 4, "canonical_peptidoform": "PEPTIDE", "charge": 2, "faims_cv": None, "q_value": 0.001, "psm_id": "best", "conflict_status": "unique"},
            {"assay_id": 2, "canonical_peptidoform": "OTHER", "charge": 2, "faims_cv": None, "q_value": 0.001, "psm_id": "zero", "conflict_status": "unique"},
            {"assay_id": 3, "canonical_peptidoform": "OTHER", "charge": 2, "faims_cv": None, "q_value": 0.001, "psm_id": "generic", "conflict_status": "unique"},
        ],
    )
    assert len(observations) == 1
    assert observations[0].psm_id == "best"
    assert observations[0].rt_apex_sec == 100.0


def _write_empty_outputs(paths):
    schemas = compact_schemas()
    pq.write_table(
        pa.Table.from_pylist([], schema=schemas["hybrid_features"]),
        paths["features"],
    )


def test_external_target_decoy_extraction_adds_one_feature_idempotently(tmp_path):
    source = tmp_path / "recipient.mzML"
    source.write_bytes(b"synthetic source fingerprint")
    cache = tmp_path / "raw-cache"
    peptidoform = parse_peptidoform("PEPTIDE")
    peaks = tuple(
        peak
        for peak in isotope_library(peptidoform.formula, 2, max_isotopes=6)
        if peak.isotope_index == 0 or peak.relative_abundance >= 0.01
    )
    builder = RawMS1StoreBuilder()
    rt_values = np.arange(94.0, 108.0, 2.0)
    profile = np.asarray([1, 4, 12, 20, 12, 4, 1], dtype=float)
    for index, (rt, scale) in enumerate(zip(rt_values, profile)):
        mz = np.asarray([peak.mz for peak in peaks], dtype=float)
        intensity = np.asarray(
            [scale * peak.relative_abundance * 1000 for peak in peaks],
            dtype=float,
        )
        builder.append(
            mz,
            intensity,
            source_scan_index=index,
            scan_number=1000 + index,
            rt_sec=rt,
            faims_cv=None,
        )
    raw_store = builder.finalize()
    save_raw_ms1_cache(raw_store, cache, source)
    ownership = save_residual_ownership_cache(
        ResidualMS1Ledger(raw_store), tmp_path / "residual-ownership", source
    )

    paths = {
        "features": str(tmp_path / "features.parquet"),
        "external_evidence": str(tmp_path / "external.parquet"),
        "raw_ms1_cache": str(cache),
        "residual_ownership_cache": str(ownership),
    }
    _write_empty_outputs(paths)
    observation = _observation("donor", "PEPTIDE-ion", 90)
    model = RTAlignmentModel(
        "donor", "recipient", "median_shift", 10, 10, (), (), 1.0, 10.0, 0.0, "accepted"
    )
    plan = ExternalPlan("recipient", "donor", "explicit:g", observation, 100.0, model)
    task = {
        "run": SimpleNamespace(run_id="recipient", mzml_path=source),
        "paths": paths,
        "plans": (plan,),
        "options": {
            "ppm": 8.0,
            "rt_tolerance_sec": 20.0,
            "min_isotope_cosine": 0.8,
            # One target has conservative (+1) q=1.0; permit it in this
            # focused mechanics test rather than weakening production FDR.
            "q_value_max": 1.0,
            "quant_method": "envelope_area",
            "baseline": "none",
        },
    }
    first = run_external_recipient(task, workers=2)
    assert first["new_external_feature_count"] == 1
    assert pq.ParquetFile(paths["features"]).metadata.num_rows == 1
    feature = pq.read_table(paths["features"]).to_pylist()[0]
    assert feature["feature_origin"] == "aligned_external"
    assert feature["quant_value"] > 0
    evidence = pq.read_table(paths["external_evidence"]).to_pylist()
    assert evidence[0]["status"] == "accepted_new_external_feature"
    assert evidence[0]["feature_id"] == 1
    assert evidence[0]["target_mono_points"] == 7
    metadata = pq.read_schema(paths["external_evidence"]).metadata
    assert metadata[b"biosaur2_external_evidence_schema_version"] == b"3"

    second = run_external_recipient(task, workers=2)
    assert second["new_external_feature_count"] == 0
    assert pq.ParquetFile(paths["features"]).metadata.num_rows == 1
    evidence = pq.read_table(paths["external_evidence"]).to_pylist()
    assert evidence[0]["status"] == "accepted_matched_existing_feature"
    assert evidence[0]["feature_id"] == 1
    assert pq.read_table(paths["features"]).to_pylist()[0] == feature
    with pytest.raises(RuntimeError, match="cannot deterministically restore"):
        run_external_recipient({**task, "plans": ()}, workers=2)
    assert pq.read_table(paths["features"]).to_pylist()[0] == feature

    tsv_paths = {**paths, "external_evidence": str(tmp_path / "external.tsv")}
    assert run_external_recipient(
        {**task, "paths": tsv_paths}, workers=2
    )["new_external_feature_count"] == 0
    with open(tsv_paths["external_evidence"], newline="", encoding="utf-8") as handle:
        tsv_columns = next(csv.reader(handle, delimiter="\t"))

    import duckdb

    database = tmp_path / "recipient.biosaur2.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "CREATE TABLE features AS SELECT * FROM read_parquet(?)",
            [paths["features"]],
        )
    duckdb_paths = {
        **paths,
        "features": str(database),
        "external_evidence": str(database),
        "run_output": str(database),
        "format": "duckdb",
    }
    assert run_external_recipient(
        {**task, "paths": duckdb_paths}, workers=2
    )["new_external_feature_count"] == 0
    with duckdb.connect(str(database), read_only=True) as connection:
        duckdb_columns = [
            row[0] for row in connection.execute(
                "DESCRIBE external_id_evidence"
            ).fetchall()
        ]
    parquet_columns = pq.read_table(paths["external_evidence"]).schema.names
    assert tsv_columns == parquet_columns == duckdb_columns


def test_raw_workers_cap_native_threads_and_restore_parent_environment(monkeypatch):
    seen = {}

    class FailingExecutor:
        def __init__(self, *args, **kwargs):
            seen.update({name: os.environ.get(name) for name in external._NATIVE_THREAD_ENVIRONMENT})

        def __enter__(self):
            raise RuntimeError("stop before worker startup")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(external, "ProcessPoolExecutor", FailingExecutor)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    monkeypatch.setenv("ARROW_IO_THREADS", "9")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    with pytest.raises(RuntimeError, match="stop before worker startup"):
        external._parallel_raw_extract(
            [(None, None, 0.0), (None, None, 0.0)], 2,
            store=None, raw_ms1_cache="unused", mzml_path="unused", options={},
        )
    assert set(seen.values()) == {"1"}
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7"
    assert os.environ["ARROW_IO_THREADS"] == "9"
    assert "OMP_NUM_THREADS" not in os.environ


def test_external_weak_recovery_accepts_two_point_mono_and_secondary(tmp_path):
    source = tmp_path / "recipient.mzML"
    source.write_bytes(b"synthetic source fingerprint")
    cache = tmp_path / "raw-cache"
    peptidoform = parse_peptidoform("PEPTIDE")
    peaks = tuple(
        peak
        for peak in isotope_library(peptidoform.formula, 2, max_isotopes=6)
        if peak.isotope_index == 0 or peak.relative_abundance >= 0.01
    )
    builder = RawMS1StoreBuilder()
    for index, (rt, scale) in enumerate(((98.0, 10.0), (100.0, 20.0))):
        builder.append(
            np.asarray([peak.mz for peak in peaks], dtype=float),
            np.asarray(
                [scale * peak.relative_abundance * 1000 for peak in peaks],
                dtype=float,
            ),
            source_scan_index=index,
            scan_number=1000 + index,
            rt_sec=rt,
            faims_cv=None,
        )
    save_raw_ms1_cache(builder.finalize(), cache, source)
    paths = {
        "features": str(tmp_path / "features.parquet"),
        "external_evidence": str(tmp_path / "external.parquet"),
        "raw_ms1_cache": str(cache),
    }
    _write_empty_outputs(paths)
    model = RTAlignmentModel(
        "donor", "recipient", "median_shift", 10, 10, (), (),
        1.0, 10.0, 0.0, "accepted",
    )
    plans = tuple(
        ExternalPlan(
            "recipient", "donor", "explicit:g",
            _observation("donor", "PEPTIDE-ion-%02d" % value, 90, value),
            100.0, model,
        )
        for value in range(1, 21)
    )
    result = run_external_recipient(
        {
            "run": SimpleNamespace(run_id="recipient", mzml_path=source),
            "paths": paths,
            "plans": plans,
            "options": {
                "ppm": 8.0,
                "rt_tolerance_sec": 20.0,
                "min_isotope_cosine": 0.8,
                "q_value_max": 0.01,
                "weak_feature": True,
                "weak_q_value_max": 0.05,
                "weak_overlap_max": 0.20,
                "quant_method": "envelope_area",
                "baseline": "none",
            },
        },
        workers=2,
    )
    assert result["new_strict_external_feature_count"] == 0
    assert result["new_weak_external_feature_count"] == 1
    feature = pq.read_table(paths["features"]).to_pylist()[0]
    assert feature["feature_origin"] == "aligned_external_weak"
    assert feature["confidence_tier"] == "external_id_weak"
    evidence = pq.read_table(paths["external_evidence"]).to_pylist()
    accepted = [row for row in evidence if row["status"] == "accepted_new_weak_external_feature"]
    assert len(accepted) == 1
    assert accepted[0]["weak_target_mono_points"] == 2
    assert accepted[0]["weak_target_secondary_channels"] >= 1
    assert accepted[0]["weak_extraction_q_value"] == 0.05
    assert accepted[0]["weak_residual_target_evaluated"] is True
    assert accepted[0]["weak_raw_target_overlap_fraction"] == accepted[0]["weak_overlap_fraction"]


def test_three_run_reference_star_weak_rescue_never_becomes_donor(tmp_path):
    source_path = tmp_path / "recipient.mzML"
    source_path.write_bytes(b"synthetic source fingerprint")
    cache = tmp_path / "raw-cache"
    peptidoform = parse_peptidoform("PEPTIDE")
    peaks = tuple(
        peak
        for peak in isotope_library(peptidoform.formula, 2, max_isotopes=6)
        if peak.isotope_index == 0 or peak.relative_abundance >= 0.01
    )
    builder = RawMS1StoreBuilder()
    for index, (rt, scale) in enumerate(((108.0, 10.0), (110.0, 20.0))):
        builder.append(
            np.asarray([peak.mz for peak in peaks], dtype=float),
            np.asarray(
                [scale * peak.relative_abundance * 1000 for peak in peaks],
                dtype=float,
            ),
            source_scan_index=index,
            scan_number=1000 + index,
            rt_sec=rt,
            faims_cv=None,
        )
    save_raw_ms1_cache(builder.finalize(), cache, source_path)

    runs = [_run("source"), _run("reference"), _run("recipient")]
    shared = tuple("anchor-%d" % index for index in range(6))
    source_donors = tuple(
        _observation("source", "donor-%02d" % index, 90.0, 50 + index)
        for index in range(20)
    )
    reference_donors = tuple(
        ExternalObservation(
            **{
                **donor.__dict__,
                "run_id": "reference",
                "rt_apex_sec": 100.0,
                "q_value": 0.005,
                "psm_id": "reference-donor-%d" % index,
            }
        )
        for index, donor in enumerate(source_donors)
    )
    observations = {
        "source": tuple(_observation("source", key, 80.0 + index, index + 1) for index, key in enumerate(shared)) + source_donors,
        "reference": tuple(_observation("reference", key, 90.0 + index, index + 1) for index, key in enumerate(shared)) + reference_donors,
        "recipient": tuple(_observation("recipient", key, 100.0 + index, index + 1) for index, key in enumerate(shared)),
    }
    forest = build_alignment_models(runs, observations, min_anchors=5)
    plans = plan_external_assays(runs, observations, forest)["recipient"]
    assert len(plans) == len(source_donors)
    assert all(value.source_run == "source" for value in plans)
    assert all(value.alignment.method == "reference_star" for value in plans)
    assert all(value.predicted_rt_sec == pytest.approx(110.0) for value in plans)

    paths = {
        "features": str(tmp_path / "features.parquet"),
        "identifications": str(tmp_path / "identifications.parquet"),
        "external_evidence": str(tmp_path / "external.parquet"),
        "external_observations": str(tmp_path / "observations-v2.parquet"),
        "raw_ms1_cache": str(cache),
    }
    feature_rows = [
        {
            "feature_idx": index + 1,
            "charge": 2,
            "mz": 100.0 + index,
            "rtStart": 1.0,
            "rtEnd": 2.0,
            "FAIMS": None,
            "rt_apex_sec": 100.0 + index,
            "quant_value": 100.0,
            "ms2_events": [{"assay_id": index + 1, "association_tier": "direct_id"}],
        }
        for index in range(len(shared))
    ]
    identification_rows = [
        {
            "run_id": "recipient",
            "psm_id": "recipient-%d" % (index + 1),
            "mapping_status": "accepted",
            "q_value": 0.001,
            "canonical_peptidoform": "PEPTIDE",
            "assay_id": index + 1,
            "assay_charge": 2,
            "assay_faims_cv": None,
            "assay_conflict_status": "unique",
        }
        for index in range(len(shared))
    ]
    pq.write_table(
        pa.Table.from_pylist(feature_rows, schema=compact_schemas()["hybrid_features"]),
        paths["features"],
    )
    pq.write_table(
        pa.Table.from_pylist(identification_rows, schema=compact_schemas()["merged_identifications"]),
        paths["identifications"],
    )
    recipient = SimpleNamespace(
        run_id="recipient", mzml_path=source_path, psm_path=None
    )
    result = run_external_recipient(
        {
            "run": recipient,
            "paths": paths,
                "plans": plans,
            "options": {
                "ppm": 8.0,
                "rt_tolerance_sec": 20.0,
                "min_isotope_cosine": 0.8,
                "q_value_max": 0.01,
                "weak_feature": True,
                "weak_q_value_max": 0.05,
                "weak_overlap_max": 0.20,
                "quant_method": "envelope_area",
                "baseline": "none",
            },
        },
        workers=1,
    )
    assert result["new_weak_external_feature_count"] == 1
    features = pq.read_table(paths["features"]).to_pylist()
    assert features[-1]["feature_origin"] == "aligned_external_weak"
    donor_observations = external.read_external_observations(recipient, paths)
    assert len(donor_observations) == 1
    assert all(item.psm_id.startswith("recipient-") for item in donor_observations)


def test_project_external_stage_rescues_synthetic_three_run_sample(tmp_path):
    """Exercise Project alignment, planning, spawned extraction and publication."""

    peptidoform = parse_peptidoform("PEPTIDE")
    peaks = tuple(
        peak
        for peak in isotope_library(peptidoform.formula, 2, max_isotopes=6)
        if peak.isotope_index == 0 or peak.relative_abundance >= 0.01
    )
    runs = []
    results = {}
    paths_by_run = {}
    for run_id in ("source", "reference", "recipient"):
        mzml_path = tmp_path / (run_id + ".mzML")
        mzml_path.write_bytes((run_id + " synthetic source").encode())
        cache = tmp_path / (run_id + ".raw-cache")
        builder = RawMS1StoreBuilder()
        if run_id == "recipient":
            for index, (rt, scale) in enumerate(((108.0, 10.0), (110.0, 20.0))):
                builder.append(
                    np.asarray([peak.mz for peak in peaks], dtype=float),
                    np.asarray(
                        [scale * peak.relative_abundance * 1000 for peak in peaks],
                        dtype=float,
                    ),
                    source_scan_index=index,
                    scan_number=1000 + index,
                    rt_sec=rt,
                    faims_cv=None,
                )
        raw_store = builder.finalize()
        save_raw_ms1_cache(raw_store, cache, mzml_path)
        ownership = save_residual_ownership_cache(
            ResidualMS1Ledger(raw_store),
            tmp_path / (run_id + ".residual-ownership"),
            mzml_path,
        )
        paths = {
            "format": "parquet",
            "features": str(tmp_path / (run_id + ".features.parquet")),
            "identifications": str(tmp_path / (run_id + ".identifications.parquet")),
            "external_evidence": str(tmp_path / (run_id + ".external.parquet")),
            "external_observations": str(tmp_path / (run_id + ".observations-v2.parquet")),
            "raw_ms1_cache": str(cache),
            "strict_stage_cache": str(tmp_path / (run_id + ".strict-stage")),
            "candidate_cache": str(tmp_path / (run_id + ".candidates")),
            "residual_ownership_cache": str(ownership),
        }
        _write_empty_outputs(paths)
        run = SimpleNamespace(
            run_id=run_id,
            mzml_path=mzml_path,
            psm_path=None,
            metadata={"alignment_group": "synthetic", "fraction": "", "batch": ""},
        )
        runs.append(run)
        paths_by_run[run_id] = paths
        results[len(runs) - 1] = {
            "run_id": run_id,
            "status": "success",
            "paths": paths,
            "command": ["biosaur2", str(mzml_path)],
        }

    shared = tuple("anchor-%d" % index for index in range(6))
    source_donors = tuple(
        _observation("source", "donor-%02d" % index, 90.0, 50 + index)
        for index in range(20)
    )
    reference_donors = tuple(
        ExternalObservation(
            **{
                **donor.__dict__,
                "run_id": "reference",
                "rt_apex_sec": 100.0,
                "q_value": 0.005,
                "psm_id": "reference-donor-%d" % index,
            }
        )
        for index, donor in enumerate(source_donors)
    )
    observations = {
        "source": tuple(
            _observation("source", key, 80.0 + index, index + 1)
            for index, key in enumerate(shared)
        ) + source_donors,
        "reference": tuple(
            _observation("reference", key, 90.0 + index, index + 1)
            for index, key in enumerate(shared)
        ) + reference_donors,
        "recipient": tuple(
            _observation("recipient", key, 100.0 + index, index + 1)
            for index, key in enumerate(shared)
        ),
    }
    for run in runs:
        external_observations.write_observation_sidecar(
            run.mzml_path,
            None,
            paths_by_run[run.run_id]["external_observations"],
            observations[run.run_id],
        )

    stage = _run_external_stage(
        runs,
        results,
        {
            "workers": 1,
            "_effective_workers": 1,
            "_max_memory_bytes": 32 * 1024 ** 3,
            "quant_method": "envelope_area",
            "feature_baseline": "none",
            "external_ppm": 8.0,
            "external_rt_tolerance_sec": 20.0,
            "external_min_isotope_cosine": 0.8,
            "external_q_value_max": 0.01,
            "external_weak_feature": True,
            "external_weak_q_value_max": 0.05,
            "external_weak_overlap_max": 0.20,
        },
    )
    summary = stage["summaries"]["recipient"]
    assert summary["planned_assay_count"] == 20
    assert summary["new_weak_external_feature_count"] == 1
    recovered = pq.read_table(paths_by_run["recipient"]["features"]).to_pylist()
    assert recovered[0]["feature_origin"] == "aligned_external_weak"
    evidence = pq.read_table(paths_by_run["recipient"]["external_evidence"]).to_pylist()
    assert len([row for row in evidence if row["status"] == "accepted_new_weak_external_feature"]) == 1
    assert len(stage["alignment_models"]) == 4
    assert all(row["status"] == "accepted" for row in stage["alignment_models"])


def test_external_weak_recovery_rejects_component_claimed_by_existing_feature(tmp_path):
    source = tmp_path / "recipient.mzML"
    source.write_bytes(b"synthetic source fingerprint")
    cache = tmp_path / "raw-cache"
    peptidoform = parse_peptidoform("PEPTIDE")
    peaks = tuple(
        peak
        for peak in isotope_library(peptidoform.formula, 2, max_isotopes=6)
        if peak.isotope_index == 0 or peak.relative_abundance >= 0.01
    )
    builder = RawMS1StoreBuilder()
    for index, (rt, scale) in enumerate(((98.0, 10.0), (100.0, 20.0))):
        builder.append(
            np.asarray([peak.mz for peak in peaks], dtype=float),
            np.asarray(
                [scale * peak.relative_abundance * 1000 for peak in peaks],
                dtype=float,
            ),
            source_scan_index=index,
            scan_number=1000 + index,
            rt_sec=rt,
            faims_cv=None,
        )
    raw_store = builder.finalize()
    save_raw_ms1_cache(raw_store, cache, source)
    ledger = ResidualMS1Ledger(raw_store)
    traces = raw_store.extract_traces(
        tuple(peak.mz for peak in peaks), 8.0, 80.0, 120.0
    )
    allocation = ledger.allocate_component(
        "existing_feature",
        traces,
        0,
        tuple(np.asarray(trace.intensity, dtype=float) for trace in traces),
    )
    assert allocation.accepted
    ownership = save_residual_ownership_cache(
        ledger, tmp_path / "residual-ownership", source
    )
    paths = {
        "features": str(tmp_path / "features.parquet"),
        "external_evidence": str(tmp_path / "external.parquet"),
        "raw_ms1_cache": str(cache),
        "residual_ownership_cache": str(ownership),
    }
    _write_empty_outputs(paths)
    model = RTAlignmentModel(
        "donor", "recipient", "median_shift", 10, 10, (), (),
        1.0, 10.0, 0.0, "accepted",
    )
    plans = tuple(
        ExternalPlan(
            "recipient", "donor", "explicit:g",
            _observation("donor", "PEPTIDE-ion-%02d" % value, 90, value),
            100.0, model,
        )
        for value in range(1, 21)
    )
    result = run_external_recipient(
        {
            "run": SimpleNamespace(run_id="recipient", mzml_path=source),
            "paths": paths,
            "plans": plans,
            "options": {
                "ppm": 8.0,
                "rt_tolerance_sec": 20.0,
                "min_isotope_cosine": 0.8,
                "q_value_max": 0.01,
                "weak_feature": True,
                "weak_q_value_max": 0.05,
                "weak_overlap_max": 0.20,
                "quant_method": "envelope_area",
                "baseline": "none",
            },
        },
        workers=2,
    )
    assert result["new_weak_external_feature_count"] == 0
    evidence = pq.read_table(paths["external_evidence"]).to_pylist()
    assert all(row["status"] == "weak_target_overlap_above_limit" for row in evidence)
    assert all(row["weak_overlap_fraction"] > 0.20 for row in evidence)
    assert all(row["weak_residual_target_evaluated"] is False for row in evidence)
    assert all(row["weak_target_extraction_status"] is None for row in evidence)
    assert all(row["weak_raw_target_extraction_status"] is not None for row in evidence)
