from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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
    first = run_external_recipient(task)
    assert first["new_external_feature_count"] == 1
    assert pq.ParquetFile(paths["features"]).metadata.num_rows == 1
    feature = pq.read_table(paths["features"]).to_pylist()[0]
    assert feature["feature_origin"] == "aligned_external"
    assert feature["quant_value"] > 0
    evidence = pq.read_table(paths["external_evidence"]).to_pylist()
    assert evidence[0]["status"] == "accepted_new_external_feature"
    assert evidence[0]["feature_id"] == 1
    assert evidence[0]["target_mono_points"] == 7

    second = run_external_recipient(task)
    assert second["new_external_feature_count"] == 0
    assert pq.ParquetFile(paths["features"]).metadata.num_rows == 1
    evidence = pq.read_table(paths["external_evidence"]).to_pylist()
    assert evidence[0]["status"] == "accepted_matched_existing_feature"
    assert evidence[0]["feature_id"] == 1


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
        }
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
        }
    )
    assert result["new_weak_external_feature_count"] == 0
    evidence = pq.read_table(paths["external_evidence"]).to_pylist()
    assert all(row["status"] == "weak_target_overlap_above_limit" for row in evidence)
    assert all(row["weak_overlap_fraction"] > 0.20 for row in evidence)
