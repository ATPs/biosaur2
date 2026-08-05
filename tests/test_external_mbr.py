from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biosaur2 import external_mbr
from biosaur2.external_alignment import ReferenceStarAlignment
from biosaur2.external_mbr import (
    FeatureRecord,
    _aggregate_support_score,
    _json_default,
    build_feature_alignment_models,
    run_feature_mbr_stage,
    write_feature_sidecars,
)
from biosaur2.schema import compact_schemas


def _run(run_id):
    return SimpleNamespace(
        run_id=run_id,
        mzml_path=None,
        metadata={"alignment_group": "g", "fraction": "", "batch": ""},
    )


def _feature(run_id, feature_id, mz, rt, quality=0.95):
    return FeatureRecord(run_id, feature_id, mz, 2, None, rt - 2, rt, rt + 2, 100.0, quality)


def _candidate_row(mz=700.0):
    return {
        "run_id": "target", "feature_idx": -1, "massCalib": 1400.0,
        "rtApex": 170.0, "intensityApex": 100.0, "intensitySum": 200.0,
        "charge": 2, "nIsotopes": 2, "nScans": 2, "mz": mz,
        "rtStart": 168.0, "rtEnd": 172.0, "FAIMS": None, "im": None,
        "scanApex": 1, "isoerror": 0.0, "isoerror2": 0.0,
        "area_sum": 100.0, "feature_origin": "aligned_external_weak",
        "confidence_tier": "external_id_weak", "quant_value": 100.0,
        "quant_method": "all", "quant_status": "quantified",
        "area_envelope_raw": 100.0, "area_envelope_corrected": 100.0,
        "area_mono_raw": 80.0, "area_mono_corrected": 80.0,
        "envelope_apex": 100.0, "quant_envelope_area": 100.0,
        "quant_mono_area": 80.0, "quant_envelope_apex": 100.0,
        "feature_quality_score": 0.75, "quality_flags": 0,
        "extraction_q_value": None, "supporting_psm_count": 0,
        "supporting_ms2_count": 0, "points_across_peak": 2,
        "rt_start_sec": 168.0, "rt_apex_sec": 170.0, "rt_end_sec": 172.0,
        "isotope_cosine": 0.75, "mass_error_ppm_median": 0.0,
        "ms2_events": [],
    }


def test_feature_mbr_rescues_weak_candidate_without_raw_cache(tmp_path):
    runs = [_run("source"), _run("reference"), _run("target")]
    source_path = tmp_path / "source.mzML"
    reference_path = tmp_path / "reference.mzML"
    target_path = tmp_path / "target.mzML"
    for run, path in zip(runs, (source_path, reference_path, target_path)):
        path.write_bytes(run.run_id.encode())
        run.mzml_path = path

    anchors = [500.0 + value for value in range(5)]
    strong = {
        "source": [_feature("source", index + 1, mz, 100.0 + index) for index, mz in enumerate(anchors)] + [_feature("source", 20 + index, 700.0 + index * 0.1, 150.0) for index in range(20)],
        "reference": [_feature("reference", index + 1, mz, 110.0 + index) for index, mz in enumerate(anchors)] + [_feature("reference", 20 + index, 700.0 + index * 0.1, 160.0) for index in range(20)],
        "target": [_feature("target", index + 1, mz, 120.0 + index) for index, mz in enumerate(anchors)],
    }
    paths_by_run = {}
    results = {}
    for index, run in enumerate(runs):
        features = tmp_path / (run.run_id + ".features.parquet")
        evidence = tmp_path / (run.run_id + ".external.parquet")
        pq.write_table(pa.Table.from_pylist([], schema=compact_schemas()["hybrid_features"]), features)
        paths = {
            "features": str(features), "external_evidence": str(evidence),
            "external_strong_features": str(tmp_path / (run.run_id + ".strong.parquet")),
            "external_weak_candidates": str(tmp_path / (run.run_id + ".weak.parquet")),
        }
        weak = []
        if run.run_id == "target":
            candidate_rows = []
            for candidate_index, candidate_id in enumerate(range(1, 21)):
                row = _candidate_row(700.0 + candidate_index * 0.1)
                row["envelope_apex"] = str(np.float32(100.0))
                candidate_rows.append({
                    **_feature(
                        "target", candidate_id,
                        700.0 + candidate_index * 0.1, 170.0, 0.75,
                    ).__dict__,
                    "candidate_id": candidate_id,
                    "row_json": __import__("json").dumps(row),
                    "mono_points": 2,
                    "secondary_points": 2,
                    "isotope_cosine": 0.75,
                })
            weak = [{
                **row
            } for row in candidate_rows]
        write_feature_sidecars(run.mzml_path, paths, [item.__dict__ for item in strong[run.run_id]], weak)
        paths_by_run[run.run_id] = paths
        results[index] = {"status": "success", "paths": paths}

    stage = run_feature_mbr_stage(runs, results, {
        "external_ppm": 8.0, "external_rt_tolerance_sec": 120.0,
        "external_q_value_max": 0.05, "external_alignment_min_anchors": 3,
        "external_alignment_max_mad_sec": 30.0, "external_alignment_max_anchors": 64,
    })
    assert stage["summaries"]["target"]["new_weak_external_feature_count"] == 20
    output = pq.read_table(paths_by_run["target"]["features"]).to_pylist()
    assert output[0]["feature_origin"] == "aligned_external_weak"
    assert output[0]["envelope_apex"] == pytest.approx(100.0)
    evidence = pq.read_table(paths_by_run["target"]["external_evidence"]).to_pylist()
    assert evidence[0]["status"] == "accepted_matched_weak_feature"
    assert evidence[0]["source_run"] in {"source", "reference"}
    for candidate_id in range(1, 21):
        sources = {
            row["source_run"] for row in evidence
            if row["weak_candidate_id"] == candidate_id
        }
        assert len(sources) == len([
            row for row in evidence
            if row["weak_candidate_id"] == candidate_id
        ])
    assert stage["scheduler_summary"][
        "component_strong_index_build_count"
    ] == 1


def test_external_competition_aggregates_distinct_run_supports():
    supports = [
        ((0.9, 0.1, 1.0, -0.9, 1, None, 100.0), "run-a", None),
        ((0.8, 0.2, 2.0, -0.8, 2, None, 101.0), "run-b", None),
    ]
    assert _aggregate_support_score(supports) == pytest.approx(1.7)
    assert _aggregate_support_score([]) is None
    assert _json_default(np.float32(1.25)) == pytest.approx(1.25)


def test_feature_alignment_lis_rejects_isobaric_distractors_and_validates_holdout():
    runs = [_run("source"), _run("target")]
    source, target = [], []
    for index in range(80):
        rt = 30.0 + index * 25.0
        aligned = rt + 18.0 + 0.00001 * rt * rt
        source.append(_feature("source", index + 1, 500.0 + index * 0.1, rt))
        target.append(_feature("target", index + 1, 500.0 + index * 0.1, aligned))
        source.append(_feature("source", 1000 + index, 700.0 + index * 0.1, rt))
        target.append(_feature(
            "target", 1000 + index, 700.0 + index * 0.1,
            30.0 + ((index * 37) % 80) * 25.0,
        ))
    models = build_feature_alignment_models(
        runs, {"source": source, "target": target},
        ppm=8.0, min_anchors=20, max_mad=30.0, max_anchors=256,
        validation_q90_limit=120.0,
    )
    forward = models[("source", "target")][1]
    assert forward.status == "accepted"
    assert forward.validation_anchor_count >= 16
    assert abs(forward.validation_median_bias_sec) < 5.0
    assert forward.validation_q90_abs_error_sec < 20.0

    random_target = [
        _feature(
            "target", index + 1, 800.0 + index * 0.1,
            30.0 + ((index * 37) % 80) * 25.0,
        )
        for index in range(80)
    ]
    random_source = [
        _feature("source", index + 1, 800.0 + index * 0.1, 30.0 + index * 25.0)
        for index in range(80)
    ]
    rejected = build_feature_alignment_models(
        runs, {"source": random_source, "target": random_target},
        ppm=8.0, min_anchors=20, max_mad=30.0, max_anchors=256,
        validation_q90_limit=120.0,
    )[("source", "target")][1]
    assert rejected.status.startswith("insufficient_")


def test_feature_alignment_forest_retries_reference_and_keeps_multihop(monkeypatch):
    runs = [_run("a"), _run("b"), _run("c")]
    strong = {
        run_id: [
            _feature(run_id, index + 1, 500.0 + index * 0.1, 30.0 + index)
            for index in range(count)
        ]
        for run_id, count in (("a", 70), ("b", 80), ("c", 90))
    }
    original = external_mbr._validated_alignment

    def reject_a_to_c(source_run, target_run, anchors, **options):
        model = original(source_run, target_run, anchors, **options)
        if {source_run, target_run} == {"a", "c"}:
            return replace(model, status="validation_mad_exceeds_limit")
        return model

    monkeypatch.setattr(
        external_mbr, "_validated_alignment", reject_a_to_c
    )
    models = build_feature_alignment_models(
        runs, strong, ppm=8.0, min_anchors=20, max_mad=30.0,
        max_anchors=256, validation_q90_limit=120.0,
    )
    assert models.component_by_run["a"] == models.component_by_run["c"]
    alignment = ReferenceStarAlignment(
        "a", "c", models.path_to_reference("a"),
        models.reference_to_run_path("c"),
    )
    assert alignment.status == "accepted"
    assert alignment.method == "reference_forest"
    assert len(alignment.source_to_reference) == 2
