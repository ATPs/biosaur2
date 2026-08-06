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
    _fit_empirical_support_llr,
    _json_default,
    _outcome_status,
    _write_evidence,
    build_feature_alignment_models,
    read_feature_sidecars,
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
        "scanStart": 100, "scanApex": 101, "scanEnd": 102,
        "isoerror": 0.0, "isoerror2": 0.0,
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


def test_external_evidence_schema_is_stable_for_empty_output(tmp_path):
    empty_path = tmp_path / "empty.parquet"
    populated_path = tmp_path / "populated.parquet"
    _write_evidence(empty_path, [])
    _write_evidence(populated_path, [{
        "target_run": "target",
        "weak_candidate_id": 1,
        "feature_id": None,
        "source_run": None,
        "source_feature_id": None,
        "support_rank": None,
        "support_score": None,
        "mz_error_ppm": None,
        "support_log_likelihood_ratio": None,
        "rt_error_sec": None,
        "predicted_rt_sec": None,
        "target_support_count": 0,
        "decoy_support_count": 0,
        "target_score": None,
        "decoy_score": None,
        "competition_winner": "none",
        "acceptance_q_value": 1.0,
        "status": "no_external_support",
        "alignment_method": None,
        "alignment_anchor_count": None,
        "alignment_residual_mad_sec": None,
    }])

    empty_schema = pq.read_schema(empty_path)
    populated_schema = pq.read_schema(populated_path)
    assert empty_schema == populated_schema
    assert empty_schema.field("weak_candidate_id").type == pa.int64()
    assert empty_schema.field("support_score").type == pa.float64()
    assert empty_schema.metadata[
        b"biosaur2_external_evidence_schema_version"
    ] == b"1"


def test_feature_mbr_rescues_weak_candidate_without_raw_cache(tmp_path):
    run_ids = ("source", "reference", "donor_c", "donor_d", "target")
    runs = [_run(run_id) for run_id in run_ids]
    offsets = {
        run_id: index * 5.0 for index, run_id in enumerate(run_ids)
    }
    for run in runs:
        path = tmp_path / (run.run_id + ".mzML")
        path.write_bytes(run.run_id.encode())
        run.mzml_path = path

    anchors = [500.0 + value for value in range(5)]
    strong = {}
    for run_id in run_ids:
        strong[run_id] = [
            _feature(
                run_id, index + 1, mz,
                100.0 + offsets[run_id] + index,
            )
            for index, mz in enumerate(anchors)
        ]
        if run_id != "target":
            strong[run_id].extend(
                _feature(
                    run_id, 20 + index, 700.0 + index * 0.1,
                    150.0 + offsets[run_id],
                )
                for index in range(20)
            )
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
    assert output[0]["quant_envelope_apex"] == pytest.approx(100.0)
    assert output[0]["rtApex"] == pytest.approx(170.0 / 60.0)
    assert (
        output[0]["scanStart"], output[0]["scanApex"], output[0]["scanEnd"]
    ) == (100, 101, 102)
    assert "rt_apex_sec" not in output[0]
    evidence = pq.read_table(paths_by_run["target"]["external_evidence"]).to_pylist()
    assert evidence[0]["status"] == "accepted_matched_weak_feature"
    assert evidence[0]["target_support_count"] == 4
    assert evidence[0]["decoy_support_count"] == 0
    assert evidence[0]["support_log_likelihood_ratio"] > 0
    assert evidence[0]["source_run"] in set(run_ids) - {"target"}
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
    assert stage["scheduler_summary"]["external_min_support_runs"] == 1
    assert stage["scheduler_summary"]["external_max_support_runs"] == 4
    assert stage["scheduler_summary"][
        "rescued_support_run_count_distribution"
    ] == {4: 20}
    max_four_score = evidence[0]["target_score"]

    capped = run_feature_mbr_stage(runs, results, {
        "external_ppm": 8.0, "external_rt_tolerance_sec": 120.0,
        "external_q_value_max": 0.10, "external_alignment_min_anchors": 3,
        "external_alignment_max_mad_sec": 30.0,
        "external_alignment_max_anchors": 64,
        "external_min_support_runs": 1,
        "external_max_support_runs": 2,
    })
    assert capped["summaries"]["target"][
        "rescued_support_run_count_distribution"
    ] == {2: 20}
    capped_evidence = pq.read_table(
        paths_by_run["target"]["external_evidence"]
    ).to_pylist()
    assert len([
        row for row in capped_evidence if row["weak_candidate_id"] == 1
    ]) == 2
    max_two_score = capped_evidence[0]["target_score"]

    limited = run_feature_mbr_stage(runs, results, {
        "external_ppm": 8.0, "external_rt_tolerance_sec": 120.0,
        "external_q_value_max": 0.05, "external_alignment_min_anchors": 3,
        "external_alignment_max_mad_sec": 30.0,
        "external_alignment_max_anchors": 64,
        "external_min_support_runs": 1,
        "external_max_support_runs": 1,
    })
    assert limited["summaries"]["target"][
        "new_weak_external_feature_count"
    ] == 20
    assert limited["summaries"]["target"][
        "rescued_support_run_count_distribution"
    ] == {1: 20}
    limited_evidence = pq.read_table(
        paths_by_run["target"]["external_evidence"]
    ).to_pylist()
    max_one_score = limited_evidence[0]["target_score"]
    assert max_four_score > max_two_score > max_one_score
    assert stage["scheduler_summary"]["support_scoring"] == (
        "empirical_log_likelihood_ratio_sum"
    )
    assert stage["scheduler_summary"]["support_llr_crossfit_folds"] == 2
    calibration_audit = next(iter(
        stage["scheduler_summary"]["support_llr_calibrations"].values()
    ))
    assert set(calibration_audit) == {"full", "crossfit"}
    assert set(calibration_audit["crossfit"]) == {"0", "1"}

    gated = run_feature_mbr_stage(runs, results, {
        "external_ppm": 8.0, "external_rt_tolerance_sec": 120.0,
        "external_q_value_max": 0.10, "external_alignment_min_anchors": 3,
        "external_alignment_max_mad_sec": 30.0,
        "external_alignment_max_anchors": 64,
        "external_min_support_runs": 5,
        "external_max_support_runs": 5,
    })
    assert gated["summaries"]["target"][
        "new_weak_external_feature_count"
    ] == 0
    assert gated["summaries"]["target"]["status_counts"] == {
        "insufficient_target_support_runs": 20
    }


def test_external_competition_aggregates_distinct_run_supports():
    supports = [
        ((0.9, 0.1, 1.0, -0.9, 1, None, 100.0), "run-a", None),
        ((0.8, 0.2, 2.0, -0.8, 2, None, 101.0), "run-b", None),
    ]
    poor = ((0.2, 0.1, 1.0, -0.9, 3, None, 100.0), "run-c", None)
    excellent = (
        (0.99, 0.1, 1.0, -0.9, 4, None, 100.0), "run-d", None
    )
    calibration_rows = []
    for _ in range(100):
        calibration_rows.append({"targets": [poor], "decoys": [poor]})
    for _ in range(40):
        calibration_rows.append({"targets": [excellent], "decoys": []})
    calibrator = _fit_empirical_support_llr(calibration_rows)
    assert _aggregate_support_score(
        [excellent], calibrator
    ) > _aggregate_support_score([poor] * 4, calibrator)
    assert _aggregate_support_score(
        supports, calibrator, min_support_runs=2
    ) == pytest.approx(sum(calibrator.score(item[0][0]) for item in supports))
    assert _aggregate_support_score(
        supports[:1], calibrator, min_support_runs=2
    ) is None
    assert _aggregate_support_score([], calibrator) is None
    assert list(calibrator.log_likelihood_ratios) == sorted(
        calibrator.log_likelihood_ratios
    )
    assert _json_default(np.float32(1.25)) == pytest.approx(1.25)


def test_support_minimum_status_and_sidecar_options_are_explicit(tmp_path):
    result = SimpleNamespace(winner="none", q_value=1.0)
    item = {
        "competition": result,
        "accepted_alignment_count": 1,
        "targets": [(None, "source", None)],
        "min_support_runs": 2,
    }
    assert _outcome_status(item, 0.10) == "insufficient_target_support_runs"

    item.update({
        "competition": SimpleNamespace(winner="target", q_value=0.08),
        "targets": [(None, "source", None), (None, "reference", None)],
    })
    assert _outcome_status(item, 0.10) == "accepted_matched_weak_feature"
    assert _outcome_status(item, 0.05) == "target_q_value_above_limit"

    mzml = tmp_path / "run.mzML"
    mzml.write_bytes(b"source")
    paths = {
        "external_strong_features": str(tmp_path / "strong.parquet"),
        "external_weak_candidates": str(tmp_path / "weak.parquet"),
    }
    run = _run("run")
    run.mzml_path = mzml
    write_feature_sidecars(mzml, paths, [], [], {
        "external_weak_max_strong_overlap": 0.30,
    })
    assert read_feature_sidecars(run, paths, {
        "external_weak_max_strong_overlap": 0.30,
    }) == ((), ())
    assert read_feature_sidecars(run, paths, {
        "external_weak_max_strong_overlap": 0.20,
    }) is None


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
