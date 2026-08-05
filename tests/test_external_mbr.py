from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from biosaur2.external_mbr import (
    FeatureRecord,
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
            weak = [{
                **_feature("target", candidate_id, 700.0 + index * 0.1, 170.0, 0.75).__dict__,
                "candidate_id": candidate_id, "row_json": __import__("json").dumps(_candidate_row(700.0 + index * 0.1)),
                "mono_points": 2, "secondary_points": 1, "isotope_cosine": 0.75,
            } for index, candidate_id in enumerate(range(1, 21))]
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
    evidence = pq.read_table(paths_by_run["target"]["external_evidence"]).to_pylist()
    assert evidence[0]["status"] == "accepted_matched_weak_feature"
    assert evidence[0]["source_run"] in {"source", "reference"}
