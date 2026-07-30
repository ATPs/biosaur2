from biosaur2.postprocess_cache import (
    load_local_candidate_pairs,
    local_candidate_fingerprint,
    save_local_candidate_pairs,
)


def _event(event_id, mz):
    return {
        "ms2_event_id": event_id,
        "selected_ion_mz": mz,
        "isolation_target_mz": mz,
        "isolation_lower_offset": 0.7,
        "isolation_upper_offset": 0.7,
        "charge": 2,
        "rt_sec": 10.0,
        "precursor_ms1_index": 3,
        "faims_cv": None,
        "ion_mobility": None,
    }


def test_local_candidate_cache_reuses_only_the_exact_residual_state(tmp_path):
    source = tmp_path / "run.mzML"
    source.write_bytes(b"source")
    targets = [_event(1, 500.0)]
    decoys = [_event(1, 520.0)]
    fingerprint = local_candidate_fingerprint(
        source,
        stage="generic_standard",
        target_events=targets,
        decoy_events=decoys,
        options={"ppm": 10.0, "width_limit_sec": 30.0},
        residual_state="state-a",
        raw_scan_count=5,
        raw_point_count=10,
    )
    cached, path = load_local_candidate_pairs(tmp_path / "cache", fingerprint)
    assert cached is None
    save_local_candidate_pairs(path, fingerprint, ("target",), ("decoy",))
    cached, reused_path = load_local_candidate_pairs(
        tmp_path / "cache", fingerprint
    )
    assert reused_path == path
    assert cached == (("target",), ("decoy",))

    changed = local_candidate_fingerprint(
        source,
        stage="generic_standard",
        target_events=targets,
        decoy_events=decoys,
        options={"ppm": 10.0, "width_limit_sec": 30.0},
        residual_state="state-b",
        raw_scan_count=5,
        raw_point_count=10,
    )
    missing, changed_path = load_local_candidate_pairs(
        tmp_path / "cache", changed
    )
    assert missing is None
    assert changed_path != path
