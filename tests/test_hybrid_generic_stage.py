from collections import Counter

import biosaur2.hybrid_generic_stage as generic_stage


def test_relaxed_recovery_disabled_preserves_standard_local_state(monkeypatch):
    sentinel = object()
    captured = {}

    def finalize(state):
        captured.update(state)
        return sentinel

    monkeypatch.setattr(generic_stage, "_finalize_generic_stage", finalize)
    state = {
        "run_id": "run",
        "ingestion": None,
        "strict_contexts": (),
        "args": {"relaxed_ms2_feature": False},
        "audit_by_event": {},
        "strict_index": None,
        "strict_hill_claims": None,
        "residual_ledger": None,
        "residual_allocation_status_counts": Counter(),
        "strict_ownership": {},
        "strict_quant_rows": [],
        "recovered": [],
        "recovered_quant_rows": [],
        "local_candidate_cache_telemetry": [],
        "next_feature_id": 7,
        "final_strict_detector": None,
        "generic_summary": {},
        "generic_recovered_feature_rows": [],
        "generic_recovered_quant_rows": [],
        "generic_recovered": [],
        "generic_score_weights": (),
        "local_events": (),
        "decoy_events": {},
        "local_workers": 1,
        "local_ppm": 10.0,
        "local_rt_tolerance": 120.0,
        "width_limit": 60.0,
        "local_competitions": (),
        "local_status_counts": Counter({"generic_local_candidate": 2}),
        "q_value_max": 0.01,
    }

    result = generic_stage._run_relaxed_generic_recovery(state)

    assert result is sentinel
    assert captured["local_status_counts"] == Counter(
        {"generic_local_candidate": 2}
    )
    assert captured["relaxed_competitions"] == ()
    assert captured["relaxed_strict_competition"]["reason"] == (
        "relaxed_retry_disabled"
    )
