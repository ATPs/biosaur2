from collections import Counter
import multiprocessing
import os
import random
from types import SimpleNamespace

import pytest

import biosaur2.hybrid_generic_stage as generic_stage
import biosaur2.hybrid_generic_local as generic_local
import biosaur2.hybrid_generic_association as generic_association
from biosaur2.parallel import WorkerProcessError


def _generic_candidate(mz, charge=2, faims_cv=None, **attributes):
    defaults = {
        "rt_start_sec": 0.0,
        "rt_end_sec": 12.0,
        "rt_apex_sec": 6.0,
        "allocation_group_key": None,
        "allocation_component_index": None,
        "intensity_conserved": False,
        "isotope_cosine": 0.9,
    }
    defaults.update(attributes)
    return SimpleNamespace(
        mono_mz=mz,
        event={"charge": charge, "faims_cv": faims_cv},
        **defaults,
    )


def test_generic_recovered_index_preserves_prior_candidate_order(monkeypatch):
    early_conflict = _generic_candidate(650.0)
    late_equivalent = _generic_candidate(500.0, faims_cv=-45.0)
    challenger = _generic_candidate(500.003, faims_cv=-44.9999995)
    points = {
        id(early_conflict): frozenset({(7, 650.0)}),
        id(late_equivalent): frozenset({(9, 500.0)}),
        id(challenger): frozenset({(7, 650.0)}),
    }
    monkeypatch.setattr(
        generic_local,
        "_local_candidate_raw_points",
        lambda candidate: points[id(candidate)],
    )

    index = generic_local._GenericRecoveredIndex(10.0)
    index.add(early_conflict, 11)
    index.add(late_equivalent, 12)

    challenger_points, entries = index.candidates(challenger)

    assert challenger_points == points[id(challenger)]
    assert entries == [
        (early_conflict, 11, True),
        (late_equivalent, 12, False),
    ]


def test_generic_recovered_index_uses_cached_raw_points_for_conflicts(monkeypatch):
    protected = _generic_candidate(700.0)
    challenger = _generic_candidate(701.0)
    points = {
        id(protected): frozenset({(3, 700.0)}),
        id(challenger): frozenset({(3, 700.0)}),
    }
    calls = []

    monkeypatch.setattr(
        generic_local,
        "_local_candidate_raw_points",
        lambda candidate: points[id(candidate)],
    )

    def conflict(left, right, protected_points=None, challenger_points=None):
        calls.append((left, right, protected_points, challenger_points))
        return True

    monkeypatch.setattr(generic_local, "_protected_local_conflict", conflict)
    index = generic_local._GenericRecoveredIndex(10.0)
    index.add(protected, 1)

    assert index.conflict(challenger)
    assert calls == [
        (
            protected,
            challenger,
            generic_local._SHARED_RAW_POINT,
            generic_local._SHARED_RAW_POINT,
        )
    ]


def test_generic_recovered_index_reuses_query_footprint_when_accepting(monkeypatch):
    previous = _generic_candidate(500.0)
    accepted = _generic_candidate(501.0)
    calls = Counter()

    def raw_points(candidate):
        calls[id(candidate)] += 1
        return frozenset({(int(candidate.mono_mz), candidate.mono_mz)})

    monkeypatch.setattr(generic_local, "_local_candidate_raw_points", raw_points)
    index = generic_local._GenericRecoveredIndex(10.0)
    index.add(previous, 1)
    accepted_points, _entries = index.candidates(accepted)
    index.add(accepted, 2, accepted_points)

    assert calls[id(previous)] == 1
    assert calls[id(accepted)] == 1


def test_generic_recovered_index_filters_charge_and_faims_without_missing_boundary(monkeypatch):
    reference = _generic_candidate(500.0, charge=2, faims_cv=10.0)
    close = _generic_candidate(500.005, charge=2, faims_cv=10.0000009)
    wrong_charge = _generic_candidate(500.005, charge=3, faims_cv=10.0)
    points = {
        id(reference): frozenset(),
        id(close): frozenset(),
        id(wrong_charge): frozenset(),
    }
    monkeypatch.setattr(
        generic_local,
        "_local_candidate_raw_points",
        lambda candidate: points[id(candidate)],
    )
    index = generic_local._GenericRecoveredIndex(10.0)
    index.add(reference, 1)

    assert [entry[1] for entry in index.candidates(close)[1]] == [1]
    assert index.candidates(wrong_charge)[1] == []


def test_generic_recovered_index_matches_linear_oracle_for_random_candidates(monkeypatch):
    generator = random.Random(20260824)
    candidates = []
    points = {}
    for position in range(80):
        candidate = _generic_candidate(
            500.0 + generator.randrange(12) * 0.0015,
            charge=generator.choice((2, 3)),
            faims_cv=generator.choice((None, -45.0, -44.9999995)),
            rt_start_sec=generator.randrange(12),
            rt_end_sec=12.0 + generator.randrange(12),
            rt_apex_sec=6.0 + generator.randrange(12),
            isotope_cosine=generator.random(),
        )
        candidates.append(candidate)
        points[id(candidate)] = frozenset(
            (generator.randrange(8), 400.0 + generator.randrange(8))
            for _value in range(generator.randrange(4))
        )
    monkeypatch.setattr(
        generic_local,
        "_local_candidate_raw_points",
        lambda candidate: points[id(candidate)],
    )

    index = generic_local._GenericRecoveredIndex(10.0)
    accepted = []
    for position, candidate in enumerate(candidates):
        _candidate_points, indexed = index.candidates(candidate)
        indexed_candidates = {id(entry[0]) for entry in indexed}
        expected = {
            id(previous)
            for previous in accepted
            if generic_local._generic_local_equivalent(previous, candidate, 10.0)
            or generic_local._protected_local_conflict(
                previous,
                candidate,
                protected_points=points[id(previous)],
                challenger_points=points[id(candidate)],
            )
        }
        assert expected <= indexed_candidates
        index.add(candidate, position, points[id(candidate)])
        accepted.append(candidate)


def test_generic_standard_link_pair_matches_serial_results(monkeypatch):
    def fake_links(ms2_rows, _ingestion, _contexts, _args):
        rows = tuple(dict(row) for row in ms2_rows)
        return rows, {"selected_mz": [row["selected_ion_mz"] for row in rows]}

    monkeypatch.setattr(generic_association, "_generic_standard_links", fake_links)
    rows = (
        {"ms2_event_id": 1, "selected_ion_mz": 500.0, "charge": 2},
        {"ms2_event_id": 2, "selected_ion_mz": 600.0, "charge": 3},
    )

    serial = generic_association.generic_standard_link_pair(
        "run", rows, None, (), {"nprocs": 1}
    )
    parallel = generic_association.generic_standard_link_pair(
        "run", rows, None, (), {"nprocs": 2}
    )

    assert [result[:2] for result in parallel] == [result[:2] for result in serial]


def test_generic_standard_link_pair_terminates_peer_on_worker_failure(monkeypatch):
    def fail_links(*_args):
        raise RuntimeError("expected pair failure")

    monkeypatch.setattr(generic_association, "_generic_standard_links", fail_links)
    before = {process.pid for process in multiprocessing.active_children()}

    with pytest.raises(WorkerProcessError, match="expected pair failure"):
        generic_association.generic_standard_link_pair(
            "run",
            ({"ms2_event_id": 1, "selected_ion_mz": 500.0, "charge": 2},),
            None,
            (),
            {"nprocs": 2},
        )

    assert {process.pid for process in multiprocessing.active_children()} == before


def test_generic_standard_link_pair_reports_clean_worker_exit(monkeypatch):
    monkeypatch.setattr(
        generic_association,
        "_generic_standard_links",
        lambda *_args: os._exit(3),
    )
    before = {process.pid for process in multiprocessing.active_children()}

    with pytest.raises(WorkerProcessError, match="ProcessExit"):
        generic_association.generic_standard_link_pair(
            "run",
            ({"ms2_event_id": 1, "selected_ion_mz": 500.0, "charge": 2},),
            None,
            (),
            {"nprocs": 2},
        )

    assert {process.pid for process in multiprocessing.active_children()} == before


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
        "q_value_max": 0.05,
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
