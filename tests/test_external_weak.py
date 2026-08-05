import copy

import pytest

from biosaur2 import main, utils
from biosaur2.external_weak import weak_feature_rows_from_contexts
from biosaur2.raw_ms1 import RawMS1StoreBuilder
from biosaur2.residual import ResidualMS1Ledger


def _candidate(mono_index, mono_id, isotope_index, isotope_id, cosine):
    return {
        "monoisotope idx": mono_index,
        "monoisotope hill idx": mono_id,
        "hill_mz_1": 500.0 + mono_index * 0.1,
        "charge": 2,
        "FAIMS": None,
        "isotopes": [{
            "isotope_idx": isotope_index,
            "isotope_hill_idx": isotope_id,
            "isotope_number": 1,
            "mass_diff_ppm": 0.0,
        }],
        "nIsotopes": 2,
        "nScans": 3,
        "cos_cor_isotopes": cosine,
        "intensity_array_for_cos_corr": [[1.0, 0.4], [1.0, 0.4]],
    }


def _hills():
    hills, _step = utils._build_hills_dict(
        hills_idx_array_unique=[10, 11, 12],
        hills_mz_median=[500.0, 500.5, 500.1],
        hills_im_median=None,
        hills_lengths=[3, 3, 3],
        hills_scan_lists=[[0, 1, 2]] * 3,
        hills_intensity_array=[
            [60.0, 120.0, 60.0],
            [50.0, 100.0, 50.0],
            [30.0, 60.0, 30.0],
        ],
        rt_start=[0.0] * 3,
        rt_end=[2.0] * 3,
        rt_apex=[1.0] * 3,
        hill_mass_accuracy=8.0,
        paseftol=0.0,
    )
    hills["tmp_mz_array"] = [
        [500.0] * 3, [500.5] * 3, [500.1] * 3
    ]
    return hills


def _ledger():
    builder = RawMS1StoreBuilder()
    for scan, scale in enumerate((1.0, 2.0, 1.0)):
        builder.append(
            [500.0, 500.1, 500.5],
            [60.0 * scale, 30.0 * scale, 50.0 * scale],
            source_scan_index=10 + scan,
            scan_number=100 + scan,
            rt_sec=float(scan),
            faims_cv=None,
        )
    return ResidualMS1Ledger(builder.finalize())


def _context(candidate):
    hills = _hills()
    hills["_external_weak_candidates"] = (candidate,)
    hills["_external_weak_detector_audit"] = {
        "initial_candidates": 3,
        "smart_filter_accepted": 2,
        "smart_filter_rejected": 1,
        "strict_selected": 1,
        "greedy_rejected": 1,
    }
    return {
        "hills": hills,
        "rt_by_local": {0: 0.0, 1: 1.0, 2: 2.0},
        "spectra": [
            {"scan_index": 10 + value, "scan_number": 100 + value}
            for value in range(3)
        ],
        "faims_cv": None,
        "candidates": (),
    }


def _args():
    return {
        "nm": 0,
        "iuse": -1,
        "no_mono_hills": False,
        "quant_method": "all",
        "external_weak_min_mono_points": 2,
        "external_weak_min_secondary_points": 2,
        "external_weak_min_isotope_cosine": 0.6,
    }


def _claim_candidate_fraction(ledger, fraction):
    contributions = []
    for scan, scale in enumerate((1.0, 2.0, 1.0)):
        contributions.extend([
            (10 + scan, 500.0, 60.0 * scale * fraction),
            (10 + scan, 500.5, 50.0 * scale * fraction),
        ])
    assert ledger.allocate_observed_points("strong", contributions).accepted


def test_weak_ownership_accepts_twenty_percent_boundary_and_rejects_above():
    candidate = _candidate(0, 10, 1, 11, 0.8)
    candidate["_external_reject_source"] = "smart_filter_reject"
    ledger = _ledger()
    _claim_candidate_fraction(ledger, 0.20)
    rows, audit = weak_feature_rows_from_contexts(
        "run", [_context(candidate)], [], _args(), ledger
    )
    assert len(rows) == 1
    assert rows[0]["external_strong_overlap_fraction"] == pytest.approx(0.20)
    assert audit["persisted_weak_candidates"] == 1
    assert audit["reject_source_counts"] == {"smart_filter_reject": 1}

    candidate = _candidate(0, 10, 1, 11, 0.8)
    candidate["_external_reject_source"] = "greedy_conflict_reject"
    ledger = _ledger()
    _claim_candidate_fraction(ledger, 0.201)
    rows, audit = weak_feature_rows_from_contexts(
        "run", [_context(candidate)], [], _args(), ledger
    )
    assert rows == []
    assert audit["strong_overlap_rejected"] == 1


def test_same_run_strong_equivalent_and_weak_mono_duplicate_are_rejected():
    first = _candidate(0, 10, 1, 11, 0.8)
    first["_external_reject_source"] = "smart_filter_reject"
    duplicate = copy.deepcopy(first)
    duplicate["cos_cor_isotopes"] = 0.7
    context = _context(first)
    context["hills"]["_external_weak_candidates"] = (first, duplicate)
    rows, audit = weak_feature_rows_from_contexts(
        "run", [context], [{
            "mz": 500.0, "charge": 2, "FAIMS": None,
            "rtStart": 0.0, "rtEnd": 2.0,
        }], _args(), _ledger()
    )
    assert rows == []
    assert audit["weak_deduplicated"] == 1
    assert audit["strong_equivalent_rejected"] == 1


def test_greedy_conflict_reject_is_retained_even_when_it_shares_an_isotope():
    hills = _hills()
    winner = _candidate(0, 10, 1, 11, 1.0)
    loser = _candidate(2, 12, 1, 11, 0.8)
    rejected = []
    _used, selected = main._select_nonconflicting_isotope_candidates(
        [winner, loser], hills, {0: 0.0, 1: 1.0, 2: 2.0}, 0,
        rejected,
    )
    assert selected == [winner]
    assert rejected == [loser]


@pytest.mark.parametrize(
    ("feature_mode", "external_id", "collects"),
    [("legacy", True, False), ("hybrid", False, False), ("hybrid", True, True)],
)
def test_only_external_hybrid_collects_weak_candidates(
    monkeypatch, feature_mode, external_id, collects
):
    candidate = _candidate(0, 10, 1, 11, 0.8)
    sinks = []
    monkeypatch.setattr(
        main, "_generate_initial_isotope_candidates",
        lambda *_args: [candidate],
    )

    def calibrate(ready, _faims, _args, rejected_sink=None):
        sinks.append(rejected_sink)
        if rejected_sink is not None:
            rejected_sink.append(ready[0])
        return []

    def select(ready, _hills, _rt, _offset, rejected_sink=None):
        sinks.append(rejected_sink)
        return set(), []

    monkeypatch.setattr(main, "_calibrate_and_filter_isotope_candidates", calibrate)
    monkeypatch.setattr(main, "_select_nonconflicting_isotope_candidates", select)
    monkeypatch.setattr(main, "_capture_direct_competitors", lambda *_args: ())
    monkeypatch.setattr(main, "_record_losing_direct_competitors", lambda *_args: None)
    monkeypatch.setattr(
        main, "_assign_feature_indices_and_write",
        lambda *_args: ({}, 1),
    )
    hills = {"hills_idx_array_unique": [10]}
    main.process_features_iteration(
        hills, None, 0.01, 0.0, {}, 0, False,
        {"feature_mode": feature_mode, "external_id": external_id},
    )
    assert all(sink is not None for sink in sinks) is collects
    assert ("_external_weak_candidates" in hills) is collects
