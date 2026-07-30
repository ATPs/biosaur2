import pytest

from biosaur2.generic_local import (
    GenericLocalCandidate,
    _averagine_probabilities,
    compete_generic_local_candidates,
    cluster_compatible_generic_candidates,
    evaluate_generic_local_candidate,
    evaluate_generic_local_candidate_pairs,
    generic_local_width_limit,
)
from biosaur2.generic_association import C13_C12_MASS_DIFF
from biosaur2.raw_ms1 import RawMS1StoreBuilder


def _event(event_id=1):
    return {
        "ms2_event_id": event_id,
        "selected_ion_mz": 500.0,
        "isolation_target_mz": 500.0,
        "isolation_lower_offset": 0.7,
        "isolation_upper_offset": 0.7,
        "charge": 2,
        "rt_sec": 2.0,
        "faims_cv": None,
    }


def _store(event, active_scans=(1, 2, 3)):
    probabilities = _averagine_probabilities(
        event["selected_ion_mz"], event["charge"], 5
    )
    builder = RawMS1StoreBuilder()
    for scan in range(5):
        mz = []
        intensity = []
        if scan in active_scans:
            scale = [1.0, 2.0, 1.0][active_scans.index(scan)]
            for isotope, probability in enumerate(probabilities):
                mz.append(
                    event["selected_ion_mz"]
                    + isotope * C13_C12_MASS_DIFF / event["charge"]
                )
                intensity.append(scale * probability * 10000.0)
        builder.append(
            mz,
            intensity,
            source_scan_index=scan,
            scan_number=100 + scan,
            rt_sec=float(scan),
            faims_cv=None,
        )
    return builder.finalize()


def _profile_store(event, profile):
    probabilities = _averagine_probabilities(
        event["selected_ion_mz"], event["charge"], 5
    )
    builder = RawMS1StoreBuilder()
    for scan, scale in enumerate(profile):
        mz = []
        intensity = []
        if scale > 0:
            for isotope, probability in enumerate(probabilities):
                mz.append(
                    event["selected_ion_mz"]
                    + isotope * C13_C12_MASS_DIFF / event["charge"]
                )
                intensity.append(scale * probability * 10000.0)
        builder.append(
            mz,
            intensity,
            source_scan_index=scan,
            scan_number=100 + scan,
            rt_sec=float(scan),
            faims_cv=None,
        )
    return builder.finalize()


def test_generic_local_candidate_requires_multiscan_coherent_envelope():
    event = _event()
    candidate = evaluate_generic_local_candidate(
        _store(event), event, width_limit_sec=20.0, rt_tolerance_sec=10.0
    )
    assert candidate.status == "candidate"
    assert candidate.quantitative_candidate
    assert candidate.isotope_error == 0
    assert candidate.mono_points == 3
    assert candidate.supported_channels >= 2
    assert candidate.isotope_cosine == pytest.approx(1.0)
    assert abs(candidate.selected_event_mz_error_ppm) < 1e-8
    assert candidate.score == pytest.approx(0.96)
    assert candidate.selected_event_apex_ratio == pytest.approx(1.0)
    assert dict(candidate.score_components) == pytest.approx(
        {
            "mass_support": 1.0,
            "isotope_cosine_support": 1.0,
            "event_apex_support": 1.0,
            "coelution_support": 1.0,
            "point_support": 0.6,
            "channel_support": 1.0,
        }
    )

    single = evaluate_generic_local_candidate(
        _store(event, active_scans=(2,)),
        event,
        width_limit_sec=20.0,
        rt_tolerance_sec=10.0,
    )
    assert single.status == "insufficient_mono_points"
    assert not single.quantitative_candidate

    relaxed = evaluate_generic_local_candidate(
        _store(event, active_scans=(1, 2)),
        event,
        width_limit_sec=20.0,
        rt_tolerance_sec=10.0,
        min_mono_points=2,
        min_channel_points=2,
        min_supported_channels=2,
        min_cosine=0.95,
        relaxed=True,
    )
    assert relaxed.quantitative_candidate
    assert relaxed.relaxed
    assert relaxed.mono_points == 2


def test_generic_local_paired_parallel_evaluation_is_ordered_and_deterministic():
    targets = [_event(11), _event(12)]
    decoys = [
        {
            **event,
            "selected_ion_mz": 520.0,
            "isolation_target_mz": 520.0,
        }
        for event in targets
    ]
    store = _store(targets[0])
    options = {"width_limit_sec": 20.0, "rt_tolerance_sec": 10.0}
    serial_target, serial_decoy = evaluate_generic_local_candidate_pairs(
        store, targets, decoys, workers=1, **options
    )
    parallel_target, parallel_decoy = evaluate_generic_local_candidate_pairs(
        store, targets, decoys, workers=2, **options
    )
    assert [value.event["ms2_event_id"] for value in parallel_target] == [11, 12]
    assert [value.status for value in parallel_target] == [
        value.status for value in serial_target
    ]
    assert [value.score for value in parallel_target] == pytest.approx(
        [value.score for value in serial_target]
    )
    assert [value.status for value in parallel_decoy] == [
        value.status for value in serial_decoy
    ]


def test_generic_local_target_decoy_q_values_remain_event_level():
    targets = []
    decoys = []
    for event_id in range(200):
        event = _event(event_id)
        targets.append(GenericLocalCandidate(event, "candidate", 1.0))
        decoys.append(GenericLocalCandidate(event, "no_component_at_event_scan", None))
    results = compete_generic_local_candidates(targets, decoys)
    assert len(results) == 200
    assert all(result.winner == "target" for result in results)
    assert all(result.q_value == pytest.approx(0.005) for result in results)


def test_generic_local_score_rewards_event_apex_locality_without_hard_forcing_it():
    apex_event = _event()
    shoulder_event = {**apex_event, "rt_sec": 1.0}
    apex = evaluate_generic_local_candidate(
        _store(apex_event),
        apex_event,
        width_limit_sec=20.0,
        rt_tolerance_sec=10.0,
    )
    shoulder = evaluate_generic_local_candidate(
        _store(shoulder_event),
        shoulder_event,
        width_limit_sec=20.0,
        rt_tolerance_sec=10.0,
    )
    assert shoulder.quantitative_candidate
    assert shoulder.selected_event_apex_ratio == pytest.approx(0.5)
    assert shoulder.score < apex.score


def test_generic_local_uses_exact_precursor_scan_before_nearest_rt():
    event = {
        **_event(),
        # The following survey scan is closer to the MS2 RT, but scan 2 is
        # the actual precursor survey recorded by mzML.
        "rt_sec": 3.8,
        "precursor_ms1_index": 2,
    }
    candidate = evaluate_generic_local_candidate(
        _store(event, active_scans=(1, 2, 3)),
        event,
        width_limit_sec=20.0,
        rt_tolerance_sec=10.0,
    )
    assert candidate.quantitative_candidate
    assert candidate.status == "candidate"
    assert candidate.selected_event_apex_ratio == pytest.approx(1.0)


def test_generic_local_width_limit_is_measured_and_bounded():
    rows = [
        {"rt_start_sec": 0.0, "rt_end_sec": float(value)}
        for value in range(1, 101)
    ]
    assert generic_local_width_limit(rows) == pytest.approx(60.0)
    assert generic_local_width_limit([]) == 30.0


def test_generic_local_boundary_truncation_is_visible_not_silently_rejected():
    event = _event()
    candidate = evaluate_generic_local_candidate(
        _store(event, active_scans=(0, 1, 2)),
        event,
        width_limit_sec=20.0,
        rt_tolerance_sec=10.0,
    )
    assert candidate.quantitative_candidate
    assert candidate.boundary_truncated


def test_generic_local_joint_segmentation_splits_a_bimodal_hill_at_ms2_component():
    event = {
        **_event(),
        "rt_sec": 8.2,
        "precursor_ms1_index": 8,
    }
    candidate = evaluate_generic_local_candidate(
        _profile_store(event, [0, 1, 4, 10, 4, 1, 1, 4, 9, 4, 1, 0]),
        event,
        width_limit_sec=20.0,
        rt_tolerance_sec=20.0,
    )

    assert candidate.quantitative_candidate
    assert candidate.rt_start_sec >= 6.0
    assert candidate.rt_apex_sec == pytest.approx(8.0)
    assert candidate.component_count == 2
    assert any(
        edit.action == "split" and edit.accepted
        for edit in candidate.edit_history
    )


def test_generic_local_joint_segmentation_repairs_one_scan_hill_gap():
    event = {
        **_event(),
        "rt_sec": 4.2,
        "precursor_ms1_index": 4,
    }
    candidate = evaluate_generic_local_candidate(
        _profile_store(event, [0, 2, 5, 0, 4, 2, 0]),
        event,
        width_limit_sec=20.0,
        rt_tolerance_sec=20.0,
    )

    assert candidate.quantitative_candidate
    assert candidate.mono_points == 4
    assert any(
        edit.action in {"merge", "relink"} and edit.accepted
        for edit in candidate.edit_history
    )


def test_repeated_compatible_ms2_cluster_without_duplicating_split_components():
    first_event = {
        **_event(1),
        "rt_sec": 2.0,
        "precursor_ms1_index": 2,
    }
    repeated_event = {
        **_event(2),
        "rt_sec": 3.0,
        "precursor_ms1_index": 3,
    }
    store = _profile_store(first_event, [0, 1, 4, 8, 4, 1, 0])
    first = evaluate_generic_local_candidate(
        store, first_event, width_limit_sec=20.0, rt_tolerance_sec=20.0
    )
    repeated = evaluate_generic_local_candidate(
        store, repeated_event, width_limit_sec=20.0, rt_tolerance_sec=20.0
    )
    groups = cluster_compatible_generic_candidates(
        [repeated, first], ppm=10.0
    )

    assert len(groups) == 1
    assert [value.event["ms2_event_id"] for value in groups[0]] == [1, 2]
