import numpy as np
import pytest

from biosaur2.raw_ms1 import RawMS1StoreBuilder
from biosaur2.residual import ResidualMS1Ledger


def _store():
    builder = RawMS1StoreBuilder()
    for scan, scale in enumerate((1.0, 2.0, 1.0)):
        builder.append(
            [500.0, 500.0005, 500.5],
            [60.0 * scale, 40.0 * scale, 50.0 * scale],
            source_scan_index=10 + scan,
            scan_number=100 + scan,
            rt_sec=float(scan),
            faims_cv=None,
        )
    return builder.finalize()


def test_residual_ledger_supports_rt_split_and_revert():
    ledger = ResidualMS1Ledger(_store())
    initial_state = ledger.state_fingerprint()
    trace = ledger.extract_trace(500.0, 2.0, 0.0, 2.0)
    np.testing.assert_allclose(trace.intensity, [100.0, 200.0, 100.0])

    result = ledger.allocate_component(
        "left-rt-component", (trace,), 0, [[100.0, 200.0]]
    )
    assert result.accepted
    assert result.requested_intensity == pytest.approx(300.0)
    np.testing.assert_allclose(
        ledger.extract_trace(500.0, 2.0, 0.0, 2.0).intensity,
        [0.0, 0.0, 100.0],
    )
    assert ledger.residual_intensity + ledger.claimed_intensity == pytest.approx(
        ledger.original_intensity
    )
    assert ledger.state_fingerprint() != initial_state

    ledger.revert("left-rt-component")
    assert ledger.state_fingerprint() == initial_state
    np.testing.assert_allclose(
        ledger.extract_trace(500.0, 2.0, 0.0, 2.0).intensity,
        [100.0, 200.0, 100.0],
    )
    assert ledger.allocation_count == 0
    assert ledger.claimed_point_count == 0


def test_residual_ledger_supports_conserved_intensity_split():
    ledger = ResidualMS1Ledger(_store())
    original = ledger.extract_trace(500.0, 2.0, 0.0, 2.0)
    first = ledger.allocate_component(
        "component-a", (original,), 0, [original.intensity * 0.6]
    )
    residual = ledger.extract_trace(500.0, 2.0, 0.0, 2.0)
    second = ledger.allocate_component(
        "component-b", (residual,), 0, [residual.intensity]
    )

    assert first.accepted and second.accepted
    np.testing.assert_allclose(
        ledger.extract_trace(500.0, 2.0, 0.0, 2.0).intensity, 0.0
    )
    assert ledger.claimed_intensity == pytest.approx(
        float(np.sum(original.intensity))
    )
    materialized = ledger.materialize()
    np.testing.assert_allclose(
        materialized.extract_trace(500.0, 2.0, 0.0, 2.0).intensity, 0.0
    )


def test_materialized_residual_matches_sparse_claims_exactly():
    ledger = ResidualMS1Ledger(_store())
    trace = ledger.extract_trace(500.0, 2.0, 0.0, 2.0)
    assert ledger.allocate_component(
        "partial-component", (trace,), 0, [trace.intensity * 0.25]
    ).accepted

    materialized = ledger.materialize()
    expected = ledger.store.intensity.copy()
    for point_index, claimed in ledger._claimed.items():
        expected[point_index] = max(0.0, expected[point_index] - claimed)
    np.testing.assert_allclose(materialized.intensity, expected)


def test_residual_overallocation_fails_atomically():
    ledger = ResidualMS1Ledger(_store())
    trace = ledger.extract_trace(500.0, 2.0, 0.0, 2.0)
    before = ledger.residual_intensity
    result = ledger.allocate_component(
        "too-much", (trace,), 0, [trace.intensity * 1.01]
    )

    assert not result.accepted
    assert result.status == "insufficient_residual_intensity"
    assert ledger.residual_intensity == pytest.approx(before)
    assert ledger.allocation_count == 0


def test_repeated_ms2_evidence_does_not_duplicate_component_allocation():
    ledger = ResidualMS1Ledger(_store())
    trace = ledger.extract_trace(500.0, 2.0, 0.0, 2.0)
    assert ledger.allocate_component(
        "shared-feature-1", (trace,), 0, [trace.intensity]
    ).accepted
    with pytest.raises(ValueError, match="already exists"):
        ledger.allocate_component(
            "shared-feature-1", (trace,), 0, [trace.intensity]
        )


def test_observed_point_allocation_is_exact_conserved_and_atomic():
    ledger = ResidualMS1Ledger(_store())
    accepted = ledger.allocate_observed_points(
        "strict-1",
        [
            (10, 500.0, 60.0),
            (11, 500.0005, 80.0),
            (12, 500.5, 50.0),
        ],
    )
    assert accepted.accepted
    assert accepted.raw_point_count == 3
    assert accepted.allocated_intensity == pytest.approx(190.0)
    assert ledger.claimed_intensity + ledger.residual_intensity == pytest.approx(
        ledger.original_intensity
    )

    before = ledger.state_fingerprint()
    rejected = ledger.allocate_observed_points(
        "strict-2",
        [(10, 500.0, 40.0), (11, 500.0005, 200.0)],
    )
    assert rejected.status == "observed_point_not_available"
    assert ledger.state_fingerprint() == before
    assert ledger.allocation_count == 1


def test_partially_shared_isotope_trace_preserves_true_areas_without_duplication():
    scans = np.arange(5, dtype=np.float64)
    feature_a = np.asarray([0.0, 20.0, 40.0, 20.0, 0.0])
    feature_b = np.asarray([0.0, 10.0, 30.0, 10.0, 0.0])
    builder = RawMS1StoreBuilder()
    for scan, (amount_a, amount_b) in enumerate(zip(feature_a, feature_b)):
        builder.append(
            [500.0, 500.5, 501.0],
            [amount_a, 0.4 * amount_a + amount_b, 0.5 * amount_b],
            source_scan_index=scan,
            scan_number=100 + scan,
            rt_sec=float(scan),
            faims_cv=None,
        )
    ledger = ResidualMS1Ledger(builder.finalize())
    mono_a = ledger.extract_trace(500.0, 2.0, 0.0, 4.0)
    shared = ledger.extract_trace(500.5, 2.0, 0.0, 4.0)
    allocated_a = np.stack([feature_a, 0.4 * feature_a])
    assert ledger.allocate_component(
        "feature-a", (mono_a, shared), 0, allocated_a
    ).accepted

    residual_shared = ledger.extract_trace(500.5, 2.0, 0.0, 4.0)
    isotope_b = ledger.extract_trace(501.0, 2.0, 0.0, 4.0)
    np.testing.assert_allclose(residual_shared.intensity, feature_b)
    allocated_b = np.stack([feature_b, 0.5 * feature_b])
    assert ledger.allocate_component(
        "feature-b", (residual_shared, isotope_b), 0, allocated_b
    ).accepted

    area_a = np.trapezoid(allocated_a.sum(axis=0), scans)
    area_b = np.trapezoid(allocated_b.sum(axis=0), scans)
    assert area_a == pytest.approx(
        np.trapezoid(1.4 * feature_a, scans)
    )
    assert area_b == pytest.approx(
        np.trapezoid(1.5 * feature_b, scans)
    )
    assert ledger.claimed_intensity == pytest.approx(ledger.original_intensity)
    assert ledger.residual_intensity == pytest.approx(0.0)
    assert ledger.claimed_point_count == 9
