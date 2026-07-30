import numpy as np

from biosaur2.optimization import (
    ConflictCandidate,
    nonnegative_deconvolution,
    select_conflict_candidates,
)


def test_exact_conflict_selection_beats_greedy_single_high_candidate():
    candidates = [
        ConflictCandidate("wide", 9.0, frozenset({1, 2})),
        ConflictCandidate("left", 5.0, frozenset({1})),
        ConflictCandidate("right", 5.0, frozenset({2})),
    ]
    result = select_conflict_candidates(candidates)
    assert result.selected_ids == ("left", "right")
    assert result.objective == 10.0
    assert result.method == "exact"


def test_protected_anchor_is_not_removed_by_a_higher_scoring_conflict():
    result = select_conflict_candidates(
        [
            ConflictCandidate("anchor", 1.0, frozenset({1}), protected=True),
            ConflictCandidate("late", 100.0, frozenset({1})),
        ]
    )
    assert result.selected_ids == ("anchor",)


def test_nnls_recovers_identifiable_components_and_rejects_rank_deficiency():
    design = np.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    observed = design @ np.asarray([4.0, 2.0])
    result = nonnegative_deconvolution(design, observed)
    assert result.status == "accepted"
    np.testing.assert_allclose(result.coefficients, [4.0, 2.0], atol=1e-10)
    assert result.intensity_conserved

    ambiguous = nonnegative_deconvolution(
        np.asarray([[1.0, 1.0], [2.0, 2.0]]), np.asarray([2.0, 4.0])
    )
    assert ambiguous.status == "unidentifiable"
    assert np.sum(ambiguous.modeled) == 0.0
