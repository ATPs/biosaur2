import numpy as np

from biosaur2.alignment import (
    AlignmentAnchor,
    choose_reference_run,
    fit_rt_alignment,
)


def test_sparse_alignment_uses_median_shift():
    model = fit_rt_alignment(
        "source",
        "target",
        [AlignmentAnchor("a", 10, 15), AlignmentAnchor("b", 20, 25)],
    )
    assert model.method == "median_shift"
    assert model.predict(30) == 35


def test_robust_alignment_rejects_outlier_and_is_monotonic():
    anchors = [
        AlignmentAnchor(str(index), float(index * 100), float(index * 105 + 7))
        for index in range(12)
    ]
    anchors.append(AlignmentAnchor("outlier", 550.0, 5000.0))
    model = fit_rt_alignment("source", "target", anchors)
    predicted = [model.predict(value) for value in np.linspace(0, 1100, 50)]
    assert model.method == "monotonic_piecewise"
    assert model.inlier_count == 12
    assert np.all(np.diff(predicted) >= 0)
    assert abs(model.predict(500) - 532) < 2


def test_reference_run_uses_coverage_then_stable_id():
    assert choose_reference_run({"b": 10, "a": 10, "c": 2}) == "a"
