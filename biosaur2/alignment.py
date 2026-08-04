"""Robust deterministic retention-time alignment for exact peptide-ion anchors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Sequence

import numpy as np
from scipy.stats import theilslopes


@dataclass(frozen=True)
class AlignmentAnchor:
    ion_key: str
    source_rt_sec: float
    target_rt_sec: float
    quality: float = 1.0


@dataclass(frozen=True)
class RTAlignmentModel:
    source_run: str
    target_run: str
    method: str
    anchor_count: int
    inlier_count: int
    x_knots: tuple[float, ...]
    y_knots: tuple[float, ...]
    slope: float
    intercept: float
    residual_mad_sec: Optional[float]
    status: str

    def predict(self, rt_sec):
        value = float(rt_sec)
        if self.status != "accepted":
            raise ValueError("alignment model is not accepted")
        if self.method == "identity":
            return value
        if self.method in {"median_shift", "robust_affine"}:
            return self.slope * value + self.intercept
        x = np.asarray(self.x_knots)
        y = np.asarray(self.y_knots)
        if value <= x[0]:
            local_slope = (y[1] - y[0]) / (x[1] - x[0]) if x.size > 1 else self.slope
            return float(y[0] + local_slope * (value - x[0]))
        if value >= x[-1]:
            local_slope = (y[-1] - y[-2]) / (x[-1] - x[-2]) if x.size > 1 else self.slope
            return float(y[-1] + local_slope * (value - x[-1]))
        return float(np.interp(value, x, y))


def _mad(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def _isotonic(values, weights):
    blocks = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([index, index + 1, float(value), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left[3] + right[3]
            mean = (left[2] * left[3] + right[2] * right[3]) / total_weight
            blocks.append([left[0], right[1], mean, total_weight])
    result = np.empty(len(values), dtype=np.float64)
    for start, end, value, _weight in blocks:
        result[start:end] = value
    return result


def fit_rt_alignment(
    source_run: str,
    target_run: str,
    anchors: Sequence[AlignmentAnchor],
    *,
    piecewise_min_anchors: int = 8,
) -> RTAlignmentModel:
    finite = [
        anchor
        for anchor in anchors
        if math.isfinite(anchor.source_rt_sec)
        and math.isfinite(anchor.target_rt_sec)
        and math.isfinite(anchor.quality)
        and anchor.quality > 0
    ]
    finite.sort(key=lambda item: (item.source_rt_sec, item.target_rt_sec, item.ion_key))
    if not finite:
        return RTAlignmentModel(source_run, target_run, "none", 0, 0, (), (), 1.0, 0.0, None, "insufficient_anchors")
    x = np.asarray([item.source_rt_sec for item in finite], dtype=np.float64)
    y = np.asarray([item.target_rt_sec for item in finite], dtype=np.float64)
    weights = np.asarray([item.quality for item in finite], dtype=np.float64)
    if len(finite) < 3:
        shift = float(np.median(y - x))
        residual = y - (x + shift)
        return RTAlignmentModel(source_run, target_run, "median_shift", len(finite), len(finite), (), (), 1.0, shift, _mad(residual), "accepted")

    slope, intercept, _low, _high = theilslopes(y, x)
    predicted = slope * x + intercept
    residual = y - predicted
    residual_mad = _mad(residual) or 0.0
    threshold = max(10.0, 4.0 * 1.4826 * residual_mad)
    inlier = np.abs(residual - np.median(residual)) <= threshold
    if np.count_nonzero(inlier) >= 3:
        slope, intercept, _low, _high = theilslopes(y[inlier], x[inlier])
    else:
        inlier[:] = True
    if not math.isfinite(slope) or slope <= 0:
        shift = float(np.median(y - x))
        return RTAlignmentModel(source_run, target_run, "median_shift", len(finite), int(np.count_nonzero(inlier)), (), (), 1.0, shift, _mad(y - (x + shift)), "accepted")
    residual = y[inlier] - (slope * x[inlier] + intercept)
    if np.count_nonzero(inlier) < piecewise_min_anchors:
        return RTAlignmentModel(source_run, target_run, "robust_affine", len(finite), int(np.count_nonzero(inlier)), (), (), float(slope), float(intercept), _mad(residual), "accepted")

    inlier_x = x[inlier]
    inlier_y = y[inlier]
    inlier_weights = weights[inlier]
    unique_x = np.unique(inlier_x)
    knot_y = []
    knot_weights = []
    for value in unique_x:
        selected = inlier_x == value
        knot_y.append(float(np.median(inlier_y[selected])))
        knot_weights.append(float(np.sum(inlier_weights[selected])))
    monotonic_y = _isotonic(knot_y, knot_weights)
    fitted = np.interp(inlier_x, unique_x, monotonic_y)
    return RTAlignmentModel(
        source_run,
        target_run,
        "monotonic_piecewise",
        len(finite),
        int(np.count_nonzero(inlier)),
        tuple(float(value) for value in unique_x),
        tuple(float(value) for value in monotonic_y),
        float(slope),
        float(intercept),
        _mad(inlier_y - fitted),
        "accepted",
    )


def choose_reference_run(anchor_counts):
    if not anchor_counts:
        raise ValueError("no runs are available for alignment")
    return min(anchor_counts, key=lambda run_id: (-anchor_counts[run_id], run_id))
