"""Robust mass-calibration fitting with explicit fallback status."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class CalibrationResult:
    status: str
    reason: Optional[str]
    shift: float
    sigma: Optional[float]
    covariance: Optional[float]
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _noisy_gaussian(x, amplitude, center, sigma, baseline):
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2)) + baseline


def fit_mass_calibration(
    values: Iterable[float],
    bin_width: float = 0.05,
) -> CalibrationResult:
    samples = np.asarray(list(values), dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    sample_count = int(samples.size)
    if sample_count < 4:
        return CalibrationResult(
            "not_applied", "insufficient_samples", 0.0, None, None, sample_count
        )
    if float(np.ptp(samples)) == 0.0:
        return CalibrationResult(
            "not_applied", "constant_samples", 0.0, None, None, sample_count
        )
    if not np.isfinite(bin_width) or bin_width <= 0:
        return CalibrationResult(
            "failed", "invalid_bin_width", 0.0, None, None, sample_count
        )

    lower = float(np.min(samples))
    upper = float(np.max(samples))
    edges = np.arange(lower, upper + bin_width * 1.5, bin_width)
    if edges.size < 4:
        edges = np.linspace(lower, upper, 5)
    histogram, edges = np.histogram(samples, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    initial_sigma = max(float(np.std(samples)), bin_width)
    initial = [
        max(float(np.max(histogram)), 1.0),
        float(np.median(samples)),
        initial_sigma,
        max(float(np.min(histogram)), 0.0),
    ]
    try:
        fitted, covariance = curve_fit(
            _noisy_gaussian,
            centers,
            histogram,
            p0=initial,
            maxfev=10000,
        )
    except (RuntimeError, TypeError, ValueError, FloatingPointError) as exc:
        return CalibrationResult(
            "failed", "fit_failed:%s" % type(exc).__name__, 0.0, None, None, sample_count
        )

    shift = float(fitted[1])
    sigma = abs(float(fitted[2]))
    covariance_value = float(covariance[0][0])
    if not all(np.isfinite([shift, sigma, covariance_value])) or sigma <= 0:
        return CalibrationResult(
            "failed", "nonfinite_fit", 0.0, None, None, sample_count
        )
    if not lower <= shift <= upper:
        return CalibrationResult(
            "failed", "center_outside_data", 0.0, None, None, sample_count
        )
    return CalibrationResult(
        "applied", None, shift, sigma, covariance_value, sample_count
    )
