import numpy as np

from biosaur2.calibration import fit_mass_calibration


def test_calibration_falls_back_for_empty_small_and_constant_values():
    assert fit_mass_calibration([]).reason == "insufficient_samples"
    assert fit_mass_calibration([1.0, 2.0, 3.0]).reason == "insufficient_samples"
    result = fit_mass_calibration([1.0] * 10)
    assert result.status == "not_applied"
    assert result.reason == "constant_samples"
    assert result.shift == 0.0
    assert result.sigma is None


def test_calibration_applies_to_a_well_sampled_gaussian():
    values = np.random.default_rng(7).normal(0.2, 0.5, 5000)
    result = fit_mass_calibration(values)
    assert result.status == "applied"
    assert abs(result.shift - 0.2) < 0.1
    assert 0.3 < result.sigma < 0.7
