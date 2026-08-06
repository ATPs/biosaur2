"""Identification-aware direct assays and bounded local feature extraction."""

from __future__ import annotations

FEATURE_ORIGIN_DIRECT_IDENTIFIED = "direct_identified"
FEATURE_ORIGIN_ALIGNED_EXTERNAL = "aligned_external"
FEATURE_ORIGIN_ALIGNED_EXTERNAL_WEAK = "aligned_external_weak"
FEATURE_ORIGIN_STRICT_UNTARGETED = "strict_untargeted"
FEATURE_ORIGIN_MS2_GUIDED_FULL = "ms2_guided_full"
FEATURE_ORIGIN_MS2_GUIDED_PARTIAL = "ms2_guided_partial"
FEATURE_ORIGIN_MS2_GUIDED_MONO_ONLY = "ms2_guided_mono_only"


RELAXED_DIRECT_Q_VALUE_MAX = 0.01
QUALITY_FLAG_RELAXED_MS2_FEATURE = 0x0001
QUALITY_FLAG_BOUNDARY_TRUNCATED = 0x0002
QUALITY_FLAG_TWO_POINT_QUANT = 0x0004
QUALITY_FLAG_RAW_BASELINE_FALLBACK = 0x0008
QUALITY_FLAG_WEAK_EXTERNAL_FEATURE = 0x0010
GENERIC_SCORE_CALIBRATION_MIN_PAIRED_ANCHORS = 40
GENERIC_SCORE_CALIBRATION_PRIOR_FRACTIONS = (0.95, 0.90, 0.80, 0.70, 0.60, 0.50)
GENERIC_LOCAL_REFINEMENT_INPUT_STATUSES = frozenset(
    {
        "generic_no_standard_candidate",
        "generic_q_value_rejected",
        "generic_decoy_won",
        "generic_decoy_only",
    }
)

__all__ = [name for name in globals() if not name.startswith("__")]
