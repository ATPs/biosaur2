"""Typed external-ID evidence output schema."""

import pyarrow as pa


EXTERNAL_EVIDENCE_SCHEMA_VERSION = "3"


def evidence_schema():
    required = False
    nullable = True
    fields = (
        ("target_run", pa.string, required), ("source_run", pa.string, required),
        ("alignment_group", pa.string, required), ("ion_key", pa.string, required),
        ("canonical_peptidoform", pa.string, required), ("charge", pa.int16, required),
        ("faims_cv", pa.float32, nullable), ("donor_assay_id", pa.int32, required),
        ("donor_psm_id", pa.string, required), ("donor_q_value", pa.float64, required),
        ("donor_rt_apex_sec", pa.float64, required), ("predicted_rt_sec", pa.float64, required),
        ("alignment_method", pa.string, required), ("alignment_anchor_count", pa.int32, required),
        ("alignment_inlier_count", pa.int32, required), ("alignment_residual_mad_sec", pa.float64, nullable),
        ("decoy_neutral_shift", pa.float64, required), ("target_score", pa.float64, nullable),
        ("decoy_score", pa.float64, nullable), ("target_extraction_status", pa.string, required),
        ("target_gate_status", pa.string, required), ("decoy_extraction_status", pa.string, required),
        ("decoy_gate_status", pa.string, required), ("competition_winner", pa.string, required),
        ("extraction_q_value", pa.float64, required), ("status", pa.string, required),
        ("feature_id", pa.int64, nullable), ("target_mono_points", pa.int32, required),
        ("target_isotope_cosine", pa.float32, nullable), ("target_mass_error_ppm", pa.float32, nullable),
        ("target_rt_error_sec", pa.float64, nullable),
        ("weak_raw_target_extraction_status", pa.string, nullable), ("weak_raw_target_gate_status", pa.string, nullable),
        ("weak_raw_decoy_extraction_status", pa.string, nullable), ("weak_raw_decoy_gate_status", pa.string, nullable),
        ("weak_raw_target_mono_points", pa.int32, nullable), ("weak_raw_target_secondary_channels", pa.int16, nullable),
        ("weak_raw_target_isotope_cosine", pa.float32, nullable), ("weak_raw_target_mass_error_ppm", pa.float32, nullable),
        ("weak_raw_target_rt_error_sec", pa.float64, nullable), ("weak_raw_target_overlap_fraction", pa.float64, nullable),
        ("weak_raw_decoy_overlap_fraction", pa.float64, nullable), ("weak_residual_target_evaluated", pa.bool_, required),
        ("weak_residual_decoy_evaluated", pa.bool_, required),
        ("weak_target_extraction_status", pa.string, nullable), ("weak_target_gate_status", pa.string, nullable),
        ("weak_decoy_extraction_status", pa.string, nullable), ("weak_decoy_gate_status", pa.string, nullable),
        ("weak_target_score", pa.float64, nullable), ("weak_decoy_score", pa.float64, nullable),
        ("weak_competition_winner", pa.string, nullable), ("weak_extraction_q_value", pa.float64, nullable),
        ("weak_target_mono_points", pa.int32, nullable), ("weak_target_secondary_channels", pa.int16, nullable),
        ("weak_target_isotope_cosine", pa.float32, nullable), ("weak_target_mass_error_ppm", pa.float32, nullable),
        ("weak_target_rt_error_sec", pa.float64, nullable), ("weak_overlap_fraction", pa.float64, nullable),
        ("acceptance_family", pa.string, nullable), ("acceptance_q_value", pa.float64, nullable),
    )
    return pa.schema(
        [pa.field(name, data_type(), nullable=is_nullable) for name, data_type, is_nullable in fields],
        metadata={b"biosaur2_external_evidence_schema_version": EXTERNAL_EVIDENCE_SCHEMA_VERSION.encode()},
    )
