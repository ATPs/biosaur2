"""Explicit compact output schemas."""

from __future__ import annotations

import pyarrow as pa


SCHEMA_VERSION = "3.0"

FEATURE_BASE_COLUMNS = (
    "massCalib",
    "rtApex",
    "intensityApex",
    "intensitySum",
    "charge",
    "nIsotopes",
    "nScans",
    "mz",
    "rtStart",
    "rtEnd",
    "FAIMS",
    "im",
)
FEATURE_MONO_COLUMNS = (
    "mono_hills_scan_lists",
    "mono_hills_intensity_list",
)
FEATURE_TRAILING_COLUMNS = (
    "scanApex",
    "isoerror",
    "isoerror2",
    "feature_idx",
    "area_sum",
)
FEATURE_EXTRA_COLUMNS = (
    "isotopes",
    "intensity_array_for_cos_corr",
    "monoisotope hill idx",
    "monoisotope idx",
)

HILL_BASE_COLUMNS = (
    "rtApex",
    "intensityApex",
    "intensitySum",
    "nScans",
    "mz",
    "rtStart",
    "rtEnd",
    "FAIMS",
    "im",
    "scanApex",
    "hill_idx",
)
HILL_LIST_COLUMNS = (
    "hills_scan_lists",
    "hills_intensity_list",
    "hills_mz_array",
    "hills_rt_list",
)
HILL_TRAILING_COLUMNS = ("feature_idx",)
MS1_COLUMNS = ("scan_id", "RT", "total_intensity")
MS2_COLUMNS = (
    "run_id",
    "ms2_event_id",
    "ms2_index",
    "spectrum_index",
    "native_id",
    "native_scan_number",
    "rt_sec",
    "precursor_ms1_index",
    "precursor_resolution",
    "selected_ion_mz",
    "isolation_target_mz",
    "isolation_lower_offset",
    "isolation_upper_offset",
    "precursor_mz",
    "precursor_mz_source",
    "charge",
    "selected_ion_intensity",
    "faims_cv",
    "ion_mobility",
    "metadata_flags",
)

MS2_FEATURE_LINK_COLUMNS = (
    "run_id", "ms2_event_id", "feature_id", "status", "seed_eligible",
    "seed_used_in_selection", "selected_ion_isotope_offset",
    "isolated_isotope_index", "mz_error_ppm", "rt_distance_sec",
    "precursor_scan_distance", "seed_support", "reason_flags",
)
MS2_FEATURE_LINK_SCHEMA_VERSION = "1"
HYBRID_SCHEMA_VERSION = "2"

HYBRID_FEATURE_QUANT_COLUMNS = (
    "run_id", "feature_id", "feature_origin", "confidence_tier",
    "quant_value", "quant_method", "quant_status", "area_envelope_raw",
    "area_envelope_corrected", "area_mono_raw", "area_mono_corrected",
    "envelope_apex", "feature_quality_score", "quality_flags",
    "extraction_q_value", "supporting_psm_count", "supporting_ms2_count",
    "points_across_peak", "rt_start_sec", "rt_apex_sec", "rt_end_sec",
    "isotope_cosine", "mass_error_ppm_median",
)
HYBRID_MS2_AUDIT_COLUMNS = (
    "run_id", "ms2_event_id", "feature_id", "association_tier", "status",
    "primary_identification_id", "assay_id", "charge_used", "charge_source",
    "selected_isotope_index", "generic_isotope_error", "mz_error_ppm",
    "rt_error_sec", "score", "extraction_q_value", "alternative_count",
    "reason_flags",
)
IDENTIFICATION_COLUMNS = (
    "run_id", "psm_id", "ms2_event_id", "mapping_status", "q_value", "pep",
    "score", "rank", "peptide_raw", "canonical_peptidoform", "formula_status",
    "assay_status", "selected_isotope_index", "selected_mz_error_ppm",
)
ID_ASSAY_COLUMNS = (
    "run_id", "assay_id", "ms2_event_id", "psm_id", "canonical_peptidoform",
    "charge", "rt_sec", "faims_cv", "monoisotopic_mz",
    "selected_isotope_index", "selected_mz_error_ppm", "q_value", "pep",
    "conflict_status",
)

MS2_SCHEMA_VERSION = "1"
MS2_MISSING_PRECURSOR_MZ = 0x0001
MS2_MISSING_CHARGE = 0x0002
MS2_UNRESOLVED_SPECTRUM_REF = 0x0004
MS2_MISSING_PRECURSOR_MS1 = 0x0008

INTENSITY_COLUMNS = {
    "intensityApex",
    "intensitySum",
    "mono_hills_intensity_list",
    "hills_intensity_list",
    "total_intensity",
}


def feature_columns(include_mono=True, extra_details=False):
    columns = list(FEATURE_BASE_COLUMNS)
    if include_mono:
        columns.extend(FEATURE_MONO_COLUMNS)
    columns.extend(FEATURE_TRAILING_COLUMNS)
    if extra_details:
        columns.extend(FEATURE_EXTRA_COLUMNS)
    return tuple(columns)


def hill_columns(include_lists=True):
    columns = list(HILL_BASE_COLUMNS)
    if include_lists:
        columns.extend(HILL_LIST_COLUMNS)
    columns.extend(HILL_TRAILING_COLUMNS)
    return tuple(columns)


def _feature_schema(use64=False, include_mono=True, extra_details=False):
    float_type = pa.float64() if use64 else pa.float32()
    wide_int = pa.int64() if use64 else pa.int32()
    fields = {
        "massCalib": float_type,
        "rtApex": float_type,
        "intensityApex": float_type,
        "intensitySum": float_type,
        "charge": pa.int64() if use64 else pa.int8(),
        "nIsotopes": pa.int64() if use64 else pa.int8(),
        "nScans": pa.int64() if use64 else pa.int16(),
        "mz": float_type,
        "rtStart": float_type,
        "rtEnd": float_type,
        "FAIMS": float_type,
        "im": float_type,
        "mono_hills_scan_lists": pa.list_(wide_int),
        "mono_hills_intensity_list": pa.list_(float_type),
        "scanApex": wide_int,
        "isoerror": float_type,
        "isoerror2": float_type,
        "feature_idx": wide_int,
        "area_sum": float_type,
    }
    if extra_details:
        isotope_struct = pa.struct(
            [
                pa.field("isotope_number", wide_int),
                pa.field("isotope_hill_idx", wide_int),
                pa.field("isotope_idx", wide_int),
                pa.field("cos_cor", float_type),
                pa.field("mass_diff_ppm", float_type),
            ]
        )
        fields.update(
            {
                "isotopes": pa.list_(isotope_struct),
                "intensity_array_for_cos_corr": pa.list_(pa.list_(float_type)),
                "monoisotope hill idx": wide_int,
                "monoisotope idx": wide_int,
            }
        )
    return pa.schema(
        pa.field(name, fields[name], nullable=True)
        for name in feature_columns(include_mono, extra_details)
    )


def _hill_schema(use64=False, include_lists=True):
    float_type = pa.float64() if use64 else pa.float32()
    wide_int = pa.int64() if use64 else pa.int32()
    fields = {
        "rtApex": float_type,
        "intensityApex": float_type,
        "intensitySum": float_type,
        "nScans": pa.int64() if use64 else pa.int16(),
        "mz": float_type,
        "rtStart": float_type,
        "rtEnd": float_type,
        "FAIMS": float_type,
        "im": float_type,
        "scanApex": wide_int,
        "hill_idx": wide_int,
        "hills_scan_lists": pa.list_(wide_int),
        "hills_intensity_list": pa.list_(float_type),
        "hills_mz_array": pa.list_(float_type),
        "hills_rt_list": pa.list_(float_type),
        "feature_idx": wide_int,
    }
    return pa.schema(
        pa.field(name, fields[name], nullable=True)
        for name in hill_columns(include_lists)
    )


def _ms1_schema(use64=False):
    float_type = pa.float64() if use64 else pa.float32()
    int_type = pa.int64() if use64 else pa.int32()
    return pa.schema(
        [
            pa.field("scan_id", int_type, nullable=True),
            pa.field("RT", float_type, nullable=True),
            pa.field("total_intensity", float_type, nullable=True),
        ]
    )


def _ms2_schema():
    category_type = pa.dictionary(pa.int8(), pa.string())
    return pa.schema(
        [
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("ms2_event_id", pa.int32(), nullable=False),
            pa.field("ms2_index", pa.int32(), nullable=False),
            pa.field("spectrum_index", pa.int32(), nullable=False),
            pa.field("native_id", pa.string(), nullable=True),
            pa.field("native_scan_number", pa.int32(), nullable=True),
            pa.field("rt_sec", pa.float32(), nullable=True),
            pa.field("precursor_ms1_index", pa.int32(), nullable=True),
            pa.field("precursor_resolution", category_type, nullable=True),
            pa.field("selected_ion_mz", pa.float64(), nullable=True),
            pa.field("isolation_target_mz", pa.float64(), nullable=True),
            pa.field("isolation_lower_offset", pa.float32(), nullable=True),
            pa.field("isolation_upper_offset", pa.float32(), nullable=True),
            pa.field("precursor_mz", pa.float64(), nullable=True),
            pa.field("precursor_mz_source", category_type, nullable=True),
            pa.field("charge", pa.int16(), nullable=True),
            pa.field("selected_ion_intensity", pa.float32(), nullable=True),
            pa.field("faims_cv", pa.float32(), nullable=True),
            pa.field("ion_mobility", pa.float32(), nullable=True),
            pa.field("metadata_flags", pa.uint16(), nullable=False),
        ]
    )


def _ms2_feature_link_schema(use64=False):
    category_type = pa.dictionary(pa.int8(), pa.string())
    feature_id_type = pa.int64() if use64 else pa.int32()
    return pa.schema(
        [
            pa.field("run_id", category_type, nullable=False),
            pa.field("ms2_event_id", pa.int32(), nullable=False),
            pa.field("feature_id", feature_id_type, nullable=True),
            pa.field("status", category_type, nullable=False),
            pa.field("seed_eligible", pa.bool_(), nullable=False),
            pa.field("seed_used_in_selection", pa.bool_(), nullable=False),
            pa.field("selected_ion_isotope_offset", pa.int8(), nullable=True),
            pa.field("isolated_isotope_index", pa.int8(), nullable=True),
            pa.field("mz_error_ppm", pa.float32(), nullable=True),
            pa.field("rt_distance_sec", pa.float32(), nullable=True),
            pa.field("precursor_scan_distance", pa.int16(), nullable=True),
            pa.field("seed_support", pa.float32(), nullable=True),
            pa.field("reason_flags", pa.uint16(), nullable=False),
        ]
    )


def _hybrid_feature_quant_schema(use64=False):
    feature_id = pa.int64() if use64 else pa.int32()
    category = pa.dictionary(pa.int8(), pa.string())
    fields = {
        "run_id": category, "feature_id": feature_id,
        "feature_origin": category, "confidence_tier": category,
        "quant_value": pa.float64(), "quant_method": category,
        "quant_status": category, "area_envelope_raw": pa.float64(),
        "area_envelope_corrected": pa.float64(), "area_mono_raw": pa.float64(),
        "area_mono_corrected": pa.float64(), "envelope_apex": pa.float64(),
        "feature_quality_score": pa.float32(), "quality_flags": pa.uint32(),
        "extraction_q_value": pa.float32(), "supporting_psm_count": pa.int32(),
        "supporting_ms2_count": pa.int32(), "points_across_peak": pa.int32(),
        "rt_start_sec": pa.float64(), "rt_apex_sec": pa.float64(),
        "rt_end_sec": pa.float64(), "isotope_cosine": pa.float32(),
        "mass_error_ppm_median": pa.float32(),
    }
    return pa.schema(pa.field(name, fields[name], nullable=name not in {"run_id", "feature_id"}) for name in HYBRID_FEATURE_QUANT_COLUMNS)


def _hybrid_ms2_audit_schema(use64=False):
    feature_id = pa.int64() if use64 else pa.int32()
    category = pa.dictionary(pa.int8(), pa.string())
    fields = {
        "run_id": category, "ms2_event_id": pa.int32(), "feature_id": feature_id,
        "association_tier": category, "status": category,
        "primary_identification_id": pa.string(), "assay_id": pa.int32(),
        "charge_used": pa.int16(), "charge_source": category,
        "selected_isotope_index": pa.int8(), "generic_isotope_error": pa.int8(),
        "mz_error_ppm": pa.float32(), "rt_error_sec": pa.float32(),
        "score": pa.float32(), "extraction_q_value": pa.float32(),
        "alternative_count": pa.int16(), "reason_flags": pa.uint32(),
    }
    required = {"run_id", "ms2_event_id", "association_tier", "status", "alternative_count", "reason_flags"}
    return pa.schema(pa.field(name, fields[name], nullable=name not in required) for name in HYBRID_MS2_AUDIT_COLUMNS)


def _identification_schema():
    category = pa.dictionary(pa.int8(), pa.string())
    fields = {
        "run_id": category, "psm_id": pa.string(), "ms2_event_id": pa.int32(),
        "mapping_status": category, "q_value": pa.float64(), "pep": pa.float64(),
        "score": pa.float64(), "rank": pa.int16(), "peptide_raw": pa.string(),
        "canonical_peptidoform": pa.string(), "formula_status": category,
        "assay_status": category, "selected_isotope_index": pa.int8(),
        "selected_mz_error_ppm": pa.float32(),
    }
    return pa.schema(pa.field(name, fields[name], nullable=name not in {"run_id", "psm_id", "mapping_status", "q_value"}) for name in IDENTIFICATION_COLUMNS)


def _id_assay_schema():
    category = pa.dictionary(pa.int8(), pa.string())
    fields = {
        "run_id": category, "assay_id": pa.int32(), "ms2_event_id": pa.int32(),
        "psm_id": pa.string(), "canonical_peptidoform": pa.string(),
        "charge": pa.int16(), "rt_sec": pa.float64(), "faims_cv": pa.float32(),
        "monoisotopic_mz": pa.float64(), "selected_isotope_index": pa.int8(),
        "selected_mz_error_ppm": pa.float32(), "q_value": pa.float64(),
        "pep": pa.float64(), "conflict_status": category,
    }
    return pa.schema(pa.field(name, fields[name], nullable=name in {"faims_cv", "pep"}) for name in ID_ASSAY_COLUMNS)


def compact_schemas(
    use64=False,
    include_mono=True,
    extra_details=False,
    include_hill_lists=True,
):
    return {
        "features": _feature_schema(use64, include_mono, extra_details),
        "hills": _hill_schema(use64, include_hill_lists),
        "ms1": _ms1_schema(use64),
        "ms2": _ms2_schema(),
        "ms2_feature_links": _ms2_feature_link_schema(use64),
        "hybrid_feature_quant": _hybrid_feature_quant_schema(use64),
        "hybrid_ms2_audit": _hybrid_ms2_audit_schema(use64),
        "identifications": _identification_schema(),
        "id_assays": _id_assay_schema(),
    }


FEATURE_COLUMNS = feature_columns()
HILL_COLUMNS = hill_columns()
HILL_TSV_COLUMNS = HILL_COLUMNS
