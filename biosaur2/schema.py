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
    }


FEATURE_COLUMNS = feature_columns()
HILL_COLUMNS = hill_columns()
HILL_TSV_COLUMNS = HILL_COLUMNS
