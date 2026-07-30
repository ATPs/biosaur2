import pyarrow as pa
import pytest

from biosaur2.legacy_output import CompactOutputManager
from biosaur2.schema import compact_schemas


duckdb = pytest.importorskip("duckdb")


DOWNSTREAM_FEATURE_COLUMNS = (
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
    "scanApex",
    "isoerror",
    "isoerror2",
    "feature_idx",
)


def test_preprocessing_projection_works_with_no_mono_hills_schema():
    schema = compact_schemas(include_mono=False)["features"]
    assert schema.names == [*DOWNSTREAM_FEATURE_COLUMNS, "area_sum"]
    assert schema.field("charge").type == pa.int8()
    assert schema.field("nIsotopes").type == pa.int8()
    assert schema.field("nScans").type == pa.int16()
    assert schema.field("feature_idx").type == pa.int32()
    assert all(
        schema.field(name).type == pa.float32()
        for name in (
            "massCalib",
            "rtApex",
            "intensityApex",
            "intensitySum",
            "mz",
            "rtStart",
            "rtEnd",
            "FAIMS",
            "im",
            "isoerror",
            "isoerror2",
            "area_sum",
        )
    )

    table = pa.Table.from_pylist([], schema=schema)
    with duckdb.connect() as connection:
        connection.register("features", table)
        projection = ", ".join('"%s"' % name for name in DOWNSTREAM_FEATURE_COLUMNS)
        result = connection.execute(
            "SELECT %s FROM features" % projection
        ).to_arrow_table()
    assert result.schema.names == list(DOWNSTREAM_FEATURE_COLUMNS)


def test_duckdb_reads_an_actual_pyarrow_fallback_file(tmp_path):
    output = tmp_path / "sample.features.parquet"
    args = {
        "file": str(tmp_path / "sample.mzML"),
        "o": str(output),
        "format": "parquet",
        "write_hills": False,
        "write_ms1": False,
        "stop_after_hills": False,
        "no_mono_hills": True,
        "no_hill_list": True,
        "write_extra_details": False,
        "overwrite": False,
        "use64": False,
        "intensity_decimals": "0",
        "parquet_engine": "pyarrow",
        "parquet_compression": "zstd",
        "parquet_compression_level": 6,
        "parquet_row_group_size": 100,
        "parquet_sort": "mz_rt",
    }
    manager = CompactOutputManager(args)
    manager.append_features(
        [
            {
                "massCalib": 997.985447,
                "rtApex": 120.0,
                "intensityApex": 2.5,
                "intensitySum": 10.5,
                "charge": 2,
                "nIsotopes": 3,
                "nScans": 5,
                "mz": 500.0,
                "rtStart": 60.0,
                "rtEnd": 180.0,
                "scan_apex_number": 101,
                "isoerror": 0.25,
                "isoerror2": -100,
                "feature_idx": 1,
                "area_sum": 42.25,
            }
        ]
    )
    manager.finalize()
    with duckdb.connect() as connection:
        projection = ", ".join(
            '"%s"' % name for name in DOWNSTREAM_FEATURE_COLUMNS
        )
        row = connection.execute(
            "SELECT %s FROM read_parquet(?)" % projection, [str(output)]
        ).fetchone()
    assert row[DOWNSTREAM_FEATURE_COLUMNS.index("feature_idx")] == 1
