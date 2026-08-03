import builtins
import json

from biosaur2.duckdb_output import DuckDBOutputManager, uses_duckdb
from biosaur2.schema import feature_columns
import pyarrow.parquet as pq
import pytest


duckdb = pytest.importorskip("duckdb")


def _args(tmp_path, database=False, **updates):
    input_path = tmp_path / "sample.mzML"
    input_path.write_bytes(b"test-input")
    values = {
        "file": str(input_path),
        "o": str(
            tmp_path / ("result.duckdb" if database else "result.features.parquet")
        ),
        "format": "duckdb" if database else "parquet",
        "parquet_engine": "duckdb",
        "parquet_compression": "zstd",
        "parquet_compression_level": 6,
        "parquet_row_group_size": 100,
        "parquet_sort": "mz_rt",
        "write_hills": False,
        "write_ms1": False,
        "stop_after_hills": False,
        "no_mono_hills": True,
        "no_hill_list": True,
        "write_extra_details": False,
        "overwrite": False,
        "use64": False,
        "intensity_decimals": "0",
    }
    values.update(updates)
    return values


def _feature(feature_idx=1):
    return {
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
        "faims_cv": None,
        "ion_mobility_1_over_k0": None,
        "scan_apex_number": 101,
        "isoerror": 0.25,
        "isoerror2": -100,
        "feature_idx": feature_idx,
        "area_sum": 42.25,
    }


def _hill(hill_idx=9):
    return {
        "rtApex": 120.0,
        "intensityApex": 8.0,
        "intensitySum": 10.0,
        "nScans": 2,
        "mz": 500.0,
        "rtStart": 60.0,
        "rtEnd": 180.0,
        "scan_apex_number": 101,
        "hill_idx": hill_idx,
        "feature_idx": -1,
    }


def test_duckdb_database_contains_only_compact_requested_tables(tmp_path):
    args = _args(tmp_path, database=True)
    manager = DuckDBOutputManager(args)
    manager.append_features([_feature()])
    args["hill_calibration"] = {"status": "applied", "shift": 0.25}
    manager.finalize()
    with duckdb.connect(str(tmp_path / "result.duckdb"), read_only=True) as connection:
        tables = {row[0] for row in connection.execute("show tables").fetchall()}
        assert tables == {"features", "runs"}
        description = connection.execute("describe features").fetchall()
        assert [row[0] for row in description] == list(feature_columns(False, False))
        assert connection.execute("select count(*) from features").fetchone()[0] == 1
        assert connection.execute("select feature_idx from features").fetchone()[0] == 1
        run_columns = [
            row[0] for row in connection.execute("describe runs").fetchall()
        ]
        assert "input_size" in run_columns
        assert "input_sha256" not in run_columns
        provenance = json.loads(
            connection.execute("select provenance_json from runs").fetchone()[0]
        )
        assert provenance["calibration"]["hill"]["shift"] == 0.25


def test_duckdb_database_includes_explicit_hills_and_ms1(tmp_path):
    args = _args(tmp_path, database=True, write_hills=True, write_ms1=True)
    manager = DuckDBOutputManager(args)
    manager.append_features([_feature()])
    manager.append_hills([_hill()])
    manager.append_ms1([{"scan_number": 101, "rt_sec": 120.0, "total_intensity": 2.5}])
    manager.finalize()
    with duckdb.connect(str(tmp_path / "result.duckdb"), read_only=True) as connection:
        assert {row[0] for row in connection.execute("show tables").fetchall()} == {
            "features",
            "hills",
            "ms1",
            "runs",
        }
        assert connection.execute("select feature_idx from hills").fetchone()[0] == -1
        assert connection.execute("select total_intensity from ms1").fetchone()[0] == 3.0


def test_duckdb_parquet_is_v2_single_file_with_compact_types(tmp_path):
    args = _args(tmp_path)
    manager = DuckDBOutputManager(args)
    manager.append_features([_feature()])
    args["hill_calibration"] = {"status": "applied", "shift": 0.25}
    args["isotope_calibration"] = {"late": {"status": "applied"}}
    args["_area_sum_approximate"] = True
    manager.finalize()
    output = tmp_path / "result.features.parquet"
    schema = pq.read_schema(output)
    assert schema.names == list(feature_columns(False, False))
    assert schema.field("charge").type.bit_width == 8
    assert schema.field("nScans").type.bit_width == 16
    assert schema.field("mz").type.bit_width == 32
    assert b"biosaur2_schema_version" in schema.metadata
    metadata = pq.ParquetFile(output).metadata.metadata
    provenance = json.loads(metadata[b"biosaur2_provenance_json"])
    assert provenance["calibration"]["hill"]["shift"] == 0.25
    assert provenance["calibration"]["isotope"]["late"]["status"] == "applied"
    assert provenance["area_sum_rt"] == "approximated_from_hill_anchors"
    assert "input_sha256" not in provenance
    assert [path.name for path in tmp_path.glob("*.parquet")] == [output.name]


def test_duckdb_hybrid_summary_is_persisted_in_merged_output_metadata(tmp_path):
    args = _args(tmp_path, feature_mode="hybrid")
    manager = DuckDBOutputManager(args)
    args["_hybrid_summary"] = {
        "audit_row_count": 2,
        "generic_summary": {
            "competition_counts": {"decoy_only_candidate_count": 1}
        },
    }
    manager.finalize()
    metadata = pq.ParquetFile(tmp_path / "result.features.parquet").metadata.metadata
    provenance = json.loads(metadata[b"biosaur2_provenance_json"])
    assert provenance["hybrid_schema_version"] == "6"
    assert json.loads(provenance["hybrid_summary_json"]) == args["_hybrid_summary"]
    assert (tmp_path / "result.identifications.parquet").is_file()
    assert not (tmp_path / "result.identifications.tsv").exists()


def test_hybrid_duckdb_writes_no_parquet_or_tsv_sidecars(tmp_path):
    args = _args(tmp_path, database=True, feature_mode="hybrid")
    manager = DuckDBOutputManager(args)
    manager.finalize()

    database_path = tmp_path / "result.duckdb"
    assert database_path.is_file()
    assert not list(tmp_path.glob("*.parquet"))
    assert not list(tmp_path.glob("*.tsv"))
    with duckdb.connect(str(database_path), read_only=True) as connection:
        assert {row[0] for row in connection.execute("show tables").fetchall()} == {
            "features",
            "identifications",
            "runs",
        }


def test_unified_tsv_format_does_not_select_duckdb(tmp_path):
    args = _args(
        tmp_path,
        o=str(tmp_path / "mixed.features.tsv"),
        format="tsv",
        write_hills=True,
        write_ms1=True,
        tsv_float_decimals="roundtrip",
    )
    assert not uses_duckdb(args)


def test_duckdb_writes_all_three_requested_parquet_products(tmp_path):
    args = _args(
        tmp_path,
        write_hills=True,
        write_ms1=True,
    )
    manager = DuckDBOutputManager(args)
    manager.append_features([_feature()])
    manager.append_hills([_hill()])
    manager.append_ms1(
        [{"scan_number": 101, "rt_sec": 120.0, "total_intensity": 2.5}]
    )
    manager.finalize()
    assert {path.name for path in tmp_path.glob("*.parquet")} == {
        "result.features.parquet",
        "result.hills.parquet",
        "result.ms1.parquet",
    }


def test_explicit_pyarrow_does_not_select_duckdb_for_unified_tsv(tmp_path):
    args = _args(
        tmp_path,
        format="tsv",
        write_hills=True,
        parquet_engine="pyarrow",
    )
    assert not uses_duckdb(args)


def test_duckdb_staging_database_uses_cache_workspace(tmp_path):
    staging = tmp_path / "staging"
    manager = DuckDBOutputManager(_args(tmp_path, _cache_workspace=str(staging)))
    assert manager.staging_path.parent == staging / "output_staging"
    manager.abort()


def test_optional_dependency_error_is_concise(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def missing_duckdb(name, *args, **kwargs):
        if name == "duckdb":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_duckdb)
    with pytest.raises(ImportError, match="simulated missing dependency"):
        DuckDBOutputManager(_args(tmp_path))
