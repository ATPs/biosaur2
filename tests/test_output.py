import json
import pyarrow.parquet as pq
import pytest

import biosaur2.output as output_module
import biosaur2.search as search_module
from biosaur2.legacy_output import CompactOutputManager, compact_ms1
from biosaur2.output import planned_output_paths, round_intensity
from biosaur2.schema import (
    LINKED_MS2_EVENT_COLUMNS,
    feature_columns,
    hybrid_feature_columns,
)


def _args(tmp_path, **updates):
    values = {
        "file": str(tmp_path / "sample.mzML.gz"),
        "o": str(tmp_path / "sample.features.tsv"),
        "format": "tsv",
        "write_hills": False,
        "write_ms1": False,
        "stop_after_hills": False,
        "no_mono_hills": True,
        "no_hill_list": True,
        "write_extra_details": False,
        "overwrite": False,
        "use64": False,
        "intensity_decimals": "0",
        "tsv_float_decimals": "roundtrip",
        "parquet_compression": "zstd",
        "parquet_compression_level": 6,
        "parquet_row_group_size": 100,
        "parquet_sort": "mz_rt",
        "parquet_engine": "pyarrow",
    }
    values.update(updates)
    return values


def _feature(feature_idx=1, mz=500.0):
    return {
        "massCalib": 997.985447,
        "rtApex": 120.0,
        "intensityApex": 2.5,
        "intensitySum": 10.5,
        "charge": 2,
        "nIsotopes": 3,
        "nScans": 5,
        "mz": mz,
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


def test_planned_hybrid_and_duckdb_paths_are_per_input(tmp_path):
    source = tmp_path / "sample.mzML.gz"
    output = tmp_path / "results"
    hybrid = {
        "file": str(source),
        "o": str(output / "sample.features.parquet"),
        "format": "parquet",
        "feature_mode": "hybrid",
    }
    assert planned_output_paths(hybrid) == [
        output / "sample.features.parquet",
        output / "sample.ms2_events.parquet",
        output / "sample.identifications.parquet",
    ]

    database = {**hybrid, "o": str(output), "format": "duckdb"}
    assert planned_output_paths(database) == [
        output / "sample.biosaur2.duckdb"
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(2.5, 3.0), (1.5, 2.0), (-1.5, -2.0), (-2.5, -3.0)],
)
def test_round_intensity_is_half_away_from_zero(value, expected):
    assert round_intensity(value, 0) == expected


def test_ms1_scan_id_prefers_native_number_then_falls_back_to_index():
    args = {"intensity_decimals": "0"}
    assert compact_ms1(
        {"scan_number": 101, "scan_index": 4, "rt_sec": 1.0}, args
    )["scan_id"] == 101
    assert compact_ms1(
        {"scan_number": None, "scan_index": 4, "rt_sec": 1.0}, args
    )["scan_id"] == 5


def test_tsv_is_atomic_compact_and_refuses_existing_output(tmp_path):
    args = _args(tmp_path)
    manager = CompactOutputManager(args)
    manager.append_features([_feature()])
    target = tmp_path / "sample.features.tsv"
    assert not target.exists()
    manager.finalize()
    lines = target.read_text().splitlines()
    assert tuple(lines[0].split("\t")) == feature_columns(False, False)
    row = dict(zip(lines[0].split("\t"), lines[1].split("\t")))
    assert row["massCalib"] == "997.985447"
    assert row["rtApex"] == "2.0"
    assert row["intensityApex"] == "3.0"
    assert row["FAIMS"] == ""
    assert row["isoerror2"] == ""
    assert row["feature_idx"] == "1"
    with pytest.raises(FileExistsError):
        CompactOutputManager(args)


def test_abort_leaves_no_final_output(tmp_path):
    manager = CompactOutputManager(_args(tmp_path))
    manager.append_features([_feature()])
    manager.abort()
    assert not (tmp_path / "sample.features.tsv").exists()


def test_empty_compact_outputs_keep_explicit_schema(tmp_path):
    tsv_manager = CompactOutputManager(_args(tmp_path))
    tsv_manager.finalize()
    assert len((tmp_path / "sample.features.tsv").read_text().splitlines()) == 1

    parquet_args = _args(
        tmp_path,
        o=str(tmp_path / "empty.features.parquet"),
        format="parquet",
    )
    parquet_manager = CompactOutputManager(parquet_args)
    parquet_manager.finalize()
    schema = pq.read_schema(tmp_path / "empty.features.parquet")
    assert schema.names == list(feature_columns(False, False))
    assert schema.field("charge").type.bit_width == 8
    assert schema.field("nScans").type.bit_width == 16
    assert schema.field("feature_idx").type.bit_width == 32
    assert schema.field("mz").type.bit_width == 32
    assert b"biosaur2_schema_version" in schema.metadata


def test_parquet_is_one_feature_file_without_sidecars(tmp_path):
    args = _args(
        tmp_path,
        o=str(tmp_path / "compact.features.parquet"),
        format="parquet",
    )
    manager = CompactOutputManager(args)
    manager.append_features([_feature()])
    manager.finalize()
    assert [path.name for path in tmp_path.glob("*.parquet")] == [
        "compact.features.parquet"
    ]
    assert not list(tmp_path.glob("*.json"))
    row = pq.read_table(tmp_path / "compact.features.parquet").to_pylist()[0]
    assert row["rtStart"] == 1.0
    assert row["rtEnd"] == 3.0
    assert row["area_sum"] == 42.25


def test_pyarrow_provenance_is_final_and_omits_input_hash(tmp_path):
    args = _args(
        tmp_path,
        o=str(tmp_path / "provenance.features.parquet"),
        format="parquet",
    )
    input_path = tmp_path / "sample.mzML.gz"
    input_path.write_bytes(b"input")
    manager = CompactOutputManager(args)
    manager.append_features([_feature()])
    args["hill_calibration"] = {"status": "applied", "shift": 0.25}
    args["isotope_calibration"] = {"late": {"status": "applied"}}
    args["_area_sum_approximate"] = True
    manager.finalize()

    metadata = pq.ParquetFile(
        tmp_path / "provenance.features.parquet"
    ).metadata.metadata
    assert metadata[b"biosaur2_input_size"] == b"5"
    assert b"biosaur2_input_sha256" not in metadata
    assert b"biosaur2_input_hash_algorithm" not in metadata
    assert b"late" in metadata[b"biosaur2_calibration"]
    assert metadata[b"biosaur2_area_sum_rt"] == b"approximated_from_hill_anchors"


def test_hybrid_summary_is_persisted_in_merged_output_metadata(tmp_path):
    args = _args(
        tmp_path,
        o=str(tmp_path / "hybrid.features.parquet"),
        format="parquet",
        feature_mode="hybrid",
    )
    (tmp_path / "sample.mzML.gz").write_bytes(b"input")
    manager = CompactOutputManager(args)
    args["_hybrid_summary"] = {
        "audit_row_count": 1,
        "audit_status_counts": {"generic_decoy_only": 1},
    }
    manager.finalize()
    metadata = pq.ParquetFile(
        tmp_path / "hybrid.features.parquet"
    ).metadata.metadata
    assert metadata[b"biosaur2_hybrid_schema_version"] == b"8"
    assert json.loads(metadata[b"biosaur2_hybrid_summary_json"]) == args[
        "_hybrid_summary"
    ]
    assert (tmp_path / "hybrid.identifications.parquet").is_file()
    assert (tmp_path / "hybrid.ms2_events.parquet").is_file()


def test_hybrid_splits_compact_linked_ms2_rows_from_features(tmp_path):
    args = _args(
        tmp_path,
        o=str(tmp_path / "hybrid.features.parquet"),
        format="parquet",
        feature_mode="hybrid",
    )
    manager = CompactOutputManager(args)
    manager.append_hybrid_results(
        [_feature()],
        [{
            "feature_id": 1,
            "quant_value": 42.0,
            "area_envelope_raw": 43.0,
            "area_envelope_corrected": 42.0,
            "area_mono_raw": 22.0,
            "area_mono_corrected": 21.0,
            "envelope_apex": 9.0,
            "quant_envelope_apex": 9.0,
            "feature_quality_score": 0.9,
            "isotope_cosine": 0.9,
        }],
        [{"ms2_event_id": 7, "feature_id": 1}],
        [{
            "ms2_event_id": 7,
            "native_id": "controllerType=0 scan=42",
            "native_scan_number": 42,
            "rt_sec": 120.5,
            "precursor_mz": 500.25,
            "charge": 2,
        }],
        [],
        [],
    )
    manager.finalize()

    feature_schema = pq.read_schema(tmp_path / "hybrid.features.parquet")
    assert feature_schema.names == list(hybrid_feature_columns())
    for removed in (
        "ms2_events",
        "mono_hills_scan_lists",
        "area_envelope_raw",
        "area_envelope_corrected",
        "area_mono_raw",
        "area_mono_corrected",
        "envelope_apex",
        "feature_quality_score",
    ):
        assert removed not in feature_schema.names

    event_table = pq.read_table(tmp_path / "hybrid.ms2_events.parquet")
    assert event_table.column_names == list(LINKED_MS2_EVENT_COLUMNS)
    assert event_table.to_pylist() == [{
        "feature_idx": 1,
        "ms2_event_id": 7,
        "native_id": "controllerType=0 scan=42",
        "native_scan_number": 42,
        "rt_sec": 120.5,
        "precursor_mz": 500.25,
        "charge": 2,
    }]


def test_hybrid_diagnostic_switches_restore_large_optional_columns(tmp_path):
    args = _args(
        tmp_path,
        o=str(tmp_path / "diagnostic.features.parquet"),
        format="parquet",
        feature_mode="hybrid",
        write_mono_hills=True,
        write_quant_details=True,
    )
    manager = CompactOutputManager(args)
    manager.finalize()
    names = pq.read_schema(tmp_path / "diagnostic.features.parquet").names
    assert "mono_hills_scan_lists" in names
    assert "mono_hills_intensity_list" in names
    assert "area_envelope_raw" in names
    assert "area_envelope_corrected" in names
    assert "area_mono_raw" in names
    assert "area_mono_corrected" in names
    assert "envelope_apex" not in names
    assert "feature_quality_score" not in names


def test_hybrid_tsv_writes_only_tsv_feature_and_identification_tables(tmp_path):
    args = _args(
        tmp_path,
        o=str(tmp_path / "hybrid.features.tsv"),
        feature_mode="hybrid",
    )
    manager = CompactOutputManager(args)
    manager.finalize()

    feature_path = tmp_path / "hybrid.features.tsv"
    ms2_events_path = tmp_path / "hybrid.ms2_events.tsv"
    identification_path = tmp_path / "hybrid.identifications.tsv"
    assert feature_path.is_file()
    assert ms2_events_path.is_file()
    assert identification_path.is_file()
    assert "canonical_peptidoform" in identification_path.read_text().splitlines()[0]
    assert not list(tmp_path.glob("*.parquet"))


def test_write_extra_details_stays_in_same_feature_file(tmp_path):
    args = _args(
        tmp_path,
        o=str(tmp_path / "details.features.parquet"),
        format="parquet",
        write_extra_details=True,
    )
    feature = _feature()
    feature.update(
        {
            "isotopes": [
                {
                    "isotope_number": 1,
                    "isotope_hill_idx": 11,
                    "isotope_idx": 2,
                    "cos_cor": 0.9,
                    "mass_diff_ppm": 0.25,
                }
            ],
            "intensity_array_for_cos_corr": [[1.0, 2.0], [3.0, 4.0]],
            "monoisotope hill idx": 10,
            "monoisotope idx": 1,
        }
    )
    manager = CompactOutputManager(args)
    manager.append_features([feature])
    manager.finalize()
    table = pq.read_table(tmp_path / "details.features.parquet")
    assert table.schema.names == list(feature_columns(False, True))
    assert table.column("isotopes")[0].as_py()[0]["isotope_hill_idx"] == 11
    assert [path.name for path in tmp_path.glob("*.parquet")] == [
        "details.features.parquet"
    ]


def test_default_nested_arrays_and_hill_point_rt_are_typed(tmp_path):
    args = _args(
        tmp_path,
        o=str(tmp_path / "nested.features.parquet"),
        format="parquet",
        no_mono_hills=False,
        no_hill_list=False,
        write_hills=True,
    )
    feature = _feature()
    feature["mono_hills_scan_lists"] = [1, 2]
    feature["mono_hills_intensity_list"] = [2.5, 3.5]
    manager = CompactOutputManager(args)
    manager.append_features([feature])
    manager.append_hills(
        [
            {
                "rtApex": 120.0,
                "intensityApex": 3.5,
                "intensitySum": 6.0,
                "nScans": 2,
                "mz": 500.0,
                "rtStart": 60.0,
                "rtEnd": 120.0,
                "scan_apex_number": 102,
                "hill_idx": 10,
                "feature_idx": 1,
                "hills_scan_lists": [1, 2],
                "hills_intensity_list": [2.5, 3.5],
                "hills_mz_array": [500.0, 500.01],
                "_hill_points": [{"rt_sec": 60.0}, {"rt_sec": 120.0}],
            }
        ]
    )
    manager.finalize()
    feature_row = pq.read_table(tmp_path / "nested.features.parquet").to_pylist()[0]
    hill_row = pq.read_table(tmp_path / "nested.hills.parquet").to_pylist()[0]
    assert feature_row["mono_hills_scan_lists"] == [1, 2]
    assert feature_row["mono_hills_intensity_list"] == [3.0, 4.0]
    assert hill_row["hills_rt_list"] == [60.0, 120.0]


def test_narrow_integer_overflow_is_an_output_error(tmp_path):
    args = _args(
        tmp_path,
        o=str(tmp_path / "overflow.features.parquet"),
        format="parquet",
    )
    feature = _feature()
    feature["charge"] = 128
    manager = CompactOutputManager(args)
    with pytest.raises((OverflowError, ValueError)):
        manager.append_features([feature])
    manager.abort()


def test_default_duckdb_selection_warns_and_falls_back_to_pyarrow(
    tmp_path, monkeypatch, caplog
):
    args = _args(
        tmp_path,
        o=str(tmp_path / "fallback.features.parquet"),
        format="parquet",
        parquet_engine="duckdb",
    )

    def unavailable(_args):
        raise ImportError("simulated missing DuckDB")

    monkeypatch.setattr(search_module, "DuckDBOutputManager", unavailable)
    manager = search_module._create_output_manager(args)
    assert isinstance(manager, CompactOutputManager)
    assert args["parquet_engine"] == "pyarrow"
    assert "falling back to PyArrow" in caplog.text
    manager.append_features([_feature()])
    manager.finalize()
    assert (tmp_path / "fallback.features.parquet").exists()


def test_failed_overwrite_restores_existing_final(tmp_path, monkeypatch):
    target = tmp_path / "sample.features.tsv"
    target.write_text("old-data\n")
    manager = CompactOutputManager(_args(tmp_path, overwrite=True))
    manager.append_features([_feature()])
    real_replace = output_module.os.replace
    calls = 0

    def fail_new_publish(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(output_module.os, "replace", fail_new_publish)
    with pytest.raises(OSError, match="simulated"):
        manager.finalize()
    assert target.read_text() == "old-data\n"
