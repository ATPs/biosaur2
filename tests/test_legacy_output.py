import pyarrow.parquet as pq

from biosaur2.legacy_output import CompactOutputManager, merge_hybrid_output_rows


def _args(tmp_path, **updates):
    values = {
        "file": str(tmp_path / "sample.mzML"),
        "o": str(tmp_path / "legacy.tsv"),
        "format": "tsv",
        "write_hills": False,
        "write_ms1": False,
        "stop_after_hills": False,
        "no_mono_hills": True,
        "no_hill_list": False,
        "write_extra_details": False,
        "tsv_float_decimals": "2",
        "intensity_decimals": "0",
        "overwrite": False,
    }
    values.update(updates)
    return values


def test_legacy_units_precision_and_intensity_rounding(tmp_path):
    manager = CompactOutputManager(_args(tmp_path))
    manager.append_features(
        [
            {
                "massCalib": 1000.123,
                "rtApex": 120.0,
                "intensityApex": 2.5,
                "intensitySum": -1.5,
                "charge": 2,
                "nIsotopes": 2,
                "nScans": 4,
                "mz": 500.123,
                "rtStart": 60.0,
                "rtEnd": 180.0,
                "FAIMS": None,
                "im": 0.0,
                "scanApex": 9,
                "isoerror": 0.0,
                "isoerror2": 0.0,
            }
        ]
    )
    manager.finalize()
    lines = (tmp_path / "legacy.tsv").read_text().splitlines()
    values = dict(zip(lines[0].split("\t"), lines[1].split("\t")))
    assert values["rtStart"] == "1"
    assert values["rtApex"] == "2"
    assert values["rtEnd"] == "3"
    assert values["intensityApex"] == "3"
    assert values["intensitySum"] == "-2"
    assert values["FAIMS"] == ""


def test_empty_legacy_parquet_has_typed_schema(tmp_path):
    args = _args(
        tmp_path,
        o=str(tmp_path / "legacy.parquet"),
        format="parquet",
    )
    manager = CompactOutputManager(args)
    manager.finalize()
    schema = pq.read_schema(tmp_path / "legacy.parquet")
    assert schema.field("charge").type.bit_width == 8
    assert schema.field("nScans").type.bit_width == 16
    assert schema.field("mz").type.bit_width == 32
    assert b"biosaur2_schema_version" in schema.metadata


def test_hybrid_merge_splits_feature_linked_ms2_and_merges_assay():
    features, ms2_events, identifications = merge_hybrid_output_rows(
        [
            {"feature_idx": 1, "rtApex": 60.0},
            {"feature_idx": 2, "rtApex": 120.0},
        ],
        [
            {"feature_id": 1, "quant_value": 10.0},
            {"feature_id": 2, "quant_value": 20.0},
        ],
        [
            {"ms2_event_id": 10, "feature_id": 1, "status": "linked"},
            {"ms2_event_id": 11, "feature_id": 1, "status": "linked"},
            {"ms2_event_id": 12, "feature_id": None, "status": "unresolved"},
        ],
        [
            {
                "run_id": "run", "ms2_event_id": 10,
                "native_id": "scan=101", "native_scan_number": 101,
                "rt_sec": 60.0, "precursor_mz": 500.0, "charge": 2,
            },
            {
                "run_id": "run", "ms2_event_id": 11,
                "native_id": "scan=102", "native_scan_number": 102,
                "rt_sec": 61.0, "precursor_mz": 600.0, "charge": 3,
            },
            {"run_id": "run", "ms2_event_id": 12, "metadata_flags": 0},
        ],
        [
            {
                "run_id": "run",
                "psm_id": "psm-1",
                "ms2_event_id": 12,
                "mapping_status": "mapped",
                "q_value": 0.001,
            }
        ],
        [
            {
                "psm_id": "psm-1",
                "ms2_event_id": 12,
                "assay_id": 7,
                "charge": 2,
                "conflict_status": "unique",
            }
        ],
        {"no_mono_hills": True, "write_extra_details": False},
    )

    assert all("ms2_events" not in row for row in features)
    assert ms2_events == [
        {
            "feature_idx": 1, "ms2_event_id": 10,
            "native_id": "scan=101", "native_scan_number": 101,
            "rt_sec": 60.0, "precursor_mz": 500.0, "charge": 2,
        },
        {
            "feature_idx": 1, "ms2_event_id": 11,
            "native_id": "scan=102", "native_scan_number": 102,
            "rt_sec": 61.0, "precursor_mz": 600.0, "charge": 3,
        },
    ]
    assert identifications[0]["ms2_event_id"] == 12
    assert identifications[0]["assay_id"] == 7
    assert identifications[0]["assay_charge"] == 2
