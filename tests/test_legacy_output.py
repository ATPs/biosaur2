import pyarrow.parquet as pq

from biosaur2.legacy_output import CompactOutputManager


def _args(tmp_path, **updates):
    values = {
        "file": str(tmp_path / "sample.mzML"),
        "o": str(tmp_path / "legacy.tsv"),
        "feature_format": "tsv",
        "hills_format": "tsv",
        "ms1_format": "tsv",
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
        feature_format="parquet",
    )
    manager = CompactOutputManager(args)
    manager.finalize()
    schema = pq.read_schema(tmp_path / "legacy.parquet")
    assert schema.field("charge").type.bit_width == 8
    assert schema.field("nScans").type.bit_width == 16
    assert schema.field("mz").type.bit_width == 32
    assert b"biosaur2_schema_version" in schema.metadata
