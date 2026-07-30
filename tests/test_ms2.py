import gzip

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biosaur2 import preprocessing, utils
from biosaur2.legacy_output import CompactOutputManager
from biosaur2.schema import (
    MS2_MISSING_CHARGE,
    MS2_MISSING_PRECURSOR_MZ,
    compact_schemas,
)


def _ms1(native_id, rt, faims=None):
    spectrum = {
        "id": native_id,
        "ms level": 1,
        "scanList": {"scan": [{"scan start time": rt}]},
        "m/z array": np.asarray([500.0]),
        "intensity array": np.asarray([100.0]),
        "total ion current": 100.0,
    }
    if faims is not None:
        spectrum["FAIMS compensation voltage"] = faims
    return spectrum


def _ms2(native_id, rt, spectrum_ref=None, selected_mz=500.0, charge=2, faims=None):
    precursor = {
        "isolationWindow": {
            "isolation window target m/z": 499.5,
            "isolation window lower offset": 0.5,
            "isolation window upper offset": 0.5,
        },
        "selectedIonList": {"selectedIon": [{}]},
    }
    if spectrum_ref is not None:
        precursor["spectrumRef"] = spectrum_ref
    selected = precursor["selectedIonList"]["selectedIon"][0]
    if selected_mz is not None:
        selected["selected ion m/z"] = selected_mz
    if charge is not None:
        selected["charge state"] = charge
    spectrum = {
        "id": native_id,
        "ms level": 2,
        "scanList": {"scan": [{"scan start time": rt}]},
        "precursorList": {"precursor": [precursor]},
    }
    if faims is not None:
        spectrum["FAIMS compensation voltage"] = faims
    return spectrum


def _args(**updates):
    args = {
        "file": "run.mzML",
        "combine_every": 1,
        "mini": 1,
        "minmz": 350,
        "maxmz": 1500,
        "input_rt_unit": "seconds",
        "write_ms1": False,
        "write_ms2": True,
    }
    args.update(updates)
    return args


def test_interleaved_ms2_indexes_and_precursor_resolution(monkeypatch):
    spectra = [
        _ms1("controller scan=10", 1.0),
        _ms2("controller scan=11", 2.0, "controller scan=10"),
        _ms2("controller scan=12", 3.0),
        _ms1("controller scan=13", 4.0),
        _ms2("controller scan=14", 5.0, "controller scan=13"),
    ]
    monkeypatch.setattr(
        utils, "iter_ms1_and_ms2_metadata", lambda _path: iter(spectra)
    )

    result = preprocessing.ingest_mzml(_args())

    assert [spectrum["scan_index"] for spectrum in result.spectra] == [0, 1]
    assert [row["ms2_index"] for row in result.ms2_rows] == [0, 1, 2]
    assert [row["spectrum_index"] for row in result.ms2_rows] == [1, 2, 4]
    assert [row["precursor_ms1_index"] for row in result.ms2_rows] == [0, 0, 1]
    assert [row["precursor_resolution"] for row in result.ms2_rows] == [
        "spectrum_ref",
        "preceding_ms1",
        "spectrum_ref",
    ]


def test_ms2_fallback_flags_and_faims_zero(monkeypatch):
    spectra = [
        _ms1("scan=1", 1.0),
        _ms2("scan=2", 2.0, selected_mz=None, charge=None, faims=0),
    ]
    spectra[1]["precursorList"]["precursor"][0]["isolationWindow"] = {}
    monkeypatch.setattr(
        utils, "iter_ms1_and_ms2_metadata", lambda _path: iter(spectra)
    )

    row = preprocessing.ingest_mzml(_args()).ms2_rows[0]

    assert row["precursor_mz"] is None
    assert row["precursor_mz_source"] is None
    assert row["charge"] is None
    assert row["faims_cv"] == 0.0
    assert row["metadata_flags"] & MS2_MISSING_PRECURSOR_MZ
    assert row["metadata_flags"] & MS2_MISSING_CHARGE


def test_selected_ion_and_isolation_target_are_distinct(monkeypatch):
    spectra = [_ms1("scan=1", 1.0), _ms2("scan=2", 2.0, selected_mz=None)]
    monkeypatch.setattr(
        utils, "iter_ms1_and_ms2_metadata", lambda _path: iter(spectra)
    )

    row = preprocessing.ingest_mzml(_args()).ms2_rows[0]

    assert row["selected_ion_mz"] is None
    assert row["isolation_target_mz"] == 499.5
    assert row["precursor_mz"] == 499.5
    assert row["precursor_mz_source"] == "isolation_target"


def test_ingestion_uses_one_reader_for_every_requested_sidecar(monkeypatch):
    calls = []

    def ms1_reader(_path):
        calls.append("ms1")
        return iter([_ms1("scan=1", 1.0)])

    def full_reader(_path):
        calls.append("full")
        return iter([_ms1("scan=1", 1.0), _ms2("scan=2", 2.0)])

    monkeypatch.setattr(utils, "iter_ms1_spectra", ms1_reader)
    monkeypatch.setattr(utils, "iter_ms1_and_ms2_metadata", full_reader)

    preprocessing.ingest_mzml(_args(write_ms1=False, write_ms2=False))
    preprocessing.ingest_mzml(_args(write_ms1=True, write_ms2=False))
    preprocessing.ingest_mzml(_args(write_ms1=False, write_ms2=True))
    preprocessing.ingest_mzml(_args(write_ms1=True, write_ms2=True))

    assert calls == ["ms1", "ms1", "full", "full"]


def test_plain_and_gzip_mzml_are_logically_equivalent(tmp_path):
    compressed = "examples/PXD010154_1554451_middle.mzML.gz"
    plain = tmp_path / "fixture.mzML"
    with gzip.open(compressed, "rb") as source:
        plain.write_bytes(source.read())

    plain_result = preprocessing.ingest_mzml(_args(file=str(plain)))
    gzip_result = preprocessing.ingest_mzml(_args(file=compressed))

    assert len(plain_result.spectra) == len(gzip_result.spectra)
    plain_rows = [{k: v for k, v in row.items() if k != "run_id"} for row in plain_result.ms2_rows]
    gzip_rows = [{k: v for k, v in row.items() if k != "run_id"} for row in gzip_result.ms2_rows]
    assert plain_rows == gzip_rows


@pytest.mark.parametrize("output_format", ["tsv", "parquet"])
def test_ms2_schema_and_atomic_sidecar_uses_unified_format(
    tmp_path, output_format
):
    source = tmp_path / "sample.mzML.gz"
    source.write_bytes(b"input")
    args = {
        "file": str(source),
        "o": str(tmp_path / ("custom.features." + output_format)),
        "format": output_format,
        "write_hills": False,
        "write_ms1": False,
        "write_ms2": True,
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
        "combine_every": 1,
    }
    manager = CompactOutputManager(args)
    manager.append_ms2(
        [
            {
                "run_id": "sample",
                "ms2_event_id": 0,
                "ms2_index": 0,
                "spectrum_index": 1,
                "metadata_flags": 0,
            }
        ]
    )
    target = tmp_path / ("sample.ms2." + output_format)
    assert not target.exists()
    manager.finalize()
    if output_format == "tsv":
        lines = target.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("run_id\tms2_event_id\tms2_index")
        assert lines[1].startswith("sample\t0\t0")
        return
    schema = pq.read_schema(target)
    assert schema == compact_schemas()["ms2"].with_metadata(schema.metadata)
    assert schema.field("selected_ion_mz").type == pa.float64()
    assert schema.field("charge").type == pa.int16()
    assert schema.field("metadata_flags").type == pa.uint16()
