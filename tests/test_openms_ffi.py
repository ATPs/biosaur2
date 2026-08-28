import json
import os
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biosaur2 import openms_ffi
from biosaur2 import search as search_module
from biosaur2.external_mbr import sidecar_rows


_FEATURE_XML = """<?xml version="1.0"?>
<featureMap><featureList count="1"><feature id="f1">
<position dim="0">120</position><position dim="1">500</position>
<intensity>1000</intensity><overallquality>1.5</overallquality><charge>2</charge>
<PeptideIdentification><UserParam name="psm_id" value="psm-1"/><UserParam name="ms2_event_id" value="7"/></PeptideIdentification>
<UserParam name="leftWidth" value="110"/><UserParam name="rightWidth" value="130"/>
<UserParam name="peak_apices_sum" value="50"/><UserParam name="raw_intensity" value="200"/>
</feature></featureList></featureMap>"""


def _candidate(**updates):
    value = {
        "psm_id": "psm-1", "ms2_event_id": 7,
        "canonical_peptidoform": "PEPTIDE", "charge": 2,
        "rt_sec": 120.0, "selected_ion_mz": 500.0,
        "q_value": 0.001, "native_scan_number": 42, "faims_cv": None,
    }
    value.update(updates)
    return value


def _fake_openms(monkeypatch, feature_xml=_FEATURE_XML):
    def fake(*args):
        args[3].write_text(feature_xml)

    monkeypatch.setattr(openms_ffi, "_run_openms", fake)
    monkeypatch.setattr(openms_ffi, "_openms_version", lambda _exe: "test")


def test_idxml_preserves_source_keys_and_escapes_peptidoform(tmp_path):
    path = tmp_path / "input.idXML"
    openms_ffi._write_idxml(path, [_candidate(
        psm_id="psm&1", canonical_peptidoform="[UNIMOD:1]-PEPTIDE",
    )])
    root = openms_ffi.etree.parse(str(path)).getroot()
    peptide = next(root.iter("PeptideIdentification"))
    assert peptide.get("spectrum_reference").endswith("scan=42")
    assert next(root.iter("PeptideHit")).get("sequence") == ".(UniMod:1)PEPTIDE"
    assert openms_ffi._value(peptide, "psm_id") == "psm&1"
    assert openms_ffi._value(peptide, "ms2_event_id") == "7"


def test_execute_rescue_maps_target_and_preserves_openms_quantification(monkeypatch):
    _fake_openms(monkeypatch)
    features, quant, links, deltas, summary = openms_ffi.execute_rescue(
        source="input.mzML.gz", run_id="run", candidates=[_candidate()],
        existing_features=[], ms1_rows=[
            {"rt_sec": 110.0, "scan_number": 1, "faims_cv": None},
            {"rt_sec": 120.0, "scan_number": 2, "faims_cv": None},
            {"rt_sec": 130.0, "scan_number": 3, "faims_cv": None},
        ], next_feature_id=9, workers=1, executable="/bin/true",
    )
    assert links == {7: 9}
    assert deltas == {}
    assert features[0]["scanApex"] == 2
    assert features[0]["area_sum"] == 1000.0
    assert quant[0]["feature_origin"] == "openms_ffi_rescue"
    assert quant[0]["feature_quality_score"] == 1.5
    assert quant[0]["supporting_psm_count"] == 1
    assert summary["mapped_psm_count"] == 1


def test_execute_rescue_reuses_matching_feature_and_returns_support_delta(monkeypatch):
    _fake_openms(monkeypatch)
    features, quant, links, deltas, summary = openms_ffi.execute_rescue(
        source="input.mzML", run_id="run", candidates=[_candidate()],
        existing_features=[{
            "feature_idx": 3, "charge": 2, "mz": 500.0,
            "rtStart": 111.0, "rtEnd": 129.0, "FAIMS": None,
        }], ms1_rows=[], next_feature_id=4, workers=1, executable="/bin/true",
    )
    assert features == []
    assert quant == []
    assert links == {7: 3}
    assert deltas == {3: (1, 1)}
    assert summary["attached_feature_count"] == 1


def test_execute_rescue_maps_multiple_psms_once(monkeypatch):
    feature_xml = _FEATURE_XML.replace(
        "</PeptideIdentification>\n<UserParam",
        "</PeptideIdentification><PeptideIdentification>"
        "<UserParam name=\"psm_id\" value=\"psm-2\"/>"
        "<UserParam name=\"ms2_event_id\" value=\"8\"/>"
        "</PeptideIdentification>\n<UserParam",
    )
    _fake_openms(monkeypatch, feature_xml)
    features, quant, links, _deltas, _summary = openms_ffi.execute_rescue(
        source="input.mzML", run_id="run",
        candidates=[_candidate(), _candidate(psm_id="psm-2", ms2_event_id=8)],
        existing_features=[], ms1_rows=[], next_feature_id=9, workers=1,
        executable="/bin/true",
    )
    assert len(features) == 1
    assert links == {7: 9, 8: 9}
    assert quant[0]["supporting_psm_count"] == 2
    assert quant[0]["supporting_ms2_count"] == 2


def test_execute_rescue_rejects_malformed_featurexml(monkeypatch):
    _fake_openms(monkeypatch, "<featureMap>")
    with pytest.raises(openms_ffi.OpenMSFFIError, match="invalid featureXML"):
        openms_ffi.execute_rescue(
            source="input.mzML", run_id="run", candidates=[_candidate()],
            existing_features=[], ms1_rows=[], next_feature_id=1, workers=1,
            executable="/bin/true",
        )


def test_faims_is_required_for_feature_reuse_and_scan_mapping(monkeypatch):
    _fake_openms(monkeypatch)
    features, _quant, links, _deltas, _summary = openms_ffi.execute_rescue(
        source="input.mzML", run_id="run", candidates=[_candidate(faims_cv=0.0)],
        existing_features=[{
            "feature_idx": 3, "charge": 2, "mz": 500.0,
            "rtStart": 111.0, "rtEnd": 129.0, "FAIMS": None,
        }], ms1_rows=[
            {"rt_sec": 120.0, "scan_number": 10, "faims_cv": None},
            {"rt_sec": 120.0, "scan_number": 20, "faims_cv": 0.0},
        ], next_feature_id=4, workers=1, executable="/bin/true",
    )
    assert links == {7: 4}
    assert features[0]["FAIMS"] == 0.0
    assert features[0]["scanApex"] == 20


def test_scan_grid_cache_keeps_faims_tolerance_queries_distinct():
    grid = openms_ffi._ScanGrid([
        {"rt_sec": 10.0, "scan_number": 1, "faims_cv": -45.0000014},
        {"rt_sec": 20.0, "scan_number": 2, "faims_cv": -44.9999986},
    ])

    assert grid.nearest(-45.00000049, 15.0) == 1
    assert grid.nearest(-44.99999951, 15.0) == 2


def test_feature_index_only_considers_nearby_mz_bucket(monkeypatch):
    calls = 0
    original = openms_ffi._faims_equal

    def counted(left, right):
        nonlocal calls
        calls += 1
        return original(left, right)

    monkeypatch.setattr(openms_ffi, "_faims_equal", counted)
    index = openms_ffi._FeatureIndex([
        {"feature_idx": value, "charge": 2, "mz": 400.0 + value,
         "rtStart": 100.0, "rtEnd": 200.0, "FAIMS": None}
        for value in range(1, 2001)
    ])
    assert index.matches({
        "charge": 2, "mz": 1500.0, "rt_start_sec": 110.0,
        "rt_end_sec": 120.0, "faims_cv": None,
    })
    assert calls < 20


def test_standalone_candidates_use_source_ms2_not_psm_id_scan_text():
    candidates = openms_ffi.standalone_candidates(
        [{
            "psm_id": "opaque-identifier", "ms2_event_id": 7,
            "formula_status": "exact", "assay_status": "accepted_direct_assay",
            "assay_charge": 2, "assay_faims_cv": -45.0,
            "canonical_peptidoform": "PEPTIDE", "q_value": 0.001,
        }], set(), [{
            "ms2_event_id": 7, "native_id": "native=scan-42",
            "native_scan_number": 42, "rt_sec": 120.0,
            "selected_ion_mz": 500.123, "faims_cv": -45.0,
        }], 0.01,
    )
    assert candidates == [{
        "psm_id": "opaque-identifier", "ms2_event_id": 7,
        "canonical_peptidoform": "PEPTIDE", "charge": 2,
        "rt_sec": 120.0, "selected_ion_mz": 500.123,
        "q_value": 0.001, "native_scan_number": 42,
        "native_id": "native=scan-42", "faims_cv": -45.0,
    }]


def test_openms_rescue_feature_is_excluded_from_project_strong_donors():
    strong, weak = sidecar_rows(
        "run",
        [{
            "feature_idx": 1, "charge": 2, "mz": 500.0,
            "rtApex": 120.0, "rtStart": 110.0, "rtEnd": 130.0,
        }, {
            "feature_idx": 2, "charge": 2, "mz": 600.0,
            "rtApex": 220.0, "rtStart": 210.0, "rtEnd": 230.0,
        }],
        [{
            "feature_id": 1, "feature_origin": "openms_ffi_rescue",
            "quant_value": 10.0, "feature_quality_score": 0.9,
        }, {
            "feature_id": 2, "feature_origin": "strict_untargeted",
            "quant_value": 20.0, "feature_quality_score": 0.9,
        }], [],
    )
    assert [row["feature_id"] for row in strong] == [2]
    assert weak == []


def test_normal_hybrid_run_skips_missing_executable_without_mutation():
    assay = SimpleNamespace(
        psm_id="psm-1", ms2_event_id=7, canonical_peptidoform="PEPTIDE",
        charge=2, rt_sec=120.0, selected_ion_mz=500.0, q_value=0.001,
        faims_cv=None, conflict_status="unique",
    )
    audit = {7: {"feature_id": None}}
    summary, next_feature_id = openms_ffi.rescue_hybrid_results(
        source="input.mzML", run_id="run", assays=[assay], audit_by_event=audit,
        feature_rows=[], quant_rows=[], ms1_rows=[],
        ms2_rows=[{"ms2_event_id": 7, "native_scan_number": 42}],
        next_feature_id=1, args={
            "feature_finder_identification": True,
            "feature_finder_identification_path": "definitely-not-an-openms-tool",
        },
    )
    assert summary["status"] == "executable_not_found"
    assert next_feature_id == 1
    assert audit[7]["feature_id"] is None


def _provenance(source):
    return {
        "input_path": str(source.resolve()), "input_size": source.stat().st_size,
        "hybrid_schema_version": "9", "hybrid_summary_json": "{}",
    }


def test_source_validation_uses_resolved_fingerprint_from_another_directory(
        monkeypatch, tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "sample.mzML"
    source.write_bytes(b"source")
    provenance = {
        "input_path": source.name,
        "input_size": source.stat().st_size,
        "input_fingerprint": openms_ffi.source_fingerprint(source),
    }
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)

    openms_ffi._validate_source_provenance(provenance, source)


def test_quality_column_is_inserted_at_schema_10_position():
    table = pa.table({
        "quant_envelope_apex": pa.array([1.0], type=pa.float64()),
        "quality_flags": pa.array([0], type=pa.uint32()),
        "unknown_column": ["preserve"],
    })

    upgraded = openms_ffi._append_quality_column(table)

    assert upgraded.column_names == [
        "quant_envelope_apex", "feature_quality_score", "quality_flags",
        "unknown_column",
    ]


def _hybrid_tables(provenance):
    metadata = {b"biosaur2_provenance_json": json.dumps(provenance).encode()}
    return {
        "features": pa.Table.from_pylist([{
            "feature_idx": 1, "mz": 500.0, "charge": 2,
            "rtStart": 1.0, "rtEnd": 2.0,
            "supporting_psm_count": 0, "supporting_ms2_count": 0,
            "quant_envelope_apex": 1.0, "quality_flags": 0,
            "unknown_feature_column": "preserve",
        }]).replace_schema_metadata(metadata),
        "ms2_events": pa.Table.from_pylist([{
            "feature_idx": 1, "ms2_event_id": 3,
        }]).replace_schema_metadata(metadata),
        "identifications": pa.Table.from_pylist([{
            "run_id": "sample", "psm_id": "opaque", "ms2_event_id": 7,
            "q_value": 0.001, "formula_status": "exact",
            "assay_status": "accepted_direct_assay", "assay_charge": 2,
            "assay_faims_cv": None, "canonical_peptidoform": "PEPTIDE",
            "unknown_identification_column": "unchanged",
        }]).replace_schema_metadata(metadata),
    }


def _write_hybrid_output(tmp_path, source):
    targets = {
        "features": tmp_path / "sample.features.parquet",
        "ms2_events": tmp_path / "sample.ms2_events.parquet",
        "identifications": tmp_path / "sample.identifications.parquet",
    }
    tables = _hybrid_tables(_provenance(source))
    for name, path in targets.items():
        pq.write_table(tables[name], path)
    return targets, tables


def test_standalone_validates_source_and_publishes_all_parquet_metadata(monkeypatch, tmp_path):
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"source")
    targets, tables = _write_hybrid_output(tmp_path, source)

    def completed(_source, features, events, _ids, *_args):
        features = openms_ffi._append_quality_column(features)
        return features, events, {}, {"status": "completed", "mapped_psm_count": 0}

    monkeypatch.setattr(openms_ffi, "_standalone_merge", completed)
    summary = openms_ffi.rescue_completed_output(
        source=source, output=targets["features"], q_value_max=0.01,
        executable_path="ignored", workers=1,
    )
    assert summary["status"] == "completed"
    features = pq.read_table(targets["features"])
    assert "feature_quality_score" in features.column_names
    assert features.column_names.index("feature_quality_score") + 1 == (
        features.column_names.index("quality_flags")
    )
    assert features.column("unknown_feature_column").to_pylist() == ["preserve"]
    identifications = pq.read_table(targets["identifications"])
    assert identifications.to_pylist() == tables["identifications"].to_pylist()
    updated = json.loads(pq.read_schema(targets["features"]).metadata[b"biosaur2_provenance_json"])
    assert json.loads(updated["hybrid_summary_json"])["openms_ffi_rescue"]["status"] == "completed"
    assert json.loads(
        pq.read_schema(targets["features"]).metadata[b"biosaur2_hybrid_summary_json"]
    )["openms_ffi_rescue"]["status"] == "completed"


def test_standalone_rejects_wrong_source_without_rewriting(tmp_path):
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"source")
    targets, _tables = _write_hybrid_output(tmp_path, source)
    wrong = tmp_path / "wrong.mzML"
    wrong.write_bytes(b"source")
    before = os.stat(targets["features"]).st_mtime_ns
    with pytest.raises(openms_ffi.OpenMSFFIError, match="does not match"):
        openms_ffi.rescue_completed_output(
            source=wrong, output=targets["features"], q_value_max=0.01,
            executable_path="ignored", workers=1,
        )
    assert os.stat(targets["features"]).st_mtime_ns == before


def test_standalone_no_candidates_does_not_rewrite_parquet(monkeypatch, tmp_path):
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"source")
    targets, tables = _write_hybrid_output(tmp_path, source)
    before = {name: os.stat(path).st_mtime_ns for name, path in targets.items()}
    monkeypatch.setattr(
        openms_ffi, "_standalone_merge",
        lambda *_args: (tables["features"], tables["ms2_events"], {}, openms_ffi._no_candidate_summary()),
    )
    assert openms_ffi.rescue_completed_output(
        source=source, output=targets["features"], q_value_max=0.01,
        executable_path="missing-is-irrelevant", workers=1,
    )["status"] == "no_candidates"
    assert {name: os.stat(path).st_mtime_ns for name, path in targets.items()} == before


def test_standalone_duckdb_preserves_unrelated_tables(monkeypatch, tmp_path):
    duckdb = pytest.importorskip("duckdb")
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"source")
    database = tmp_path / "sample.biosaur2.duckdb"
    tables = _hybrid_tables(_provenance(source))
    with duckdb.connect(str(database)) as connection:
        for name, table in tables.items():
            connection.register("input_table", table)
            connection.execute('CREATE TABLE "%s" AS SELECT * FROM input_table' % name)
            connection.unregister("input_table")
        connection.execute(
            "CREATE TABLE runs (schema_version VARCHAR, input_path VARCHAR, input_size BIGINT, parameters_json VARCHAR, provenance_json VARCHAR)"
        )
        connection.execute(
            "INSERT INTO runs VALUES ('6.0', ?, ?, '{}', ?)",
            [str(source.resolve()), source.stat().st_size, json.dumps(_provenance(source))],
        )
        connection.execute("CREATE TABLE unrelated (value VARCHAR)")
        connection.execute("INSERT INTO unrelated VALUES ('keep')")

    def completed(_source, features, events, _ids, *_args):
        return (
            openms_ffi._append_quality_column(features),
            events, {}, {"status": "completed", "mapped_psm_count": 0},
        )

    monkeypatch.setattr(openms_ffi, "_standalone_merge", completed)
    assert openms_ffi.rescue_completed_output(
        source=source, output=database, q_value_max=0.01,
        executable_path="ignored", workers=1,
    )["status"] == "completed"
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT * FROM unrelated").fetchall() == [("keep",)]
        feature_columns = [
            row[1] for row in connection.execute("PRAGMA table_info('features')").fetchall()
        ]
        assert feature_columns.index("feature_quality_score") + 1 == (
            feature_columns.index("quality_flags")
        )
        provenance = json.loads(connection.execute("SELECT provenance_json FROM runs").fetchone()[0])
    assert json.loads(provenance["hybrid_summary_json"])["openms_ffi_rescue"]["status"] == "completed"


def test_cli_returns_nonzero_for_standalone_failure(monkeypatch, tmp_path):
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"")
    monkeypatch.setattr(
        openms_ffi, "rescue_completed_output",
        lambda **_kwargs: (_ for _ in ()).throw(openms_ffi.OpenMSFFIError("missing executable")),
    )
    assert search_module._run_rescue([
        str(source), "--output", str(tmp_path / "sample.features.parquet"),
    ]) == 1


def test_cli_parses_exact_feature_finder_identification_spelling(monkeypatch, tmp_path):
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"")
    captured = {}
    monkeypatch.setattr(
        search_module, "_execute_inputs", lambda args, *_unused: captured.update(args)
    )
    monkeypatch.setattr(
        search_module.sys, "argv",
        ["biosaur2", str(source), "--feature-mode", "hybrid", "--no-FeatureFinderIdentification"],
    )
    search_module.run()
    assert captured["feature_finder_identification"] is False
    assert captured["feature_finder_identification_path"] == "FeatureFinderIdentification"
