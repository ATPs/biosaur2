import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from biosaur2 import project_cli
from biosaur2 import project
from biosaur2 import project_validation
from biosaur2.schema import compact_schemas
from biosaur2.project import (
    _command_for_run,
    _command_worker_allocation,
    _project_worker,
    _local_resume_option_signature,
    _resume_option_signature,
    _input_fingerprint,
    _read_successful_runs,
    _scientific_command,
    _write_project_database,
    _run_paths,
    _summary_for_result,
    _summary_worker_count,
    validate_project,
)
from biosaur2.project_validation import _validation_worker_count


def test_project_worker_streams_output_and_preserves_tails(capsys):
    result = _project_worker(
        {
            "run_id": "sample_01",
            "paths": {},
            "command": [
                sys.executable,
                "-c",
                (
                    "import os, sys; "
                    "print('run=' + os.environ['BIOSAUR2_LOG_RUN_ID']); "
                    "print('child-error', file=sys.stderr)"
                ),
            ],
        }
    )
    captured = capsys.readouterr()
    assert result["status"] == "success"
    assert "run=sample_01" in captured.out
    assert "child-error" in captured.err
    assert "run=sample_01" in result["stdout_tail"]
    assert "child-error" in result["stderr_tail"]


def _summary_paths(tmp_path, output_format="parquet"):
    if output_format == "duckdb":
        output = tmp_path / "run.biosaur2.duckdb"
        return {
            "format": output_format,
            "features": str(output),
            "ms2_events": str(output),
            "identifications": str(output),
        }
    return {
        "format": output_format,
        "features": str(tmp_path / ("features." + output_format)),
        "ms2_events": str(tmp_path / ("ms2_events." + output_format)),
        "identifications": str(tmp_path / ("identifications." + output_format)),
    }


def _summary_result(paths):
    return {"run_id": "summary-run", "status": "success", "paths": paths}


def test_parquet_project_summary_uses_metadata_and_assay_batches(tmp_path, monkeypatch):
    paths = _summary_paths(tmp_path)
    schemas = compact_schemas()
    hybrid_summary = {"strict_feature_count": 2, "audit_status_counts": {}}
    features = pa.Table.from_pylist(
        [
            {"feature_idx": 1, "quant_value": 10.0, "scanStart": 1, "scanApex": 2, "scanEnd": 3},
            {"feature_idx": 2, "quant_value": 20.0, "scanStart": 4, "scanApex": 5, "scanEnd": 6},
        ],
        schema=schemas["hybrid_features"],
    ).replace_schema_metadata(
        {b"biosaur2_hybrid_summary_json": json.dumps(hybrid_summary).encode()}
    )
    pq.write_table(features, paths["features"], row_group_size=1)
    pq.write_table(
        pa.Table.from_pylist(
            [{"feature_idx": 1, "ms2_event_id": 1}, {"feature_idx": 2, "ms2_event_id": 2}],
            schema=schemas["linked_ms2_events"],
        ),
        paths["ms2_events"],
        row_group_size=1,
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"run_id": "summary-run", "psm_id": "one", "mapping_status": "mapped", "q_value": 0.01, "assay_id": 1},
                {"run_id": "summary-run", "psm_id": "two", "mapping_status": "mapped", "q_value": 0.01, "assay_id": None},
                {"run_id": "summary-run", "psm_id": "three", "mapping_status": "mapped", "q_value": 0.01, "assay_id": 0},
            ],
            schema=schemas["merged_identifications"],
        ),
        paths["identifications"],
        row_group_size=1,
    )
    monkeypatch.setattr(
        project.pq,
        "read_table",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full table read")),
    )

    summary = _summary_for_result(_summary_result(paths), mode="hybrid")

    assert summary == {
        "feature_count": 2,
        "ms2_count": 2,
        "audit_count": 2,
        "linked_ms2_count": 2,
        "quant_feature_count": 2,
        "accepted_identification_count": 3,
        "direct_assay_count": 2,
        "hybrid_summary": hybrid_summary,
    }

    features = features.replace_schema_metadata(
        {b"biosaur2_provenance_json": json.dumps({"hybrid_summary_json": json.dumps(hybrid_summary)}).encode()}
    )
    pq.write_table(features, paths["features"], row_group_size=1)
    assert _summary_for_result(_summary_result(paths), mode="hybrid")["hybrid_summary"] == hybrid_summary


def test_tsv_and_duckdb_project_summaries_remain_compatible(tmp_path):
    tsv_paths = _summary_paths(tmp_path / "tsv", "tsv")
    for path, text in (
        (tsv_paths["features"], "quant_value\textra\n1\tx\n\ty\n"),
        (tsv_paths["ms2_events"], "feature_idx\n1\n2\n"),
        (tsv_paths["identifications"], "assay_id\textra\n1\tx\n\ty\n"),
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    tsv_summary = _summary_for_result(_summary_result(tsv_paths), mode="hybrid")
    assert tsv_summary["feature_count"] == 2
    assert tsv_summary["quant_feature_count"] == 1
    assert tsv_summary["direct_assay_count"] == 1

    duckdb_paths = _summary_paths(tmp_path / "duckdb", "duckdb")
    Path(duckdb_paths["features"]).parent.mkdir(parents=True)
    import duckdb

    with duckdb.connect(duckdb_paths["features"]) as connection:
        connection.execute("CREATE TABLE features (feature_idx INTEGER)")
        connection.execute("INSERT INTO features VALUES (1), (2)")
        connection.execute("CREATE TABLE ms2_events (feature_idx INTEGER)")
        connection.execute("INSERT INTO ms2_events VALUES (1)")
        connection.execute("CREATE TABLE identifications (assay_id INTEGER)")
        connection.execute("INSERT INTO identifications VALUES (1), (NULL), (0)")
        connection.execute("CREATE TABLE runs (provenance_json VARCHAR)")
        connection.execute("INSERT INTO runs VALUES ('{}')")
    duckdb_summary = _summary_for_result(
        _summary_result(duckdb_paths), mode="hybrid"
    )
    assert duckdb_summary["feature_count"] == 2
    assert duckdb_summary["linked_ms2_count"] == 1
    assert duckdb_summary["accepted_identification_count"] == 3
    assert duckdb_summary["direct_assay_count"] == 2


def test_project_summary_missing_output_fails_before_publication(tmp_path):
    run = SimpleNamespace(
        run_id="missing-run", mzml_path=tmp_path / "missing.mzML", psm_path=None
    )
    paths = _summary_paths(tmp_path)
    result = {
        "run_id": run.run_id,
        "status": "success",
        "paths": {**paths, "raw_ms1_cache": str(tmp_path / "cache")},
        "command": [],
    }
    database = tmp_path / "project.duckdb"
    try:
        _write_project_database(
            database, [run], {0: result}, {"mode": "hybrid", "workers": 1}
        )
    except RuntimeError as error:
        assert "missing-run" in str(error)
        assert paths["features"] in str(error)
    else:
        raise AssertionError("missing summary output did not fail")
    assert not database.exists()


def test_project_summary_reader_cap_matches_server5_budget():
    assert _summary_worker_count(
        32, 512, 1200 * 1024 ** 3, cpu_count_value=512
    ) == 32


def test_project_validator_uses_bounded_readers():
    assert _validation_worker_count(
        40, 512, 1200 * 1024 ** 3, cpu_count_value=512
    ) == 32


def test_project_validate_workers_are_forwarded(monkeypatch, tmp_path):
    captured = {}

    def fake_validate(project_db, workers=None):
        captured.update(project_db=project_db, workers=workers)
        return {"run_count": 4, "problems": ()}

    monkeypatch.setattr(project_cli, "validate_project", fake_validate)
    assert project_cli.run_project_cli(
        ["validate", "--project-db", str(tmp_path / "project.duckdb"), "--workers", "4"]
    ) == 0
    assert captured == {"project_db": str(tmp_path / "project.duckdb"), "workers": 4}


def test_log_level_does_not_affect_project_resume_signature():
    base = {
        "mode": "hybrid",
        "format": "parquet",
        "workers": 4,
        "resume": False,
        "log_level": "info",
    }
    debug = dict(base, log_level="debug")
    assert _resume_option_signature(base) == _resume_option_signature(debug)


def test_scheduler_resource_mode_does_not_affect_project_resume_signature():
    base = {"mode": "hybrid", "format": "parquet", "workers": 4}
    detailed = dict(base, scheduler_resource_mode="detailed")
    assert _resume_option_signature(base) == _resume_option_signature(detailed)


def test_recorded_worker_allocation_uses_the_execution_command():
    assert _command_worker_allocation(["python", "-m", "biosaur2.search"]) == 4
    assert _command_worker_allocation(["biosaur2", "run", "--workers", "8"]) == 8
def test_project_hybrid_mode_is_explicit_opt_in(monkeypatch, tmp_path):
    captured = {}

    def fake_run_project(manifest, output_dir, project_db, **options):
        captured.update(
            {
                "manifest": manifest,
                "output_dir": output_dir,
                "project_db": project_db,
                "options": options,
            }
        )

    monkeypatch.setattr(project_cli, "run_project", fake_run_project)
    assert project_cli.run_project_cli(
        [
            "run",
            "--manifest",
            str(tmp_path / "manifest.tsv"),
            "--output-dir",
            str(tmp_path / "output"),
            "--project-db",
            str(tmp_path / "project.duckdb"),
        ]
    ) == 0
    assert captured["options"]["mode"] == "legacy"
    assert captured["options"]["write_ms1"] is False
    assert captured["options"]["resume"] is True
    assert captured["options"]["max_memory"] > 0
    assert captured["options"]["scheduler_heartbeat_seconds"] == 60
    assert captured["options"]["scheduler_resource_mode"] == "auto"

    project_cli.run_project_cli(
        [
            "run",
            "--manifest",
            str(tmp_path / "manifest.tsv"),
            "--output-dir",
            str(tmp_path / "hybrid-output"),
            "--project-db",
            str(tmp_path / "hybrid.duckdb"),
            "--mode",
            "hybrid",
        ]
    )
    assert captured["options"]["mode"] == "hybrid"
    assert captured["options"]["write_ms1"] is True
    assert captured["options"]["external_q_value_max"] == 0.10
    assert captured["options"]["external_weak_max_strong_overlap"] == 0.30
    assert captured["options"]["external_min_support_runs"] == 1
    assert captured["options"]["external_max_support_runs"] == 4


def test_project_hybrid_command_propagates_rt_tolerance(tmp_path):
    run = SimpleNamespace(
        mzml_path=tmp_path / "run.mzML.gz",
        psm_path=None,
        q_value_max=None,
        fixed_mods=None,
    )
    paths = {
        "features": str(tmp_path / "features.parquet"),
        "raw_ms1_cache": str(tmp_path / "raw_ms1_cache"),
        "candidate_cache": str(tmp_path / "candidate_cache"),
    }
    options = {
        "mode": "hybrid",
        "overwrite": False,
        "psm_q_value_max": 0.01,
        "psm_pep_max": None,
        "fixed_mod": [],
        "quant_method": "envelope_area",
        "write_mono_hills": True,
        "write_quant_details": True,
        "feature_baseline": "edge_linear",
        "direct_id": True,
        "external_id": False,
        "generic_ms2_refine": True,
        "generic_q_value_max": 0.01,
        "max_charge": 8,
        "relaxed_ms2_feature": True,
        "ms2_rt_tolerance_sec": 90.0,
        "psm_run_column": "idn",
    }
    command = _command_for_run(run, paths, options)
    position = command.index("--ms2-rt-tolerance-sec")
    assert command[position + 1] == "90.0"
    position = command.index("--max-charge")
    assert command[position + 1] == "8"
    assert "--relaxed-ms2-feature" in command
    assert "--write-mono-hills" in command
    assert "--write-quant-details" in command
    assert "--write-ms1" in command
    assert "--write-ms2" not in command
    assert command[command.index("--format") + 1] == "parquet"
    assert command[command.index("--generic-ms2-isotope-errors") + 1] == "0,1,2,3"
    assert command[command.index("--external-weak-max-strong-overlap") + 1] == "0.3"
    assert command[command.index("--psm-run-column") + 1] == "idn"
    assert "--workers" not in command
    assert "--cache-dir" not in command


def test_project_hybrid_command_propagates_disabled_ms1(tmp_path):
    run = SimpleNamespace(
        mzml_path=tmp_path / "run.mzML.gz",
        psm_path=None,
        q_value_max=None,
        fixed_mods=None,
    )
    command = _command_for_run(
        run,
        {"features": str(tmp_path / "features.parquet")},
        {
            "mode": "legacy",
            "overwrite": False,
            "write_ms1": False,
            "max_charge": 7,
        },
    )
    assert "--no-write-ms1" in command


def test_project_run_paths_include_ms1_only_when_enabled(tmp_path):
    run = SimpleNamespace(
        run_id="run", mzml_path=tmp_path / "run.mzML.gz"
    )
    parquet = _run_paths(
        run, tmp_path / "output", tmp_path / "cache", "parquet",
        write_ms1=True,
    )
    assert parquet["ms1"].endswith("/run/run.ms1.parquet")
    assert _run_paths(
        run, tmp_path / "output", tmp_path / "cache", "parquet",
        write_ms1=False,
    )["ms1"] is None
    duckdb = _run_paths(
        run, tmp_path / "output", tmp_path / "cache", "duckdb",
        write_ms1=True,
    )
    assert duckdb["ms1"] == duckdb["run_output"]


def test_resume_signature_ignores_scheduling_but_tracks_external_science():
    previous = {
        "mode": "hybrid",
        "resume": False,
        "workers": 4,
        "cache_dir": "cache-a",
        "keep_cache": False,
        "external_q_value_max": 0.01,
    }
    resumed = {
        **previous,
        "resume": True,
        "workers": 12,
        "cache_dir": "cache-b",
        "keep_cache": True,
    }
    assert _resume_option_signature(previous) == _resume_option_signature(resumed)
    resumed["external_q_value_max"] = 0.02
    assert _resume_option_signature(previous) != _resume_option_signature(resumed)


def test_local_resume_tracks_weak_sidecar_options_only():
    base = {
        "mode": "hybrid",
        "external_id": True,
        "external_q_value_max": 0.10,
        "external_min_support_runs": 1,
        "external_max_support_runs": 4,
        "external_weak_min_mono_points": 2,
        "external_weak_min_secondary_points": 2,
        "external_weak_min_isotope_cosine": 0.6,
        "external_weak_max_strong_overlap": 0.30,
    }
    signature = _local_resume_option_signature(base)
    for key, value in (
        ("external_q_value_max", 0.05),
        ("external_min_support_runs", 2),
        ("external_max_support_runs", 8),
    ):
        assert signature == _local_resume_option_signature({**base, key: value})
    for key, value in (
        ("external_id", False),
        ("external_weak_min_mono_points", 3),
        ("external_weak_min_secondary_points", 3),
        ("external_weak_min_isotope_cosine", 0.7),
        ("external_weak_max_strong_overlap", 0.25),
    ):
        assert signature != _local_resume_option_signature({**base, key: value})


def test_resume_command_ignores_worker_and_cache_location():
    base = ["python", "-m", "biosaur2.search", "run.mzML"]
    executed = base + [
        "--workers", "8", "--cache-dir", "/tmp/cache", "--keep-cache"
    ]
    assert _scientific_command(executed) == base


def test_project_worker_captures_cpu_and_peak_rss(tmp_path):
    result = _project_worker(
        {
            "run_id": "resource-smoke",
            "command": [sys.executable, "-c", "sum(range(100000))"],
            "paths": {"run_dir": str(tmp_path)},
        }
    )
    assert result["status"] == "success"
    assert result["cpu_user_sec"] >= 0
    assert result["cpu_system_sec"] >= 0
    assert result["peak_rss_kib"] > 0


def test_project_database_records_cache_command_and_resume_fingerprints(tmp_path):
    mzml = tmp_path / "run.mzML.gz"
    psm = tmp_path / "run.psms.tsv"
    mzml.write_bytes(b"mzML-source")
    psm.write_text("psm-source", encoding="utf-8")
    run = SimpleNamespace(run_id="run", mzml_path=mzml, psm_path=psm)
    run_dir = tmp_path / "run-output"
    run_dir.mkdir()
    paths = {
        "format": "parquet",
        "run_output": None,
        "features": str(run_dir / "features.parquet"),
        "ms2_events": str(run_dir / "ms2_events.parquet"),
        "identifications": str(run_dir / "identifications.parquet"),
        "ms1": str(run_dir / "ms1.parquet"),
        "external_evidence": str(run_dir / "external.parquet"),
        "raw_ms1_cache": str(run_dir / "raw_ms1_cache"),
    }
    hybrid_summary = {
        "strict_feature_count": 1,
        "direct_assay_count": 1,
        "recovered_feature_count": 0,
        "audit_row_count": 1,
        "audit_status_counts": {"generic_decoy_only": 1},
        "generic_summary": {
            "competition_counts": {"decoy_only_candidate_count": 1}
        },
    }
    schemas = compact_schemas()
    feature_table = pa.Table.from_pylist(
        [{
            "feature_idx": 1,
            "quant_value": 10.0,
            "scanStart": 100,
            "scanApex": 101,
            "scanEnd": 102,
        }],
        schema=schemas["hybrid_features"],
    ).replace_schema_metadata(
        {
            b"biosaur2_provenance_json": json.dumps({
                "hybrid_summary_json": json.dumps(
                    hybrid_summary, sort_keys=True
                )
            }, sort_keys=True).encode()
        }
    )
    pq.write_table(feature_table, paths["features"])
    pq.write_table(
        pa.Table.from_pylist(
            [{"feature_idx": 1, "ms2_event_id": 7}],
            schema=schemas["linked_ms2_events"],
        ),
        paths["ms2_events"],
    )
    identification_table = pa.Table.from_pylist(
        [
            {
                "run_id": "run",
                "psm_id": "psm-1",
                "mapping_status": "mapped",
                "q_value": 0.001,
                "assay_id": 1,
            }
        ],
        schema=schemas["merged_identifications"],
    )
    pq.write_table(identification_table, paths["identifications"])
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"scan_id": 100, "RT": 60.0, "total_intensity": 10.0},
                {"scan_id": 101, "RT": 120.0, "total_intensity": 20.0},
                {"scan_id": 102, "RT": 180.0, "total_intensity": 10.0},
            ],
            schema=schemas["ms1"],
        ),
        paths["ms1"],
    )
    pq.write_table(
        pa.Table.from_pylist([], schema=schemas["external_evidence"]),
        paths["external_evidence"],
    )
    Path(paths["raw_ms1_cache"]).mkdir()
    Path(paths["raw_ms1_cache"], "manifest.json").write_text(
        "{}\n", encoding="utf-8"
    )
    command = ["python", "-m", "biosaur2.search", str(mzml)]
    results = {
        0: {
            "run_id": "run",
            "status": "success",
            "runtime_sec": 1.0,
            "cpu_user_sec": 2.0,
            "cpu_system_sec": 0.5,
            "peak_rss_kib": 123456,
            "error": None,
            "paths": paths,
            "command": command,
        }
    }
    database = tmp_path / "project.duckdb"
    external_stage = {
        "summaries": {
            "run": {
                "planned_assay_count": 10,
                "evaluated_assay_count": 9,
                "new_external_feature_count": 2,
                "status_counts": {"accepted_new_external_feature": 2},
            }
        },
        "alignment_models": [
            {
                "alignment_group": "explicit:g",
                "reference_run": "run",
                "source_run": "run",
                "target_run": "other",
                "method": "median_shift",
                "anchor_count": 5,
                "inlier_count": 5,
                "slope": 1.0,
                "intercept": 2.0,
                "residual_mad_sec": 1.0,
                "validation_anchor_count": 7,
                "validation_median_bias_sec": 0.5,
                "validation_mad_sec": 1.0,
                "validation_q90_abs_error_sec": 2.5,
                "status": "accepted",
                "x_knots_json": "[]",
                "y_knots_json": "[]",
            }
        ],
    }
    _write_project_database(
        database,
        [run],
        results,
        {"mode": "hybrid", "external_id": True, "write_ms1": True},
        external_stage=external_stage,
    )

    successful = _read_successful_runs(database)
    assert successful["run"]["command"] == command
    assert successful["run"]["input_fingerprint"] == _input_fingerprint(run)
    assert successful["run"]["peak_rss_kib"] == 123456
    assert validate_project(database) == {"run_count": 1, "problems": ()}

    Path(paths["ms1"]).unlink()
    try:
        validate_project(database)
    except ValueError as error:
        assert "missing MS1 output" in str(error)
    else:
        raise AssertionError("missing MS1 output was not detected")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"scan_id": 100, "RT": 60.0, "total_intensity": 10.0},
                {"scan_id": 101, "RT": 120.0, "total_intensity": 20.0},
                {"scan_id": 102, "RT": 180.0, "total_intensity": 10.0},
            ],
            schema=schemas["ms1"],
        ),
        paths["ms1"],
    )

    pq.write_table(
        pa.Table.from_pylist(
            [{"feature_idx": 999, "ms2_event_id": 7}],
            schema=schemas["linked_ms2_events"],
        ),
        paths["ms2_events"],
    )
    try:
        validate_project(database)
    except ValueError as error:
        assert "orphan linked MS2 feature IDs" in str(error)
    else:
        raise AssertionError("orphan linked MS2 feature was not detected")
    pq.write_table(
        pa.Table.from_pylist(
            [{"feature_idx": 1, "ms2_event_id": 7}],
            schema=schemas["linked_ms2_events"],
        ),
        paths["ms2_events"],
    )

    Path(paths["external_evidence"]).unlink()
    try:
        validate_project(database)
    except ValueError as error:
        assert "missing external evidence output" in str(error)
    else:
        raise AssertionError("missing external evidence was not detected")
    pq.write_table(
        pa.Table.from_pylist([], schema=schemas["external_evidence"]),
        paths["external_evidence"],
    )

    import duckdb

    with duckdb.connect(str(database), read_only=True) as connection:
        resources = connection.execute(
            "SELECT cpu_user_sec, cpu_system_sec, peak_rss_kib FROM runs"
        ).fetchone()
        persisted = connection.execute(
            "SELECT generic_decoy_only_count, generic_summary_json "
            "FROM hybrid_summary"
        ).fetchone()
        external = connection.execute(
            "SELECT planned_assay_count, evaluated_assay_count, "
            "new_external_feature_count FROM external_summary"
        ).fetchone()
        alignment = connection.execute(
            "SELECT alignment_group, reference_run, source_run, target_run, "
            "status, validation_anchor_count, validation_median_bias_sec, "
            "validation_mad_sec, validation_q90_abs_error_sec "
            "FROM rt_alignment_models"
        ).fetchone()
        schema_version = connection.execute(
            "SELECT value_json FROM project_metadata "
            "WHERE key='project_schema_version'"
        ).fetchone()[0]
        strict_cache_stage = connection.execute(
            "SELECT status FROM stage_status "
            "WHERE run_id='run' AND stage='strict_stage_cache'"
        ).fetchone()[0]
    assert resources == (2.0, 0.5, 123456)
    assert persisted[0] == 1
    assert json.loads(persisted[1]) == hybrid_summary["generic_summary"]
    assert external == (10, 9, 2)
    assert alignment == (
        "explicit:g", "run", "run", "other", "accepted", 7, 0.5, 1.0, 2.5
    )
    assert schema_version == "12"
    assert strict_cache_stage == "missing"

    mzml.write_bytes(b"changed-mzML-source")
    assert successful["run"]["input_fingerprint"] != _input_fingerprint(run)


def test_project_validator_rejects_old_schema_before_new_column_queries(
    tmp_path,
):
    import duckdb

    database = tmp_path / "old-project.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "CREATE TABLE project_metadata (key VARCHAR, value_json VARCHAR)"
        )
        connection.execute(
            "INSERT INTO project_metadata VALUES "
            "('project_schema_version', '11')"
        )

    try:
        validate_project(database)
    except ValueError as error:
        assert "Unsupported Project schema 11; expected 12" in str(error)
    else:
        raise AssertionError("old Project schema was not rejected")


def _four_run_validation_project(tmp_path):
    import duckdb

    database = tmp_path / "validation-project.duckdb"
    feature_paths = []
    for index in range(4):
        feature_path = tmp_path / ("run-%02d.features.parquet" % index)
        pq.write_table(
            pa.Table.from_pylist([{"feature_idx": 1}]), feature_path
        )
        feature_paths.append(feature_path)
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "CREATE TABLE project_metadata (key VARCHAR, value_json VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE runs (run_id VARCHAR, run_order INTEGER, "
            "status VARCHAR, output_format VARCHAR, run_output_path VARCHAR, "
            "features_path VARCHAR, ms2_events_path VARCHAR, "
            "identification_path VARCHAR, ms1_path VARCHAR, "
            "external_evidence_path VARCHAR)"
        )
        connection.executemany(
            "INSERT INTO project_metadata VALUES (?, ?)",
            [
                ("project_schema_version", "12"),
                ("resolved_options", json.dumps({"mode": "legacy", "workers": 4})),
            ],
        )
        connection.executemany(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "run-%02d" % index,
                    index,
                    "success",
                    "parquet",
                    None,
                    str(feature_path),
                    None,
                    None,
                    None,
                    None,
                )
                for index, feature_path in enumerate(feature_paths)
            ],
        )
    return database, feature_paths


def test_project_validator_parallel_matches_serial_and_preserves_order(
    tmp_path, monkeypatch
):
    database, feature_paths = _four_run_validation_project(tmp_path)
    database_mtime = database.stat().st_mtime_ns
    expected = {"run_count": 4, "problems": ()}
    assert validate_project(database, workers=1) == expected
    assert validate_project(database) == expected
    monkeypatch.setattr(
        project_validation, "VALIDATION_MIN_PARALLEL_OUTPUT_BYTES", 0
    )
    assert validate_project(database, workers=4) == expected
    assert database.stat().st_mtime_ns == database_mtime

    for index in (1, 3):
        pq.write_table(
            pa.Table.from_pylist([{"feature_idx": 1}, {"feature_idx": 1}]),
            feature_paths[index],
        )
    messages = []
    for workers in (1, None, 4):
        try:
            validate_project(database, workers=workers)
        except ValueError as error:
            messages.append(str(error))
        else:
            raise AssertionError("duplicate features were not rejected")
    assert messages[0] == messages[1] == messages[2]
    assert messages[0].endswith(
        "run-01 has invalid/duplicate feature IDs; "
        "run-03 has invalid/duplicate feature IDs"
    )


def test_project_validator_duckdb_readers_use_one_thread(tmp_path, monkeypatch):
    import duckdb

    feature_path = tmp_path / "features.duckdb"
    external_path = tmp_path / "external.duckdb"
    with duckdb.connect(str(feature_path)) as connection:
        connection.execute("CREATE TABLE features (feature_idx INTEGER)")
        connection.execute("INSERT INTO features VALUES (1)")
    with duckdb.connect(str(external_path)) as connection:
        connection.execute(
            "CREATE TABLE external_id_evidence "
            "(status VARCHAR, feature_id INTEGER, acceptance_q_value DOUBLE)"
        )

    real_connect = duckdb.connect
    connections = []

    class TrackingConnection:
        def __init__(self, connection):
            self.connection = connection
            self.statements = []

        def __enter__(self):
            self.connection.__enter__()
            return self

        def __exit__(self, *args):
            return self.connection.__exit__(*args)

        def execute(self, statement, *args, **kwargs):
            self.statements.append(statement)
            return self.connection.execute(statement, *args, **kwargs)

    def tracking_connect(*args, **kwargs):
        connection = TrackingConnection(real_connect(*args, **kwargs))
        connections.append((kwargs, connection))
        return connection

    monkeypatch.setattr(duckdb, "connect", tracking_connect)
    assert project_validation._read_output_table(feature_path, "features").num_rows == 1
    assert project_validation._validate_run(
        (
            0,
            "run",
            "success",
            "duckdb",
            str(feature_path),
            str(feature_path),
            None,
            None,
            None,
            str(external_path),
            "legacy",
            False,
            False,
        )
    ) == (0, ())
    assert len(connections) == 3
    for config, connection in connections:
        assert config["config"] == {"threads": "1"}
        assert "SET threads TO 1" in connection.statements


def test_project_validator_caps_recorded_memory_to_current_host(
    tmp_path, monkeypatch
):
    database, _feature_paths = _four_run_validation_project(tmp_path)
    recorded_memory = 1200 * 1024 ** 3
    current_memory = 16 * 1024 ** 3
    import duckdb

    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "UPDATE project_metadata SET value_json = ? "
            "WHERE key = 'resolved_options'",
            [json.dumps({"mode": "legacy", "_max_memory_bytes": recorded_memory})],
        )
    captured_budgets = []
    monkeypatch.setattr(
        project_validation, "physical_memory_bytes", lambda: current_memory
    )
    monkeypatch.setattr(
        project_validation,
        "_validation_worker_count",
        lambda _runs, _workers, budget, **_kwargs: captured_budgets.append(budget) or 1,
    )
    assert validate_project(database) == {"run_count": 4, "problems": ()}
    assert captured_budgets == [current_memory]

    captured_budgets.clear()
    monkeypatch.setattr(project_validation, "physical_memory_bytes", lambda: 0)
    assert validate_project(database) == {"run_count": 4, "problems": ()}
    assert captured_budgets == [recorded_memory]
