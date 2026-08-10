import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq

from biosaur2 import project_cli
from biosaur2.schema import compact_schemas
from biosaur2.project import (
    _command_for_run,
    _project_worker,
    _local_resume_option_signature,
    _resume_option_signature,
    _input_fingerprint,
    _read_successful_runs,
    _scientific_command,
    _write_project_database,
    _run_paths,
    validate_project,
)


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
