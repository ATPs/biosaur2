import re
import subprocess
import sys

import pytest

import biosaur2.search as search_module


def _run(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "biosaur2.search", *map(str, arguments)],
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    ("extra", "expected"),
    [([], "tsv"), (["--feature-mode", "hybrid"], "parquet")],
)
def test_mode_selects_default_format(monkeypatch, tmp_path, extra, expected):
    captured = {}
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        search_module,
        "_execute_inputs",
        lambda args, parser, logger: captured.update(args),
    )
    monkeypatch.setattr(
        sys, "argv", ["biosaur2", str(source), *extra]
    )
    search_module.run()
    assert captured["format"] == expected


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ([], "info"),
        (["--log-level", "quiet"], "quiet"),
        (["--log-level", "warning"], "warning"),
        (["--log-level", "debug"], "debug"),
        (["-debug"], "debug"),
    ],
)
def test_log_level_is_parsed_and_legacy_debug_is_accepted(
    monkeypatch, tmp_path, extra, expected
):
    captured = {}
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"")
    monkeypatch.setattr(
        search_module,
        "_execute_inputs",
        lambda args, parser, logger: captured.update(args),
    )
    monkeypatch.setattr(sys, "argv", ["biosaur2", str(source), *extra])
    search_module.run()
    assert captured["log_level"] == expected


def test_log_level_validation_and_debug_format(tmp_path):
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"")
    result = _run(source, "--log-level", "loud")
    assert result.returncode != 0
    assert "invalid choice" in result.stderr

    result = _run(source, "--log-level", "info", "-debug")
    assert result.returncode != 0
    assert "not allowed with argument" in result.stderr

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import logging; "
                "from biosaur2.search import _configure_logging; "
                "from biosaur2.main import _debug_stage_complete, _debug_stage_start; "
                "_configure_logging('debug', 'sample'); "
                "logging.getLogger(__name__).debug('probe'); "
                "stage = _debug_stage_start('probe_stage'); "
                "_debug_stage_complete('probe_stage', stage)"
            ),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert re.search(
        r"DEBUG: \[\d{2}:\d{2}:\d{2}\] \[run=sample pid=\d+\] probe",
        result.stderr,
    )
    assert "Stage complete: probe_stage runtime_sec=" in result.stderr


def test_invalid_output_controls_fail_during_argument_validation(tmp_path):
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"")
    result = _run(
        source,
        "--format",
        "parquet",
        "--parquet-compression",
        "snappy",
        "--parquet-compression-level",
        "3",
    )
    assert result.returncode != 0
    assert "supported only by zstd and brotli" in result.stderr

    result = _run(
        source,
        "--stop-after-hills",
        "--format", "tsv",
        "--parquet-engine",
        "duckdb",
    )
    assert result.returncode != 0
    assert "requires at least one Parquet output" in result.stderr


@pytest.mark.parametrize(
    "removed_option",
    [
        "--parquet-layout",
        "--legacy-columns",
        "--scalar-float",
        "--intensity-float",
        "--quantification",
        "--output-rt-unit",
        "--duckdb-parquet-version",
        "--intensity-decimals",
        "-nprocs",
        "--nprocs",
        "--run-workers",
        "--raw-ms1-cache-dir",
        "--hybrid-stage-cache-dir",
        "--hybrid-candidate-cache-dir",
        "--feature-format",
        "--feature_format",
        "--hills-format",
        "--hills_format",
        "--ms1-format",
        "--ms1_format",
        "--duckdb-output",
        "--duckdb_output",
        "--parquet-temp-dir",
        "--parquet_temp_dir",
    ],
)
def test_removed_output_options_are_rejected(tmp_path, removed_option):
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"")
    result = _run(source, removed_option, "value")
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr


def test_help_documents_everyday_output_contract():
    result = _run("--help")
    assert result.returncode == 0
    for text in (
        "features.tsv",
        "--format",
        "--workers",
        "--log-level",
        "--cache-dir",
        "--keep-cache",
        "--continue-on-error",
        "--feature-mode",
        "--generic-ms2-ppm",
        "--ms2-rt-tolerance-sec",
        "--quant-method",
        "--max-charge",
        "Input notes:",
        "same-run Percolator target",
        "<stem>.identifications.parquet",
        "docs/",
    ):
        assert text in result.stdout
    for advanced in (
        "--write-ms2",
        "--write-extra-details",
        "--generic-ms2-isotope-errors",
        "--generic-local-max-width-sec",
        "--parquet-compression",
    ):
        assert advanced not in result.stdout


def test_help_all_explains_defaults_and_advanced_evidence_controls():
    result = _run("--help-all")
    assert result.returncode == 0
    help_text = " ".join(result.stdout.split())
    for text in (
        "(default: legacy)",
        "(default: 0.01)",
        "(default: 120.0)",
        "(default: 7)",
        "(default: all)",
        "(default: False)",
        "same-run Percolator",
        "target/decoy",
        "no donor runs",
        "0,1,2,3",
        "99th percentile",
        "clamped to 15-60 s",
        "separate from --ms2-rt-tolerance-sec",
    ):
        assert text in help_text


def test_ms2_rejects_unsupported_modes(tmp_path):
    hills = tmp_path / "sample.hills.parquet"
    hills.write_bytes(b"not read")
    result = _run(hills, "--write-ms2")
    assert result.returncode != 0
    assert "--write-ms2 cannot be used with hills input" in result.stderr

    mzml = tmp_path / "sample.mzML"
    mzml.write_bytes(b"not read")
    result = _run(mzml, "--feature-mode", "hybrid", "--write-ms2")
    assert result.returncode != 0
    assert "legacy-only diagnostic" in result.stderr



def test_all_multi_input_collisions_are_checked_before_processing(tmp_path):
    first = tmp_path / "first.mzML"
    second = tmp_path / "second.mzML"
    first.write_bytes(b"not read")
    second.write_bytes(b"not read")
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "second.features.tsv").write_text("existing\n")

    result = _run(first, second, "-o", output)
    assert result.returncode != 0
    assert "Output already exists" in result.stderr
    assert not (output / "first.features.tsv").exists()


@pytest.mark.parametrize(
    "option", ["--stop-after-hills", "--write-hills", "--write-ms1"]
)
def test_hills_input_rejects_mzml_only_output_options(tmp_path, option):
    source = tmp_path / "sample.hills.parquet"
    source.write_bytes(b"not read")
    result = _run(source, option)
    assert result.returncode != 0
    assert "%s cannot be used with hills input" % option in result.stderr


def test_mixed_inputs_are_validated_before_any_output_manager(tmp_path):
    mzml_source = tmp_path / "first.mzML"
    hills_source = tmp_path / "second.hills.parquet"
    mzml_source.write_bytes(b"not read")
    hills_source.write_bytes(b"not read")
    output = tmp_path / "outputs"
    result = _run(mzml_source, hills_source, "--write-ms1", "-o", output)
    assert result.returncode != 0
    assert "--write-ms1 cannot be used with hills input" in result.stderr
    assert not output.exists()


def test_removed_weak_mode_and_hybrid_mode_validation(tmp_path):
    hills = tmp_path / "sample.hills.parquet"
    hills.write_bytes(b"not read")
    result = _run(hills, "--feature-mode", "hybrid")
    assert result.returncode != 0
    assert "normal mzML" in result.stderr

    mzml = tmp_path / "sample.mzML"
    mzml.write_bytes(b"not read")
    result = _run(mzml, "--feature-mode", "weak-ms2")
    assert result.returncode != 0
    assert "invalid choice" in result.stderr

    result = _run(mzml, "--ms2-seed")
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr

    result = _run(mzml, "--ms2-seed-ppm", "10")
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr

    result = _run(mzml, "--ms2-seed-isotope-errors", "0,1")
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr

    result = _run(mzml, "--generic-ms2-ppm", "0")
    assert result.returncode != 0
    assert "finite positive number" in result.stderr

    result = _run(
        "project", "run",
        "--manifest", tmp_path / "missing.tsv",
        "--output-dir", tmp_path / "runs",
        "--project-db", tmp_path / "project.duckdb",
        "--mode", "weak-ms2",
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr

    result = _run(mzml, "-cmin", "8", "--max-charge", "7")
    assert result.returncode != 0
    assert "must be at least -cmin" in result.stderr


def test_project_rt_tolerance_is_exposed_and_validated(tmp_path):
    help_result = _run("project", "run", "--help")
    assert help_result.returncode == 0
    assert "--ms2-rt-tolerance-sec" in help_result.stdout
    assert "--workers" in help_result.stdout
    assert "--log-level" in help_result.stdout
    assert "--cache-dir" in help_result.stdout
    assert "--keep-cache" in help_result.stdout
    assert "--max-charge" in help_result.stdout
    assert "--relaxed-ms2-feature" not in help_result.stdout
    assert "(default: 120.0)" in help_result.stdout
    assert "(default: 0.01)" in help_result.stdout
    assert "(default: all)" in help_result.stdout
    assert "input project manifest TSV (required)" in help_result.stdout
    assert "aligned external assays" in help_result.stdout
    assert "README.md and examples/hybrid_project_manifest.tsv" in help_result.stdout

    all_help = _run("project", "run", "--help-all")
    assert all_help.returncode == 0
    assert "--relaxed-ms2-feature" in all_help.stdout
    assert "recipient-run m/z tolerance" in all_help.stdout
    assert "--external-q-value-max" in all_help.stdout
    assert "--external-weak-max-strong-overlap" in all_help.stdout
    assert "--external-min-support-runs" in all_help.stdout
    assert "--external-max-support-runs" in all_help.stdout
    assert "(default: 0.1)" in all_help.stdout
    assert "(default: 0.3)" in all_help.stdout
    assert "explicit positive value disables adaptation" in all_help.stdout

    result = _run(
        "project",
        "run",
        "--manifest",
        tmp_path / "missing.tsv",
        "--output-dir",
        tmp_path / "runs",
        "--project-db",
        tmp_path / "project.duckdb",
        "--ms2-rt-tolerance-sec",
        "-1",
    )
    assert result.returncode != 0
    assert "finite nonnegative number" in result.stderr


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (("--external-weak-max-strong-overlap", "nan"), "finite and in [0, 1]"),
        (("--external-weak-max-strong-overlap", "1.1"), "finite and in [0, 1]"),
        (("--external-min-support-runs", "0"), "positive integer"),
        (("--external-max-support-runs", "17"), "at most 16"),
        (("--external-min-support-runs", "5", "--external-max-support-runs", "4"), "cannot exceed"),
    ],
)
def test_project_external_rescue_options_are_validated(tmp_path, options, message):
    result = _run(
        "project", "run",
        "--manifest", tmp_path / "missing.tsv",
        "--output-dir", tmp_path / "runs",
        "--project-db", tmp_path / "project.duckdb",
        *options,
    )
    assert result.returncode != 0
    assert message in result.stderr


def test_project_manifest_help_documents_inputs_and_defaults():
    result = _run("project", "make-manifest", "--help")
    assert result.returncode == 0
    for text in (
        "directory containing .mzML or .mzML.gz files",
        "directory containing PSM tables paired by exact",
        "normalized stem (required)",
        "built-in Percolator/PSM suffixes",
        "(default: None)",
        "output manifest TSV path (required)",
        "atomically replace an existing manifest",
        "(default:",
    ):
        assert text in result.stdout
