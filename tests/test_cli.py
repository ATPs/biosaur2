import subprocess
import sys

import pytest


def _run(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "biosaur2.search", *map(str, arguments)],
        text=True,
        capture_output=True,
    )


def test_invalid_output_controls_fail_during_argument_validation(tmp_path):
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"")
    result = _run(source, "--intensity-decimals", "bad")
    assert result.returncode != 0
    assert "nonnegative integer" in result.stderr

    result = _run(
        source,
        "--feature-format",
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
        "--feature-format",
        "parquet",
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
    ],
)
def test_removed_output_options_are_rejected(tmp_path, removed_option):
    source = tmp_path / "sample.mzML"
    source.write_bytes(b"")
    result = _run(source, removed_option, "value")
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr


def test_help_documents_compact_output_contract():
    result = _run("--help")
    assert result.returncode == 0
    for text in (
        "one compact features.parquet file",
        "DuckDB V2",
        "float32",
        "feature_idx",
        "area_sum",
        "minutes",
        "half-away-from-zero",
        "--write-extra-details",
        "--run-workers",
        "--continue-on-error",
        "--write-ms2",
        "--feature-mode",
        "--generic-ms2-ppm",
        "--ms2-rt-tolerance-sec",
        "--quant-method",
        "--max-charge",
        "--relaxed-ms2-feature",
        "--hybrid-candidate-cache-dir",
        "Input notes:",
        "Hybrid DDA with q-filtered same-run Percolator PSMs",
        "design.md",
        "updates/2026-07-30.md",
    ):
        assert text in result.stdout


def test_hybrid_help_explains_defaults_and_evidence_controls():
    result = _run("--help")
    assert result.returncode == 0
    help_text = " ".join(result.stdout.split())
    for text in (
        "(default: legacy)",
        "(default: 0.01)",
        "(default: 120.0)",
        "(default: 7)",
        "(default: envelope_area)",
        "(default: False)",
        "same-run Percolator",
        "target/decoy",
        "no donor runs",
    ):
        assert text in help_text


def test_ms2_and_file_parallelism_reject_unsupported_modes(tmp_path):
    hills = tmp_path / "sample.hills.parquet"
    hills.write_bytes(b"not read")
    result = _run(hills, "--write-ms2")
    assert result.returncode != 0
    assert "--write-ms2 cannot be used with hills input" in result.stderr

    mzml = tmp_path / "sample.mzML"
    mzml.write_bytes(b"not read")
    result = _run(mzml, "-dia", "--run-workers", "2")
    assert result.returncode != 0
    assert "normal mzML" in result.stderr


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
    assert "--allow-nested-parallelism" in help_result.stdout
    assert "--max-charge" in help_result.stdout
    assert "--relaxed-ms2-feature" in help_result.stdout
    assert "(default: 120.0)" in help_result.stdout
    assert "(default: 0.01)" in help_result.stdout
    assert "(default: envelope_area)" in help_result.stdout
    assert "input project manifest TSV (required)" in help_result.stdout
    assert "recipient-run m/z tolerance" in help_result.stdout
    assert "aligned external assays" in help_result.stdout
    assert "README.md and examples/hybrid_project_manifest.tsv" in help_result.stdout

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
