import csv
import subprocess
import sys

import pytest

from biosaur2.project_manifest import (
    auto_pair_runs,
    normalized_mzml_stem,
    normalized_psm_stem,
    read_manifest,
    write_manifest,
)


def test_known_suffix_normalization_removes_only_the_expected_tail():
    assert normalized_mzml_stem("sample.part.mzML.gz") == "sample.part"
    assert normalized_mzml_stem("sample.mzML") == "sample"
    assert normalized_psm_stem("sample.part.percolator.target.psms.tsv") == "sample.part"
    assert normalized_mzml_stem("sample.mzML.txt") is None


def test_auto_pairing_is_exact_deterministic_and_reports_both_unmatched_sets(tmp_path):
    mzml = tmp_path / "mzml"
    psm = tmp_path / "psm"
    mzml.mkdir()
    psm.mkdir()
    (mzml / "b.mzML.gz").write_bytes(b"")
    (mzml / "a.part.mzML").write_bytes(b"")
    (mzml / "missing.mzML").write_bytes(b"")
    (psm / "b.percolator.target.psms.tsv").write_text("header\n")
    (psm / "a.part.percolator.target.psms.tsv").write_text("header\n")
    (psm / "orphan.percolator.target.psms.tsv").write_text("header\n")

    report = auto_pair_runs(mzml, psm)

    assert [row["run_id"] for row in report.rows] == ["a.part", "b", "missing"]
    assert report.rows[-1]["psm_path"] == ""
    assert [path.name for path in report.mzml_without_psm] == ["missing.mzML"]
    assert [path.name for path in report.orphan_psms] == [
        "orphan.percolator.target.psms.tsv"
    ]


def test_auto_pairing_rejects_duplicate_normalized_stems(tmp_path):
    mzml = tmp_path / "mzml"
    psm = tmp_path / "psm"
    mzml.mkdir()
    psm.mkdir()
    (mzml / "run.mzML").write_bytes(b"")
    (mzml / "RUN.mzML.gz").write_bytes(b"")
    with pytest.raises(ValueError, match="duplicate normalized mzML"):
        auto_pair_runs(mzml, psm)


def test_manifest_read_resolves_relative_paths_and_rejects_duplicates(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "run.mzML.gz").write_bytes(b"")
    manifest = tmp_path / "runs.tsv"
    write_manifest(
        manifest,
        [{"run_id": "run", "mzml_path": "data/run.mzML.gz", "q_value_max": "0.01"}],
    )
    rows = read_manifest(manifest)
    assert rows[0].mzml_path == (data / "run.mzML.gz").resolve()
    assert rows[0].q_value_max == 0.01

    with manifest.open("a", encoding="utf-8") as handle:
        handle.write("run\tdata/run.mzML.gz\n")
    with pytest.raises(ValueError, match="duplicate run_id"):
        read_manifest(manifest)


def test_project_make_manifest_cli_and_alias_write_same_rows(tmp_path):
    mzml = tmp_path / "mzml"
    psm = tmp_path / "psm"
    mzml.mkdir()
    psm.mkdir()
    (mzml / "run.mzML.gz").write_bytes(b"")
    (psm / "run.percolator.target.psms.tsv").write_text("header\n")
    first = tmp_path / "first.tsv"
    second = tmp_path / "second.tsv"
    common = ["--mzml-dir", str(mzml), "--psm-dir", str(psm)]
    project = subprocess.run(
        [sys.executable, "-m", "biosaur2.search", "project", "make-manifest", *common, "--output", str(first)],
        text=True,
        capture_output=True,
    )
    alias = subprocess.run(
        [sys.executable, "-m", "biosaur2.search", "build-manifest", *common, "--output", str(second)],
        text=True,
        capture_output=True,
    )
    assert project.returncode == alias.returncode == 0
    with first.open(newline="", encoding="utf-8") as handle:
        first_rows = list(csv.DictReader(handle, delimiter="\t"))
    with second.open(newline="", encoding="utf-8") as handle:
        second_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert first_rows == second_rows
