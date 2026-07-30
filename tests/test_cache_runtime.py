from pathlib import Path

from biosaur2.cache_runtime import CacheWorkspace, run_cache_paths


def test_temporary_cache_workspace_cleans_only_its_invocation(tmp_path):
    root = tmp_path / ".biosaur2_cache"
    retained = root / "runs" / "retained"
    retained.mkdir(parents=True)
    (retained / "manifest.json").write_text("{}\n", encoding="utf-8")

    session = CacheWorkspace.create(root, keep=False)
    paths = session.paths_for(tmp_path / "sample.mzML")
    Path(paths["raw_ms1_cache"]).mkdir(parents=True)
    temporary_workspace = session.workspace
    session.cleanup()

    assert not temporary_workspace.exists()
    assert (retained / "manifest.json").is_file()


def test_retained_cache_path_is_stable_and_layered(tmp_path):
    source = tmp_path / "sample.mzML.gz"
    first = run_cache_paths(tmp_path / "cache", source)
    second = run_cache_paths(tmp_path / "cache", source)
    assert first == second
    assert Path(first["raw_ms1_cache"]).name == "raw-ms1"
    assert Path(first["strict_stage_cache"]).name == "strict-stage"
    assert Path(first["candidate_cache"]).name == "candidates"
