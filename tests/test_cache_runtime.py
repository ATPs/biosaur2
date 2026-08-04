from pathlib import Path

import pytest

from biosaur2.cache_runtime import (
    CacheWorkspace,
    ProjectCacheWorkspace,
    ProjectCheckpoint,
    remove_cache_layers,
    run_cache_paths,
)


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


def test_project_workspace_retains_interrupted_cache_then_cleans_success(tmp_path):
    workspace = ProjectCacheWorkspace.create(
        tmp_path / "cache", tmp_path / "result" / "project.duckdb", keep=False
    )
    marker = workspace.workspace / "runs" / "run" / "raw-ms1" / "manifest.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("{}\n", encoding="utf-8")
    workspace.cleanup(success=False)
    assert marker.is_file()
    workspace.cleanup(success=True)
    assert not workspace.workspace.exists()

    retained = ProjectCacheWorkspace.create(
        tmp_path / "retained-cache", tmp_path / "result" / "project.duckdb", keep=True
    )
    retained.checkpoint_path.write_text("{}\n", encoding="utf-8")
    retained.cleanup(success=True)
    assert retained.checkpoint_path.is_file()


def test_project_checkpoint_is_atomic_and_reopenable(tmp_path):
    path = tmp_path / "project-state.json"
    identity = {"project_db": "project.duckdb", "scientific_options": {}}
    checkpoint = ProjectCheckpoint(path).open(identity, resume=True)
    checkpoint.put_run("run", {"status": "success", "result": {"runtime_sec": 1}})
    checkpoint.release()
    resumed = ProjectCheckpoint(path).open(identity, resume=True)
    assert resumed.run_record("run")["result"]["runtime_sec"] == 1
    resumed.release()


def test_project_checkpoint_updates_only_the_changed_record(tmp_path):
    checkpoint = ProjectCheckpoint(tmp_path / "project-state.json").open(
        {"project_db": "project.duckdb"}, resume=True
    )
    checkpoint.put_run("first", {"status": "success", "result": {"id": 1}})
    first = checkpoint.runs_dir / checkpoint._record_name("first")
    first_payload = first.read_bytes()
    checkpoint.put_run("second", {"status": "success", "result": {"id": 2}})
    assert first.read_bytes() == first_payload
    assert len(list(checkpoint.runs_dir.glob("*.json"))) == 2
    metadata = checkpoint.path.read_text(encoding="utf-8")
    assert '"runs"' not in metadata
    checkpoint.release()


def test_project_checkpoint_does_not_replace_an_active_lease(tmp_path):
    path = tmp_path / "project-state.json"
    first = ProjectCheckpoint(path).open({"project_db": "project.duckdb"}, resume=True)
    with pytest.raises(RuntimeError, match="owned"):
        ProjectCheckpoint(path).open({"project_db": "project.duckdb"}, resume=False)
    first.release()


def test_remove_cache_layers_removes_only_requested_paths(tmp_path):
    paths = run_cache_paths(tmp_path, tmp_path / "source.mzML")
    for path in paths.values():
        if path != paths["cache_run_dir"]:
            Path(path).mkdir(parents=True, exist_ok=True)
    remove_cache_layers(paths, ("strict", "candidate"))
    assert Path(paths["raw_ms1_cache"]).is_dir()
    assert not Path(paths["strict_stage_cache"]).exists()
    assert not Path(paths["candidate_cache"]).exists()
