"""Unified cache roots and per-invocation cleanup."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import tempfile
import time
import uuid

from .output import input_stem


def default_cache_dir() -> Path:
    return Path.cwd() / ".biosaur2_cache"


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return cleaned or "run"


def run_cache_key(source_path, run_id=None) -> str:
    source = Path(source_path).resolve()
    label = _safe_name(run_id or input_stem(str(source)))
    path_hash = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    return "%s-%s" % (label, path_hash)


def run_cache_paths(workspace, source_path, run_id=None):
    run_dir = Path(workspace).resolve() / "runs" / run_cache_key(
        source_path, run_id=run_id
    )
    return {
        "cache_run_dir": str(run_dir),
        "raw_ms1_cache": str(run_dir / "raw-ms1"),
        "strict_stage_cache": str(run_dir / "strict-stage"),
        "candidate_cache": str(run_dir / "candidates"),
        # Final ownership is consumed only by the project-level external-ID
        # stage.  Keep it separate from reusable upstream caches because it
        # represents the complete postprocessing population.
        "residual_ownership_cache": str(run_dir / "residual-ownership"),
    }


@dataclass(frozen=True)
class CacheWorkspace:
    root: Path
    workspace: Path
    keep: bool

    @classmethod
    def create(cls, cache_dir, *, keep: bool):
        root = Path(cache_dir).resolve()
        if keep:
            workspace = root
        else:
            identifier = "%d-%d-%s" % (
                int(time.time()),
                os.getpid(),
                uuid.uuid4().hex[:8],
            )
            workspace = root / "temporary" / identifier
        workspace.mkdir(parents=True, exist_ok=True)
        return cls(root=root, workspace=workspace, keep=keep)

    def paths_for(self, source_path, run_id=None):
        return run_cache_paths(self.workspace, source_path, run_id=run_id)

    def cleanup(self):
        if self.keep:
            return
        shutil.rmtree(self.workspace, ignore_errors=True)
        temporary = self.root / "temporary"
        for directory in (temporary, self.root):
            try:
                directory.rmdir()
            except OSError:
                pass


def _project_key(project_db) -> str:
    target = Path(project_db).resolve()
    label = _safe_name(target.stem)
    digest = hashlib.sha256(str(target).encode("utf-8")).hexdigest()[:12]
    return "%s-%s" % (label, digest)


@dataclass(frozen=True)
class ProjectCacheWorkspace:
    """Project cache layout that survives interruption but not success."""

    root: Path
    workspace: Path
    state_dir: Path
    keep: bool

    @classmethod
    def create(cls, cache_dir, project_db, *, keep: bool):
        root = Path(cache_dir).resolve()
        state_dir = root / "projects" / _project_key(project_db)
        workspace = root if keep else state_dir
        workspace.mkdir(parents=True, exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        return cls(root=root, workspace=workspace, state_dir=state_dir, keep=keep)

    @property
    def checkpoint_path(self):
        return self.state_dir / "project-state.json"

    def cleanup(self, *, success):
        if not success or self.keep:
            return
        shutil.rmtree(self.workspace, ignore_errors=True)
        projects = self.root / "projects"
        try:
            projects.rmdir()
        except OSError:
            pass


def remove_cache_layers(paths, layers):
    """Remove selected known cache layers without following symlinks."""

    keys = {
        "raw": "raw_ms1_cache",
        "strict": "strict_stage_cache",
        "candidate": "candidate_cache",
        "ownership": "residual_ownership_cache",
    }
    for layer in layers:
        path = Path(paths[keys[layer]])
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


class ProjectCheckpoint:
    """Atomic project progress state, including an interruption-safe lease."""

    VERSION = 1

    def __init__(self, path):
        self.path = Path(path)
        self.state = {}

    def open(self, identity, *, resume):
        if self.path.is_file() and resume:
            with self.path.open(encoding="utf-8") as handle:
                self.state = json.load(handle)
            if self.state.get("identity") != identity:
                raise ValueError(
                    "project checkpoint has different scientific inputs/options; "
                    "use --no-resume with --overwrite for a fresh project"
                )
        else:
            self.state = {
                "version": self.VERSION,
                "identity": identity,
                "runs": {},
                "external": {},
                "lease": None,
            }
        self._claim()
        self.save()
        return self

    def _claim(self):
        lease = self.state.get("lease")
        now = time.time()
        if lease:
            same_host = lease.get("hostname") == socket.gethostname()
            active = False
            if same_host and int(lease.get("pid", -1)) != os.getpid():
                try:
                    os.kill(int(lease.get("pid", -1)), 0)
                    active = True
                except (OSError, ValueError):
                    pass
            fresh_remote = not same_host and now - float(lease.get("heartbeat", 0)) < 120
            if active or fresh_remote:
                raise RuntimeError(
                    "project checkpoint is owned by %s pid %s"
                    % (lease.get("hostname"), lease.get("pid"))
                )
        self.state["lease"] = {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "heartbeat": now,
        }

    def save(self):
        if self.state.get("lease") is not None:
            self.state["lease"]["heartbeat"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="." + self.path.name + ".", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def release(self):
        if self.state:
            self.state["lease"] = None
            self.save()

    def run_record(self, run_id):
        return self.state.get("runs", {}).get(run_id)

    def put_run(self, run_id, record):
        self.state.setdefault("runs", {})[run_id] = record
        self.save()

    def external_record(self, run_id):
        return self.state.get("external", {}).get(run_id)

    def put_external(self, run_id, record):
        self.state.setdefault("external", {})[run_id] = record
        self.save()
