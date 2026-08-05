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
import threading
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
        "external_observations": str(
            run_dir / "external" / "observations-v2.parquet"
        ),
        "external_strong_features": str(
            run_dir / "external" / "strong-features-v2.parquet"
        ),
        "external_weak_candidates": str(
            run_dir / "external" / "weak-candidates-v2.parquet"
        ),
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
    """Crash-safe Project state with O(1) record updates and a host lease.

    Project workspaces can live on NFS, where SQLite WAL is a poor fit.  Keep
    the identity, each run record, and the lease as separate atomically
    published files instead.  Completing one of thousands of runs therefore
    never serializes the rest of the project state.
    """

    VERSION = 2
    HEARTBEAT_SECONDS = 30.0
    STALE_SECONDS = 120.0

    def __init__(self, path):
        self.path = Path(path)
        self.records_dir = self.path.with_name(self.path.stem + ".records")
        self.runs_dir = self.records_dir / "runs"
        self.external_dir = self.records_dir / "external"
        self.lease_dir = self.path.with_name(self.path.stem + ".lease")
        self.state = {}
        self._owner = None
        self._heartbeat_stop = None
        self._heartbeat_thread = None

    def __del__(self):
        # ``run_project`` normally releases explicitly.  This is the fallback
        # for exceptions raised before that point, so a handled failure does
        # not leave a live process holding the lease until it becomes stale.
        try:
            self.release()
        except Exception:
            pass

    @staticmethod
    def _record_name(run_id):
        label = _safe_name(str(run_id))
        digest = hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()[:12]
        return "%s-%s.json" % (label, digest)

    @staticmethod
    def _read_json(path):
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _sync_directory(directory):
        try:
            descriptor = os.open(str(directory), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @classmethod
    def _write_json(cls, path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="." + path.name + ".", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
            cls._sync_directory(path.parent)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def _load_records(self, directory):
        records = {}
        if not directory.is_dir():
            return records
        for path in sorted(directory.glob("*.json")):
            try:
                record = self._read_json(path)
                records[record["run_id"]] = record["record"]
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                # Atomic replacement guarantees that only a manually damaged
                # record can reach this branch.  Treat it as unfinished.
                continue
        return records

    def _write_record(self, directory, run_id, record):
        self._write_json(
            directory / self._record_name(run_id),
            {"run_id": run_id, "record": record},
        )

    def _migrate_v1(self, legacy, identity):
        self.records_dir.mkdir(parents=True, exist_ok=True)
        for run_id, record in legacy.get("runs", {}).items():
            self._write_record(self.runs_dir, run_id, record)
        for run_id, record in legacy.get("external", {}).items():
            # v1 external completion has no dependency/output fingerprints.
            # It must be recomputed before it can be safely reused.
            migrated = dict(record)
            migrated.pop("status", None)
            self._write_record(self.external_dir, run_id, migrated)
        self._write_json(
            self.path,
            {"version": self.VERSION, "identity": identity},
        )

    def _load_or_create(self, identity, resume):
        if self.path.is_file() and resume:
            metadata = self._read_json(self.path)
            saved_identity = metadata.get("identity", {})
            if metadata.get("version") == 1:
                saved_identity = {
                    key: saved_identity.get(key)
                    for key in ("manifest", "output_dir", "project_db")
                }
            if saved_identity != identity:
                raise ValueError(
                    "project checkpoint has different scientific inputs/options; "
                    "use --no-resume with --overwrite for a fresh project"
                )
            if metadata.get("version") == 1:
                self._migrate_v1(metadata, identity)
                metadata = self._read_json(self.path)
            if metadata.get("version") != self.VERSION:
                raise ValueError("unsupported project checkpoint version")
        else:
            if self.records_dir.exists():
                shutil.rmtree(self.records_dir)
            self._write_json(
                self.path, {"version": self.VERSION, "identity": identity}
            )
        self.state = {
            "version": self.VERSION,
            "identity": identity,
            "runs": self._load_records(self.runs_dir),
            "external": self._load_records(self.external_dir),
        }

    def _lease_owner(self):
        try:
            return self._read_json(self.lease_dir / "owner.json")
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _pid_active(pid):
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ValueError):
            return False

    def _lease_is_active(self, owner):
        if not owner:
            try:
                return time.time() - self.lease_dir.stat().st_mtime < self.STALE_SECONDS
            except OSError:
                return False
        if owner.get("hostname") == socket.gethostname():
            return self._pid_active(owner.get("pid", -1))
        return time.time() - float(owner.get("heartbeat", 0)) < self.STALE_SECONDS

    def _claim(self):
        for _attempt in range(4):
            try:
                self.lease_dir.mkdir(parents=False)
            except FileExistsError:
                owner = self._lease_owner()
                if self._lease_is_active(owner):
                    raise RuntimeError(
                        "project checkpoint is owned by %s pid %s"
                        % ((owner or {}).get("hostname"), (owner or {}).get("pid"))
                    )
                stale = self.lease_dir.with_name(
                    self.lease_dir.name + ".stale-" + uuid.uuid4().hex
                )
                try:
                    os.replace(self.lease_dir, stale)
                except FileNotFoundError:
                    continue
                except OSError:
                    continue
                shutil.rmtree(stale, ignore_errors=True)
                continue
            self._owner = {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "heartbeat": time.time(),
            }
            self._write_json(self.lease_dir / "owner.json", self._owner)
            return
        raise RuntimeError("could not claim project checkpoint lease")

    def touch(self):
        if self._owner is None:
            return
        self._owner = {**self._owner, "heartbeat": time.time()}
        self._write_json(self.lease_dir / "owner.json", self._owner)

    def _heartbeat(self):
        while not self._heartbeat_stop.wait(self.HEARTBEAT_SECONDS):
            try:
                self.touch()
            except OSError:
                # A later foreground checkpoint operation will surface an I/O
                # failure; do not silently relinquish a live lease here.
                pass

    def _start_heartbeat(self):
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat, name="biosaur2-project-lease", daemon=True
        )
        self._heartbeat_thread.start()

    def open(self, identity, *, resume):
        self.lease_dir.parent.mkdir(parents=True, exist_ok=True)
        self._claim()
        try:
            self._load_or_create(identity, resume)
            self._start_heartbeat()
            return self
        except BaseException:
            self.release()
            raise

    def release(self):
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=self.HEARTBEAT_SECONDS + 1)
        self._heartbeat_stop = self._heartbeat_thread = None
        owner = self._lease_owner()
        if owner == self._owner:
            shutil.rmtree(self.lease_dir, ignore_errors=True)
        self._owner = None

    def run_record(self, run_id):
        return self.state.get("runs", {}).get(run_id)

    def put_run(self, run_id, record):
        self.state.setdefault("runs", {})[run_id] = record
        self._write_record(self.runs_dir, run_id, record)

    def external_record(self, run_id):
        return self.state.get("external", {}).get(run_id)

    def put_external(self, run_id, record):
        self.state.setdefault("external", {})[run_id] = record
        self._write_record(self.external_dir, run_id, record)
