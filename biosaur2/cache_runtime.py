"""Unified cache roots and per-invocation cleanup."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
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
