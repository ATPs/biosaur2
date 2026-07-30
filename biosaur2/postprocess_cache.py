"""Fingerprint-safe cache for expensive hybrid local candidate extraction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile

from .raw_ms1 import source_fingerprint
from .generic_association import C13_C12_MASS_DIFF


LOCAL_CANDIDATE_CACHE_VERSION = 1
PAYLOAD_NAME = "candidate_pairs.pkl"
MANIFEST_NAME = "manifest.json"


def _implementation_signature():
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "generic_association.py",
        "generic_local.py",
        "local_refinement.py",
        "optimization.py",
        "confidence.py",
        "chemistry.py",
        "raw_ms1.py",
        "residual.py",
        "postprocess_cache.py",
    ):
        path = package / name
        digest.update(name.encode("ascii"))
        digest.update(path.read_bytes())
    digest.update(
        ("C13_C12_MASS_DIFF=%.17g" % C13_C12_MASS_DIFF).encode("ascii")
    )
    return digest.hexdigest()


def _file_sha256(path, block_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _event_signature(events):
    digest = hashlib.sha256()
    fields = (
        "ms2_event_id",
        "selected_ion_mz",
        "isolation_target_mz",
        "isolation_lower_offset",
        "isolation_upper_offset",
        "charge",
        "rt_sec",
        "precursor_ms1_index",
        "faims_cv",
        "ion_mobility",
    )
    for event in events:
        digest.update(
            json.dumps(
                [event.get(field) for field in fields],
                allow_nan=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def local_candidate_fingerprint(
    source_path,
    *,
    stage,
    target_events,
    decoy_events,
    options,
    residual_state,
    raw_scan_count,
    raw_point_count,
):
    return {
        "cache_version": LOCAL_CANDIDATE_CACHE_VERSION,
        "source_fingerprint": source_fingerprint(source_path),
        "implementation_signature": _implementation_signature(),
        "stage": str(stage),
        "target_event_signature": _event_signature(target_events),
        "decoy_event_signature": _event_signature(decoy_events),
        "event_count": len(target_events),
        "options": dict(sorted(options.items())),
        "residual_state": str(residual_state),
        "raw_scan_count": int(raw_scan_count),
        "raw_point_count": int(raw_point_count),
    }


def cache_key(fingerprint):
    encoded = json.dumps(
        fingerprint, sort_keys=True, separators=(",", ":"), allow_nan=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_local_candidate_pairs(root, fingerprint):
    if not root:
        return None, None
    path = Path(root).resolve() / (
        "%s-%s" % (fingerprint["stage"], cache_key(fingerprint)[:20])
    )
    if not path.is_dir():
        return None, path
    with (path / MANIFEST_NAME).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("fingerprint") != fingerprint:
        return None, path
    payload_path = path / PAYLOAD_NAME
    if payload_path.stat().st_size != int(manifest["payload_bytes"]):
        raise ValueError("local candidate cache payload size mismatch")
    if _file_sha256(payload_path) != manifest.get("payload_sha256"):
        raise ValueError("local candidate cache payload hash mismatch")
    with payload_path.open("rb") as handle:
        payload = pickle.load(handle)
    return (tuple(payload["target"]), tuple(payload["decoy"])), path


def save_local_candidate_pairs(path, fingerprint, targets, decoys):
    target = Path(path).resolve()
    if target.exists():
        cached, _ = load_local_candidate_pairs(target.parent, fingerprint)
        if cached is None:
            raise FileExistsError("candidate cache path already exists: %s" % target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".%s." % target.name, dir=target.parent)
    )
    try:
        payload_path = temporary / PAYLOAD_NAME
        with payload_path.open("wb") as handle:
            pickle.dump(
                {"target": tuple(targets), "decoy": tuple(decoys)},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {
            "fingerprint": fingerprint,
            "payload": PAYLOAD_NAME,
            "payload_bytes": payload_path.stat().st_size,
            "payload_sha256": _file_sha256(payload_path),
        }
        with (temporary / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
