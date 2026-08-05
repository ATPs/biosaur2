"""Fingerprint-safe reusable strict-stage cache for hybrid development runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import shutil
import tempfile

import numpy as np

from .raw_ms1 import source_fingerprint
STRICT_STAGE_CACHE_VERSION = 3
PAYLOAD_NAME = "strict_stage.pkl"
MANIFEST_NAME = "manifest.json"
logger = logging.getLogger(__name__)

UPSTREAM_ARGUMENTS = (
    "mini",
    "minmz",
    "maxmz",
    "pasefmini",
    "htol",
    "itol",
    "ignore_iso_calib",
    "use_hill_calib",
    "paseftol",
    "nm",
    "hvf",
    "ivf",
    "minlh",
    "pasefminlh",
    "cmin",
    "cmax",
    "tof",
    "profile",
    "md_correction",
    "combine_every",
    "iuse",
    "external_id",
)


def strict_stage_argument_signature(args):
    return {
        name: args.get(name)
        for name in UPSTREAM_ARGUMENTS
    }


def _implementation_signature():
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "main.py",
        "preprocessing.py",
        "calibration.py",
        "cutils.pyx",
        "direct_competitors.py",
        "stage_cache.py",
        "external_weak.py",
    ):
        path = package / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
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


def strict_stage_fingerprint(source_path, args):
    return {
        "cache_version": STRICT_STAGE_CACHE_VERSION,
        "source_fingerprint": source_fingerprint(source_path),
        "upstream_arguments": strict_stage_argument_signature(args),
        "implementation_signature": _implementation_signature(),
    }


def invalidate_stale_strict_stage_cache(directory, source_path, args):
    """Remove an incompatible strict cache while preserving other layers."""

    if not directory:
        return None
    cache = Path(directory).resolve()
    if not cache.is_dir():
        return None
    try:
        with (cache / MANIFEST_NAME).open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    expected = strict_stage_fingerprint(source_path, args)
    mismatch = next(
        (key for key, value in expected.items() if manifest.get(key) != value),
        None,
    )
    if mismatch is None:
        return None
    shutil.rmtree(cache)
    logger.info(
        "Invalidated incompatible strict-stage cache %s: %s",
        cache,
        mismatch,
    )
    return mismatch


def _compact_context(context, hill_mass_accuracy, paseftol):
    source = context["hills"]
    candidates = deepcopy(list(context["candidates"]))
    # Candidate envelopes can share nested isotope dictionaries.  Copy each
    # envelope independently so remapping one cannot mutate another.
    weak_candidates = [
        deepcopy(candidate)
        for candidate in source.get("_external_weak_candidates", ())
    ]
    direct_competitors = [
        replace(value, candidate=deepcopy(value.candidate))
        for value in context.get("direct_competitors", ())
    ]
    used = set()
    for candidate in candidates + weak_candidates + [
        value.candidate for value in direct_competitors
    ]:
        used.add(int(candidate["monoisotope idx"]))
        used.update(
            int(value["isotope_idx"])
            for value in candidate["isotopes"]
        )
    ordered = sorted(used)
    remap = {old: new for new, old in enumerate(ordered)}
    remapped_candidates = set()

    def remap_candidate(candidate):
        identity = id(candidate)
        if identity in remapped_candidates:
            return
        candidate["monoisotope idx"] = remap[
            int(candidate["monoisotope idx"])
        ]
        for isotope in candidate["isotopes"]:
            isotope["isotope_idx"] = remap[int(isotope["isotope_idx"])]
        remapped_candidates.add(identity)

    for candidate in candidates:
        remap_candidate(candidate)
    for candidate in weak_candidates:
        remap_candidate(candidate)
    for position, competitor in enumerate(direct_competitors):
        candidate = competitor.candidate
        remap_candidate(candidate)
        direct_competitors[position] = replace(
            competitor, candidate=candidate
        )

    rt_by_local = {
        int(key): float(value)
        for key, value in context["rt_by_local"].items()
    }
    # Preserve the detector's scalar types.  In particular, converting the
    # original numpy intensity scalars to Python float changes the accumulation
    # behavior of legacy intensitySum/intensityApex for large features by a few
    # units.  A reusable cache must replay quantitative values exactly.
    scans = [list(source["hills_scan_lists"][index]) for index in ordered]
    rt_start = [rt_by_local[int(values[0])] for values in scans]
    rt_end = [rt_by_local[int(values[-1])] for values in scans]
    rt_apex = []
    for old_index, values in zip(ordered, scans):
        apex_scan = source["hills_scan_apex"][old_index]
        if apex_scan is None:
            intensities = source["hills_intensity_array"][old_index]
            apex_position = int(np.argmax(intensities)) if intensities else 0
            apex_scan = values[apex_position]
        rt_apex.append(rt_by_local[int(apex_scan)])
    compact = {
        "hills_idx_array_unique": np.asarray(
            [source["hills_idx_array_unique"][index] for index in ordered]
        ),
        "hills_mz_median": np.asarray(
            [source["hills_mz_median"][index] for index in ordered]
        ),
        "hills_lengths": np.asarray(
            [source["hills_lengths"][index] for index in ordered]
        ),
        "hills_scan_lists": scans,
        "hills_scan_sets": [set(values) for values in scans],
        "hills_intensity_array": [
            list(source["hills_intensity_array"][index])
            for index in ordered
        ],
        "hills_idict": [None] * len(ordered),
        "hill_sqrt_of_i": [None] * len(ordered),
        "hills_intensity_apex": [
            source["hills_intensity_apex"][index]
            for index in ordered
        ],
        "hills_scan_apex": [
            source["hills_scan_apex"][index]
            for index in ordered
        ],
        "rtStart": np.asarray(rt_start),
        "rtEnd": np.asarray(rt_end),
        "rtApex": np.asarray(rt_apex),
        "_external_weak_candidates": tuple(weak_candidates),
        "_external_weak_detector_audit": deepcopy(
            source.get("_external_weak_detector_audit", {})
        ),
    }
    if "hills_im_median" in source:
        compact["hills_im_median"] = np.asarray(
            [source["hills_im_median"][index] for index in ordered]
        )
    if "hills_point_rt_array" in source:
        compact["hills_point_rt_array"] = [
            list(source["hills_point_rt_array"][index])
            for index in ordered
        ]
    compact["tmp_mz_array"] = [
        list(source["tmp_mz_array"][index])
        for index in ordered
    ]
    # Candidate generation has already completed.  These three-neighbor m/z
    # indexes are large and are not used by strict replay, hybrid matching or
    # generic association, so do not persist them.
    compact.pop("hills_mz_median_fast_dict", None)
    compact.pop("hills_im_median_fast_dict", None)
    if "hills_scan_number_array" in source:
        compact["hills_scan_number_array"] = [
            list(source["hills_scan_number_array"][index])
            for index in ordered
        ]
    spectra = []
    for spectrum in context["spectra"]:
        spectra.append(
            {
                key: spectrum.get(key)
                for key in ("scan_index", "scan_number", "rt_sec")
            }
        )
    return {
        "hills": compact,
        "rt_by_local": rt_by_local,
        "spectra": spectra,
        "faims_cv": context["faims_cv"],
        "candidates": candidates,
        "direct_competitors": tuple(direct_competitors),
        "mz_step": float(hill_mass_accuracy) * 1e-6 * float(
            np.max(compact["hills_mz_median"])
            if compact["hills_mz_median"].size else 1.0
        ),
        "paseftol": float(paseftol),
    }


def build_strict_stage_payload(
    ingestion,
    strict_contexts,
    next_feature_id,
    args,
):
    return {
        "ms1_rows": list(ingestion.ms1_rows),
        "ms2_rows": list(ingestion.ms2_rows),
        "ms1_metadata": dict(ingestion.ms1_metadata),
        "strict_contexts": tuple(
            _compact_context(
                context,
                args["htol"],
                context.get("paseftol", args.get("paseftol", 0.0)),
            )
            for context in strict_contexts
        ),
        "next_feature_id": int(next_feature_id),
    }


def save_strict_stage_cache(directory, source_path, args, payload):
    target = Path(directory).resolve()
    if target.exists():
        raise FileExistsError("strict stage cache already exists: %s" % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".%s." % target.name, dir=target.parent)
    )
    try:
        payload_path = temporary / PAYLOAD_NAME
        with payload_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        payload_sha256 = _file_sha256(payload_path)
        manifest = strict_stage_fingerprint(source_path, args)
        manifest.update(
            {
                "payload": PAYLOAD_NAME,
                "payload_bytes": payload_path.stat().st_size,
                "payload_sha256": payload_sha256,
                "context_count": len(payload["strict_contexts"]),
                "strict_feature_count": sum(
                    len(context["candidates"])
                    for context in payload["strict_contexts"]
                ),
                "direct_competitor_count": sum(
                    len(context.get("direct_competitors", ()))
                    for context in payload["strict_contexts"]
                ),
                "weak_candidate_count": sum(
                    len(context["hills"].get(
                        "_external_weak_candidates", ()
                    ))
                    for context in payload["strict_contexts"]
                ),
            }
        )
        manifest_path = temporary / MANIFEST_NAME
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        for path in temporary.iterdir():
            path.unlink(missing_ok=True)
        temporary.rmdir()
        raise
    return target


def load_strict_stage_cache(directory, source_path, args):
    cache = Path(directory).resolve()
    with (cache / MANIFEST_NAME).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = strict_stage_fingerprint(source_path, args)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                "strict stage cache fingerprint mismatch for %s" % key
            )
    payload_path = cache / manifest.get("payload", PAYLOAD_NAME)
    if payload_path.stat().st_size != int(manifest["payload_bytes"]):
        raise ValueError("strict stage cache payload size mismatch")
    if _file_sha256(payload_path) != manifest.get(
        "payload_sha256"
    ):
        raise ValueError("strict stage cache payload hash mismatch")
    with payload_path.open("rb") as handle:
        return pickle.load(handle), manifest
