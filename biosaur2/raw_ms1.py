"""Compact one-read raw MS1 storage and bounded local trace extraction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Optional

import numpy as np


RAW_MS1_CACHE_VERSION = 1
_CACHE_ARRAYS = (
    "offsets",
    "mz",
    "intensity",
    "source_scan_index",
    "scan_number",
    "rt_sec",
    "faims_cv",
)


def source_fingerprint(path, *, block_bytes=1024 * 1024):
    """Return a bounded content/stat fingerprint for cache invalidation."""

    source = Path(path).resolve()
    stat = source.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    with source.open("rb") as handle:
        digest.update(handle.read(block_bytes))
        if stat.st_size > block_bytes:
            handle.seek(max(0, stat.st_size - block_bytes))
            digest.update(handle.read(block_bytes))
    return {
        "resolved_path": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "edge_sha256": digest.hexdigest(),
    }


@dataclass(frozen=True)
class ExtractedTrace:
    scan_index: np.ndarray
    scan_number: np.ndarray
    rt_sec: np.ndarray
    intensity: np.ndarray
    observed_mz: np.ndarray
    point_present: np.ndarray
    target_mz: float
    ppm: float


def event_position_in_trace(
    trace: ExtractedTrace,
    event_rt_sec: float,
    precursor_ms1_index: Optional[int],
) -> int:
    """Locate an MS2 event on an extracted MS1 grid.

    mzML precursor metadata identifies the actual source survey scan.  Use it
    before an RT-nearest fallback because the survey scan after an MS2 can be
    closer in time than the survey scan that produced the precursor.
    """

    if trace.rt_sec.size == 0:
        raise ValueError("cannot locate an event on an empty trace")
    if precursor_ms1_index is not None:
        exact = np.flatnonzero(
            trace.scan_index == int(precursor_ms1_index)
        )
        if exact.size:
            return int(exact[0])
    return int(np.argmin(np.abs(trace.rt_sec - float(event_rt_sec))))


@dataclass(frozen=True)
class RawMS1Store:
    offsets: np.ndarray
    mz: np.ndarray
    intensity: np.ndarray
    source_scan_index: np.ndarray
    scan_number: np.ndarray
    rt_sec: np.ndarray
    faims_cv: np.ndarray

    @property
    def scan_count(self):
        return int(self.rt_sec.size)

    @property
    def point_count(self):
        return int(self.mz.size)

    @property
    def memory_bytes(self):
        return int(
            sum(
                value.nbytes
                for value in (
                    self.offsets,
                    self.mz,
                    self.intensity,
                    self.source_scan_index,
                    self.scan_number,
                    self.rt_sec,
                    self.faims_cv,
                )
            )
        )

    def scan(self, local_index: int):
        start = int(self.offsets[local_index])
        end = int(self.offsets[local_index + 1])
        return self.mz[start:end], self.intensity[start:end]

    def detector_spectra(
        self,
        *,
        min_intensity: float,
        min_mz: float,
        max_mz: float,
    ):
        """Materialize ordinary centroid spectra from this immutable store."""

        spectra = []
        for local_index in range(self.scan_count):
            mz, intensity = self.scan(local_index)
            keep = (
                (intensity >= float(min_intensity))
                & (mz >= float(min_mz))
                & (mz <= float(max_mz))
            )
            if not np.any(keep):
                continue
            faims_cv = float(self.faims_cv[local_index])
            spectrum = {
                "m/z array": np.asarray(mz[keep], dtype=np.float64).copy(),
                "intensity array": np.asarray(
                    intensity[keep], dtype=np.float64
                ).copy(),
                "mean inverse reduced ion mobility array": np.zeros(
                    int(np.count_nonzero(keep)), dtype=np.float64
                ),
                "ignore_ion_mobility": True,
                "scan_index": int(self.source_scan_index[local_index]),
                "scan_number": (
                    None
                    if int(self.scan_number[local_index]) < 0
                    else int(self.scan_number[local_index])
                ),
                "scan_id": int(self.source_scan_index[local_index]),
                "rt_sec": float(self.rt_sec[local_index]),
            }
            if math.isfinite(faims_cv):
                spectrum["FAIMS compensation voltage"] = faims_cv
            spectra.append(spectrum)
        return spectra

    def extract_trace(
        self,
        target_mz: float,
        ppm: float,
        rt_start_sec: float,
        rt_end_sec: float,
        *,
        faims_cv: Optional[float] = None,
    ) -> ExtractedTrace:
        """Extract one zero-filled centroid XIC on the real MS1 scan grid."""

        if not math.isfinite(target_mz) or target_mz <= 0:
            raise ValueError("target_mz must be finite and positive")
        if not math.isfinite(ppm) or ppm <= 0:
            raise ValueError("ppm must be finite and positive")
        if rt_end_sec < rt_start_sec:
            raise ValueError("RT end must not precede RT start")
        selected = (self.rt_sec >= rt_start_sec) & (self.rt_sec <= rt_end_sec)
        if faims_cv is None:
            selected &= np.isnan(self.faims_cv)
        else:
            selected &= np.isfinite(self.faims_cv) & np.isclose(
                self.faims_cv, float(faims_cv), atol=1e-6, rtol=0.0
            )
        local_indices = np.flatnonzero(selected)
        intensities = np.zeros(local_indices.size, dtype=np.float64)
        observed = np.full(local_indices.size, np.nan, dtype=np.float64)
        tolerance = target_mz * ppm * 1e-6
        lower = target_mz - tolerance
        upper = target_mz + tolerance
        for output_index, local_index in enumerate(local_indices):
            mz_values, intensity_values = self.scan(int(local_index))
            start = int(np.searchsorted(mz_values, lower, side="left"))
            end = int(np.searchsorted(mz_values, upper, side="right"))
            if end <= start:
                continue
            local_intensity = np.asarray(intensity_values[start:end], dtype=np.float64)
            positive = local_intensity > 0
            if not np.any(positive):
                continue
            local_mz = np.asarray(mz_values[start:end], dtype=np.float64)[positive]
            local_intensity = local_intensity[positive]
            total = float(np.sum(local_intensity, dtype=np.float64))
            intensities[output_index] = total
            observed[output_index] = float(np.average(local_mz, weights=local_intensity))
        present = intensities > 0
        return ExtractedTrace(
            scan_index=self.source_scan_index[local_indices].copy(),
            scan_number=self.scan_number[local_indices].copy(),
            rt_sec=self.rt_sec[local_indices].copy(),
            intensity=intensities,
            observed_mz=observed,
            point_present=present,
            target_mz=float(target_mz),
            ppm=float(ppm),
        )


class RawMS1StoreBuilder:
    def __init__(self):
        self.mz = []
        self.intensity = []
        self.source_scan_index = []
        self.scan_number = []
        self.rt_sec = []
        self.faims_cv = []

    def append(
        self,
        mz,
        intensity,
        *,
        source_scan_index: int,
        scan_number: Optional[int],
        rt_sec: float,
        faims_cv: Optional[float],
    ):
        mz_values = np.asarray(mz, dtype=np.float64)
        intensity_values = np.asarray(intensity, dtype=np.float64)
        if mz_values.size != intensity_values.size:
            raise ValueError("raw MS1 m/z and intensity arrays have different lengths")
        finite = np.isfinite(mz_values) & np.isfinite(intensity_values)
        mz_values = mz_values[finite]
        intensity_values = intensity_values[finite]
        order = np.argsort(mz_values, kind="stable")
        self.mz.append(mz_values[order])
        self.intensity.append(intensity_values[order])
        self.source_scan_index.append(int(source_scan_index))
        self.scan_number.append(-1 if scan_number is None else int(scan_number))
        self.rt_sec.append(float(rt_sec))
        self.faims_cv.append(np.nan if faims_cv is None else float(faims_cv))

    def finalize(self) -> RawMS1Store:
        lengths = np.asarray([value.size for value in self.mz], dtype=np.int64)
        offsets = np.empty(lengths.size + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        return RawMS1Store(
            offsets=offsets,
            mz=np.concatenate(self.mz) if self.mz else np.empty(0, dtype=np.float64),
            intensity=(
                np.concatenate(self.intensity)
                if self.intensity
                else np.empty(0, dtype=np.float64)
            ),
            source_scan_index=np.asarray(self.source_scan_index, dtype=np.int32),
            scan_number=np.asarray(self.scan_number, dtype=np.int64),
            rt_sec=np.asarray(self.rt_sec, dtype=np.float64),
            faims_cv=np.asarray(self.faims_cv, dtype=np.float64),
        )


def save_raw_ms1_cache(store: RawMS1Store, directory, source_path):
    """Atomically publish a directory of mmap-compatible NumPy arrays."""

    target = Path(directory).resolve()
    if target.exists():
        raise FileExistsError("raw MS1 cache already exists: %s" % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix="." + target.name + ".tmp-", dir=target.parent)
    )
    try:
        for name in _CACHE_ARRAYS:
            np.save(staging / (name + ".npy"), getattr(store, name), allow_pickle=False)
        manifest = {
            "cache_version": RAW_MS1_CACHE_VERSION,
            "source_fingerprint": source_fingerprint(source_path),
            "scan_count": store.scan_count,
            "point_count": store.point_count,
            "arrays": list(_CACHE_ARRAYS),
        }
        with (staging / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def load_raw_ms1_cache(directory, source_path, *, mmap=True):
    """Load and validate a persisted RawMS1Store, optionally using mmap."""

    cache = Path(directory).resolve()
    with (cache / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("cache_version") != RAW_MS1_CACHE_VERSION:
        raise ValueError("unsupported raw MS1 cache version")
    if manifest.get("arrays") != list(_CACHE_ARRAYS):
        raise ValueError("raw MS1 cache array manifest is incomplete")
    actual = source_fingerprint(source_path)
    if manifest.get("source_fingerprint") != actual:
        raise ValueError("raw MS1 cache source fingerprint does not match")
    arrays = {
        name: np.load(
            cache / (name + ".npy"),
            mmap_mode="r" if mmap else None,
            allow_pickle=False,
        )
        for name in _CACHE_ARRAYS
    }
    store = RawMS1Store(**arrays)
    if (
        store.scan_count != manifest.get("scan_count")
        or store.point_count != manifest.get("point_count")
        or store.offsets.size != store.scan_count + 1
        or (store.offsets.size and int(store.offsets[-1]) != store.point_count)
    ):
        raise ValueError("raw MS1 cache dimensions do not match its manifest")
    return store
