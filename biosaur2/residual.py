"""Reversible, intensity-conserving residual MS1 allocation.

The ledger peels accepted feature components from raw centroid intensity
without deleting whole hills.  A feature can therefore own only an RT slice,
only a fitted fraction of an overlapping centroid, or both.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Hashable, Sequence

import numpy as np

from .raw_ms1 import ExtractedTrace, RawMS1Store


@dataclass(frozen=True)
class RawPointAllocation:
    point_index: int
    intensity: float


@dataclass(frozen=True)
class AllocationResult:
    allocation_id: Hashable
    status: str
    requested_intensity: float
    allocated_intensity: float
    raw_point_count: int
    conservation_error: float

    @property
    def accepted(self):
        return self.status == "accepted"


@dataclass(frozen=True)
class ComponentOverlap:
    """Preview of how much one candidate is already explained by ownership."""

    status: str
    requested_intensity: float
    overlapping_intensity: float

    @property
    def fraction(self):
        if self.requested_intensity <= 0:
            return 1.0
        return self.overlapping_intensity / self.requested_intensity


@dataclass(frozen=True)
class ComponentFootprint:
    """Compact raw-point representation of a fitted component.

    Footprints deliberately use raw, rather than residual, intensity when
    distributing each fitted trace value.  They can therefore be built by a
    spawned raw-extraction worker and compared against the recipient's later,
    current ownership ledger without transferring candidate traces.
    """

    status: str
    requested_intensity: float
    allocations: tuple[RawPointAllocation, ...]


class ResidualMS1Ledger:
    """Sparse reversible ownership ledger over one immutable RawMS1Store."""

    def __init__(self, store: RawMS1Store, *, tolerance: float = 1e-9):
        self.store = store
        self.tolerance = float(tolerance)
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("residual tolerance must be finite and nonnegative")
        self._claimed = {}
        self._allocations = {}

    @property
    def allocation_count(self):
        return len(self._allocations)

    @property
    def claimed_point_count(self):
        return len(self._claimed)

    @property
    def claimed_intensity(self):
        return float(sum(self._claimed.values()))

    @property
    def original_intensity(self):
        return float(np.sum(self.store.intensity, dtype=np.float64))

    @property
    def residual_intensity(self):
        return self.original_intensity - self.claimed_intensity

    def state_fingerprint(self):
        """Return an exact deterministic signature of current sparse claims."""

        digest = hashlib.sha256()
        digest.update(struct.pack("<Q", len(self._claimed)))
        for point_index, amount in sorted(self._claimed.items()):
            digest.update(struct.pack("<Qd", int(point_index), float(amount)))
        return digest.hexdigest()

    def footprint_overlap(self, footprint: ComponentFootprint) -> ComponentOverlap:
        """Measure a raw footprint against this ledger's current claims."""

        if footprint.status != "accepted":
            return ComponentOverlap(
                footprint.status, footprint.requested_intensity, 0.0
            )
        overlap = float(
            sum(
                min(value.intensity, self._claimed.get(value.point_index, 0.0))
                for value in footprint.allocations
            )
        )
        return ComponentOverlap("accepted", footprint.requested_intensity, overlap)

    def observed_point_footprint(
        self, contributions, *, mz_tolerance_ppm: float = 0.01
    ) -> ComponentFootprint:
        """Map detector hill centroids to raw points without consuming them."""

        if not math.isfinite(mz_tolerance_ppm) or mz_tolerance_ppm < 0:
            raise ValueError("mz tolerance must be finite and nonnegative")
        normalized = tuple(
            (int(scan), float(mz), float(intensity))
            for scan, mz, intensity in contributions
        )
        if any(
            not math.isfinite(mz)
            or mz <= 0
            or not math.isfinite(intensity)
            or intensity < 0
            for _scan, mz, intensity in normalized
        ):
            raise ValueError(
                "observed-point contributions must have finite positive m/z "
                "and finite nonnegative intensity"
            )
        requested_total = float(
            sum(intensity for _scan, _mz, intensity in normalized)
        )
        if requested_total <= self.tolerance:
            return ComponentFootprint(
                "no_candidate_intensity", requested_total, ()
            )
        by_point = {}
        for source_scan, observed_mz, requested in normalized:
            if requested <= self.tolerance:
                continue
            local_scan = self._local_scan_index(source_scan)
            if local_scan is None:
                return ComponentFootprint(
                    "source_scan_not_found", requested_total, ()
                )
            scan_start = int(self.store.offsets[local_scan])
            scan_end = int(self.store.offsets[local_scan + 1])
            mz_values = self.store.mz[scan_start:scan_end]
            tolerance = max(
                np.finfo(np.float64).eps * max(abs(observed_mz), 1.0) * 8.0,
                observed_mz * float(mz_tolerance_ppm) * 1e-6,
            )
            start = int(np.searchsorted(
                mz_values, observed_mz - tolerance, side="left"
            ))
            end = int(np.searchsorted(
                mz_values, observed_mz + tolerance, side="right"
            ))
            candidates = []
            for point_index in range(scan_start + start, scan_start + end):
                raw = float(self.store.intensity[point_index])
                proposed = by_point.get(point_index, 0.0)
                if requested + proposed <= raw + max(
                    self.tolerance, self.tolerance * raw
                ):
                    candidates.append((
                        abs(float(self.store.mz[point_index]) - observed_mz),
                        abs(raw - proposed - requested),
                        point_index,
                    ))
            if not candidates:
                return ComponentFootprint(
                    "observed_point_not_found", requested_total, ()
                )
            point_index = min(candidates)[2]
            by_point[point_index] = by_point.get(point_index, 0.0) + requested
        return ComponentFootprint(
            "accepted",
            requested_total,
            tuple(
                RawPointAllocation(point_index, intensity)
                for point_index, intensity in sorted(by_point.items())
                if intensity > self.tolerance
            ),
        )

    def _local_scan_index(self, source_scan_index: int):
        values = self.store.source_scan_index
        position = int(np.searchsorted(values, int(source_scan_index)))
        if position >= values.size or int(values[position]) != int(source_scan_index):
            return None
        return position

    def _residual_slice(self, start: int, end: int):
        values = np.asarray(self.store.intensity[start:end], dtype=np.float64).copy()
        for local, point_index in enumerate(range(start, end)):
            claimed = self._claimed.get(point_index)
            if claimed:
                values[local] = max(0.0, values[local] - claimed)
        return values

    def scan(self, local_index: int):
        start = int(self.store.offsets[local_index])
        end = int(self.store.offsets[local_index + 1])
        return self.store.mz[start:end], self._residual_slice(start, end)

    def extract_trace(
        self,
        target_mz: float,
        ppm: float,
        rt_start_sec: float,
        rt_end_sec: float,
        *,
        faims_cv=None,
    ) -> ExtractedTrace:
        """Extract an XIC from only the currently unallocated intensity."""

        if not math.isfinite(target_mz) or target_mz <= 0:
            raise ValueError("target_mz must be finite and positive")
        if not math.isfinite(ppm) or ppm <= 0:
            raise ValueError("ppm must be finite and positive")
        if rt_end_sec < rt_start_sec:
            raise ValueError("RT end must not precede RT start")
        local_indices = self.store.select_local_indices(
            rt_start_sec, rt_end_sec, faims_cv=faims_cv
        )
        intensities = np.zeros(local_indices.size, dtype=np.float64)
        observed = np.full(local_indices.size, np.nan, dtype=np.float64)
        tolerance = float(target_mz) * float(ppm) * 1e-6
        lower = float(target_mz) - tolerance
        upper = float(target_mz) + tolerance
        for output_index, local_index in enumerate(local_indices):
            mz_values, intensity_values = self.scan(int(local_index))
            start = int(np.searchsorted(mz_values, lower, side="left"))
            end = int(np.searchsorted(mz_values, upper, side="right"))
            if end <= start:
                continue
            local_intensity = np.asarray(
                intensity_values[start:end], dtype=np.float64
            )
            positive = local_intensity > self.tolerance
            if not np.any(positive):
                continue
            local_mz = np.asarray(
                mz_values[start:end], dtype=np.float64
            )[positive]
            local_intensity = local_intensity[positive]
            total = float(np.sum(local_intensity, dtype=np.float64))
            intensities[output_index] = total
            observed[output_index] = float(
                np.average(local_mz, weights=local_intensity)
            )
        return ExtractedTrace(
            scan_index=self.store.source_scan_index[local_indices].copy(),
            scan_number=self.store.scan_number[local_indices].copy(),
            rt_sec=self.store.rt_sec[local_indices].copy(),
            intensity=intensities,
            observed_mz=observed,
            point_present=intensities > self.tolerance,
            target_mz=float(target_mz),
            ppm=float(ppm),
        )

    def extract_traces(
        self, target_mzs, ppm, rt_start_sec, rt_end_sec, *, faims_cv=None
    ):
        """Compatibility batch API; sparse claims currently require per-trace scans."""

        return tuple(
            self.extract_trace(
                target_mz, ppm, rt_start_sec, rt_end_sec, faims_cv=faims_cv
            )
            for target_mz in target_mzs
        )

    def _trace_point_distribution(
        self,
        trace: ExtractedTrace,
        trace_position: int,
        requested_intensity: float,
    ):
        local_scan = self._local_scan_index(
            int(trace.scan_index[trace_position])
        )
        if local_scan is None:
            return None
        scan_start = int(self.store.offsets[local_scan])
        scan_end = int(self.store.offsets[local_scan + 1])
        mz_values = self.store.mz[scan_start:scan_end]
        tolerance = float(trace.target_mz) * float(trace.ppm) * 1e-6
        start = int(
            np.searchsorted(
                mz_values, float(trace.target_mz) - tolerance, side="left"
            )
        )
        end = int(
            np.searchsorted(
                mz_values, float(trace.target_mz) + tolerance, side="right"
            )
        )
        if end <= start:
            return None
        point_indices = np.arange(
            scan_start + start, scan_start + end, dtype=np.int64
        )
        residual = np.asarray(
            [
                max(
                    0.0,
                    float(self.store.intensity[index])
                    - self._claimed.get(int(index), 0.0),
                )
                for index in point_indices
            ],
            dtype=np.float64,
        )
        positive = residual > self.tolerance
        if not np.any(positive):
            return None
        point_indices = point_indices[positive]
        residual = residual[positive]
        available = float(np.sum(residual, dtype=np.float64))
        if requested_intensity > available + max(
            self.tolerance, self.tolerance * available
        ):
            return None
        requested = min(float(requested_intensity), available)
        fractions = residual / available
        allocated = requested * fractions
        # Make the sum exact at float64 scale without violating nonnegativity.
        allocated[-1] += requested - float(np.sum(allocated, dtype=np.float64))
        return tuple(
            RawPointAllocation(int(index), float(value))
            for index, value in zip(point_indices, allocated)
            if value > self.tolerance
        )

    def _commit_point_allocations(self, allocation_id, requested_total, by_point):
        accepted = tuple(
            RawPointAllocation(point_index, amount)
            for point_index, amount in sorted(by_point.items())
            if amount > self.tolerance
        )
        allocated_total = float(sum(value.intensity for value in accepted))
        conservation_error = abs(float(requested_total) - allocated_total)
        allowed_error = max(
            self.tolerance,
            self.tolerance * max(float(requested_total), 1.0),
        )
        if conservation_error > allowed_error:
            return AllocationResult(
                allocation_id,
                "conservation_failed",
                float(requested_total),
                allocated_total,
                len(accepted),
                conservation_error,
            )
        for point_index, amount in by_point.items():
            available = max(
                0.0,
                float(self.store.intensity[point_index])
                - self._claimed.get(point_index, 0.0),
            )
            if amount > available + max(
                self.tolerance, self.tolerance * available
            ):
                return AllocationResult(
                    allocation_id,
                    "raw_point_overallocation",
                    float(requested_total),
                    0.0,
                    0,
                    float(requested_total),
                )
        for value in accepted:
            self._claimed[value.point_index] = (
                self._claimed.get(value.point_index, 0.0) + value.intensity
            )
        self._allocations[allocation_id] = accepted
        return AllocationResult(
            allocation_id,
            "accepted",
            float(requested_total),
            allocated_total,
            len(accepted),
            conservation_error,
        )

    def allocate_observed_points(
        self,
        allocation_id: Hashable,
        contributions,
        *,
        mz_tolerance_ppm: float = 0.01,
    ) -> AllocationResult:
        """Atomically claim exact observed centroids from accepted detector hills.

        Each contribution is ``(source_scan_index, observed_mz, intensity)``.
        The very small m/z tolerance only accommodates serialization roundoff;
        it is not a feature matching tolerance.
        """

        if allocation_id in self._allocations:
            raise ValueError("allocation ID already exists: %r" % (allocation_id,))
        if not math.isfinite(mz_tolerance_ppm) or mz_tolerance_ppm < 0:
            raise ValueError("m/z tolerance must be finite and nonnegative")
        normalized = tuple(
            (int(scan), float(mz), float(intensity))
            for scan, mz, intensity in contributions
        )
        if any(
            not math.isfinite(mz)
            or mz <= 0
            or not math.isfinite(intensity)
            or intensity < 0
            for _scan, mz, intensity in normalized
        ):
            raise ValueError(
                "observed-point contributions must have finite positive m/z "
                "and finite nonnegative intensity"
            )
        requested_total = float(
            sum(intensity for _scan, _mz, intensity in normalized)
        )
        by_point = {}
        for source_scan, observed_mz, requested in normalized:
            if requested <= self.tolerance:
                continue
            local_scan = self._local_scan_index(source_scan)
            if local_scan is None:
                return AllocationResult(
                    allocation_id,
                    "source_scan_not_found",
                    requested_total,
                    0.0,
                    0,
                    requested_total,
                )
            scan_start = int(self.store.offsets[local_scan])
            scan_end = int(self.store.offsets[local_scan + 1])
            mz_values = self.store.mz[scan_start:scan_end]
            tolerance = max(
                np.finfo(np.float64).eps * max(abs(observed_mz), 1.0) * 8.0,
                observed_mz * float(mz_tolerance_ppm) * 1e-6,
            )
            start = int(
                np.searchsorted(mz_values, observed_mz - tolerance, side="left")
            )
            end = int(
                np.searchsorted(mz_values, observed_mz + tolerance, side="right")
            )
            candidates = []
            for point_index in range(scan_start + start, scan_start + end):
                already_proposed = by_point.get(point_index, 0.0)
                available = max(
                    0.0,
                    float(self.store.intensity[point_index])
                    - self._claimed.get(point_index, 0.0)
                    - already_proposed,
                )
                if requested <= available + max(
                    self.tolerance, self.tolerance * available
                ):
                    candidates.append(
                        (
                            abs(float(self.store.mz[point_index]) - observed_mz),
                            abs(available - requested),
                            point_index,
                        )
                    )
            if not candidates:
                return AllocationResult(
                    allocation_id,
                    "observed_point_not_available",
                    requested_total,
                    0.0,
                    0,
                    requested_total,
                )
            point_index = min(candidates)[2]
            by_point[point_index] = by_point.get(point_index, 0.0) + requested
        return self._commit_point_allocations(
            allocation_id, requested_total, by_point
        )

    def commit_observed_point_footprint(
        self,
        allocation_id: Hashable,
        footprint: ComponentFootprint,
    ) -> AllocationResult:
        """Commit a previously mapped observed-point footprint atomically.

        Callers may build footprints against the immutable raw store in worker
        processes, but commits remain ordered in this ledger's owner process.
        """

        if allocation_id in self._allocations:
            raise ValueError("allocation ID already exists: %r" % (allocation_id,))
        if footprint.status != "accepted":
            return AllocationResult(
                allocation_id,
                footprint.status,
                float(footprint.requested_intensity),
                0.0,
                0,
                float(footprint.requested_intensity),
            )
        by_point = {}
        for value in footprint.allocations:
            point_index = int(value.point_index)
            amount = float(value.intensity)
            if (
                point_index < 0
                or point_index >= self.store.intensity.size
                or not math.isfinite(amount)
                or amount < 0
            ):
                raise ValueError("observed footprint contains an invalid allocation")
            by_point[point_index] = by_point.get(point_index, 0.0) + amount
        return self._commit_point_allocations(
            allocation_id, float(footprint.requested_intensity), by_point
        )

    def allocate_component(
        self,
        allocation_id: Hashable,
        traces: Sequence[ExtractedTrace],
        segment_start: int,
        allocated_trace_values,
    ) -> AllocationResult:
        """Atomically claim one RT/intensity-split feature component."""

        if allocation_id in self._allocations:
            raise ValueError("allocation ID already exists: %r" % (allocation_id,))
        matrix = np.asarray(allocated_trace_values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(traces):
            raise ValueError("allocated component must be channels by scans")
        if np.any(matrix < 0) or not np.all(np.isfinite(matrix)):
            raise ValueError("allocated component intensity must be finite and nonnegative")
        if any(
            segment_start < 0
            or segment_start + matrix.shape[1] > trace.intensity.size
            for trace in traces
        ):
            raise ValueError("allocated component falls outside an extracted trace")

        requested_total = float(np.sum(matrix, dtype=np.float64))
        proposed = []
        for channel, trace in enumerate(traces):
            for local_position in np.flatnonzero(matrix[channel] > self.tolerance):
                request = float(matrix[channel, local_position])
                distribution = self._trace_point_distribution(
                    trace,
                    segment_start + int(local_position),
                    request,
                )
                if distribution is None:
                    return AllocationResult(
                        allocation_id,
                        "insufficient_residual_intensity",
                        requested_total,
                        0.0,
                        0,
                        requested_total,
                    )
                proposed.extend(distribution)

        by_point = {}
        for allocation in proposed:
            by_point[allocation.point_index] = (
                by_point.get(allocation.point_index, 0.0)
                + allocation.intensity
            )
        for point_index, amount in by_point.items():
            available = max(
                0.0,
                float(self.store.intensity[point_index])
                - self._claimed.get(point_index, 0.0),
            )
            if amount > available + max(
                self.tolerance, self.tolerance * available
            ):
                return AllocationResult(
                    allocation_id,
                    "overlapping_trace_overallocation",
                    requested_total,
                    0.0,
                    0,
                    requested_total,
                )

        return self._commit_point_allocations(
            allocation_id, requested_total, by_point
        )

    def revert(self, allocation_id: Hashable):
        """Remove one accepted allocation and restore its residual intensity."""

        allocations = self._allocations.pop(allocation_id)
        for value in allocations:
            remaining = self._claimed[value.point_index] - value.intensity
            if remaining <= self.tolerance:
                self._claimed.pop(value.point_index, None)
            else:
                self._claimed[value.point_index] = remaining

    def materialize(self):
        """Return a RawMS1Store with the current residual intensity."""

        residual = np.asarray(self.store.intensity, dtype=np.float64).copy()
        if self._claimed:
            point_indices = np.fromiter(
                self._claimed, dtype=np.intp, count=len(self._claimed)
            )
            claimed = np.fromiter(
                self._claimed.values(),
                dtype=np.float64,
                count=len(self._claimed),
            )
            residual[point_indices] = np.maximum(
                0.0, residual[point_indices] - claimed
            )
        return RawMS1Store(
            offsets=self.store.offsets,
            mz=self.store.mz,
            intensity=residual,
            source_scan_index=self.store.source_scan_index,
            scan_number=self.store.scan_number,
            rt_sec=self.store.rt_sec,
            faims_cv=self.store.faims_cv,
        )
