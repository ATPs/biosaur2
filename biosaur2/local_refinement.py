"""Bounded reversible trace repair and joint isotope-component segmentation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .cutils import local_segment_objective_values
from .optimization import nonnegative_deconvolution


@dataclass(frozen=True)
class SegmentEdit:
    action: str
    before: tuple[tuple[int, int], ...]
    after: tuple[tuple[int, int], ...]
    objective_before: float
    objective_after: float
    accepted: bool
    reason: str

    @property
    def objective_delta(self):
        return self.objective_after - self.objective_before

    def revert(self):
        return self.before


@dataclass(frozen=True)
class JointComponent:
    start: int
    end: int
    apex: int
    allocated_matrix: np.ndarray
    score: float
    bic: float
    source: str
    allocation_group: str
    allocation_index: int
    deconvolution_status: str
    condition_number: Optional[float]
    intensity_conserved: bool


@dataclass(frozen=True)
class LocalRefinementResult:
    components: tuple[JointComponent, ...]
    edits: tuple[SegmentEdit, ...]
    initial_segments: tuple[tuple[int, int], ...]
    repaired_segments: tuple[tuple[int, int], ...]
    proposal_count: int
    accepted_edit_count: int


def _contiguous_segments(present):
    positions = np.flatnonzero(present)
    if positions.size == 0:
        return []
    result = []
    start = previous = int(positions[0])
    for position in positions[1:]:
        position = int(position)
        if position != previous + 1:
            result.append((start, previous + 1))
            start = position
        previous = position
    result.append((start, previous + 1))
    return result


def _segment_objective(matrix, segments, theoretical):
    if not segments:
        return 0.0
    return float(local_segment_objective_values(
        np.ascontiguousarray(matrix, dtype=np.float64),
        np.asarray(tuple(segments), dtype=np.int64),
        np.ascontiguousarray(theoretical, dtype=np.float64),
    ))


def _append_edit(
    edits,
    action,
    before,
    after,
    matrix,
    theoretical,
    reason,
    *,
    evidence_gain=0.0,
):
    before = tuple(before)
    after = tuple(after)
    before_value = _segment_objective(matrix, before, theoretical)
    after_value = _segment_objective(matrix, after, theoretical) + float(
        evidence_gain
    )
    # Structural edits are already gated by explicit evidence below.  Require
    # monotonic objective within floating precision and store rejected edits.
    accepted = after_value + 1e-12 >= before_value
    edit = SegmentEdit(
        action,
        before,
        after if accepted else before,
        before_value,
        after_value if accepted else before_value,
        accepted,
        reason,
    )
    edits.append(edit)
    return list(edit.after)


def repair_local_trace_segments(
    matrix,
    theoretical,
    *,
    max_gap_scans: int = 2,
    max_edits: int = 32,
    split_valley_ratio: float = 0.25,
    min_split_points: int = 3,
):
    """Propose bounded local edits without inventing raw intensity points."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("local trace matrix must be channels by scans")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("local trace intensities must be finite and nonnegative")
    theoretical = np.asarray(theoretical, dtype=np.float64)
    if theoretical.shape != (values.shape[0],):
        raise ValueError("theoretical isotope vector does not match channels")

    combined = np.sum(values, axis=0, dtype=np.float64)
    mono_segments = _contiguous_segments(values[0] > 0)
    combined_segments = _contiguous_segments(combined > 0)
    initial = tuple(mono_segments)
    segments = list(initial)
    edits = []

    # Raw coherent points not represented by a mono fragment are explicit
    # new-trace proposals rather than silently appearing in a segment.
    if not mono_segments:
        for segment in combined_segments[:max_edits]:
            before = tuple(segments)
            after = sorted(set(segments + [segment]))
            segments = _append_edit(
                edits,
                "new_trace",
                before,
                after,
                values,
                theoretical,
                "coherent exact-isotope raw points without a mono fragment",
            )
    else:
        for combined_segment in combined_segments:
            if len(edits) >= max_edits:
                break
            overlapping = [
                segment
                for segment in segments
                if max(segment[0], combined_segment[0])
                < min(segment[1], combined_segment[1])
            ]
            if not overlapping:
                before = tuple(segments)
                after = sorted(set(segments + [combined_segment]))
                segments = _append_edit(
                    edits,
                    "new_trace",
                    before,
                    after,
                    values,
                    theoretical,
                    "additional coherent exact-isotope raw component",
                )
                continue
            for segment in overlapping:
                expanded = (
                    min(segment[0], combined_segment[0]),
                    max(segment[1], combined_segment[1]),
                )
                if expanded == segment:
                    continue
                before = tuple(segments)
                after = [expanded if item == segment else item for item in segments]
                segments = _append_edit(
                    edits,
                    "extend",
                    before,
                    sorted(set(after)),
                    values,
                    theoretical,
                    "adjacent exact-isotope support extends a mono boundary",
                )

    # Merge/relink only short gaps with support in at least two channels on
    # both sides and tolerable intensity discontinuity. No zero is imputed.
    changed = True
    while changed and len(edits) < max_edits:
        changed = False
        segments = sorted(segments)
        for index in range(len(segments) - 1):
            left, right = segments[index : index + 2]
            gap = right[0] - left[1]
            if gap < 0 or gap > max_gap_scans:
                continue
            left_channels = values[:, left[0] : left[1]].sum(axis=1) > 0
            right_channels = values[:, right[0] : right[1]].sum(axis=1) > 0
            if np.count_nonzero(left_channels & right_channels) < 2:
                continue
            left_edge = combined[left[1] - 1]
            right_edge = combined[right[0]]
            if min(left_edge, right_edge) <= 0:
                continue
            if max(left_edge, right_edge) / min(left_edge, right_edge) > 10.0:
                continue
            merged = (left[0], right[1])
            before = tuple(segments)
            after = segments[:index] + [merged] + segments[index + 2 :]
            action = "merge" if gap <= 1 else "relink"
            updated = _append_edit(
                edits,
                action,
                before,
                after,
                values,
                theoretical,
                "short gap with mass-fixed multi-isotope and intensity continuity",
                evidence_gain=4.0 if gap <= 1 else 3.5,
            )
            if updated != segments:
                segments = updated
                changed = True
                break

    # Split clearly bimodal joint envelopes at a disjoint valley boundary.
    for segment in tuple(sorted(segments)):
        if len(edits) >= max_edits:
            break
        start, end = segment
        local = combined[start:end]
        if local.size < 2 * min_split_points + 1:
            continue
        apexes = [
            position
            for position in range(1, local.size - 1)
            if local[position] > local[position - 1]
            and local[position] >= local[position + 1]
        ]
        accepted_boundary = None
        for left, right in zip(apexes, apexes[1:]):
            if right - left < min_split_points:
                continue
            valley = left + int(np.argmin(local[left : right + 1]))
            if local[valley] > split_valley_ratio * min(
                local[left], local[right]
            ):
                continue
            boundary = start + valley + 1
            if (
                np.count_nonzero(combined[start:boundary]) >= min_split_points
                and np.count_nonzero(combined[boundary:end]) >= min_split_points
            ):
                accepted_boundary = boundary
                break
        if accepted_boundary is None:
            continue
        before = tuple(segments)
        after = []
        for item in segments:
            if item == segment:
                after.extend(
                    ((start, accepted_boundary), (accepted_boundary, end))
                )
            else:
                after.append(item)
        segments = _append_edit(
            edits,
            "split",
            before,
            sorted(after),
            values,
            theoretical,
            "two joint-isotope apexes separated by a deep valley",
            evidence_gain=4.0,
        )

    return tuple(sorted(set(segments))), tuple(edits), initial


def _gaussian_basis(length, center, sigma):
    positions = np.arange(length, dtype=np.float64)
    values = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
    total = float(np.sum(values))
    return values / total if total else values


def _bic(observed, modeled, parameter_count):
    residual = np.asarray(observed) - np.asarray(modeled)
    n = max(1, residual.size)
    mse = float(np.dot(residual, residual) / n)
    scale = max(float(np.max(observed)) ** 2, 1.0)
    return float(n * math.log(mse / scale + 1e-12) + parameter_count * math.log(n))


def _local_apexes(envelope):
    smoothed = np.convolve(
        np.asarray(envelope, dtype=np.float64),
        np.asarray([0.25, 0.5, 0.25]),
        mode="same",
    )
    apexes = [
        position
        for position in range(1, len(smoothed) - 1)
        if smoothed[position] > smoothed[position - 1]
        and smoothed[position] >= smoothed[position + 1]
    ]
    return smoothed, apexes


def _joint_components_for_segment(
    matrix,
    start,
    end,
    *,
    bic_improvement_min,
    condition_max,
    min_component_fraction,
):
    local_matrix = np.asarray(matrix[:, start:end], dtype=np.float64)
    envelope = np.sum(local_matrix, axis=0, dtype=np.float64)
    length = envelope.size
    apex = int(np.argmax(envelope))
    group_prefix = "%d:%d" % (start, end)
    if length < 7 or np.count_nonzero(envelope) < 5:
        return (
            JointComponent(
                start,
                end,
                start + apex,
                local_matrix.copy(),
                float(np.sum(envelope)),
                _bic(envelope, envelope, 1),
                "single_component",
                group_prefix + ":single",
                0,
                "not_attempted",
                None,
                True,
            ),
        )

    positions = np.arange(length, dtype=np.float64)
    total = float(np.sum(envelope))
    weighted_variance = float(
        np.sum(envelope * (positions - apex) ** 2) / max(total, 1e-12)
    )
    sigma_single = min(max(math.sqrt(weighted_variance), 1.0), length / 2.0)
    single_design = _gaussian_basis(length, apex, sigma_single)[:, None]
    single = nonnegative_deconvolution(
        single_design, envelope, condition_max=condition_max,
        conservation_tolerance=0.05,
    )
    single_bic = _bic(envelope, single.modeled, 3)

    _smoothed, apexes = _local_apexes(envelope)
    ordered = sorted(apexes, key=lambda value: (-envelope[value], value))
    pair = None
    for first in ordered:
        for second in ordered:
            if second <= first or second - first < 3:
                continue
            candidate = (first, second)
            if pair is None or (
                -(envelope[first] + envelope[second]), candidate
            ) < (-(envelope[pair[0]] + envelope[pair[1]]), pair):
                pair = candidate
    if pair is None:
        return (
            JointComponent(
                start,
                end,
                start + apex,
                local_matrix.copy(),
                float(np.sum(envelope)),
                single_bic,
                "single_component",
                group_prefix + ":single",
                0,
                "not_identifiable",
                single.condition_number,
                True,
            ),
        )

    distance = pair[1] - pair[0]
    sigma = max(1.0, min(float(distance) / 3.0, sigma_single))
    design = np.column_stack(
        [_gaussian_basis(length, center, sigma) for center in pair]
    )
    deconvolution = nonnegative_deconvolution(
        design,
        envelope,
        condition_max=condition_max,
        conservation_tolerance=0.05,
    )
    two_bic = _bic(envelope, deconvolution.modeled, 6)
    coefficient_total = float(np.sum(deconvolution.coefficients))
    sufficiently_large = (
        coefficient_total > 0
        and np.all(
            deconvolution.coefficients
            >= min_component_fraction * coefficient_total
        )
    )
    if (
        deconvolution.status != "accepted"
        or not sufficiently_large
        or single_bic - two_bic < bic_improvement_min
    ):
        return (
            JointComponent(
                start,
                end,
                start + apex,
                local_matrix.copy(),
                float(np.sum(envelope)),
                single_bic,
                "single_component",
                group_prefix + ":single",
                0,
                deconvolution.status,
                deconvolution.condition_number,
                True,
            ),
        )

    modeled_components = design * deconvolution.coefficients[None, :]
    modeled_total = np.sum(modeled_components, axis=1, dtype=np.float64)
    fractions = np.zeros_like(modeled_components)
    positive = modeled_total > 0
    fractions[positive] = (
        modeled_components[positive] / modeled_total[positive, None]
    )
    for position in np.flatnonzero(~positive):
        closest = min(range(2), key=lambda index: (abs(position - pair[index]), index))
        fractions[position, closest] = 1.0
    allocations = local_matrix[:, :, None] * fractions[None, :, :]
    conserved = bool(
        np.allclose(
            np.sum(allocations, axis=2),
            local_matrix,
            rtol=1e-12,
            atol=1e-12,
        )
    )
    if not conserved:
        return (
            JointComponent(
                start,
                end,
                start + apex,
                local_matrix.copy(),
                float(np.sum(envelope)),
                single_bic,
                "single_component",
                group_prefix + ":single",
                0,
                "intensity_conservation_failed",
                deconvolution.condition_number,
                False,
            ),
        )

    group = group_prefix + ":%d,%d" % pair
    result = []
    for index, center in enumerate(pair):
        allocated = allocations[:, :, index]
        allocated_envelope = np.sum(allocated, axis=0, dtype=np.float64)
        local_apex = int(np.argmax(allocated_envelope))
        result.append(
            JointComponent(
                start,
                end,
                start + local_apex,
                allocated.copy(),
                float(np.sum(allocated)),
                two_bic,
                "identifiable_nnls",
                group,
                index,
                "accepted",
                deconvolution.condition_number,
                True,
            )
        )
    return tuple(result)


def refine_local_isotope_components(
    matrix,
    theoretical,
    *,
    bic_improvement_min: float = 6.0,
    condition_max: float = 1e4,
    min_component_fraction: float = 0.05,
    max_components: int = 2,
    max_edits: int = 32,
    max_gap_scans: int = 2,
):
    """Repair trace segments and fit at most two shared components per region."""

    if max_components not in {1, 2}:
        raise ValueError("max_components must be 1 or 2")
    matrix = np.asarray(matrix, dtype=np.float64)
    theoretical = np.asarray(theoretical, dtype=np.float64)
    segments, edits, initial = repair_local_trace_segments(
        matrix,
        theoretical,
        max_edits=max_edits,
        max_gap_scans=max_gap_scans,
    )
    components = []
    for start, end in segments:
        if max_components == 1:
            local = matrix[:, start:end].copy()
            envelope = np.sum(local, axis=0, dtype=np.float64)
            components.append(
                JointComponent(
                    start,
                    end,
                    start + int(np.argmax(envelope)),
                    local,
                    float(np.sum(envelope)),
                    _bic(envelope, envelope, 1),
                    "single_component",
                    "%d:%d:single" % (start, end),
                    0,
                    "disabled",
                    None,
                    True,
                )
            )
        else:
            components.extend(
                _joint_components_for_segment(
                    matrix,
                    start,
                    end,
                    bic_improvement_min=bic_improvement_min,
                    condition_max=condition_max,
                    min_component_fraction=min_component_fraction,
                )
            )
    return LocalRefinementResult(
        tuple(components),
        edits,
        initial,
        segments,
        len(edits),
        sum(edit.accepted for edit in edits),
    )
