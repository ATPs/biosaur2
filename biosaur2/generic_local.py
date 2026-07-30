"""Bounded raw-envelope candidates for generic unidentified MS2 events."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence

import numpy as np

from .chemistry import PROTON_MASS
from .confidence import TargetDecoyCompetition, target_decoy_q_values
from .local_refinement import SegmentEdit, refine_local_isotope_components
from .generic_association import C13_C12_MASS_DIFF
from .parallel import balanced_ranges, run_process_tasks
from .raw_ms1 import ExtractedTrace, RawMS1Store, event_position_in_trace


GENERIC_LOCAL_ISOTOPE_ERRORS = (0, 1, 2, 3)
GENERIC_LOCAL_ISOTOPE_COUNT = 5


@dataclass(frozen=True)
class GenericLocalCandidate:
    event: Mapping
    status: str
    score: Optional[float]
    isotope_error: Optional[int] = None
    mono_mz: Optional[float] = None
    neutral_mass: Optional[float] = None
    rt_start_sec: Optional[float] = None
    rt_apex_sec: Optional[float] = None
    rt_end_sec: Optional[float] = None
    scan_apex: Optional[int] = None
    width_sec: Optional[float] = None
    mono_points: int = 0
    point_count: int = 0
    supported_channels: int = 0
    isotope_cosine: Optional[float] = None
    isotope_apex_spread_sec: Optional[float] = None
    selected_event_mz_error_ppm: Optional[float] = None
    traces: tuple[ExtractedTrace, ...] = ()
    segment_slice: Optional[tuple[int, int]] = None
    relaxed: bool = False
    selected_event_apex_ratio: Optional[float] = None
    score_components: tuple[tuple[str, float], ...] = ()
    boundary_truncated: bool = False
    allocated_trace_values: Optional[tuple[np.ndarray, ...]] = None
    edit_history: tuple[SegmentEdit, ...] = ()
    component_count: int = 0
    allocation_group_key: Optional[str] = None
    allocation_component_index: Optional[int] = None
    deconvolution_status: Optional[str] = None
    intensity_conserved: bool = True

    @property
    def quantitative_candidate(self):
        return self.status == "candidate" and self.score is not None


@dataclass(frozen=True)
class GenericLocalCompetition:
    event_id: int
    target: GenericLocalCandidate
    decoy: GenericLocalCandidate
    winner: str
    q_value: float


def generic_local_width_limit(strict_quant_rows: Sequence[Mapping]) -> float:
    widths = np.asarray(
        [
            float(row["rt_end_sec"]) - float(row["rt_start_sec"])
            for row in strict_quant_rows
            if row.get("rt_start_sec") is not None
            and row.get("rt_end_sec") is not None
            and float(row["rt_end_sec"]) >= float(row["rt_start_sec"])
        ],
        dtype=np.float64,
    )
    if widths.size == 0:
        return 30.0
    return float(min(60.0, max(15.0, np.quantile(widths, 0.99))))


def _positive_segments(present, max_gap=1):
    positions = np.flatnonzero(present)
    if positions.size == 0:
        return []
    result = []
    start = previous = int(positions[0])
    for value in positions[1:]:
        value = int(value)
        if value - previous - 1 > max_gap:
            result.append((start, previous + 1))
            start = value
        previous = value
    result.append((start, previous + 1))
    return result


def _averagine_probabilities(mono_mz, charge, count):
    neutral_mass = max(0.0, (float(mono_mz) - PROTON_MASS) * int(charge))
    carbon_count = max(1, round(neutral_mass / 111.1254 * 4.9384))
    probability = 0.0107
    values = np.asarray(
        [
            math.comb(carbon_count, index)
            * probability**index
            * (1.0 - probability) ** (carbon_count - index)
            for index in range(count)
        ],
        dtype=np.float64,
    )
    total = float(np.sum(values))
    return values / total if total else values


def evaluate_generic_local_candidate(
    store: RawMS1Store,
    event: Mapping,
    *,
    width_limit_sec: float,
    ppm: float = 10.0,
    rt_tolerance_sec: float = 120.0,
    isotope_count: int = GENERIC_LOCAL_ISOTOPE_COUNT,
    min_mono_points: int = 3,
    min_channel_points: int = 3,
    min_supported_channels: int = 2,
    min_cosine: float = 0.90,
    relaxed: bool = False,
) -> GenericLocalCandidate:
    """Evaluate one natural-envelope generic hypothesis without forcing it."""

    selected = event.get("selected_ion_mz")
    charge = event.get("charge")
    rt = event.get("rt_sec")
    if selected is None or not math.isfinite(float(selected)) or float(selected) <= 0:
        return GenericLocalCandidate(event, "invalid_selected_mz", None)
    if charge is None or int(charge) <= 0:
        return GenericLocalCandidate(event, "invalid_charge", None)
    if rt is None or not math.isfinite(float(rt)):
        return GenericLocalCandidate(event, "invalid_rt", None)
    selected = float(selected)
    charge = int(charge)
    rt = float(rt)
    target = event.get("isolation_target_mz")
    lower = event.get("isolation_lower_offset")
    upper = event.get("isolation_upper_offset")
    if target is not None and lower is not None and upper is not None:
        if not float(target) - float(lower) <= selected <= float(target) + float(upper):
            return GenericLocalCandidate(
                event, "selected_outside_isolation_window", None
            )

    relative_traces = {}
    for relative in range(-3, isotope_count):
        mz = selected + relative * C13_C12_MASS_DIFF / charge
        relative_traces[relative] = store.extract_trace(
            mz,
            ppm,
            rt - rt_tolerance_sec,
            rt + rt_tolerance_sec,
            faims_cv=event.get("faims_cv"),
        )
    reference = relative_traces[0]
    if reference.rt_sec.size == 0:
        return GenericLocalCandidate(event, "no_ms1_scans_in_window", None)
    event_position = event_position_in_trace(
        reference, rt, event.get("precursor_ms1_index")
    )
    best = None
    failures = Counter()
    for isotope_error in GENERIC_LOCAL_ISOTOPE_ERRORS:
        relatives = tuple(range(-isotope_error, isotope_count - isotope_error))
        traces = tuple(relative_traces[value] for value in relatives)
        matrix = np.stack([trace.intensity for trace in traces])
        mono_mz = selected - isotope_error * C13_C12_MASS_DIFF / charge
        theoretical = _averagine_probabilities(mono_mz, charge, isotope_count)
        refinement = refine_local_isotope_components(
            matrix,
            theoretical,
            max_gap_scans=2,
        )
        containing = [
            component
            for component in refinement.components
            if component.start <= event_position < component.end
        ]
        if not containing:
            failures["no_component_at_event_scan"] += 1
            continue
        selected_trace = traces[isotope_error]
        if not bool(selected_trace.point_present[event_position]):
            failures["selected_isotope_absent_at_event_scan"] += 1
            continue
        selected_component_found = False
        for component in containing:
            start, end = component.start, component.end
            allocated = np.asarray(component.allocated_matrix, dtype=np.float64)
            event_local = event_position - start
            if allocated[isotope_error, event_local] <= 0:
                continue
            selected_component_found = True
            channel_counts = [
                int(np.count_nonzero(values > 0)) for values in allocated
            ]
            mono_points = channel_counts[0]
            if mono_points < min_mono_points:
                failures["insufficient_mono_points"] += 1
                continue
            supported_channels = sum(
                count >= min_channel_points for count in channel_counts
            )
            if supported_channels < min_supported_channels:
                failures["insufficient_isotope_channel_support"] += 1
                continue
            segment_rt = reference.rt_sec[start:end]
            width = float(segment_rt[-1] - segment_rt[0])
            if width > width_limit_sec:
                failures["component_too_wide"] += 1
                continue
            integrated = np.asarray(
                [np.trapezoid(values, segment_rt) for values in allocated],
                dtype=np.float64,
            )
            denominator = float(
                np.linalg.norm(integrated) * np.linalg.norm(theoretical)
            )
            cosine = None if denominator == 0 else float(
                np.dot(integrated, theoretical) / denominator
            )
            if cosine is None or cosine < min_cosine:
                failures["low_averagine_cosine"] += 1
                continue
            apex_rts = []
            for values, count in zip(allocated, channel_counts):
                if count >= min_channel_points:
                    apex_rts.append(float(segment_rt[int(np.argmax(values))]))
            apex_spread = max(apex_rts) - min(apex_rts)
            allowed_apex_spread = max(3.0, min(10.0, width * 0.5))
            if apex_spread > allowed_apex_spread:
                failures["isotope_apex_spread"] += 1
                continue
            observed_mz = float(selected_trace.observed_mz[event_position])
            mass_error = (observed_mz - selected) * 1e6 / selected
            envelope = np.sum(allocated, axis=0, dtype=np.float64)
            apex_local = int(np.argmax(envelope))
            mass_support = max(0.0, 1.0 - abs(mass_error) / ppm)
            cosine_denominator = max(1e-12, 1.0 - float(min_cosine))
            cosine_support = min(
                1.0,
                max(
                    0.0,
                    (float(cosine) - float(min_cosine)) / cosine_denominator,
                ),
            )
            event_apex_ratio = min(
                1.0,
                max(
                    0.0,
                    float(envelope[event_local])
                    / max(float(envelope[apex_local]), 1e-12),
                ),
            )
            coelution_support = max(
                0.0, 1.0 - apex_spread / max(allowed_apex_spread, 1e-12)
            )
            point_support = min(
                1.0, mono_points / max(float(min_mono_points + 2), 1.0)
            )
            channel_support = min(
                1.0,
                supported_channels
                / max(float(min_supported_channels + 1), 1.0),
            )
            score_components = (
                ("mass_support", mass_support),
                ("isotope_cosine_support", cosine_support),
                ("event_apex_support", event_apex_ratio),
                ("coelution_support", coelution_support),
                ("point_support", point_support),
                ("channel_support", channel_support),
            )
            weights = (0.30, 0.25, 0.20, 0.10, 0.10, 0.05)
            score = float(
                sum(
                    weight * value
                    for weight, (_name, value) in zip(weights, score_components)
                )
            )
            boundary_truncated = bool(
                (start == 0 and envelope.size and envelope[0] > 0)
                or (
                    end == reference.rt_sec.size
                    and envelope.size
                    and envelope[-1] > 0
                )
            )
            allocation_group_key = None
            if component.source == "identifiable_nnls":
                related = [
                    value
                    for value in refinement.components
                    if value.allocation_group == component.allocation_group
                ]
                apex_scans = sorted(
                    int(reference.scan_index[value.apex]) for value in related
                )
                allocation_group_key = "%d:%.8f:%d:%d:%s" % (
                    charge,
                    mono_mz,
                    int(reference.scan_index[start]),
                    int(reference.scan_index[end - 1]),
                    ",".join(str(value) for value in apex_scans),
                )
            candidate = GenericLocalCandidate(
                event=event,
                status="candidate",
                score=score,
                isotope_error=isotope_error,
                mono_mz=mono_mz,
                neutral_mass=(mono_mz - PROTON_MASS) * charge,
                rt_start_sec=float(segment_rt[0]),
                rt_apex_sec=float(segment_rt[apex_local]),
                rt_end_sec=float(segment_rt[-1]),
                scan_apex=int(reference.scan_number[start + apex_local]),
                width_sec=width,
                mono_points=mono_points,
                point_count=int(np.count_nonzero(envelope > 0)),
                supported_channels=supported_channels,
                isotope_cosine=cosine,
                isotope_apex_spread_sec=apex_spread,
                selected_event_mz_error_ppm=mass_error,
                traces=traces,
                segment_slice=(start, end),
                relaxed=bool(relaxed),
                selected_event_apex_ratio=event_apex_ratio,
                score_components=score_components,
                boundary_truncated=boundary_truncated,
                allocated_trace_values=tuple(
                    np.asarray(values, dtype=np.float64) for values in allocated
                ),
                edit_history=refinement.edits,
                component_count=len(refinement.components),
                allocation_group_key=allocation_group_key,
                allocation_component_index=component.allocation_index,
                deconvolution_status=component.deconvolution_status,
                intensity_conserved=component.intensity_conserved,
            )
            if best is None or (score, -isotope_error, -component.start) > (
                best.score,
                -best.isotope_error,
                -best.segment_slice[0],
            ):
                best = candidate
        if containing and not selected_component_found:
            failures["selected_isotope_absent_at_event_scan"] += 1
    if best is not None:
        return best
    failure_order = (
        "no_component_at_event_scan",
        "selected_isotope_absent_at_event_scan",
        "insufficient_mono_points",
        "insufficient_isotope_channel_support",
        "component_too_wide",
        "low_averagine_cosine",
        "isotope_apex_spread",
    )
    status = max(
        (value for value in failure_order if failures[value]),
        key=failure_order.index,
    )
    return GenericLocalCandidate(event, status, None)


def _evaluate_generic_local_pair_batch(store, event_pairs, options):
    return tuple(
        (
            original_index,
            evaluate_generic_local_candidate(store, target, **options),
            evaluate_generic_local_candidate(store, decoy, **options),
        )
        for original_index, target, decoy in event_pairs
    )


def evaluate_generic_local_candidate_pairs(
    store,
    target_events,
    decoy_events,
    *,
    workers=1,
    **options,
):
    """Evaluate paired hypotheses in deterministic event order.

    Linux workers inherit the read-only raw/residual store, so large MS1
    arrays are not serialized once per event.  Each result batch is returned
    in its original range order, independent of worker completion order.
    """

    targets = tuple(target_events)
    decoys = tuple(decoy_events)
    if len(targets) != len(decoys):
        raise ValueError("target and decoy event counts differ")
    pairs = tuple(
        (index, target, decoy)
        for index, (target, decoy) in enumerate(zip(targets, decoys))
    )
    ranges = balanced_ranges(len(pairs), int(workers))
    if not ranges:
        return (), ()
    if len(ranges) == 1:
        batches = [
            _evaluate_generic_local_pair_batch(store, pairs, dict(options))
        ]
    else:
        # Local refinement cost is highly heterogeneous and nearby RT events
        # can share the same difficult region.  Striding prevents one
        # contiguous worker range from becoming the long-tail bottleneck.
        strided_batches = [
            tuple(pairs[worker_id::len(ranges)])
            for worker_id in range(len(ranges))
        ]
        batches = run_process_tasks(
            _evaluate_generic_local_pair_batch,
            [
                (store, batch, dict(options))
                for batch in strided_batches
            ],
        )
    ordered = sorted(
        (pair for batch in batches for pair in batch), key=lambda pair: pair[0]
    )
    return (
        tuple(pair[1] for pair in ordered),
        tuple(pair[2] for pair in ordered),
    )


def compete_generic_local_candidates(targets, decoys):
    decoys_by_id = {
        int(candidate.event["ms2_event_id"]): candidate for candidate in decoys
    }
    pairs = []
    ordered = []
    for target in targets:
        event_id = int(target.event["ms2_event_id"])
        decoy = decoys_by_id[event_id]
        ordered.append((event_id, target, decoy))
        pairs.append(
            TargetDecoyCompetition(
                str(event_id),
                target.score if target.quantitative_candidate else None,
                decoy.score if decoy.quantitative_candidate else None,
            )
        )
    results = {
        int(result.seed_id): result for result in target_decoy_q_values(pairs)
    }
    return tuple(
        GenericLocalCompetition(
            event_id,
            target,
            decoy,
            results[event_id].winner,
            results[event_id].q_value,
        )
        for event_id, target, decoy in ordered
    )


def cluster_compatible_generic_candidates(candidates, *, ppm: float = 10.0):
    """Cluster repeated MS2 events that support one local MS1 component.

    Clustering is evidence-only: event-level target/decoy competitions remain
    separate, while accepted events in one cluster can reuse one feature ID
    and one quantitative row.
    """

    eligible = [value for value in candidates if value.quantitative_candidate]
    eligible.sort(
        key=lambda value: (
            int(value.event["charge"]),
            math.inf
            if value.event.get("faims_cv") is None
            else float(value.event["faims_cv"]),
            float(value.mono_mz),
            float(value.rt_apex_sec),
            int(value.event["ms2_event_id"]),
        )
    )
    parents = list(range(len(eligible)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        left = find(left)
        right = find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    for left_index, left in enumerate(eligible):
        for right_index in range(left_index + 1, len(eligible)):
            right = eligible[right_index]
            if int(left.event["charge"]) != int(right.event["charge"]):
                break
            if (
                left.event.get("faims_cv") is None
            ) != (right.event.get("faims_cv") is None):
                continue
            if left.event.get("faims_cv") is not None and not math.isclose(
                float(left.event["faims_cv"]),
                float(right.event["faims_cv"]),
                abs_tol=1e-6,
            ):
                continue
            mz_error = (
                abs(float(right.mono_mz) - float(left.mono_mz))
                * 1e6
                / float(left.mono_mz)
            )
            if mz_error > ppm:
                # Within one charge/FAIMS ordering, later values only move
                # farther away in m/z.
                break
            if max(left.rt_start_sec, right.rt_start_sec) > min(
                left.rt_end_sec, right.rt_end_sec
            ):
                continue
            left_group = left.allocation_group_key
            right_group = right.allocation_group_key
            if left_group is not None and left_group == right_group:
                same_component = (
                    left.allocation_component_index
                    == right.allocation_component_index
                )
            else:
                width = min(left.width_sec, right.width_sec)
                same_component = abs(
                    left.rt_apex_sec - right.rt_apex_sec
                ) <= max(3.0, 0.25 * width)
            if same_component:
                union(left_index, right_index)

    groups = {}
    for index, candidate in enumerate(eligible):
        groups.setdefault(find(index), []).append(candidate)
    result = [
        tuple(
            sorted(
                values,
                key=lambda value: int(value.event["ms2_event_id"]),
            )
        )
        for values in groups.values()
    ]
    result.sort(
        key=lambda values: tuple(
            int(value.event["ms2_event_id"]) for value in values
        )
    )
    return tuple(result)
