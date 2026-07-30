"""Bounded deterministic local conflict selection and non-negative fitting."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import FrozenSet, Hashable, Sequence

import numpy as np
from scipy.optimize import nnls


@dataclass(frozen=True)
class ConflictCandidate:
    candidate_id: str
    score: float
    raw_points: FrozenSet[Hashable]
    protected: bool = False


@dataclass(frozen=True)
class ConflictSelection:
    selected_ids: tuple[str, ...]
    objective: float
    method: str
    component_count: int


@dataclass(frozen=True)
class DeconvolutionResult:
    status: str
    coefficients: np.ndarray
    modeled: np.ndarray
    residual: np.ndarray
    condition_number: float
    intensity_conserved: bool


def _components(candidates):
    adjacency = {item.candidate_id: set() for item in candidates}
    by_id = {item.candidate_id: item for item in candidates}
    for left, right in itertools.combinations(candidates, 2):
        if left.raw_points & right.raw_points:
            adjacency[left.candidate_id].add(right.candidate_id)
            adjacency[right.candidate_id].add(left.candidate_id)
    components = []
    unseen = set(adjacency)
    while unseen:
        root = min(unseen)
        stack = [root]
        ids = []
        unseen.remove(root)
        while stack:
            current = stack.pop()
            ids.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(([by_id[value] for value in sorted(ids)], adjacency))
    return components


def _exact_component(component, adjacency):
    best = None
    count = len(component)
    for mask in range(1 << count):
        selected = [component[index] for index in range(count) if mask & (1 << index)]
        ids = {item.candidate_id for item in selected}
        if any(adjacency[item.candidate_id] & ids for item in selected):
            continue
        if any(item.protected and item.candidate_id not in ids for item in component):
            continue
        objective = float(sum(item.score for item in selected))
        key = (objective, tuple(sorted(ids)))
        if best is None or key[0] > best[0] or (
            key[0] == best[0] and key[1] < best[1]
        ):
            best = key
    return (0.0, ()) if best is None else best


def select_conflict_candidates(
    candidates: Sequence[ConflictCandidate], *, exact_limit: int = 18
) -> ConflictSelection:
    """Select a maximum-score non-conflicting set component by component."""

    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("candidate IDs must be unique")
    selected = []
    objective = 0.0
    methods = set()
    components = _components(candidates)
    for component, adjacency in components:
        if len(component) <= exact_limit:
            score, ids = _exact_component(component, adjacency)
            methods.add("exact")
        else:
            methods.add("greedy")
            ordered = sorted(
                component,
                key=lambda item: (
                    not item.protected,
                    -item.score,
                    item.candidate_id,
                ),
            )
            ids_list = []
            blocked = set()
            score = 0.0
            for item in ordered:
                if item.candidate_id in blocked:
                    continue
                ids_list.append(item.candidate_id)
                score += item.score
                blocked.update(adjacency[item.candidate_id])
            ids = tuple(sorted(ids_list))
        selected.extend(ids)
        objective += score
    return ConflictSelection(
        tuple(sorted(selected)),
        objective,
        "+".join(sorted(methods)) if methods else "empty",
        len(components),
    )


def nonnegative_deconvolution(
    design,
    observed,
    *,
    condition_max: float = 1e8,
    conservation_tolerance: float = 1e-8,
) -> DeconvolutionResult:
    """Fit identifiable non-negative local components without duplicating signal."""

    matrix = np.asarray(design, dtype=np.float64)
    signal = np.asarray(observed, dtype=np.float64)
    if matrix.ndim != 2 or signal.ndim != 1 or matrix.shape[0] != signal.size:
        raise ValueError("design rows must match the observed vector")
    if np.any(matrix < 0) or np.any(signal < 0) or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(signal)):
        raise ValueError("deconvolution inputs must be finite and nonnegative")
    if matrix.shape[1] == 0:
        return DeconvolutionResult(
            "no_components", np.empty(0), np.zeros_like(signal), signal.copy(), np.inf, True
        )
    condition = float(np.linalg.cond(matrix))
    rank = int(np.linalg.matrix_rank(matrix))
    if rank < matrix.shape[1] or not np.isfinite(condition) or condition > condition_max:
        return DeconvolutionResult(
            "unidentifiable",
            np.zeros(matrix.shape[1], dtype=np.float64),
            np.zeros_like(signal),
            signal.copy(),
            condition,
            True,
        )
    coefficients, _residual_norm = nnls(matrix, signal)
    modeled = matrix @ coefficients
    residual = signal - modeled
    conserved = bool(
        np.sum(modeled) <= np.sum(signal) * (1.0 + conservation_tolerance)
        and np.all(modeled >= -conservation_tolerance)
    )
    status = "accepted" if conserved else "intensity_conservation_failed"
    return DeconvolutionResult(
        status, coefficients, modeled, residual, condition, conserved
    )
