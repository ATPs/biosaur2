"""Bounded RT-alignment forests and deterministic external-donor planning."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Mapping, Optional

import numpy as np
from scipy import sparse

from .alignment import AlignmentAnchor, RTAlignmentModel, choose_reference_run, fit_rt_alignment


MAX_REFERENCE_CANDIDATES = 4


@dataclass(frozen=True)
class ExternalObservation:
    run_id: str
    ion_key: str
    canonical_peptidoform: str
    charge: int
    faims_cv: Optional[float]
    rt_apex_sec: float
    q_value: float
    assay_id: int
    psm_id: str


@dataclass(frozen=True)
class ReferenceStarAlignment:
    """A donor-to-recipient map composed through a component reference.

    Most paths contain one leg on each side.  A bounded fallback edge can make
    either side multi-hop while preserving one deterministic component root.
    """

    source_run: str
    target_run: str
    source_to_reference: tuple[RTAlignmentModel, ...]
    reference_to_target: tuple[RTAlignmentModel, ...]

    @property
    def method(self):
        if len(self.source_to_reference) <= 1 and len(self.reference_to_target) <= 1:
            return "reference_star"
        return "reference_forest"

    @property
    def status(self):
        if all(
            model.status == "accepted"
            for model in self.source_to_reference + self.reference_to_target
        ):
            return "accepted"
        return "missing_reference_leg"

    @property
    def anchor_count(self):
        values = [
            model.anchor_count
            for model in self.source_to_reference + self.reference_to_target
            if model.method != "identity"
        ]
        return min(values) if values else 0

    @property
    def inlier_count(self):
        values = [
            model.inlier_count
            for model in self.source_to_reference + self.reference_to_target
            if model.method != "identity"
        ]
        return min(values) if values else 0

    @property
    def residual_mad_sec(self):
        return sum(
            model.residual_mad_sec
            for model in self.source_to_reference + self.reference_to_target
            if model.residual_mad_sec is not None
        )

    def predict(self, rt_sec):
        if self.status != "accepted":
            raise ValueError("reference alignment is not accepted")
        value = rt_sec
        for model in self.source_to_reference + self.reference_to_target:
            value = model.predict(value)
        return value


@dataclass(frozen=True)
class ExternalPlan:
    target_run: str
    source_run: str
    alignment_group: str
    observation: ExternalObservation
    predicted_rt_sec: float
    alignment: ReferenceStarAlignment


@dataclass
class AlignmentForest:
    """Accepted reference-rooted components plus all attempted edge models."""

    models: dict
    component_by_run: dict[str, str]
    reference_runs: dict[str, str]
    parent_by_run: dict[str, Optional[str]]

    # Keep the old mapping-like surface for callers and focused tests.
    def __getitem__(self, key):
        return self.models[key]

    def __len__(self):
        return len(self.models)

    def __contains__(self, key):
        return key in self.models

    def values(self):
        return self.models.values()

    def items(self):
        return self.models.items()

    def path_to_reference(self, run_id):
        result = []
        current = run_id
        while self.parent_by_run[current] is not None:
            parent = self.parent_by_run[current]
            result.append(self.models[(current, parent)][1])
            current = parent
        return tuple(result)

    def reference_to_run_path(self, run_id):
        upward = []
        current = run_id
        while self.parent_by_run[current] is not None:
            parent = self.parent_by_run[current]
            upward.append((parent, current))
            current = parent
        return tuple(self.models[edge][1] for edge in reversed(upward))


def alignment_group_for_run(run) -> str:
    explicit = (run.metadata.get("alignment_group") or "").strip()
    if explicit:
        return "explicit:" + explicit
    fraction = (run.metadata.get("fraction") or "").strip()
    batch = (run.metadata.get("batch") or "").strip()
    if fraction or batch:
        return "derived:fraction=%s|batch=%s" % (fraction, batch)
    return "derived:default"


def faims_key(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "none"
    return format(float(value), ".9g")


def exact_ion_key(canonical_peptidoform, charge, faims_cv) -> str:
    return "%s\x1f%d\x1f%s" % (
        canonical_peptidoform, int(charge), faims_key(faims_cv)
    )


def _rejected_alignment(source_run, target_run, count, status):
    return RTAlignmentModel(
        source_run, target_run, "none", count, 0, (), (), 1.0, 0.0, None, status
    )


def _downsample_alignment_anchors(anchors, max_anchors):
    limit = int(max_anchors)
    if limit < 1:
        raise ValueError("max_anchors must be positive")
    if len(anchors) <= limit:
        return anchors
    ordered = sorted(
        anchors,
        key=lambda value: (value.source_rt_sec, value.target_rt_sec, value.ion_key),
    )
    selected = []
    for index in range(limit):
        start = index * len(ordered) // limit
        end = (index + 1) * len(ordered) // limit
        selected.append(
            min(ordered[start:end], key=lambda value: (-value.quality, value.ion_key))
        )
    return sorted(
        selected,
        key=lambda value: (value.source_rt_sec, value.target_rt_sec, value.ion_key),
    )


def _fit_reference_edge(source_run, target_run, indexed, min_anchors, max_mad, max_anchors):
    common = sorted(set(indexed[source_run]) & set(indexed[target_run]))
    if len(common) < min_anchors:
        return _rejected_alignment(source_run, target_run, len(common), "insufficient_anchors")
    anchors = [
        AlignmentAnchor(
            key,
            indexed[source_run][key].rt_apex_sec,
            indexed[target_run][key].rt_apex_sec,
            max(1e-6, 1.0 - max(indexed[source_run][key].q_value, indexed[target_run][key].q_value)),
        )
        for key in common
    ]
    model = fit_rt_alignment(
        source_run, target_run, _downsample_alignment_anchors(anchors, max_anchors)
    )
    if model.residual_mad_sec is None or model.residual_mad_sec > max_mad:
        return RTAlignmentModel(**{**model.__dict__, "status": "residual_mad_exceeds_limit"})
    return model


def choose_group_reference_runs(runs, observations_by_run):
    """Retained for callers that need declared-group reference reporting."""

    grouped = defaultdict(dict)
    for run in runs:
        grouped[alignment_group_for_run(run)][run.run_id] = len(
            observations_by_run.get(run.run_id, ())
        )
    return {
        group: choose_reference_run(counts)
        for group, counts in sorted(grouped.items())
    }


def _candidate_edges(run_ids, indexed, min_anchors, max_candidates):
    """Return top-K exact shared-ion candidates per run using sparse overlap."""

    ion_ids = {
        ion_key: index
        for index, ion_key in enumerate(
            sorted({ion for run_id in run_ids for ion in indexed[run_id]})
        )
    }
    row_indices = []
    column_indices = []
    for row, run_id in enumerate(run_ids):
        row_indices.extend([row] * len(indexed[run_id]))
        column_indices.extend(ion_ids[key] for key in indexed[run_id])
    matrix = sparse.csr_matrix(
        (
            np.ones(len(row_indices), dtype=np.int32),
            (
                np.asarray(row_indices, dtype=np.int32),
                np.asarray(column_indices, dtype=np.int32),
            ),
        ),
        shape=(len(run_ids), len(ion_ids)),
        dtype=np.int32,
    )
    overlaps = (matrix @ matrix.T).tocsr()
    overlaps.setdiag(0)
    overlaps.eliminate_zeros()
    observation_counts = [len(indexed[run_id]) for run_id in run_ids]
    edges = {}
    for source_index, source_run in enumerate(run_ids):
        start, end = overlaps.indptr[source_index : source_index + 2]
        choices = [
            (int(overlaps.data[position]), int(overlaps.indices[position]))
            for position in range(start, end)
            if int(overlaps.data[position]) >= min_anchors
        ]
        choices.sort(
            key=lambda value: (
                -value[0], -observation_counts[value[1]], run_ids[value[1]]
            )
        )
        for shared_count, target_index in choices[:max_candidates]:
            target_run = run_ids[target_index]
            pair = tuple(sorted((source_run, target_run)))
            edges[pair] = max(edges.get(pair, 0), shared_count)
    return sorted(
        ((shared_count, source_run, target_run) for (source_run, target_run), shared_count in edges.items()),
        key=lambda value: (-value[0], value[1], value[2]),
    )


def _component_id(declared_group, reference_run):
    return "%s|component=%s" % (declared_group, reference_run)


def build_alignment_models(
    runs,
    observations_by_run,
    *,
    min_anchors=5,
    max_residual_mad_sec=30.0,
    max_anchors=256,
    max_reference_candidates=MAX_REFERENCE_CANDIDATES,
):
    """Fit a bounded sparse candidate graph into deterministic RT forests."""

    candidate_limit = int(max_reference_candidates)
    if candidate_limit < 1:
        raise ValueError("max_reference_candidates must be positive")
    grouped = defaultdict(list)
    for run in runs:
        grouped[alignment_group_for_run(run)].append(run.run_id)
    all_models = {}
    component_by_run = {}
    reference_runs = {}
    parent_by_run = {}
    for declared_group, unsorted_run_ids in sorted(grouped.items()):
        run_ids = sorted(unsorted_run_ids)
        indexed = {
            run_id: {item.ion_key: item for item in observations_by_run.get(run_id, ())}
            for run_id in run_ids
        }
        parents = {run_id: run_id for run_id in run_ids}
        ranks = {run_id: 0 for run_id in run_ids}
        accepted_adjacency = defaultdict(list)

        def find(run_id):
            while parents[run_id] != run_id:
                parents[run_id] = parents[parents[run_id]]
                run_id = parents[run_id]
            return run_id

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root == right_root:
                return
            if ranks[left_root] < ranks[right_root]:
                left_root, right_root = right_root, left_root
            parents[right_root] = left_root
            if ranks[left_root] == ranks[right_root]:
                ranks[left_root] += 1

        for _shared_count, left, right in _candidate_edges(
            run_ids, indexed, int(min_anchors), candidate_limit
        ):
            if find(left) == find(right):
                continue
            forward = _fit_reference_edge(
                left, right, indexed, min_anchors, max_residual_mad_sec, max_anchors
            )
            reverse = _fit_reference_edge(
                right, left, indexed, min_anchors, max_residual_mad_sec, max_anchors
            )
            all_models[(left, right)] = (declared_group, forward)
            all_models[(right, left)] = (declared_group, reverse)
            if forward.status == "accepted" and reverse.status == "accepted":
                union(left, right)
                accepted_adjacency[left].append(right)
                accepted_adjacency[right].append(left)

        components = defaultdict(list)
        for run_id in run_ids:
            components[find(run_id)].append(run_id)
        for members in components.values():
            reference = min(members, key=lambda run_id: (-len(indexed[run_id]), run_id))
            resolved_group = _component_id(declared_group, reference)
            reference_runs[resolved_group] = reference
            queue = deque([reference])
            parent_by_run[reference] = None
            component_by_run[reference] = resolved_group
            while queue:
                parent = queue.popleft()
                for child in sorted(accepted_adjacency[parent]):
                    if child in component_by_run:
                        continue
                    component_by_run[child] = resolved_group
                    parent_by_run[child] = parent
                    queue.append(child)
    return AlignmentForest(all_models, component_by_run, reference_runs, parent_by_run)


def plan_external_assays(runs, observations_by_run, models, reference_runs=None):
    """Choose one direct-ID donor per recipient-only ion within one component."""

    if not isinstance(models, AlignmentForest):
        raise TypeError("external planning requires an AlignmentForest")
    run_by_id = {run.run_id: run for run in runs}
    donors = defaultdict(lambda: defaultdict(list))
    for source_run, observations in observations_by_run.items():
        component = models.component_by_run[source_run]
        for observation in observations:
            donors[component][observation.ion_key].append(observation)
    source_paths = {
        run_id: models.path_to_reference(run_id) for run_id in run_by_id
    }
    target_paths = {
        run_id: models.reference_to_run_path(run_id) for run_id in run_by_id
    }
    for by_ion in donors.values():
        for ion_key, observations in by_ion.items():
            by_ion[ion_key] = sorted(
                observations,
                key=lambda observation: (
                    observation.q_value,
                    sum(
                        math.inf
                        if model.residual_mad_sec is None
                        else model.residual_mad_sec
                        for model in source_paths[observation.run_id]
                    ),
                    observation.run_id,
                    observation.assay_id,
                ),
            )
    result = {run.run_id: [] for run in runs}
    for target_run in sorted(run_by_id):
        component = models.component_by_run[target_run]
        target_ions = {item.ion_key for item in observations_by_run.get(target_run, ())}
        for ion_key in sorted(donors[component]):
            if ion_key in target_ions:
                continue
            for observation in donors[component][ion_key]:
                if observation.run_id == target_run:
                    continue
                alignment = ReferenceStarAlignment(
                    observation.run_id,
                    target_run,
                    source_paths[observation.run_id],
                    target_paths[target_run],
                )
                predicted = alignment.predict(observation.rt_apex_sec)
                if math.isfinite(predicted):
                    result[target_run].append(
                        ExternalPlan(
                            target_run,
                            observation.run_id,
                            component,
                            observation,
                            predicted,
                            alignment,
                        )
                    )
                    break
    return {key: tuple(value) for key, value in result.items()}
