"""Shared structures for feature-only external RT-alignment forests."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from .alignment import RTAlignmentModel


MAX_REFERENCE_CANDIDATES = 4


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


@dataclass
class AlignmentForest:
    """Accepted reference-rooted components plus all attempted edge models."""

    models: dict
    component_by_run: dict[str, str]
    reference_runs: dict[str, str]
    parent_by_run: dict[str, Optional[str]]

    def __getitem__(self, key):
        return self.models[key]

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
