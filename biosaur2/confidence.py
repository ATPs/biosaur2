"""Deterministic paired decoys and target-decoy extraction q-values."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Optional, Sequence


DECOY_NEUTRAL_SHIFTS = (11.0, 13.0, 17.0, 19.0, 23.0, 29.0)


@dataclass(frozen=True)
class TargetDecoyCompetition:
    seed_id: str
    target_score: Optional[float]
    decoy_score: Optional[float]


@dataclass(frozen=True)
class CompetitionResult:
    seed_id: str
    target_score: Optional[float]
    decoy_score: Optional[float]
    winner: str
    q_value: float


def deterministic_decoy_shift(run_id: str, seed_id: str) -> float:
    digest = hashlib.blake2b(
        (run_id + "\0" + seed_id).encode("utf-8"), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "little")
    shift = DECOY_NEUTRAL_SHIFTS[value % len(DECOY_NEUTRAL_SHIFTS)]
    return shift if (value // len(DECOY_NEUTRAL_SHIFTS)) % 2 == 0 else -shift


def target_decoy_q_values(
    competitions: Sequence[TargetDecoyCompetition],
) -> tuple[CompetitionResult, ...]:
    """Perform paired competition and monotonic q-value estimation.

    The explicit +1 decoy correction is conservative for small samples.
    Missing/non-finite scores lose to a finite score and otherwise yield no
    accepted target (q=1).
    """

    winners = []
    for item in competitions:
        target = item.target_score
        decoy = item.decoy_score
        target = float(target) if target is not None and math.isfinite(target) else None
        decoy = float(decoy) if decoy is not None and math.isfinite(decoy) else None
        if target is None and decoy is None:
            winner = "none"
            score = -math.inf
        elif decoy is None or (target is not None and target > decoy):
            winner = "target"
            score = target
        else:
            # Exact ties go to the decoy to avoid optimistic stable-ID effects.
            winner = "decoy"
            score = decoy
        winners.append((score, item.seed_id, winner, target, decoy))
    winners.sort(key=lambda value: (-value[0], value[1]))

    target_count = 0
    decoy_count = 0
    raw_fdr = [1.0] * len(winners)
    group_start = 0
    while group_start < len(winners):
        score = winners[group_start][0]
        group_end = group_start + 1
        while group_end < len(winners) and winners[group_end][0] == score:
            group_end += 1
        for _score, _seed_id, winner, _target, _decoy in winners[
            group_start:group_end
        ]:
            if winner == "target":
                target_count += 1
            elif winner == "decoy":
                decoy_count += 1
        value = min(1.0, (decoy_count + 1.0) / max(target_count, 1))
        raw_fdr[group_start:group_end] = [value] * (group_end - group_start)
        group_start = group_end
    q_values = [1.0] * len(winners)
    running = 1.0
    for index in range(len(winners) - 1, -1, -1):
        running = min(running, raw_fdr[index])
        q_values[index] = running

    by_id = {}
    for value, q_value in zip(winners, q_values):
        _score, seed_id, winner, target, decoy = value
        by_id[seed_id] = CompetitionResult(
            seed_id,
            target,
            decoy,
            winner,
            q_value if winner == "target" else 1.0,
        )
    return tuple(by_id[item.seed_id] for item in competitions)
