import math

from biosaur2.confidence import (
    DECOY_NEUTRAL_SHIFTS,
    TargetDecoyCompetition,
    deterministic_decoy_shift,
    target_decoy_q_values,
)


def test_decoy_shift_is_stable_signed_and_non_isotopic():
    first = deterministic_decoy_shift("run", "event-7")
    assert first == deterministic_decoy_shift("run", "event-7")
    assert abs(first) in DECOY_NEUTRAL_SHIFTS
    assert abs(first) > 10
    assert deterministic_decoy_shift("other", "event-7") != first or first in {
        -value for value in DECOY_NEUTRAL_SHIFTS
    }


def test_paired_q_values_use_decoy_ties_and_reverse_monotonic_minimum():
    result = target_decoy_q_values(
        [
            TargetDecoyCompetition("a", 10.0, 1.0),
            TargetDecoyCompetition("b", 9.0, 2.0),
            TargetDecoyCompetition("c", 3.0, 8.0),
            TargetDecoyCompetition("d", 7.0, 7.0),
            TargetDecoyCompetition("e", None, None),
        ]
    )
    by_id = {item.seed_id: item for item in result}
    assert by_id["a"].winner == "target"
    assert by_id["d"].winner == "decoy"
    assert by_id["e"].winner == "none"
    assert by_id["a"].q_value <= by_id["b"].q_value
    assert by_id["c"].q_value == by_id["d"].q_value == 1.0


def test_all_null_or_decoy_winners_never_get_an_accepted_target_q_value():
    result = target_decoy_q_values(
        [
            TargetDecoyCompetition("a", 1.0, 2.0),
            TargetDecoyCompetition("b", math.nan, None),
        ]
    )
    assert all(item.q_value == 1.0 for item in result)


def test_equal_score_group_has_one_id_order_independent_q_value():
    competitions = [
        TargetDecoyCompetition("target-%03d" % index, 1.0, None)
        for index in range(200)
    ] + [
        TargetDecoyCompetition("decoy-a", None, 1.0),
        TargetDecoyCompetition("decoy-z", None, 1.0),
    ]
    forward = target_decoy_q_values(competitions)
    reverse = target_decoy_q_values(list(reversed(competitions)))
    expected = 3.0 / 200.0
    assert {
        item.q_value for item in forward if item.winner == "target"
    } == {expected}
    assert {
        item.seed_id: item.q_value for item in forward
    } == {
        item.seed_id: item.q_value for item in reverse
    }
