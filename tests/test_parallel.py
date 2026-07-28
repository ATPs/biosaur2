import time

import pytest

from biosaur2.parallel import (
    WorkerProcessError,
    balanced_ranges,
    run_process_tasks,
)


def _identity(value):
    return value


def _fail(message):
    raise ValueError(message)


def _sleep(value):
    time.sleep(value)
    return value


@pytest.mark.parametrize(
    ("item_count", "workers", "expected"),
    [
        (10, 3, [(0, 4), (4, 7), (7, 10)]),
        (2, 4, [(0, 1), (1, 2)]),
        (0, 4, []),
    ],
)
def test_balanced_ranges_cover_every_item_once(item_count, workers, expected):
    ranges = balanced_ranges(item_count, workers, cpu_count_value=workers)
    assert ranges == expected
    covered = [index for start, end in ranges for index in range(start, end)]
    assert covered == list(range(item_count))


def test_balanced_ranges_cap_cpu_count():
    assert balanced_ranges(10, 8, cpu_count_value=2) == [(0, 5), (5, 10)]


@pytest.mark.parametrize("workers", [0, -1])
def test_balanced_ranges_reject_invalid_workers(workers):
    with pytest.raises(ValueError, match="positive"):
        balanced_ranges(1, workers)


def test_process_results_return_in_task_order():
    assert run_process_tasks(_identity, [(2,), (1,), (3,)]) == [2, 1, 3]


def test_child_exception_propagates_with_traceback_without_hang():
    started = time.monotonic()
    with pytest.raises(WorkerProcessError) as error:
        run_process_tasks(_fail, [("expected failure",)])
    assert time.monotonic() - started < 5
    assert error.value.failure.exception_type == "ValueError"
    assert "expected failure" in str(error.value)
    assert "_fail" in error.value.failure.traceback_text
