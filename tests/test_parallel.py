import time

import pytest

from biosaur2.parallel import (
    GIB,
    ResourceSample,
    WorkerProcessError,
    balanced_ranges,
    effective_worker_budget,
    run_adaptive_process_tasks,
    run_bounded_process_tasks,
    worker_slot_allocations,
    run_process_tasks,
)


def _identity(value):
    return value


def _fail(message):
    raise ValueError(message)


def _sleep(value):
    time.sleep(value)
    return value


def _failed_status(value):
    return {"value": value, "status": "failed"}


def _slow_resource_task(value):
    time.sleep(0.35)
    return {"value": value, "peak_rss_kib": 1024}


class _IdleSampler:
    def sample(self, _pids):
        return ResourceSample(
            process_cpu_cores=0.0,
            host_busy_cores=0.0,
            process_pss_bytes=0,
            mem_available_bytes=512 * GIB,
            physical_memory_bytes=512 * GIB,
            cpu_count=80,
        )


class _MemoryBoundSampler(_IdleSampler):
    def sample(self, _pids):
        return ResourceSample(
            process_cpu_cores=0.0,
            host_busy_cores=0.0,
            process_pss_bytes=16 * GIB,
            mem_available_bytes=512 * GIB,
            physical_memory_bytes=512 * GIB,
            cpu_count=80,
        )


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


def test_total_worker_budget_and_run_slot_allocations():
    assert effective_worker_budget(8, cpu_count_value=6) == 6
    assert worker_slot_allocations(4, 10) == [4]
    assert worker_slot_allocations(10, 10) == [4, 3, 3]
    assert worker_slot_allocations(10, 2) == [5, 5]


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


def test_bounded_file_scheduler_does_not_eagerly_consume_large_batches():
    consumed = []

    def task_args():
        for value in range(1808):
            consumed.append(value)
            yield (value,)

    results, started = run_bounded_process_tasks(
        _failed_status,
        task_args(),
        max_workers=4,
        stop_on_result=lambda result: result["status"] == "failed",
    )

    assert started == [0, 1, 2, 3]
    assert consumed == [0, 1, 2, 3]
    assert set(results) == {0, 1, 2, 3}


def test_adaptive_scheduler_adds_one_worker_runs_only_after_cpu_warmup():
    results, started, summary = run_adaptive_process_tasks(
        _slow_resource_task,
        ((value,) for value in range(3)),
        target_workers=4,
        max_memory_bytes=128 * GIB,
        resource_sampler=_IdleSampler(),
        poll_seconds=0.02,
    )
    assert set(results) == {0, 1, 2}
    assert started == [0, 1, 2]
    assert summary["allocation_ceiling"] == 6
    assert summary["peak_allocated_workers"] == 6


def test_adaptive_scheduler_records_memory_admission_pauses():
    results, started, summary = run_adaptive_process_tasks(
        _slow_resource_task,
        ((value,) for value in range(2)),
        target_workers=4,
        max_memory_bytes=32 * GIB,
        resource_sampler=_MemoryBoundSampler(),
        poll_seconds=0.02,
    )
    assert set(results) == {0, 1}
    assert started == [0, 1]
    assert summary["memory_pause_count"] > 0
