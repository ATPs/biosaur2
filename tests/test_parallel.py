import os
import time

import pytest

from biosaur2.parallel import (
    GIB,
    LinuxResourceSampler,
    ResourceSample,
    WorkerFailure,
    WorkerProcessError,
    balanced_ranges,
    effective_worker_budget,
    run_adaptive_process_tasks,
    run_bounded_process_tasks,
    run_budgeted_process_tasks,
    worker_slot_allocations,
    run_process_tasks,
)
from biosaur2.parallel import _MemoryEstimator


def _identity(value):
    return value


def _fail(message):
    raise ValueError(message)


def _sleep(value):
    time.sleep(value)
    return value


def _failed_status(value):
    return {"value": value, "status": "failed"}


def _slow_resource_task(value, _allocated_workers):
    time.sleep(0.35)
    return {"value": value, "peak_rss_kib": 1024}


def _exit_without_result(_value, _allocated_workers):
    os._exit(0)


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
    def sample(self, pids):
        return ResourceSample(
            process_cpu_cores=0.0,
            host_busy_cores=0.0,
            process_pss_bytes=17 * GIB if pids else 0,
            mem_available_bytes=512 * GIB,
            physical_memory_bytes=512 * GIB,
            cpu_count=80,
        )


class _BootstrapMemorySampler(_IdleSampler):
    def __init__(self):
        self.calls = 0

    def sample(self, pids):
        self.calls += 1
        available = 0 if not pids and self.calls < 3 else 512 * GIB
        return ResourceSample(
            process_cpu_cores=0.0,
            host_busy_cores=0.0,
            process_pss_bytes=0,
            mem_available_bytes=available,
            physical_memory_bytes=512 * GIB,
            cpu_count=80,
        )


class _PreemptOnceSampler(_IdleSampler):
    def __init__(self):
        self.preempted = False

    def sample(self, pids):
        if pids and not self.preempted:
            self.preempted = True
            pss = 65 * GIB
        else:
            pss = 0
        return ResourceSample(
            process_cpu_cores=0.0,
            host_busy_cores=0.0,
            process_pss_bytes=pss,
            mem_available_bytes=512 * GIB,
            physical_memory_bytes=512 * GIB,
            cpu_count=80,
            root_pss_bytes={pid: pss for pid in pids},
        )


class _ObservedHeadroomSampler(_IdleSampler):
    """Expose low current PSS while normal admission uses 16 GiB peaks."""

    def sample(self, pids):
        pss = len(pids) * 4 * GIB
        return ResourceSample(
            process_cpu_cores=0.0,
            host_busy_cores=0.0,
            process_pss_bytes=pss,
            mem_available_bytes=512 * GIB,
            physical_memory_bytes=512 * GIB,
            cpu_count=80,
            root_pss_bytes={pid: 4 * GIB for pid in pids},
        )


class _FastSampler:
    def __init__(self):
        self.host_calls = 0
        self.light_calls = 0

    @staticmethod
    def _sample(kind, pids=()):
        return ResourceSample(
            process_cpu_cores=0.0,
            host_busy_cores=0.0,
            process_pss_bytes=0,
            mem_available_bytes=512 * GIB,
            physical_memory_bytes=512 * GIB,
            cpu_count=80,
            root_rss_bytes={pid: 0 for pid in pids},
            sample_kind=kind,
        )

    def sample_host(self):
        self.host_calls += 1
        return self._sample("host")

    def sample_light(self, pids):
        self.light_calls += 1
        return self._sample("light", pids)


class _ObservedRssFastSampler:
    def __init__(self, rss_bytes=None):
        self.rss_bytes = rss_bytes

    @staticmethod
    def _sample(kind, pids=(), rss_bytes=None):
        return ResourceSample(
            process_cpu_cores=0.0,
            host_busy_cores=0.0,
            process_pss_bytes=0,
            mem_available_bytes=24 * GIB,
            physical_memory_bytes=64 * GIB,
            cpu_count=80,
            root_rss_bytes=(
                {pid: rss_bytes for pid in pids} if rss_bytes is not None else {}
            ),
            sample_kind=kind,
        )

    def sample_host(self):
        return self._sample("host")

    def sample_light(self, pids):
        return self._sample("light", pids, self.rss_bytes)


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
    assert effective_worker_budget(8, cpu_count_value=6) == 8
    assert effective_worker_budget(24, cpu_count_value=6) == 18
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


def test_adaptive_scheduler_reuses_four_worker_allocation_after_completion():
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
    assert summary["peak_allocated_workers"] == 4
    assert summary["peak_normal_allocated_workers"] == 4


def test_adaptive_scheduler_records_memory_admission_pauses():
    results, started, summary = run_adaptive_process_tasks(
        _slow_resource_task,
        ((value,) for value in range(2)),
        target_workers=4,
        max_memory_bytes=32 * GIB,
        resource_sampler=_MemoryBoundSampler(),
        poll_seconds=0.02,
        heartbeat_seconds=0.02,
    )
    assert set(results) == {0, 1}
    assert started == [0, 1]
    assert summary["memory_pause_count"] > 0


def test_adaptive_scheduler_uses_observed_pss_after_normal_admission_blocks():
    allocations = []
    results, started, summary = run_adaptive_process_tasks(
        _slow_resource_task,
        ((value,) for value in range(3)),
        target_workers=12,
        max_memory_bytes=32 * GIB,
        resource_sampler=_ObservedHeadroomSampler(),
        on_start=lambda _worker_id, _args, allocation: allocations.append(allocation),
        poll_seconds=0.02,
        heartbeat_seconds=0.02,
    )
    assert set(results) == {0, 1, 2}
    assert started == [0, 1, 2]
    assert allocations == [4, 4, 4]
    assert summary["observed_memory_admission_count"] == 1
    assert summary["peak_unobserved_reservation_bytes"] == 16 * GIB
    assert any(sample["event"] == "heartbeat" for sample in summary["resource_samples"])


@pytest.mark.parametrize("rss_bytes", [None, 4 * GIB])
def test_auto_scheduler_reserves_runs_for_a_fixed_startup_window(
    monkeypatch, rss_bytes
):
    monkeypatch.setattr(
        "biosaur2.parallel._AUTO_MEMORY_RESERVATION_SECONDS", 0.08
    )
    start_times = []
    results, started, summary = run_adaptive_process_tasks(
        _slow_resource_task,
        ((value,) for value in range(2)),
        target_workers=8,
        max_memory_bytes=64 * GIB,
        resource_sampler=_ObservedRssFastSampler(rss_bytes),
        on_start=lambda *_args: start_times.append(time.monotonic()),
        poll_seconds=0.01,
        host_poll_seconds=1.0,
        heartbeat_seconds=0.02,
    )
    assert set(results) == {0, 1}
    assert started == [0, 1]
    assert summary["peak_active_tasks"] == 2
    assert start_times[1] - start_times[0] >= 0.08


def test_adaptive_scheduler_limits_speculative_starts_per_resource_sample():
    allocations = []
    results, _started, summary = run_adaptive_process_tasks(
        _slow_resource_task,
        ((value,) for value in range(6)),
        target_workers=8,
        max_memory_bytes=128 * GIB,
        resource_sampler=_IdleSampler(),
        on_start=lambda _worker_id, _args, allocation: allocations.append(allocation),
        poll_seconds=0.02,
        heartbeat_seconds=0.02,
    )
    assert set(results) == set(range(6))
    assert allocations[:2] == [4, 4]
    assert summary["peak_speculative_starts_per_sample"] == 2
    assert summary["peak_allocated_workers"] == 12


def test_adaptive_scheduler_waits_for_memory_before_bootstrap():
    results, started, summary = run_adaptive_process_tasks(
        _slow_resource_task,
        ((value,) for value in range(1)),
        target_workers=1,
        max_memory_bytes=64 * GIB,
        resource_sampler=_BootstrapMemorySampler(),
        poll_seconds=0.01,
        heartbeat_seconds=0.01,
    )
    assert set(results) == {0}
    assert started == [0]
    assert summary["memory_wait_seconds"] >= 0.01


def test_adaptive_scheduler_rejects_impossible_memory_limit():
    with pytest.raises(ValueError, match="initial per-run admission"):
        run_adaptive_process_tasks(
            _slow_resource_task,
            ((0,),),
            target_workers=1,
            max_memory_bytes=2 * GIB,
            resource_sampler=_IdleSampler(),
        )


def test_adaptive_scheduler_preempts_and_requeues_memory_pressure():
    results, started, summary = run_adaptive_process_tasks(
        _slow_resource_task,
        ((0,),),
        target_workers=4,
        max_memory_bytes=64 * GIB,
        resource_sampler=_PreemptOnceSampler(),
        poll_seconds=0.02,
        heartbeat_seconds=0.02,
    )
    assert set(results) == {0}
    assert results[0]["value"] == 0
    assert started == [0, 0]
    assert summary["memory_preemption_count"] == 1
    assert summary["memory_requeue_count"] == 1


def test_adaptive_scheduler_adds_a_bounded_eight_worker_run_when_cpu_is_idle():
    allocations = []
    results, _started, summary = run_adaptive_process_tasks(
        _slow_resource_task,
        ((value,) for value in range(5)),
        target_workers=16,
        max_memory_bytes=256 * GIB,
        resource_sampler=_IdleSampler(),
        on_start=lambda _worker_id, _args, allocation: allocations.append(allocation),
        poll_seconds=0.02,
        heartbeat_seconds=0.02,
    )
    assert set(results) == {0, 1, 2, 3, 4}
    assert allocations[:4] == [4, 4, 4, 4]
    assert 8 in allocations
    assert summary["peak_scaled_tasks"] == 1


def test_adaptive_scheduler_reports_a_clean_child_exit_without_result():
    results, started, _summary = run_adaptive_process_tasks(
        _exit_without_result,
        ((0,),),
        target_workers=1,
        max_memory_bytes=64 * GIB,
        resource_sampler=_IdleSampler(),
        poll_seconds=0.02,
    )
    assert started == [0]
    assert isinstance(results[0], WorkerFailure)
    assert results[0].exception_type == "ProcessExit"


def test_budgeted_scheduler_reports_a_clean_child_exit_without_result():
    results, started, _allocations = run_budgeted_process_tasks(
        _exit_without_result,
        ((0,),),
        total_workers=1,
        poll_seconds=0.02,
    )
    assert started == [0]
    assert isinstance(results[0], WorkerFailure)
    assert results[0].exception_type == "ProcessExit"


def test_adaptive_scheduler_defaults_to_sixty_second_heartbeats():
    _results, _started, summary = run_adaptive_process_tasks(
        _slow_resource_task,
        ((0,),),
        target_workers=1,
        max_memory_bytes=64 * GIB,
        resource_sampler=_IdleSampler(),
    )
    assert summary["heartbeat_seconds"] == 60.0


def test_memory_estimator_clamps_completed_peaks_to_one_through_thirty_gib():
    estimator = _MemoryEstimator()
    estimator.observe(4, 0, {"peak_rss_kib": 128})
    assert estimator.estimate_bytes(4) == int(1.2 * GIB)
    estimator.observe(4, 0, {"peak_rss_kib": 64 * 1024 ** 2})
    assert estimator.estimate_bytes(4) == 30 * GIB


def test_auto_scheduler_uses_host_ticks_and_owned_tree_heartbeats():
    sampler = _FastSampler()
    _results, _started, summary = run_adaptive_process_tasks(
        _slow_resource_task,
        ((0,),),
        target_workers=1,
        max_memory_bytes=64 * GIB,
        resource_sampler=sampler,
        poll_seconds=0.01,
        host_poll_seconds=0.02,
        heartbeat_seconds=60,
    )
    assert summary["resource_mode"] == "auto"
    assert summary["host_poll_seconds"] == 0.02
    assert summary["host_sample_count"] >= 2
    assert summary["light_sample_count"] == 1
    assert summary["detailed_sample_count"] == 0


def test_light_sampler_reports_the_current_process_tree_rss():
    sample = LinuxResourceSampler().sample_light([os.getpid()])
    assert sample.sample_kind == "light"
    assert sample.root_rss_bytes[os.getpid()] > 0
