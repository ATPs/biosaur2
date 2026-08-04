"""Small multiprocessing helpers used by the feature workflow."""

from __future__ import annotations

import multiprocessing
import math
import os
import queue
import traceback
import logging
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerFailure:
    worker_id: int
    exception_type: str
    message: str
    traceback_text: str


class WorkerProcessError(RuntimeError):
    """Raised when a multiprocessing child fails."""

    def __init__(self, failure: WorkerFailure):
        self.failure = failure
        super().__init__(
            "Worker %d failed with %s: %s\n%s"
            % (
                failure.worker_id,
                failure.exception_type,
                failure.message,
                failure.traceback_text,
            )
        )


GIB = 1024 ** 3


@dataclass(frozen=True)
class ResourceSample:
    """A point-in-time view of the manager's child process tree."""

    process_cpu_cores: float
    host_busy_cores: float
    process_pss_bytes: int
    mem_available_bytes: int
    physical_memory_bytes: int
    cpu_count: int


def _read_proc_stat(pid):
    """Return ``(ppid, cpu_ticks)`` for a Linux process, or ``None``."""

    try:
        payload = Path("/proc") / str(pid) / "stat"
        text = payload.read_text(encoding="utf-8")
        fields = text[text.rfind(")") + 2:].split()
        return int(fields[1]), int(fields[11]) + int(fields[12])
    except (FileNotFoundError, IndexError, PermissionError, ProcessLookupError, OSError, ValueError):
        return None


def _read_proc_memory_bytes(pid):
    """Read proportional memory where Linux exposes it, falling back to RSS."""

    rollup = Path("/proc") / str(pid) / "smaps_rollup"
    status = Path("/proc") / str(pid) / "status"
    for path, label in ((rollup, "Pss:"), (status, "VmRSS:")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith(label):
                    return int(line.split()[1]) * 1024
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
            continue
    return 0


def _read_meminfo():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0]) * 1024
    except (FileNotFoundError, ValueError):
        return 0, 0
    return values.get("MemTotal", 0), values.get("MemAvailable", 0)


def physical_memory_bytes():
    """Return usable physical RAM, deliberately excluding swap."""

    total, _available = _read_meminfo()
    try:
        limit = (Path("/sys/fs/cgroup/memory.max").read_text().strip())
        if limit != "max":
            total = min(total, int(limit)) if total else int(limit)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return total


class LinuxResourceSampler:
    """Sample CPU and memory for a set of manager-owned process roots."""

    def __init__(self):
        self._previous_time = None
        self._previous_process_ticks = None
        self._previous_host = None
        self._ticks_per_second = max(1, os.sysconf("SC_CLK_TCK"))

    @staticmethod
    def _host_counters():
        try:
            fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
            values = [int(value) for value in fields]
            total = sum(values)
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            return total, idle
        except (FileNotFoundError, IndexError, ValueError):
            return 0, 0

    @staticmethod
    def _descendants(root_pids):
        stats = {}
        try:
            entries = list(Path("/proc").iterdir())
        except FileNotFoundError:
            return set()
        for entry in entries:
            if not entry.name.isdigit():
                continue
            record = _read_proc_stat(int(entry.name))
            if record is not None:
                stats[int(entry.name)] = record
        wanted = {int(pid) for pid in root_pids if int(pid) in stats}
        changed = True
        while changed:
            changed = False
            for pid, (parent, _ticks) in stats.items():
                if parent in wanted and pid not in wanted:
                    wanted.add(pid)
                    changed = True
        return wanted, stats

    def sample(self, root_pids):
        now = time.monotonic()
        descendants, stats = self._descendants(root_pids)
        process_ticks = sum(stats[pid][1] for pid in descendants)
        host_total, host_idle = self._host_counters()
        cpu_count = max(1, os.cpu_count() or 1)
        total, available = _read_meminfo()
        physical = physical_memory_bytes() or total
        process_cpu = 0.0
        host_busy = 0.0
        if self._previous_time is not None:
            elapsed = max(now - self._previous_time, 1e-6)
            process_cpu = max(
                0.0,
                (process_ticks - self._previous_process_ticks)
                / self._ticks_per_second
                / elapsed,
            )
            previous_total, previous_idle = self._previous_host
            total_delta = host_total - previous_total
            idle_delta = host_idle - previous_idle
            if total_delta > 0:
                host_busy = cpu_count * max(0.0, 1.0 - idle_delta / total_delta)
        self._previous_time = now
        self._previous_process_ticks = process_ticks
        self._previous_host = (host_total, host_idle)
        return ResourceSample(
            process_cpu_cores=process_cpu,
            host_busy_cores=host_busy,
            process_pss_bytes=sum(_read_proc_memory_bytes(pid) for pid in descendants),
            mem_available_bytes=available,
            physical_memory_bytes=physical,
            cpu_count=cpu_count,
        )


class _MemoryEstimator:
    """Conservative stage-local peak-memory estimator."""

    def __init__(self, default_bytes=16 * GIB):
        self.default_bytes = default_bytes
        self.completed = []

    @property
    def estimate_bytes(self):
        if len(self.completed) < 4:
            return self.default_bytes
        ordered = sorted(self.completed)
        index = min(len(ordered) - 1, int(math.ceil(len(ordered) * 0.9)) - 1)
        return max(4 * GIB, int(ordered[index] * 1.2))

    def observe(self, result):
        if not isinstance(result, dict):
            return
        peak_kib = result.get("peak_rss_kib")
        if peak_kib is not None:
            self.completed.append(max(1, int(peak_kib)) * 1024)


def balanced_ranges(
    item_count: int,
    requested_workers: int,
    cpu_count_value: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Return balanced, non-empty half-open ranges covering every item once."""

    if item_count < 0:
        raise ValueError("item_count must be nonnegative")
    if requested_workers < 1:
        raise ValueError("requested_workers must be a positive integer")
    if item_count == 0:
        return []

    available_cpus = cpu_count_value
    if available_cpus is None:
        available_cpus = os.cpu_count() or 1
    if available_cpus < 1:
        available_cpus = 1

    worker_count = min(requested_workers, available_cpus, item_count)
    base_size, remainder = divmod(item_count, worker_count)
    ranges = []
    start = 0
    for worker_index in range(worker_count):
        size = base_size + (1 if worker_index < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def effective_worker_budget(requested_workers, cpu_count_value=None):
    """Return a positive CPU-bounded worker budget."""

    if requested_workers < 1:
        raise ValueError("requested_workers must be a positive integer")
    available = cpu_count_value if cpu_count_value is not None else os.cpu_count()
    return min(requested_workers, max(1, available or 1))


def worker_slot_allocations(total_workers, task_count, target_per_task=4):
    """Split a total budget across reusable run slots."""

    if total_workers < 1:
        raise ValueError("total_workers must be a positive integer")
    if task_count < 0:
        raise ValueError("task_count must be nonnegative")
    if target_per_task < 1:
        raise ValueError("target_per_task must be a positive integer")
    if task_count == 0:
        return []
    slot_count = min(
        task_count,
        max(1, int(math.ceil(total_workers / target_per_task))),
    )
    base, remainder = divmod(total_workers, slot_count)
    return [base + (1 if index < remainder else 0) for index in range(slot_count)]


def _worker_entry(
    result_queue: multiprocessing.Queue,
    worker_id: int,
    function: Callable[..., Any],
    args: Sequence[Any],
) -> None:
    try:
        result = function(*args)
    except BaseException as exc:  # child boundary: forward full failure context
        result_queue.put(
            (
                "error",
                worker_id,
                WorkerFailure(
                    worker_id=worker_id,
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    traceback_text=traceback.format_exc(),
                ),
            )
        )
        return
    result_queue.put(("ok", worker_id, result))


def run_process_tasks(
    function: Callable[..., Any],
    task_args: Iterable[Sequence[Any]],
    poll_seconds: float = 0.1,
) -> List[Any]:
    """Run one process per task and return results in task order.

    A child-reported exception or an unexpected nonzero exit terminates the
    remaining children and raises in the parent. Polling process exit codes
    prevents an unbounded queue wait if a worker is killed before reporting.
    """

    tasks = list(task_args)
    if not tasks:
        return []

    result_queue = multiprocessing.Queue()
    processes = []
    results = {}
    try:
        for worker_id, args in enumerate(tasks):
            process = multiprocessing.Process(
                target=_worker_entry,
                args=(result_queue, worker_id, function, tuple(args)),
            )
            process.start()
            processes.append(process)

        while len(results) < len(processes):
            try:
                status, worker_id, payload = result_queue.get(
                    timeout=poll_seconds
                )
            except queue.Empty:
                failed = [
                    (worker_id, process.exitcode)
                    for worker_id, process in enumerate(processes)
                    if process.exitcode not in (None, 0) and worker_id not in results
                ]
                if failed:
                    worker_id, exit_code = failed[0]
                    raise WorkerProcessError(
                        WorkerFailure(
                            worker_id=worker_id,
                            exception_type="ProcessExit",
                            message="child exited with code %s before reporting"
                            % exit_code,
                            traceback_text="",
                        )
                    )
                continue

            if status == "error":
                raise WorkerProcessError(payload)
            results[worker_id] = payload

        return [results[worker_id] for worker_id in range(len(processes))]
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in processes:
            process.join()
        result_queue.close()
        result_queue.join_thread()


def run_bounded_process_tasks(
    function: Callable[..., Any],
    task_args: Iterable[Sequence[Any]],
    max_workers: int,
    stop_on_result: Optional[Callable[[Any], bool]] = None,
    poll_seconds: float = 0.1,
) -> Tuple[Dict[int, Any], List[int]]:
    """Run one spawned process per task with a bounded active set.

    ``task_args`` is consumed only as process slots become available. A worker
    never processes a second task, so its entire file-level memory footprint is
    released on exit. Returned keys are the input task order indexes.
    """

    if max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    task_iterator = enumerate(task_args)
    active = {}
    results = {}
    started = []
    stop_submitting = False

    def start_next():
        try:
            worker_id, args = next(task_iterator)
        except StopIteration:
            return False
        process = context.Process(
            target=_worker_entry,
            args=(result_queue, worker_id, function, tuple(args)),
        )
        process.start()
        active[worker_id] = process
        started.append(worker_id)
        return True

    try:
        while len(active) < max_workers and start_next():
            pass
        while active:
            try:
                status, worker_id, payload = result_queue.get(timeout=poll_seconds)
            except queue.Empty:
                failed = [
                    (worker_id, process.exitcode)
                    for worker_id, process in active.items()
                    if process.exitcode not in (None, 0)
                ]
                if not failed:
                    continue
                worker_id, exit_code = failed[0]
                payload = WorkerFailure(
                    worker_id=worker_id,
                    exception_type="ProcessExit",
                    message="child exited with code %s before reporting" % exit_code,
                    traceback_text="",
                )
                status = "error"

            process = active.pop(worker_id, None)
            if process is None:
                continue
            process.join()
            result = payload if status == "ok" else payload
            results[worker_id] = result
            if status == "error" or (
                stop_on_result is not None and stop_on_result(result)
            ):
                stop_submitting = True
            if not stop_submitting:
                while len(active) < max_workers and start_next():
                    pass
        return results, started
    finally:
        for process in active.values():
            process.join()
        result_queue.close()
        result_queue.join_thread()


def run_budgeted_process_tasks(
    function: Callable[..., Any],
    task_args: Iterable[Sequence[Any]],
    total_workers: int,
    stop_on_result: Optional[Callable[[Any], bool]] = None,
    target_per_task: int = 4,
    poll_seconds: float = 0.1,
) -> Tuple[Dict[int, Any], List[int], List[int]]:
    """Run fresh task processes while sharing one total worker budget.

    The allocated worker count is appended to each task's positional arguments.
    A completed task's allocation is handed to the next pending task.
    """

    tasks = list(task_args)
    allocations = worker_slot_allocations(
        total_workers, len(tasks), target_per_task=target_per_task
    )
    if not tasks:
        return {}, [], allocations
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    task_iterator = enumerate(tasks)
    active = {}
    results = {}
    started = []
    stop_submitting = False

    def start_next(allocation):
        try:
            worker_id, args = next(task_iterator)
        except StopIteration:
            return False
        process = context.Process(
            target=_worker_entry,
            args=(
                result_queue,
                worker_id,
                function,
                tuple(args) + (allocation,),
            ),
        )
        process.start()
        active[worker_id] = (process, allocation)
        started.append(worker_id)
        return True

    try:
        for allocation in allocations:
            if not start_next(allocation):
                break
        while active:
            try:
                status, worker_id, payload = result_queue.get(timeout=poll_seconds)
            except queue.Empty:
                failed = [
                    (worker_id, process.exitcode)
                    for worker_id, (process, _allocation) in active.items()
                    if process.exitcode not in (None, 0)
                ]
                if not failed:
                    continue
                worker_id, exit_code = failed[0]
                payload = WorkerFailure(
                    worker_id=worker_id,
                    exception_type="ProcessExit",
                    message="child exited with code %s before reporting" % exit_code,
                    traceback_text="",
                )
                status = "error"

            active_value = active.pop(worker_id, None)
            if active_value is None:
                continue
            process, allocation = active_value
            process.join()
            result = payload
            results[worker_id] = result
            if status == "error" or (
                stop_on_result is not None and stop_on_result(result)
            ):
                stop_submitting = True
            if not stop_submitting:
                start_next(allocation)
        return results, started, allocations
    finally:
        for process, _allocation in active.values():
            if process.is_alive():
                process.terminate()
            process.join()
        result_queue.close()
        result_queue.join_thread()


def run_adaptive_process_tasks(
    function: Callable[..., Any],
    task_args: Iterable[Sequence[Any]],
    target_workers: int,
    max_memory_bytes: int,
    stop_on_result: Optional[Callable[[Any], bool]] = None,
    on_result: Optional[Callable[[int, Any], None]] = None,
    on_start: Optional[Callable[[int, Sequence[Any], int], None]] = None,
    poll_seconds: float = 5.0,
    resource_sampler=None,
) -> Tuple[Dict[int, Any], List[int], Dict[str, Any]]:
    """Run fresh file tasks with CPU/memory-aware bounded soft overcommit.

    The initial cohort preserves the existing four-workers-per-run behavior.
    When its measured CPU use remains below the requested target, the manager
    adds one-worker tasks.  Their declared allocation is capped at 1.5 times
    the requested target, so temporary phase overlap is bounded rather than an
    unbounded response to a serial stage.
    """

    if target_workers < 1:
        raise ValueError("target_workers must be a positive integer")
    if max_memory_bytes < 1:
        raise ValueError("max_memory_bytes must be positive")
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    task_iterator = enumerate(task_args)
    next_task = None
    next_task_ready = False
    active = {}
    results = {}
    started = []
    stop_submitting = False
    exhausted = False
    initial_slots = max(1, int(math.ceil(target_workers / 4.0)))
    initial_started = 0
    allocation_ceiling = min(
        max(1, os.cpu_count() or 1),
        int(math.ceil(target_workers * 1.5)),
    )
    estimator = _MemoryEstimator()
    sampler = resource_sampler or LinuxResourceSampler()
    cpu_history = []
    summary = {
        "target_workers": target_workers,
        "max_memory_bytes": max_memory_bytes,
        "allocation_ceiling": allocation_ceiling,
        "peak_active_tasks": 0,
        "peak_allocated_workers": 0,
        "peak_process_cpu_cores": 0.0,
        "peak_process_pss_bytes": 0,
        "cpu_pause_count": 0,
        "memory_pause_count": 0,
        "memory_wait_seconds": 0.0,
        "last_wait_reason": None,
    }

    def allocation_total():
        return sum(allocation for _process, allocation in active.values())

    def sample():
        value = sampler.sample(
            [process.pid for process, _allocation in active.values()]
        )
        if active:
            cpu_history.append(value.process_cpu_cores)
            del cpu_history[:-3]
        summary["peak_process_cpu_cores"] = max(
            summary["peak_process_cpu_cores"], value.process_cpu_cores
        )
        summary["peak_process_pss_bytes"] = max(
            summary["peak_process_pss_bytes"], value.process_pss_bytes
        )
        return value

    def ensure_next_task():
        nonlocal exhausted, next_task, next_task_ready
        if next_task_ready or exhausted:
            return not exhausted
        try:
            next_task = next(task_iterator)
            next_task_ready = True
            return True
        except StopIteration:
            exhausted = True
            return False

    def may_start(allocation, resource, *, bootstrap):
        if allocation_total() + allocation > allocation_ceiling:
            return False, "allocation"
        estimate = estimator.estimate_bytes
        predicted = max(
            resource.process_pss_bytes if resource is not None else 0,
            len(active) * estimate,
        ) + estimate
        available = resource.mem_available_bytes
        physical = resource.physical_memory_bytes
        reserve = max(8 * GIB, int(physical * 0.05))
        if predicted > max_memory_bytes or available < estimate + reserve:
            summary["memory_pause_count"] += 1
            return False, "memory"
        if bootstrap:
            return True, "ok"
        if not active:
            return True, "ok"
        if resource is None or len(cpu_history) < 3:
            return False, "warming_up"
        median_cpu = sorted(cpu_history)[len(cpu_history) // 2]
        if median_cpu >= target_workers * 0.90:
            summary["cpu_pause_count"] += 1
            return False, "cpu"
        if resource.host_busy_cores >= resource.cpu_count * 0.95:
            summary["cpu_pause_count"] += 1
            return False, "host_cpu"
        return True, "ok"

    def start_next(allocation, *, bootstrap):
        nonlocal initial_started, next_task, next_task_ready
        if not ensure_next_task():
            return False
        worker_id, args = next_task
        if on_start is not None:
            on_start(worker_id, args, allocation)
        process = context.Process(
            target=_worker_entry,
            args=(
                result_queue,
                worker_id,
                function,
                tuple(args) + (allocation,),
            ),
        )
        process.start()
        next_task = None
        next_task_ready = False
        active[worker_id] = (process, allocation)
        started.append(worker_id)
        if bootstrap:
            initial_started += 1
        summary["peak_active_tasks"] = max(summary["peak_active_tasks"], len(active))
        summary["peak_allocated_workers"] = max(
            summary["peak_allocated_workers"], allocation_total()
        )
        return True

    def fill(resource):
        launches = 0
        while ensure_next_task() and not stop_submitting:
            bootstrap = initial_started < initial_slots
            allocation = min(4, target_workers) if bootstrap else 1
            allowed, _reason = may_start(allocation, resource, bootstrap=bootstrap)
            if not allowed:
                break
            if not start_next(allocation, bootstrap=bootstrap):
                break
            launches += 1
            if not bootstrap and launches >= 4:
                break

    try:
        resource = sample()
        if estimator.estimate_bytes > max_memory_bytes:
            raise ValueError(
                "--max-memory is below the initial per-run admission estimate "
                "of %.1f GiB" % (estimator.estimate_bytes / float(GIB))
            )
        while active or ensure_next_task():
            resource = sample()
            fill(resource)
            if not active:
                if not exhausted and not stop_submitting:
                    summary["memory_wait_seconds"] += poll_seconds
                    summary["last_wait_reason"] = "memory"
                    if int(summary["memory_wait_seconds"]) % 60 < poll_seconds:
                        logger.info(
                            "Project scheduler waiting for memory: available=%.1f GiB",
                            resource.mem_available_bytes / float(GIB),
                        )
                    time.sleep(poll_seconds)
                    continue
                break
            try:
                status, worker_id, payload = result_queue.get(timeout=poll_seconds)
            except queue.Empty:
                failed = [
                    (task_id, process.exitcode)
                    for task_id, (process, _allocation) in active.items()
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    worker_id, exit_code = failed[0]
                    status = "error"
                    payload = WorkerFailure(
                        worker_id=worker_id,
                        exception_type="ProcessExit",
                        message="child exited with code %s before reporting" % exit_code,
                        traceback_text="",
                    )
                else:
                    continue
            active_value = active.pop(worker_id, None)
            if active_value is None:
                continue
            process, _allocation = active_value
            process.join()
            result = payload
            results[worker_id] = result
            estimator.observe(result)
            if on_result is not None:
                on_result(worker_id, result)
            if status == "error" or (
                stop_on_result is not None and stop_on_result(result)
            ):
                stop_submitting = True
            if not stop_submitting:
                fill(sample())
        return results, started, summary
    finally:
        for process, _allocation in active.values():
            if process.is_alive():
                process.terminate()
            process.join()
        result_queue.close()
        result_queue.join_thread()
