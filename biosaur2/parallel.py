"""Small multiprocessing helpers used by the feature workflow."""

from __future__ import annotations

import multiprocessing
import math
import os
import queue
import signal
import traceback
import logging
from collections import deque
from dataclasses import dataclass, field
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
    root_pss_bytes: Dict[int, int] = field(default_factory=dict)
    root_rss_bytes: Dict[int, int] = field(default_factory=dict)
    process_thread_count: int = 0
    sample_kind: str = "detailed"


def _read_proc_stat(pid):
    """Return parent, CPU ticks, RSS bytes and threads for one Linux PID."""

    try:
        payload = Path("/proc") / str(pid) / "stat"
        text = payload.read_text(encoding="utf-8")
        fields = text[text.rfind(")") + 2:].split()
        return (
            int(fields[1]),
            int(fields[11]) + int(fields[12]),
            int(fields[21]) * os.sysconf("SC_PAGE_SIZE"),
            int(fields[17]),
        )
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


def _read_proc_children(pid):
    """Return direct children without enumerating unrelated host processes."""

    try:
        text = (Path("/proc") / str(pid) / "task" / str(pid) / "children").read_text(
            encoding="utf-8"
        )
        return [int(value) for value in text.split()]
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
        return []


def _read_meminfo():
    values = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.split()[0]) * 1024
    except (FileNotFoundError, ValueError):
        return 0, 0
    return values.get("MemTotal", 0), values.get("MemAvailable", 0)


def _cgroup_memory_available_bytes():
    """Return this cgroup's remaining memory when a finite limit is present."""

    try:
        limit_text = (Path("/sys/fs/cgroup/memory.max").read_text()).strip()
        if limit_text == "max":
            return 0
        limit = int(limit_text)
        current = int((Path("/sys/fs/cgroup/memory.current").read_text()).strip())
        return max(0, limit - current)
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return 0


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
        self._previous_pid_ticks = {}
        self._previous_host = None
        self._ticks_per_second = max(1, os.sysconf("SC_CLK_TCK"))
        self._physical_memory_bytes = physical_memory_bytes()

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
            for pid, (parent, _ticks, _rss, _threads) in stats.items():
                if parent in wanted and pid not in wanted:
                    wanted.add(pid)
                    changed = True
        return wanted, stats

    @staticmethod
    def _owned_descendants(root_pids):
        """Walk only the manager-owned tree using Linux ``children`` files."""

        stats = {}
        owners = {}
        pending = deque((int(pid), int(pid)) for pid in root_pids)
        seen = set()
        while pending:
            pid, owner = pending.popleft()
            if pid in seen:
                continue
            seen.add(pid)
            record = _read_proc_stat(pid)
            if record is None:
                continue
            stats[pid] = record
            owners[pid] = owner
            pending.extend((child, owner) for child in _read_proc_children(pid))
        return set(stats), stats, owners

    def _sample_host(self, *, include_cpu=False, include_cgroup=False):
        now = time.monotonic()
        total, available = _read_meminfo()
        host_busy = 0.0
        if include_cgroup:
            cgroup_available = _cgroup_memory_available_bytes()
            if cgroup_available:
                available = min(available, cgroup_available) if available else cgroup_available
        if include_cpu:
            host_total, host_idle = self._host_counters()
            if self._previous_host is not None:
                previous_total, previous_idle = self._previous_host
                total_delta = host_total - previous_total
                idle_delta = host_idle - previous_idle
                if total_delta > 0:
                    host_busy = max(1, os.cpu_count() or 1) * max(
                        0.0, 1.0 - idle_delta / total_delta
                    )
            self._previous_host = (host_total, host_idle)
        return now, ResourceSample(
            process_cpu_cores=0.0,
            host_busy_cores=host_busy,
            process_pss_bytes=0,
            mem_available_bytes=available,
            physical_memory_bytes=self._physical_memory_bytes or total,
            cpu_count=max(1, os.cpu_count() or 1),
            sample_kind="host",
        )

    def sample_host(self):
        """Read only host available memory for the five-second fast path."""

        _now, value = self._sample_host()
        return value

    def _finish_process_sample(self, now, value, descendants, stats, owners, memory_reader, kind):
        pid_ticks = {pid: stats[pid][1] for pid in descendants}
        process_cpu = 0.0
        if self._previous_time is not None:
            elapsed = max(now - self._previous_time, 1e-6)
            process_cpu = (
                sum(
                    max(0, ticks - self._previous_pid_ticks.get(pid, ticks))
                    for pid, ticks in pid_ticks.items()
                )
                / self._ticks_per_second
                / elapsed
            )
        self._previous_time = now
        self._previous_pid_ticks = pid_ticks
        root_memory = {}
        process_threads = 0
        for pid in descendants:
            owner = owners.get(pid)
            if owner is None:
                continue
            root_memory[owner] = root_memory.get(owner, 0) + memory_reader(pid, stats[pid])
            process_threads += stats[pid][3]
        return ResourceSample(
            process_cpu_cores=process_cpu,
            host_busy_cores=value.host_busy_cores,
            process_pss_bytes=sum(root_memory.values()),
            mem_available_bytes=value.mem_available_bytes,
            physical_memory_bytes=value.physical_memory_bytes,
            cpu_count=value.cpu_count,
            root_pss_bytes=root_memory if kind == "detailed" else {},
            root_rss_bytes=root_memory if kind == "light" else {},
            process_thread_count=process_threads,
            sample_kind=kind,
        )

    def sample_light(self, root_pids):
        """Read RSS for only manager-owned descendants, not all host PIDs."""

        now, value = self._sample_host(include_cpu=True, include_cgroup=True)
        descendants, stats, owners = self._owned_descendants(root_pids)
        return self._finish_process_sample(
            now, value, descendants, stats, owners,
            lambda _pid, record: record[2], "light",
        )

    def sample(self, root_pids):
        """Return an expensive complete PSS sample for detailed mode."""

        now, value = self._sample_host(include_cpu=True, include_cgroup=True)
        descendants, stats = self._descendants(root_pids)
        root_set = {int(pid) for pid in root_pids}

        def root_owner(pid):
            seen = set()
            while pid in stats and pid not in seen:
                if pid in root_set:
                    return pid
                seen.add(pid)
                pid = stats[pid][0]
            return None

        owners = {pid: root_owner(pid) for pid in descendants}
        return self._finish_process_sample(
            now, value, descendants, stats, owners,
            lambda pid, _record: _read_proc_memory_bytes(pid), "detailed",
        )


class _MemoryEstimator:
    """Conservative stage-local peak-memory estimator."""

    def __init__(self, default_bytes=16 * GIB):
        self.default_bytes = default_bytes
        self.completed = {}

    @staticmethod
    def _clamp(value):
        return min(30 * GIB, max(1 * GIB, int(value)))

    def estimate_bytes(self, allocation=4):
        allocation = max(1, int(allocation))
        values = self.completed.get(allocation, ())
        if values:
            if len(values) < 4:
                return self._clamp(max(values) * 1.2)
            ordered = sorted(values)
            index = min(len(ordered) - 1, int(math.ceil(len(ordered) * 0.9)) - 1)
            return self._clamp(ordered[index] * 1.2)
        lower = [value for value in self.completed if value < allocation and self.completed[value]]
        if lower:
            source = max(lower)
            return self._clamp(self.estimate_bytes(source) * allocation / float(source))
        return self._clamp(self.default_bytes)

    def speculative_estimate_bytes(self, allocation=4):
        allocation = max(1, int(allocation))
        values = self.completed.get(allocation, ())
        if len(values) < 4:
            lower = [value for value in self.completed if value < allocation and self.completed[value]]
            if not lower:
                return self.estimate_bytes(allocation)
            source = max(lower)
            source_values = sorted(self.completed[source])
            median = source_values[len(source_values) // 2]
            # Larger per-run pools often share input/output and have a
            # sublinear PSS increase. A hard-limit breach remains preemptible.
            return max(4 * GIB, int(median * 1.35))
        ordered = sorted(values)
        median = ordered[len(ordered) // 2]
        return max(4 * GIB, int(median * 1.1))

    def observe(self, allocation, peak_pss_bytes, result):
        observed = int(peak_pss_bytes or 0)
        if not observed and isinstance(result, dict):
            observed = max(1, int(result.get("peak_rss_kib") or 0)) * 1024
        if observed:
            self.completed.setdefault(max(1, int(allocation)), []).append(
                self._clamp(observed)
            )

    def seed(self, observations):
        for allocation, peak_bytes in observations:
            if peak_bytes:
                self.completed.setdefault(max(1, int(allocation)), []).append(
                    self._clamp(peak_bytes)
                )


def _process_tree_pids(root_pid):
    descendants, _stats = LinuxResourceSampler._descendants([root_pid])
    return descendants | {int(root_pid)}


def terminate_process_tree(root_pid, grace_seconds=10.0):
    """Terminate a manager worker and every currently reachable descendant."""

    pids = _process_tree_pids(root_pid)
    for pid in sorted(pids - {root_pid}, reverse=True):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        os.kill(root_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        alive = set()
        for pid in pids:
            try:
                state = Path("/proc") / str(pid) / "stat"
                fields = state.read_text(encoding="utf-8").split(")", 1)[1].split()
                if fields and fields[0] != "Z":
                    alive.add(pid)
            except ProcessLookupError:
                pass
            except (FileNotFoundError, IndexError, OSError):
                pass
        if not alive:
            return
        time.sleep(0.1)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _adaptive_worker_entry(result_queue, worker_id, generation, function, args):
    try:
        result = function(*args)
    except BaseException as exc:
        result_queue.put((
            "error", worker_id, generation,
            WorkerFailure(worker_id, type(exc).__name__, str(exc), traceback.format_exc()),
        ))
        return
    result_queue.put(("ok", worker_id, generation, result))


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
    """Return a positive worker budget capped at three times logical CPU capacity."""

    if requested_workers < 1:
        raise ValueError("requested_workers must be a positive integer")
    available = cpu_count_value if cpu_count_value is not None else os.cpu_count()
    return min(requested_workers, max(1, available or 1) * 3)


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


def run_adaptive_process_tasks(
    function: Callable[..., Any],
    task_args: Iterable[Sequence[Any]],
    target_workers: int,
    max_memory_bytes: int,
    stop_on_result: Optional[Callable[[Any], bool]] = None,
    on_result: Optional[Callable[[int, Any], None]] = None,
    on_start: Optional[Callable[[int, Sequence[Any], int], None]] = None,
    poll_seconds: float = 0.5,
    heartbeat_seconds: float = 60.0,
    resource_sampler=None,
    resource_mode: str = "auto",
    host_poll_seconds: float = 5.0,
    initial_memory_observations: Iterable[Tuple[int, int]] = (),
    on_preempt: Optional[Callable[[int, Sequence[Any], int, int], None]] = None,
) -> Tuple[Dict[int, Any], List[int], Dict[str, Any]]:
    """Schedule Project tasks with fast host-memory polling and recovery."""

    if target_workers < 1:
        raise ValueError("target_workers must be a positive integer")
    if max_memory_bytes < 1:
        raise ValueError("max_memory_bytes must be positive")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    if host_poll_seconds <= 0:
        raise ValueError("host_poll_seconds must be positive")
    if resource_mode not in {"auto", "detailed"}:
        raise ValueError("resource_mode must be 'auto' or 'detailed'")
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    task_iterator = enumerate(task_args)
    active = {}
    retry_queue = deque()
    generations = {}
    results = {}
    started = []
    exhausted = False
    stop_submitting = False
    cooldown_until = 0.0
    allocation_ceiling = min(
        max(1, os.cpu_count() or 1) * 3,
        int(math.ceil(target_workers * 1.5)),
    )
    sampler = resource_sampler or LinuxResourceSampler()
    legacy_sampler = resource_sampler is not None and not hasattr(sampler, "sample_host")
    estimator = _MemoryEstimator(16 * GIB)
    estimator.seed(initial_memory_observations)
    cpu_history = []
    sample_sequence = 0
    unobserved_reservation_bytes = 0
    speculative_starts_in_sample = 0
    last_tree_sample = None
    summary = {
        "target_workers": target_workers,
        "max_memory_bytes": max_memory_bytes,
        "heartbeat_seconds": heartbeat_seconds,
        "host_poll_seconds": host_poll_seconds,
        "resource_mode": resource_mode,
        "allocation_ceiling": allocation_ceiling,
        "peak_active_tasks": 0,
        "peak_allocated_workers": 0,
        "peak_normal_allocated_workers": 0,
        "peak_speculative_tasks": 0,
        "peak_overcommit_tasks": 0,
        "peak_scaled_tasks": 0,
        "peak_process_cpu_cores": 0.0,
        "peak_process_pss_bytes": 0,
        "peak_process_thread_count": 0,
        "peak_process_rss_bytes": 0,
        "peak_unobserved_reservation_bytes": 0,
        "cpu_pause_count": 0,
        "memory_pause_count": 0,
        "observed_memory_admission_count": 0,
        "speculative_admission_batches": 0,
        "peak_speculative_starts_per_sample": 0,
        "memory_preemption_count": 0,
        "memory_requeue_count": 0,
        "completion_batch_count": 0,
        "peak_completion_batch_size": 0,
        "host_sample_count": 0,
        "light_sample_count": 0,
        "detailed_sample_count": 0,
        "host_memory_floor_bytes": 0,
        "heartbeat_count": 0,
        "memory_wait_seconds": 0.0,
        "last_wait_reason": None,
        "last_admission_reason": None,
        "resource_samples": [],
    }

    def allocation_total():
        return sum(record["allocation"] for record in active.values())

    def normal_allocation_total():
        return sum(
            record["allocation"]
            for record in active.values()
            if record["tier"] != "cpu_overcommit"
        )

    def reserve(value):
        return max(8 * GIB, int(value.physical_memory_bytes * 0.05))

    def host_memory_floor(value):
        configured = max(0, value.physical_memory_bytes - min(
            max_memory_bytes, value.physical_memory_bytes
        ))
        floor = max(reserve(value), configured)
        summary["host_memory_floor_bytes"] = max(
            summary["host_memory_floor_bytes"], floor
        )
        return floor

    def median_cpu():
        if len(cpu_history) < 3:
            return None
        return sorted(cpu_history)[len(cpu_history) // 2]

    def root_pids():
        return [record["process"].pid for record in active.values()]

    def record_sample(event, value, reset_admission_window=False):
        nonlocal sample_sequence, speculative_starts_in_sample, unobserved_reservation_bytes, last_tree_sample
        sample_sequence += 1
        if reset_admission_window:
            speculative_starts_in_sample = 0
            unobserved_reservation_bytes = 0
        if value.sample_kind == "host":
            summary["host_sample_count"] += 1
        elif value.sample_kind == "light":
            summary["light_sample_count"] += 1
        else:
            summary["detailed_sample_count"] += 1
        if value.sample_kind != "host":
            last_tree_sample = value
        if active and value.sample_kind != "host":
            cpu_history.append(value.process_cpu_cores)
            del cpu_history[:-5]
        for record in active.values():
            pid = record["process"].pid
            if value.sample_kind == "detailed":
                observed = value.root_pss_bytes.get(pid)
                if observed is not None:
                    record["peak_pss_bytes"] = max(record["peak_pss_bytes"], observed)
                    record["observed_memory_bytes"] = observed
            elif value.sample_kind == "light":
                observed = value.root_rss_bytes.get(pid)
                if observed is not None:
                    record["observed_memory_bytes"] = observed
                else:
                    # A raced or unreadable /proc entry must not reduce the
                    # reservation for a still-active task.
                    record["observed_memory_bytes"] = 0
        summary["peak_process_cpu_cores"] = max(
            summary["peak_process_cpu_cores"], value.process_cpu_cores
        )
        summary["peak_process_pss_bytes"] = max(
            summary["peak_process_pss_bytes"], value.process_pss_bytes
        )
        summary["peak_process_thread_count"] = max(
            summary["peak_process_thread_count"], value.process_thread_count
        )
        if value.sample_kind == "light":
            summary["peak_process_rss_bytes"] = max(
                summary["peak_process_rss_bytes"], value.process_pss_bytes
            )
        summary["resource_samples"].append(
            {
                "sequence": sample_sequence,
                "event": event,
                "kind": value.sample_kind,
                "active_tasks": len(active),
                "allocated_workers": allocation_total(),
                "process_cpu_cores": value.process_cpu_cores,
                "host_busy_cores": value.host_busy_cores,
                "process_pss_bytes": value.process_pss_bytes,
                "mem_available_bytes": value.mem_available_bytes,
                "process_thread_count": value.process_thread_count,
            }
        )
        del summary["resource_samples"][:-256]
        return value

    def sample_host(event):
        if legacy_sampler:
            value = sampler.sample(root_pids())
        elif resource_mode == "detailed" and last_tree_sample is not None:
            host = sampler.sample_host()
            value = ResourceSample(
                process_cpu_cores=last_tree_sample.process_cpu_cores,
                host_busy_cores=host.host_busy_cores,
                process_pss_bytes=last_tree_sample.process_pss_bytes,
                mem_available_bytes=host.mem_available_bytes,
                physical_memory_bytes=host.physical_memory_bytes,
                cpu_count=host.cpu_count,
                root_pss_bytes=last_tree_sample.root_pss_bytes,
                process_thread_count=last_tree_sample.process_thread_count,
                sample_kind="host",
            )
        elif resource_mode == "detailed":
            value = sampler.sample(root_pids())
        else:
            value = sampler.sample_host()
        return record_sample(event, value)

    def sample_heartbeat(event):
        if legacy_sampler or resource_mode == "detailed":
            value = sampler.sample(root_pids())
        else:
            value = sampler.sample_light(root_pids())
        return record_sample(event, value, reset_admission_window=True)

    def pull_task():
        nonlocal exhausted
        if retry_queue:
            return retry_queue.popleft()
        if exhausted:
            return None
        try:
            worker_id, args = next(task_iterator)
        except StopIteration:
            exhausted = True
            return None
        return worker_id, args, 0

    def memory_allowed(allocation, value, speculative=False):
        estimate = estimator.estimate_bytes(allocation)
        if resource_mode == "auto" and not legacy_sampler:
            remaining_growth = sum(
                max(
                    0,
                    estimator.estimate_bytes(record["allocation"])
                    - record["observed_memory_bytes"],
                )
                for record in active.values()
            )
            return value.mem_available_bytes >= (
                host_memory_floor(value) + remaining_growth + estimate
            )
        if speculative:
            # Completed peaks protect the next small batch while current PSS
            # represents all already-observed active process trees.
            predicted = value.process_pss_bytes + unobserved_reservation_bytes + estimate
            limit = int(max_memory_bytes * 0.90)
            return (
                predicted <= limit
                and value.mem_available_bytes
                >= reserve(value) + unobserved_reservation_bytes + estimate
            )
        predicted = max(value.process_pss_bytes, len(active) * estimate) + estimate
        return (
            predicted <= max_memory_bytes
            and value.mem_available_bytes >= reserve(value) + estimate
        )

    def cpu_allowed(value, speculative=False):
        if value.host_busy_cores >= value.cpu_count * 0.95:
            summary["cpu_pause_count"] += 1
            return False
        if resource_mode == "detailed" or legacy_sampler:
            median = median_cpu()
            if median is None:
                return not speculative
            if median is not None and median >= value.cpu_count * 0.90:
                summary["cpu_pause_count"] += 1
                return False
        return True

    def choose(value):
        if (
            resource_mode == "auto"
            and not legacy_sampler
            and retry_queue
            and time.monotonic() < cooldown_until
        ):
            return None, None, "memory_recovery"
        remaining = target_workers - normal_allocation_total()
        if remaining > 0:
            allocation = min(4, remaining)
            if (
                allocation_total() + allocation <= allocation_ceiling
                and memory_allowed(allocation, value)
                and cpu_allowed(value)
            ):
                return allocation, "normal", "ok"
            summary["memory_pause_count"] += 1
            if (
                time.monotonic() >= cooldown_until
                and allocation_total() + allocation <= allocation_ceiling
                and memory_allowed(allocation, value, speculative=True)
                and cpu_allowed(value, speculative=True)
            ):
                return allocation, "memory_speculative", "memory_speculative"
            return None, None, "memory"
        if allocation_total() + 1 > allocation_ceiling:
            return None, None, "allocation"
        if (
            time.monotonic() >= cooldown_until
            and allocation_total() + 8 <= allocation_ceiling
            and memory_allowed(8, value, speculative=True)
            and cpu_allowed(value, speculative=True)
        ):
            return 8, "cpu_overcommit", "scaled"
        if not memory_allowed(1, value, speculative=True):
            summary["memory_pause_count"] += 1
            return None, None, "memory"
        if (
            time.monotonic() >= cooldown_until
            and cpu_allowed(value, speculative=True)
        ):
            return 1, "cpu_overcommit", "cpu_overcommit"
        return None, None, "cpu"

    def start_next(allocation, tier, reason):
        nonlocal unobserved_reservation_bytes, speculative_starts_in_sample
        task = pull_task()
        if task is None:
            return False
        worker_id, args, preemptions = task
        generation = generations.get(worker_id, 0) + 1
        generations[worker_id] = generation
        if on_start is not None:
            on_start(worker_id, args, allocation)
        process = context.Process(
            target=_adaptive_worker_entry,
            args=(result_queue, worker_id, generation, function, tuple(args) + (allocation,)),
        )
        process.start()
        active[worker_id] = {
            "args": args,
            "allocation": allocation,
            "generation": generation,
            "peak_pss_bytes": 0,
            "observed_memory_bytes": 0,
            "preemptions": preemptions,
            "process": process,
            "started_at": time.monotonic(),
            "tier": tier,
        }
        started.append(worker_id)
        summary["peak_active_tasks"] = max(summary["peak_active_tasks"], len(active))
        summary["peak_allocated_workers"] = max(summary["peak_allocated_workers"], allocation_total())
        summary["peak_normal_allocated_workers"] = max(
            summary["peak_normal_allocated_workers"], normal_allocation_total()
        )
        summary["peak_speculative_tasks"] = max(
            summary["peak_speculative_tasks"],
            sum(item["tier"] == "memory_speculative" for item in active.values()),
        )
        summary["peak_overcommit_tasks"] = max(
            summary["peak_overcommit_tasks"],
            sum(item["tier"] == "cpu_overcommit" for item in active.values()),
        )
        summary["peak_scaled_tasks"] = max(
            summary["peak_scaled_tasks"],
            sum(item["tier"] == "cpu_overcommit" and item["allocation"] > 1 for item in active.values()),
        )
        summary["last_admission_reason"] = reason
        if tier != "normal":
            if not speculative_starts_in_sample:
                summary["speculative_admission_batches"] += 1
            speculative_starts_in_sample += 1
            summary["peak_speculative_starts_per_sample"] = max(
                summary["peak_speculative_starts_per_sample"],
                speculative_starts_in_sample,
            )
            unobserved_reservation_bytes += estimator.estimate_bytes(allocation)
            summary["peak_unobserved_reservation_bytes"] = max(
                summary["peak_unobserved_reservation_bytes"],
                unobserved_reservation_bytes,
            )
            if tier == "memory_speculative":
                summary["observed_memory_admission_count"] += 1
        return True

    def fill(value):
        scaled_starts = 0
        while not stop_submitting:
            allocation, tier, reason = choose(value)
            if allocation is None:
                summary["last_wait_reason"] = reason
                return
            if tier != "normal" and speculative_starts_in_sample >= 2:
                return
            if reason == "scaled" and scaled_starts >= 1:
                return
            if not start_next(allocation, tier, reason):
                return
            if reason == "scaled":
                scaled_starts += 1

    def preempt_for_memory(value):
        nonlocal cooldown_until, stop_submitting
        fast_mode = resource_mode == "auto" and not legacy_sampler
        if fast_mode and value.mem_available_bytes >= host_memory_floor(value):
            return
        if not fast_mode and (
            value.process_pss_bytes <= max_memory_bytes
            and value.mem_available_bytes >= reserve(value)
        ):
            return
        projected = value.process_pss_bytes
        projected_available = value.mem_available_bytes
        recovery = host_memory_floor(value) if fast_mode else int(max_memory_bytes * 0.95)
        victims = sorted(
            active.items(),
            key=lambda item: -item[1]["started_at"],
        )
        for worker_id, record in victims:
            if fast_mode and projected_available >= recovery:
                break
            if not fast_mode and projected <= recovery:
                break
            active.pop(worker_id, None)
            terminate_process_tree(record["process"].pid)
            record["process"].join(timeout=1.0)
            reclaimed = max(
                record["observed_memory_bytes"],
                record["peak_pss_bytes"],
                estimator.estimate_bytes(record["allocation"]),
            )
            if fast_mode:
                projected_available += reclaimed
            else:
                projected -= reclaimed
            preemptions = record["preemptions"] + 1
            if on_preempt is not None:
                on_preempt(worker_id, record["args"], record["allocation"], preemptions)
            if preemptions >= 3:
                failure = WorkerFailure(
                    worker_id, "MemoryLimitExceeded",
                    "task exceeded the memory safety floor after repeated resource preemption", "",
                )
                results[worker_id] = failure
                if on_result is not None:
                    on_result(worker_id, failure)
                if stop_on_result is not None and stop_on_result(failure):
                    stop_submitting = True
            else:
                retry_queue.appendleft((worker_id, record["args"], preemptions))
                summary["memory_requeue_count"] += 1
            summary["memory_preemption_count"] += 1
        cooldown_until = time.monotonic() + 60.0
        logger.warning(
            "Project scheduler memory recovery: pss=%.1f GiB available=%.1f GiB preempted=%d",
            value.process_pss_bytes / float(GIB),
            value.mem_available_bytes / float(GIB), summary["memory_preemption_count"],
        )

    try:
        value = sample_host("initial")
        if estimator.estimate_bytes(4) > max_memory_bytes:
            raise ValueError(
                "--max-memory is below the initial per-run admission estimate "
                "of %.1f GiB" % (estimator.estimate_bytes(4) / float(GIB))
            )
        next_host_sample = time.monotonic() + host_poll_seconds
        next_heartbeat = time.monotonic()
        while active or retry_queue or not exhausted:
            now = time.monotonic()
            if now >= next_host_sample:
                value = sample_host("host")
                next_host_sample = now + host_poll_seconds
                preempt_for_memory(value)
                if not stop_submitting:
                    fill(value)
            if now >= next_heartbeat:
                value = sample_heartbeat("heartbeat")
                next_heartbeat = now + heartbeat_seconds
                summary["heartbeat_count"] += 1
                logger.info(
                    "Project scheduler heartbeat: active=%d allocated=%d cpu=%.1f memory=%.1f GiB available=%.1f GiB kind=%s",
                    len(active), allocation_total(), value.process_cpu_cores,
                    value.process_pss_bytes / float(GIB), value.mem_available_bytes / float(GIB),
                    value.sample_kind,
                )
                preempt_for_memory(value)
                if not stop_submitting:
                    fill(value)
            if not active:
                if not stop_submitting:
                    fill(value)
                if not active and not stop_submitting and (retry_queue or not exhausted):
                    wait_seconds = min(
                        poll_seconds,
                        max(0.001, next_host_sample - time.monotonic()),
                    )
                    summary["memory_wait_seconds"] += wait_seconds
                    time.sleep(wait_seconds)
                    continue
                break
            try:
                timeout = min(
                    poll_seconds,
                    max(0.001, next_host_sample - time.monotonic()),
                    max(0.001, next_heartbeat - time.monotonic()),
                )
                first = result_queue.get(timeout=timeout)
            except queue.Empty:
                failed = next(
                    (
                        (worker_id, record)
                        for worker_id, record in active.items()
                        if record["process"].exitcode is not None
                    ),
                    None,
                )
                if failed is not None:
                    worker_id, record = failed
                    active.pop(worker_id)
                    record["process"].join()
                    payload = WorkerFailure(
                        worker_id=worker_id,
                        exception_type="ProcessExit",
                        message="child exited with code %s before reporting" % record["process"].exitcode,
                        traceback_text="",
                    )
                    results[worker_id] = payload
                    if on_result is not None:
                        on_result(worker_id, payload)
                    if stop_on_result is None or stop_on_result(payload):
                        stop_submitting = True
                continue
            batch = [first]
            for _index in range(255):
                try:
                    batch.append(result_queue.get_nowait())
                except queue.Empty:
                    break
            summary["completion_batch_count"] += 1
            summary["peak_completion_batch_size"] = max(
                summary["peak_completion_batch_size"], len(batch)
            )
            accepted = []
            for status, worker_id, generation, payload in batch:
                record = active.get(worker_id)
                if record is not None and record["generation"] == generation:
                    accepted.append((status, worker_id, record, payload))
            for status, worker_id, record, payload in accepted:
                active.pop(worker_id, None)
                record["process"].join()
                estimator.observe(record["allocation"], record["peak_pss_bytes"], payload)
                results[worker_id] = payload
                if on_result is not None:
                    on_result(worker_id, payload)
                if status == "error" or (
                    stop_on_result is not None and stop_on_result(payload)
                ):
                    stop_submitting = True
            if not stop_submitting:
                fill(value)
        return results, started, summary
    finally:
        for record in active.values():
            terminate_process_tree(record["process"].pid)
            record["process"].join()
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
                    if process.exitcode is not None
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


def _run_adaptive_process_tasks_legacy(
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
        max(1, os.cpu_count() or 1) * 3,
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
