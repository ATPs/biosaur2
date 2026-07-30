"""Small multiprocessing helpers used by the feature workflow."""

from __future__ import annotations

import multiprocessing
import math
import os
import queue
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


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
