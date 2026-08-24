"""Asynchronous publication of reusable strict-stage cache snapshots."""

from __future__ import annotations

import json
import multiprocessing
import queue
import traceback

from .stage_cache import build_strict_stage_payload, save_strict_stage_cache


def _strict_cache_writer_entry(
    result_queue, directory, source_path, strict_stage_cache_args, ingestion,
    strict_contexts, next_feature_id, args,
):
    try:
        payload = build_strict_stage_payload(
            ingestion, strict_contexts, next_feature_id, args
        )
        cache_path = save_strict_stage_cache(
            directory, source_path, strict_stage_cache_args, payload
        )
        manifest = json.loads(
            (cache_path / "manifest.json").read_text(encoding="utf-8")
        )
        result_queue.put((
            "ok", {
                "path": str(cache_path),
                "payload_bytes": int(manifest["payload_bytes"]),
                "strict_feature_count": int(manifest["strict_feature_count"]),
            },
        ))
    except BaseException as exc:
        result_queue.put((
            "error", {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        ))


def start_strict_cache_writer(
    directory, source_path, strict_stage_cache_args, ingestion,
    strict_contexts, next_feature_id, args,
):
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_strict_cache_writer_entry,
        args=(
            result_queue, directory, source_path, strict_stage_cache_args,
            ingestion, strict_contexts, next_feature_id, args,
        ),
    )
    process.start()
    return process, result_queue


def finish_strict_cache_writer(process, result_queue):
    try:
        process.join()
        try:
            status, payload = result_queue.get(timeout=1.0)
        except queue.Empty as exc:
            raise RuntimeError(
                "strict-stage cache writer exited without reporting a result "
                "(exit code %s)" % process.exitcode
            ) from exc
        if status != "ok":
            raise RuntimeError(
                "strict-stage cache writer failed with %(exception_type)s: "
                "%(message)s\n%(traceback)s" % payload
            )
        if process.exitcode != 0:
            raise RuntimeError(
                "strict-stage cache writer exited with code %s" % process.exitcode
            )
        return payload
    finally:
        result_queue.close()
        result_queue.join_thread()


def cancel_strict_cache_writer(process, result_queue):
    if process.is_alive():
        process.terminate()
    process.join()
    result_queue.close()
    result_queue.join_thread()
