"""Bounded project execution, compact project metadata and validation."""

from __future__ import annotations

import csv
import json
import logging
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import pyarrow as pa
import pyarrow.parquet as pq

from .output import publish_staged_files, _temporary_neighbor
from .cache_runtime import ProjectCheckpoint, remove_cache_layers, run_cache_paths
from .parallel import (
    WorkerFailure,
    effective_worker_budget,
    physical_memory_bytes,
    run_adaptive_process_tasks,
)
from .thread_runtime import configure_cli_thread_pools
from .project_manifest import read_manifest
from .raw_ms1 import source_fingerprint
from .external_mbr import run_feature_mbr_stage
from .project_validation import validate_project as validate_project
from .schema import PROJECT_SCHEMA_VERSION
from .identifications import PSM_COLUMN_OPTIONS


logger = logging.getLogger(__name__)


def _scientific_command(command):
    """Remove scheduling/cache-location arguments from a run command."""

    normalized = []
    index = 0
    while index < len(command):
        option = command[index]
        if option in {"--workers", "--cache-dir", "--log-level"}:
            index += 2
            continue
        if option in {"--keep-cache", "--overwrite"}:
            index += 1
            continue
        normalized.append(option)
        index += 1
    return normalized


def _resume_option_signature(options):
    """Return science/output options while ignoring scheduling-only controls."""

    ignored = {
        "resume",
        "continue_on_error",
        "workers",
        "max_memory",
        "scheduler_heartbeat_seconds",
        "cache_dir",
        "keep_cache",
        "log_level",
        "_cache_workspace",
        "_project_checkpoint_path",
        "_max_memory_bytes",
        "_effective_workers",
        "_local_scheduler_summary",
    }
    signature = {
        key: value
        for key, value in options.items()
        if key not in ignored
    }
    # Project metadata is JSON. Normalize tuples such as isotope-error lists
    # before comparing a freshly parsed CLI namespace with persisted options.
    return json.loads(json.dumps(signature, sort_keys=True, default=str))


def _local_resume_option_signature(options):
    """Return options that can change per-run local output."""

    signature = _resume_option_signature(options)
    local_external_options = {
        "external_id",
        "external_weak_min_mono_points",
        "external_weak_min_secondary_points",
        "external_weak_min_isotope_cosine",
        "external_weak_max_strong_overlap",
    }
    for key in tuple(signature):
        if key.startswith("external_") and key not in local_external_options:
            signature.pop(key, None)
    return signature


def _checkpoint_identity(manifest, output_dir, database, options):
    return {
        "manifest": str(Path(manifest).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "project_db": str(Path(database).resolve()),
    }


def _tail_append(chunks, chunk, limit):
    chunks.append(chunk)
    retained = "".join(chunks)
    if len(retained) > limit:
        chunks[:] = [retained[-limit:]]


def _project_worker(task):
    started = time.monotonic()
    command = task.get("execution_command", task["command"])
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_env = dict(os.environ)
    child_env["BIOSAUR2_LOG_RUN_ID"] = task["run_id"]
    completed = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        env=child_env,
    )
    stdout_tail = []
    stderr_tail = []

    def forward(stream, destination, tail, limit):
        try:
            for line in iter(stream.readline, ""):
                destination.write(line)
                destination.flush()
                _tail_append(tail, line, limit)
        finally:
            stream.close()

    stdout_thread = threading.Thread(
        target=forward,
        args=(completed.stdout, sys.stdout, stdout_tail, 4000),
    )
    stderr_thread = threading.Thread(
        target=forward,
        args=(completed.stderr, sys.stderr, stderr_tail, 8000),
    )
    stdout_thread.start()
    stderr_thread.start()
    returncode = completed.wait()
    stdout_thread.join()
    stderr_thread.join()
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "run_id": task["run_id"],
        "status": "success" if returncode == 0 else "failed",
        "runtime_sec": time.monotonic() - started,
        "cpu_user_sec": usage_after.ru_utime - usage_before.ru_utime,
        "cpu_system_sec": usage_after.ru_stime - usage_before.ru_stime,
        # Linux reports ru_maxrss in KiB. Project mode currently launches one
        # analysis subprocess per fresh worker, so this is that run's peak RSS.
        "peak_rss_kib": usage_after.ru_maxrss,
        "returncode": returncode,
        "error": None if returncode == 0 else "".join(stderr_tail),
        "stdout_tail": "".join(stdout_tail),
        "stderr_tail": "".join(stderr_tail),
        "paths": task["paths"],
        "command": command,
    }


def _project_worker_budgeted(task, allocated_workers):
    execution_task = dict(task)
    execution_command = list(task["command"]) + [
        "--workers",
        str(allocated_workers),
    ]
    if task.get("force_overwrite"):
        execution_command.append("--overwrite")
    if task.get("cache_root"):
        execution_command.extend(
            ("--cache-dir", task["cache_root"], "--keep-cache")
        )
    execution_task["execution_command"] = execution_command
    result = _project_worker(execution_task)
    result["allocated_workers"] = allocated_workers
    return result


def _run_paths(
    run,
    output_dir,
    cache_workspace=None,
    output_format="parquet",
    write_ms1=False,
):
    directory = output_dir / run.run_id
    cache_paths = run_cache_paths(
        cache_workspace or (output_dir / ".biosaur2_cache"),
        run.mzml_path,
    )
    if output_format == "duckdb":
        run_output = directory / (run.run_id + ".biosaur2.duckdb")
        features = ms2_events = identifications = str(run_output)
        ms1 = str(run_output) if write_ms1 else None
    else:
        run_output = None
        features = str(directory / (run.run_id + ".features." + output_format))
        ms2_events = str(
            directory / (run.run_id + ".ms2_events." + output_format)
        )
        identifications = str(
            directory / (run.run_id + ".identifications." + output_format)
        )
        ms1 = (
            str(directory / (run.run_id + ".ms1." + output_format))
            if write_ms1
            else None
        )
    paths = {
        "run_dir": str(directory),
        "format": output_format,
        "run_output": None if run_output is None else str(run_output),
        "features": features,
        "ms2_events": ms2_events,
        "identifications": identifications,
        "ms1": ms1,
        "external_evidence": (
            str(run_output)
            if output_format == "duckdb"
            else str(directory / (run.run_id + ".external_id_evidence." + output_format))
        ),
    }
    paths.update(cache_paths)
    return paths


def _command_for_run(run, paths, options):
    command = [
        sys.executable,
        "-m",
        "biosaur2.search",
        str(run.mzml_path),
        "-o",
        paths.get("run_output") or paths["features"],
        "--format",
        options.get("format", "parquet"),
        "--feature-mode",
        options["mode"],
        "--max-charge",
        str(options.get("max_charge", 7)),
        "--log-level",
        options.get("log_level", "info"),
    ]
    if options["overwrite"]:
        command.append("--overwrite")
    write_ms1 = options.get("write_ms1")
    if write_ms1 is None:
        write_ms1 = options.get("mode", "legacy") == "hybrid"
    command.append("--write-ms1" if write_ms1 else "--no-write-ms1")
    if options["mode"] == "hybrid":
        if run.psm_path is not None:
            command.extend(("--psm-path", str(run.psm_path)))
        q_value = run.q_value_max if run.q_value_max is not None else options["psm_q_value_max"]
        command.extend(("--psm-q-value-max", str(q_value)))
        if options["psm_pep_max"] is not None:
            command.extend(("--psm-pep-max", str(options["psm_pep_max"])))
        for _semantic, option, _description in PSM_COLUMN_OPTIONS:
            value = options.get(option[2:].replace("-", "_"))
            if value is not None and str(value).strip():
                command.extend((option, str(value).strip()))
        fixed = list(options["fixed_mod"])
        if run.fixed_mods:
            fixed.extend(value.strip() for value in run.fixed_mods.split(";") if value.strip())
        for value in fixed:
            command.extend(("--fixed-mod", value))
        command.extend(("--quant-method", options["quant_method"]))
        if options.get("write_mono_hills"):
            command.append("--write-mono-hills")
        if options.get("write_quant_details"):
            command.append("--write-quant-details")
        command.extend(("--feature-baseline", options["feature_baseline"]))
        command.append("--direct-id" if options["direct_id"] else "--no-direct-id")
        command.append("--external-id" if options["external_id"] else "--no-external-id")
        for key, default in (
            ("external_weak_min_mono_points", 2),
            ("external_weak_min_secondary_points", 2),
            ("external_weak_min_isotope_cosine", 0.6),
            ("external_weak_max_strong_overlap", 0.30),
        ):
            command.extend(("--" + key.replace("_", "-"), str(options.get(key, default))))
        command.append(
            "--generic-ms2-refine"
            if options["generic_ms2_refine"]
            else "--no-generic-ms2-refine"
        )
        command.append(
            "--relaxed-ms2-feature"
            if options.get("relaxed_ms2_feature", False)
            else "--no-relaxed-ms2-feature"
        )
        command.extend(("--generic-q-value-max", str(options["generic_q_value_max"])))
        command.extend(("--generic-ms2-ppm", str(options.get("generic_ms2_ppm", 10.0))))
        command.extend(("--generic-ms2-isotope-errors", ",".join(map(str, options.get("generic_ms2_isotope_errors", (0, 1, 2, 3))))))
        generic_defaults = {
            "generic_local_isotope_count": 5,
            "generic_local_min_mono_points": 3,
            "generic_local_min_channel_points": 3,
            "generic_local_min_supported_channels": 2,
            "generic_local_min_isotope_cosine": 0.90,
            "generic_local_max_width_sec": "auto",
            "generic_relaxed_min_mono_points": 2,
            "generic_relaxed_min_channel_points": 2,
            "generic_relaxed_min_supported_channels": 2,
            "generic_relaxed_min_isotope_cosine": 0.95,
        }
        for option, default in generic_defaults.items():
            command.extend(
                (
                    "--" + option.replace("_", "-"),
                    str(options.get(option, default)),
                )
            )
        command.extend(
            (
                "--ms2-rt-tolerance-sec",
                str(options.get("ms2_rt_tolerance_sec", 120.0)),
            )
        )
    return command


def _read_successful_runs(database):
    if not database.is_file():
        return {}
    import duckdb

    try:
        with duckdb.connect(str(database), read_only=True) as connection:
            metadata = dict(
                connection.execute(
                    "SELECT key, value_json FROM project_metadata"
                ).fetchall()
            )
            prior_options = json.loads(
                metadata.get("resolved_options", "{}")
            )
            option_signature = _local_resume_option_signature(prior_options)
            run_columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info('runs')").fetchall()
            }
            resource_columns = (
                "cpu_user_sec, cpu_system_sec, peak_rss_kib"
                if "peak_rss_kib" in run_columns
                else "NULL, NULL, NULL"
            )
            return {
                row[0]: {
                    "input_fingerprint": json.loads(row[1]),
                    "command": json.loads(row[2]),
                    "cpu_user_sec": row[3],
                    "cpu_system_sec": row[4],
                    "peak_rss_kib": row[5],
                    "project_option_signature": option_signature,
                }
                for row in connection.execute(
                    "SELECT run_id, input_fingerprint_json, command_json, "
                    + resource_columns
                    + " FROM runs "
                    "WHERE status IN ('success', 'skipped_resume')"
                ).fetchall()
            }
    except Exception:
        return {}


def _input_fingerprint(run):
    return {
        "mzml": source_fingerprint(run.mzml_path),
        "psm": (
            None
            if run.psm_path is None
            else source_fingerprint(run.psm_path)
        ),
    }


SUMMARY_PARQUET_BATCH_SIZE = 65536
SUMMARY_READER_MEMORY_BYTES = 256 * 1024 ** 2
SUMMARY_READER_MAX_WORKERS = 32


def _empty_summary():
    return {
        "feature_count": None,
        "ms2_count": None,
        "audit_count": None,
        "linked_ms2_count": None,
        "quant_feature_count": None,
        "accepted_identification_count": None,
        "direct_assay_count": None,
        "hybrid_summary": None,
    }


def _summary_error(run_id, path, table_name, error):
    return RuntimeError(
        "Project summary read failed for run %s table %s at %s: %s"
        % (run_id, table_name, path, error)
    )


def _require_summary_path(run_id, path, table_name):
    if not path.is_file():
        raise _summary_error(run_id, path, table_name, "missing required output")


def _parquet_file(run_id, path, table_name):
    try:
        return pq.ParquetFile(path)
    except Exception as error:
        raise _summary_error(run_id, path, table_name, error) from error


def _tsv_count(run_id, path, table_name, value_column=None):
    count = 0
    value_count = 0
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                count += 1
                if value_column is not None and bool(row.get(value_column)):
                    value_count += 1
    except Exception as error:
        raise _summary_error(run_id, path, table_name, error) from error
    return count, value_count


def _read_parquet_summary(result, run_id, mode):
    paths = result["paths"]
    summary = _empty_summary()
    feature_path = Path(paths["features"])
    _require_summary_path(run_id, feature_path, "features")
    features = _parquet_file(run_id, feature_path, "features")
    summary["feature_count"] = features.metadata.num_rows
    summary["quant_feature_count"] = features.metadata.num_rows

    ms2_events_path = Path(paths["ms2_events"])
    if mode == "hybrid":
        _require_summary_path(run_id, ms2_events_path, "ms2_events")
    linked = 0
    if ms2_events_path.is_file():
        linked = _parquet_file(
            run_id, ms2_events_path, "ms2_events"
        ).metadata.num_rows
    summary["ms2_count"] = linked
    summary["audit_count"] = linked
    summary["linked_ms2_count"] = linked

    identification_path = Path(paths["identifications"])
    if mode == "hybrid":
        _require_summary_path(run_id, identification_path, "identifications")
    if identification_path.is_file():
        identifications = _parquet_file(
            run_id, identification_path, "identifications"
        )
        summary["accepted_identification_count"] = identifications.metadata.num_rows
        direct = 0
        try:
            for batch in identifications.iter_batches(
                batch_size=SUMMARY_PARQUET_BATCH_SIZE,
                columns=["assay_id"],
                use_threads=False,
            ):
                direct += batch.num_rows - batch.column(0).null_count
        except Exception as error:
            raise _summary_error(
                run_id, identification_path, "identifications", error
            ) from error
        summary["direct_assay_count"] = direct

    metadata = features.metadata.metadata or {}
    try:
        encoded = metadata.get(b"biosaur2_hybrid_summary_json")
        if not encoded and metadata.get(b"biosaur2_provenance_json"):
            provenance = json.loads(metadata[b"biosaur2_provenance_json"])
            encoded = provenance.get("hybrid_summary_json")
        if encoded:
            summary["hybrid_summary"] = json.loads(encoded)
    except Exception as error:
        raise _summary_error(run_id, feature_path, "features metadata", error) from error
    return summary


def _read_tsv_summary(result, run_id, mode):
    paths = result["paths"]
    summary = _empty_summary()
    feature_path = Path(paths["features"])
    _require_summary_path(run_id, feature_path, "features")
    count, quant_count = _tsv_count(
        run_id, feature_path, "features", value_column="quant_value"
    )
    summary["feature_count"] = count
    summary["quant_feature_count"] = quant_count

    ms2_events_path = Path(paths["ms2_events"])
    if mode == "hybrid":
        _require_summary_path(run_id, ms2_events_path, "ms2_events")
    if ms2_events_path.is_file():
        linked, _unused = _tsv_count(run_id, ms2_events_path, "ms2_events")
        summary["linked_ms2_count"] = linked
    summary["ms2_count"] = summary["linked_ms2_count"]
    summary["audit_count"] = summary["linked_ms2_count"]

    identification_path = Path(paths["identifications"])
    if mode == "hybrid":
        _require_summary_path(run_id, identification_path, "identifications")
    if identification_path.is_file():
        accepted, direct = _tsv_count(
            run_id, identification_path, "identifications", value_column="assay_id"
        )
        summary["accepted_identification_count"] = accepted
        summary["direct_assay_count"] = direct
    return summary


def _duckdb_table_exists(connection, table_name):
    return connection.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_name = ?",
        [table_name],
    ).fetchone() is not None


def _read_duckdb_summary(result, run_id, mode):
    import duckdb

    paths = result["paths"]
    summary = _empty_summary()
    feature_path = Path(paths["features"])
    _require_summary_path(run_id, feature_path, "features")
    try:
        with duckdb.connect(
            str(feature_path), read_only=True, config={"threads": "1"}
        ) as connection:
            connection.execute("SET threads TO 1")
            summary["feature_count"] = connection.execute(
                'SELECT COUNT(*) FROM "features"'
            ).fetchone()[0]
            summary["quant_feature_count"] = summary["feature_count"]
            has_ms2_events = _duckdb_table_exists(connection, "ms2_events")
            has_identifications = _duckdb_table_exists(
                connection, "identifications"
            )
            if mode == "hybrid" and not has_ms2_events:
                raise ValueError("missing required ms2_events table")
            if mode == "hybrid" and not has_identifications:
                raise ValueError("missing required identifications table")
            linked = (
                connection.execute(
                    'SELECT COUNT(*) FROM "ms2_events"'
                ).fetchone()[0]
                if has_ms2_events
                else 0
            )
            summary["ms2_count"] = linked
            summary["audit_count"] = linked
            summary["linked_ms2_count"] = linked
            if has_identifications:
                accepted, direct = connection.execute(
                    'SELECT COUNT(*), COUNT("assay_id") FROM "identifications"'
                ).fetchone()
                summary["accepted_identification_count"] = accepted
                summary["direct_assay_count"] = direct
            provenance_row = connection.execute(
                "SELECT provenance_json FROM runs LIMIT 1"
            ).fetchone()
            if provenance_row:
                encoded = json.loads(provenance_row[0]).get(
                    "hybrid_summary_json"
                )
                if encoded:
                    summary["hybrid_summary"] = json.loads(encoded)
    except Exception as error:
        raise _summary_error(run_id, feature_path, "DuckDB output", error) from error
    return summary


def _summary_for_result(result, *, run_id=None, mode="legacy"):
    if result["status"] not in {"success", "skipped_resume"}:
        return _empty_summary()
    run_id = run_id or result.get("run_id", "unknown")
    paths = result["paths"]
    output_format = paths.get("format")
    feature_path = Path(paths["features"])
    if output_format == "tsv" or feature_path.suffix.lower() == ".tsv":
        return _read_tsv_summary(result, run_id, mode)
    if output_format == "duckdb" or feature_path.suffix.lower() == ".duckdb":
        return _read_duckdb_summary(result, run_id, mode)
    return _read_parquet_summary(result, run_id, mode)


def _configure_summary_worker():
    configure_cli_thread_pools()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def _summary_worker(task):
    index, run_id, result, mode = task
    return index, _summary_for_result(result, run_id=run_id, mode=mode)


def _summary_worker_count(
    run_count, effective_workers, max_memory_bytes, cpu_count_value=None
):
    if run_count < 1:
        return 0
    cpu_count_value = cpu_count_value if cpu_count_value is not None else os.cpu_count()
    cpu_slots = max(1, int(cpu_count_value or 1))
    memory_slots = max(
        1,
        int(int(max_memory_bytes) * 0.80) // SUMMARY_READER_MEMORY_BYTES,
    )
    return min(
        int(run_count),
        int(effective_workers),
        cpu_slots,
        memory_slots,
        SUMMARY_READER_MAX_WORKERS,
    )


def _summaries_for_results(runs, results, options):
    mode = options.get("mode", "legacy")
    summaries = {}
    tasks = []
    for index, run in enumerate(runs):
        result = results[index]
        if result["status"] in {"success", "skipped_resume"}:
            tasks.append((index, run.run_id, result, mode))
        else:
            summaries[index] = _empty_summary()
    worker_count = _summary_worker_count(
        len(tasks),
        options.get(
            "_effective_workers",
            effective_worker_budget(int(options.get("workers", 4))),
        ),
        options.get("_max_memory_bytes", physical_memory_bytes()),
    )
    logger.info(
        "Project summary readers: completed_runs=%d workers=%d",
        len(tasks), worker_count,
    )
    if worker_count <= 1:
        for task in tasks:
            index, summary = _summary_worker(task)
            summaries[index] = summary
        return summaries

    # ``spawn`` imports this module before running the initializer, so set the
    # inherited native-pool environment before any child imports NumPy/Arrow.
    configure_cli_thread_pools()
    context = multiprocessing.get_context("spawn")
    pool = context.Pool(worker_count, initializer=_configure_summary_worker)
    try:
        for index, summary in pool.imap_unordered(
            _summary_worker, tasks, chunksize=1
        ):
            summaries[index] = summary
    except BaseException:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()
    return summaries


def _strict_cache_stage_row(run_id, paths):
    strict_cache_path = paths.get("strict_stage_cache")
    manifest_path = (
        Path(strict_cache_path, "manifest.json") if strict_cache_path else None
    )
    if manifest_path is not None and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return (
                run_id,
                "strict_stage_cache",
                "success",
                json.dumps(
                    {
                        "path": strict_cache_path,
                        "payload_bytes": manifest.get("payload_bytes"),
                        "payload_sha256": manifest.get("payload_sha256"),
                        "strict_feature_count": manifest.get("strict_feature_count"),
                        "context_count": manifest.get("context_count"),
                    },
                    sort_keys=True,
                ),
            )
        except (OSError, TypeError, ValueError) as error:
            return (
                run_id,
                "strict_stage_cache",
                "invalid",
                "%s: %s" % (strict_cache_path, error),
            )
    return run_id, "strict_stage_cache", "missing", strict_cache_path


def _project_database_rows(
    runs, results, summaries, options, external_stage, input_fingerprints
):
    rows = {
        "runs": [],
        "stage_status": [],
        "identification_summary": [],
        "qc_metrics": [],
        "hybrid_summary": [],
        "external_summary": [],
    }
    external_summaries = (external_stage or {}).get("summaries", {})
    for index, run in enumerate(runs):
        result = results[index]
        paths = result["paths"]
        summary = summaries[index]
        fingerprint = (
            input_fingerprints[index]
            if input_fingerprints is not None
            else _input_fingerprint(run)
        )
        rows["runs"].append([
            index,
            run.run_id,
            str(run.mzml_path),
            None if run.psm_path is None else str(run.psm_path),
            result["status"],
            result.get("runtime_sec"),
            result.get("error"),
            result.get("cpu_user_sec"),
            result.get("cpu_system_sec"),
            result.get("peak_rss_kib"),
            result.get("allocated_workers"),
            paths.get("format", options.get("format", "parquet")),
            paths.get("run_output"),
            paths["features"],
            paths["ms2_events"],
            paths["identifications"],
            paths.get("ms1"),
            paths.get("external_evidence"),
            paths["raw_ms1_cache"],
            json.dumps(fingerprint, sort_keys=True),
            json.dumps(result.get("command", [])),
        ])
        rows["stage_status"].append(
            [run.run_id, "run", result["status"], result.get("error")]
        )
        if options.get("mode") == "hybrid":
            cache_status = (
                "success"
                if Path(paths["raw_ms1_cache"], "manifest.json").is_file()
                else "missing"
            )
            rows["stage_status"].append(
                [run.run_id, "raw_ms1_cache", cache_status, paths["raw_ms1_cache"]]
            )
            rows["stage_status"].append(_strict_cache_stage_row(run.run_id, paths))
            external_summary = external_summaries.get(run.run_id)
            external_status = (
                "disabled"
                if not options.get("external_id", False)
                else "success" if external_summary is not None else "not_run"
            )
            rows["stage_status"].append(
                [run.run_id, "external_id", external_status, paths.get("external_evidence")]
            )
        rows["identification_summary"].append([
            run.run_id,
            summary["accepted_identification_count"],
            summary["direct_assay_count"],
        ])
        rows["qc_metrics"].append([
            run.run_id,
            summary["feature_count"],
            summary["ms2_count"],
            summary["audit_count"],
            summary["linked_ms2_count"],
            summary["quant_feature_count"],
        ])
        hybrid = summary["hybrid_summary"] or {}
        statuses = hybrid.get("audit_status_counts", {})
        generic = hybrid.get("generic_summary") or {}
        rows["hybrid_summary"].append([
            run.run_id,
            hybrid.get("strict_feature_count"),
            hybrid.get("direct_assay_count"),
            hybrid.get("recovered_feature_count"),
            hybrid.get("audit_row_count"),
            sum(
                int(statuses.get(status, 0))
                for status in (
                    "matched_strict_feature",
                    "recovered_direct_feature",
                    "recovered_direct_relaxed_two_point",
                    "recovered_direct_relaxed_partial_envelope",
                    "matched_recovered_feature",
                    "matched_recovered_feature_ambiguous_identity",
                )
            ),
            int(statuses.get("generic_matched_strict_feature", 0)),
            sum(
                int(statuses.get(status, 0))
                for status in (
                    "generic_local_matched_strict_feature",
                    "generic_local_matched_direct_feature",
                    "generic_recovered_local_feature",
                    "generic_matched_recovered_local_feature",
                    "generic_relaxed_recovered_local_feature",
                    "generic_relaxed_matched_recovered_local_feature",
                )
            ),
            hybrid.get("generic_recovered_feature_count"),
            int(statuses.get("generic_decoy_only", 0)),
            int(statuses.get("generic_local_decoy_only", 0)),
            json.dumps(statuses, sort_keys=True),
            json.dumps(generic, sort_keys=True),
            json.dumps(hybrid, sort_keys=True),
        ])
        external = external_summaries.get(run.run_id, {})
        rows["external_summary"].append([
            run.run_id,
            external.get("planned_assay_count"),
            external.get("evaluated_assay_count"),
            external.get("new_external_feature_count"),
            external.get("new_strict_external_feature_count"),
            external.get("new_weak_external_feature_count"),
            json.dumps(external.get("status_counts", {}), sort_keys=True),
        ])
    rows["rt_alignment_models"] = [
        [
            row["alignment_group"],
            row["reference_run"],
            row["source_run"],
            row["target_run"],
            row["method"],
            row["anchor_count"],
            row["inlier_count"],
            row["slope"],
            row["intercept"],
            row["residual_mad_sec"],
            row["status"],
            row.get("validation_anchor_count", 0),
            row.get("validation_median_bias_sec"),
            row.get("validation_mad_sec"),
            row.get("validation_q90_abs_error_sec"),
            row["x_knots_json"],
            row["y_knots_json"],
        ]
        for row in (external_stage or {}).get("alignment_models", ())
    ]
    rows["scheduler_summary"] = [
        [stage, json.dumps(summary, sort_keys=True)]
        for stage, summary in (
            ("local", options.get("_local_scheduler_summary", {})),
            ("external", (external_stage or {}).get("scheduler_summary", {})),
        )
    ]
    return rows


def _create_project_tables(connection):
    connection.execute(
        "CREATE TABLE runs (run_order INTEGER, run_id VARCHAR, mzml_path VARCHAR, "
        "psm_path VARCHAR, status VARCHAR, runtime_sec DOUBLE, error VARCHAR, "
        "cpu_user_sec DOUBLE, cpu_system_sec DOUBLE, peak_rss_kib BIGINT, "
        "allocated_workers INTEGER, output_format VARCHAR, run_output_path VARCHAR, "
        "features_path VARCHAR, ms2_events_path VARCHAR, identification_path VARCHAR, "
        "ms1_path VARCHAR, external_evidence_path VARCHAR, raw_ms1_cache_path VARCHAR, "
        "input_fingerprint_json VARCHAR, command_json VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE stage_status (run_id VARCHAR, stage VARCHAR, status VARCHAR, detail VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE identification_summary (run_id VARCHAR, accepted_identification_count BIGINT, direct_assay_count BIGINT)"
    )
    connection.execute(
        "CREATE TABLE qc_metrics (run_id VARCHAR, feature_count BIGINT, ms2_count BIGINT, audit_count BIGINT, linked_ms2_count BIGINT, quant_feature_count BIGINT)"
    )
    connection.execute(
        "CREATE TABLE hybrid_summary (run_id VARCHAR, strict_feature_count BIGINT, direct_assay_count BIGINT, recovered_feature_count BIGINT, audit_row_count BIGINT, direct_linked_count BIGINT, generic_strict_linked_count BIGINT, generic_local_linked_count BIGINT, generic_local_new_feature_count BIGINT, generic_decoy_only_count BIGINT, generic_local_decoy_only_count BIGINT, audit_status_counts_json VARCHAR, generic_summary_json VARCHAR, summary_json VARCHAR)"
    )
    connection.execute("CREATE TABLE project_metadata (key VARCHAR, value_json VARCHAR)")
    connection.execute("CREATE TABLE scheduler_summary (stage VARCHAR, summary_json VARCHAR)")
    connection.execute(
        "CREATE TABLE rt_alignment_models (alignment_group VARCHAR, reference_run VARCHAR, source_run VARCHAR, target_run VARCHAR, method VARCHAR, anchor_count INTEGER, inlier_count INTEGER, slope DOUBLE, intercept DOUBLE, residual_mad_sec DOUBLE, status VARCHAR, validation_anchor_count INTEGER, validation_median_bias_sec DOUBLE, validation_mad_sec DOUBLE, validation_q90_abs_error_sec DOUBLE, x_knots_json VARCHAR, y_knots_json VARCHAR)"
    )
    connection.execute(
        "CREATE TABLE external_summary (run_id VARCHAR, planned_assay_count BIGINT, evaluated_assay_count BIGINT, new_external_feature_count BIGINT, new_strict_external_feature_count BIGINT, new_weak_external_feature_count BIGINT, status_counts_json VARCHAR)"
    )


def _write_project_database(
    database, runs, results, options, *, external_stage=None,
    input_fingerprints=None,
):
    import duckdb

    summaries = _summaries_for_results(runs, results, options)
    rows = _project_database_rows(
        runs, results, summaries, options, external_stage, input_fingerprints
    )
    temporary = _temporary_neighbor(database)
    connection = duckdb.connect(str(temporary), config={"threads": "1"})
    try:
        connection.execute("SET threads TO 1")
        connection.execute("BEGIN TRANSACTION")
        _create_project_tables(connection)
        for table_name, values in rows.items():
            if values:
                connection.executemany(
                    "INSERT INTO %s VALUES (%s)" % (
                        table_name,
                        ", ".join("?" for _ in values[0]),
                    ),
                    values,
                )
        connection.executemany(
            "INSERT INTO project_metadata VALUES (?, ?)",
            [
                ["project_schema_version", PROJECT_SCHEMA_VERSION],
                ["resolved_options", json.dumps(options, sort_keys=True, default=str)],
            ],
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    publish_staged_files([(temporary, database)])


def _run_external_stage(runs, results, options):
    return run_feature_mbr_stage(runs, results, options)


def run_project(manifest, output_dir, project_db, **options):
    runs = read_manifest(manifest)
    output_dir = Path(output_dir).resolve()
    database = Path(project_db).resolve()
    if options.get("write_ms1") is None:
        options["write_ms1"] = options.get("mode", "legacy") == "hybrid"
    cache_workspace = Path(
        options.get("_cache_workspace")
        or options.get("cache_dir")
        or ".biosaur2_cache"
    ).resolve()
    effective_workers = effective_worker_budget(int(options.get("workers", 4)))
    options["_effective_workers"] = effective_workers
    options["_max_memory_bytes"] = int(
        options.get("_max_memory_bytes")
        or int(options.get("max_memory", 0) or 0) * (1024 ** 3)
        or physical_memory_bytes()
    )
    checkpoint = None
    checkpoint_path = options.get("_project_checkpoint_path")
    if checkpoint_path:
        checkpoint = ProjectCheckpoint(checkpoint_path).open(
            _checkpoint_identity(manifest, output_dir, database, options),
            resume=bool(options.get("resume", True)),
        )
    logger.debug(
        "Project start: manifest=%s output_dir=%s project_db=%s runs=%d mode=%s "
        "format=%s requested_workers=%d effective_workers=%d resume=%s",
        manifest,
        output_dir,
        database,
        len(runs),
        options.get("mode"),
        options.get("format"),
        options.get("workers"),
        effective_workers,
        options.get("resume"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        if not run.mzml_path.is_file():
            raise FileNotFoundError("mzML input does not exist: %s" % run.mzml_path)
        if run.psm_path is not None and not run.psm_path.is_file():
            raise FileNotFoundError("PSM input does not exist: %s" % run.psm_path)
        if run.psm_format not in {None, "percolator_tsv", "percolator_parquet"}:
            raise ValueError("unsupported psm_format for %s: %s" % (run.run_id, run.psm_format))

    successful = _read_successful_runs(database) if options["resume"] else {}
    if checkpoint and options["resume"]:
        for run_id, record in checkpoint.state.get("runs", {}).items():
            if record.get("status") != "success":
                continue
            result = record["result"]
            successful[run_id] = {
                "input_fingerprint": record["input_fingerprint"],
                "command": record["command"],
                "cpu_user_sec": result.get("cpu_user_sec"),
                "cpu_system_sec": result.get("cpu_system_sec"),
                "peak_rss_kib": result.get("peak_rss_kib"),
                "project_option_signature": record["project_option_signature"],
            }
    input_fingerprints = {
        index: _input_fingerprint(run) for index, run in enumerate(runs)
    }
    tasks = []
    skipped = {}
    for index, run in enumerate(runs):
        paths = _run_paths(
            run,
            output_dir,
            cache_workspace,
            options.get("format", "parquet"),
            write_ms1=bool(options.get("write_ms1")),
        )
        Path(paths["run_dir"]).mkdir(parents=True, exist_ok=True)
        command = _command_for_run(run, paths, options)
        logger.debug(
            "Project run prepared: run_id=%s input=%s output=%s cache=%s command=%s",
            run.run_id,
            run.mzml_path,
            paths["run_output"] or paths["features"],
            paths["raw_ms1_cache"],
            command,
        )
        resume_record = successful.get(run.run_id)
        required_paths = [paths["run_output"] or paths["features"]]
        if options["mode"] == "hybrid":
            required_paths.append(paths["run_output"] or paths["ms2_events"])
            required_paths.append(paths["run_output"] or paths["identifications"])
        if options.get("write_ms1"):
            required_paths.append(paths["run_output"] or paths["ms1"])
        resume_valid = (
            resume_record is not None
            and resume_record["input_fingerprint"] == input_fingerprints[index]
            and _scientific_command(resume_record["command"])
            == _scientific_command(command)
            and resume_record.get("project_option_signature")
            == _local_resume_option_signature(options)
            and all(Path(path).is_file() for path in required_paths)
        )
        if resume_valid:
            logger.debug("Project run %s: resume signature and outputs match", run.run_id)
            skipped[index] = {
                "run_id": run.run_id,
                "status": "skipped_resume",
                "runtime_sec": 0.0,
                "returncode": 0,
                "error": None,
                "cpu_user_sec": resume_record.get("cpu_user_sec"),
                "cpu_system_sec": resume_record.get("cpu_system_sec"),
                "peak_rss_kib": resume_record.get("peak_rss_kib"),
                "paths": paths,
                "command": resume_record["command"],
            }
            continue
        tasks.append(
            (
                index,
                {
                    "run_id": run.run_id,
                    "paths": paths,
                    "command": command,
                    "cache_root": (
                        str(Path(paths["cache_run_dir"]).parent.parent)
                        if options["mode"] == "hybrid"
                        else None
                    ),
                    "force_overwrite": bool(
                        checkpoint and checkpoint.run_record(run.run_id)
                    ),
                },
            )
        )

    logger.debug(
        "Project scheduling: pending_runs=%d resumed_runs=%d target_workers=%d max_memory_gib=%.1f",
        len(tasks),
        len(skipped),
        effective_workers,
        options["_max_memory_bytes"] / float(1024 ** 3),
    )

    def task_arguments():
        for _index, task in tasks:
            yield (task,)

    def checkpoint_local_start(task_position, _args, _allocation):
        if not checkpoint:
            return
        task = tasks[task_position][1]
        manifest_index = tasks[task_position][0]
        run = runs[manifest_index]
        checkpoint.put_run(
            run.run_id,
            {
                "status": "pending",
                "input_fingerprint": input_fingerprints[manifest_index],
                "command": task["command"],
                "project_option_signature": _local_resume_option_signature(options),
            },
        )

    def checkpoint_local_result(task_position, value):
        if not checkpoint or isinstance(value, WorkerFailure):
            return
        task = tasks[task_position][1]
        manifest_index = tasks[task_position][0]
        run = runs[manifest_index]
        checkpoint.put_run(
            run.run_id,
            {
                "status": value.get("status", "failed"),
                "input_fingerprint": input_fingerprints[manifest_index],
                "command": task["command"],
                "project_option_signature": _local_resume_option_signature(options),
                "result": value,
            },
        )
        if value.get("status") == "success" and not options.get("keep_cache"):
            if options["mode"] == "hybrid":
                remove_cache_layers(
                    task["paths"], ("raw", "strict", "candidate")
                )

    raw, _started, scheduler_summary = run_adaptive_process_tasks(
        _project_worker_budgeted,
        task_arguments(),
        effective_workers,
        options["_max_memory_bytes"],
        None if options["continue_on_error"] else lambda result: (
            isinstance(result, WorkerFailure) or result.get("status") == "failed"
        ),
        on_result=checkpoint_local_result,
        on_start=checkpoint_local_start,
        heartbeat_seconds=float(options.get("scheduler_heartbeat_seconds", 30)),
    )
    options["_local_scheduler_summary"] = {
        "initial": scheduler_summary,
        "refreshes": [],
    }
    logger.info(
        "Project adaptive manager: requested=%d effective=%d summary=%s",
        int(options.get("workers", 4)),
        effective_workers,
        scheduler_summary,
    )
    results = dict(skipped)
    for task_position, value in raw.items():
        manifest_index = tasks[task_position][0]
        if isinstance(value, WorkerFailure):
            task = tasks[task_position][1]
            value = {
                "run_id": task["run_id"],
                "status": "failed",
                "runtime_sec": None,
                "returncode": None,
                "error": "%s: %s\n%s" % (value.exception_type, value.message, value.traceback_text),
                "paths": task["paths"],
                "command": task["command"],
            }
        logger.debug(
            "Project child complete: run_id=%s status=%s runtime_sec=%s returncode=%s "
            "allocated_workers=%s stdout_tail_chars=%d stderr_tail_chars=%d",
            value["run_id"],
            value["status"],
            value.get("runtime_sec"),
            value.get("returncode"),
            value.get("allocated_workers"),
            len(value.get("stdout_tail", "")),
            len(value.get("stderr_tail", "")),
        )
        results[manifest_index] = value
    for index, run in enumerate(runs):
        if index not in results:
            results[index] = {
                "run_id": run.run_id,
                "status": "not_run",
                "runtime_sec": None,
                "returncode": None,
                "error": "not scheduled after an earlier failure",
                "paths": _run_paths(
                    run,
                    output_dir,
                    cache_workspace,
                    options.get("format", "parquet"),
                    write_ms1=bool(options.get("write_ms1")),
                ),
                "command": [],
            }
        logger.info(
            "Project run %s: %s%s",
            run.run_id,
            results[index]["status"],
            "" if not results[index].get("error") else " - " + results[index]["error"].splitlines()[-1],
        )

    external_stage = None
    if (
        options["mode"] == "hybrid"
        and options.get("external_id", False)
        and not any(
            result["status"] in {"failed", "not_run"}
            for result in results.values()
        )
    ):
        logger.info("Starting project-level RT alignment and external-ID stage")
        external_stage = _run_external_stage(runs, results, options)
        logger.info("Project-level external-ID stage complete")
    _write_project_database(
        database,
        runs,
        results,
        options,
        external_stage=external_stage,
        input_fingerprints=input_fingerprints,
    )
    if any(result["status"] in {"failed", "not_run"} for result in results.values()):
        raise RuntimeError("one or more project runs failed; inspect the project database")
    if checkpoint:
        checkpoint.release()
    return results
