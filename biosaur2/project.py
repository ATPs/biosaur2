"""Bounded project execution, compact project metadata and validation."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import pyarrow.parquet as pq

from .output import publish_staged_files, _temporary_neighbor
from .cache_runtime import ProjectCheckpoint, remove_cache_layers, run_cache_paths
from .parallel import (
    WorkerFailure,
    effective_worker_budget,
    physical_memory_bytes,
    run_adaptive_process_tasks,
)
from .project_manifest import read_manifest
from .raw_ms1 import source_fingerprint
from .external_mbr import run_feature_mbr_stage
from .project_validation import _read_output_table, validate_project as validate_project
from .schema import PROJECT_SCHEMA_VERSION


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


def _summary_for_result(result):
    paths = result["paths"]
    summary = {
        "feature_count": None,
        "ms2_count": None,
        "audit_count": None,
        "linked_ms2_count": None,
        "quant_feature_count": None,
        "accepted_identification_count": None,
        "direct_assay_count": None,
        "hybrid_summary": None,
    }
    if result["status"] not in {"success", "skipped_resume"}:
        return summary
    feature_path = Path(paths["features"])
    if not feature_path.is_file():
        return summary
    if paths.get("format") == "tsv":
        with feature_path.open("r", encoding="utf-8", newline="") as handle:
            feature_rows = list(csv.DictReader(handle, delimiter="\t"))
        summary["feature_count"] = len(feature_rows)
        summary["quant_feature_count"] = sum(
            bool(row.get("quant_value")) for row in feature_rows
        )
        ms2_events_path = Path(paths["ms2_events"])
        if ms2_events_path.is_file():
            with ms2_events_path.open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                summary["linked_ms2_count"] = sum(
                    1 for _ in csv.DictReader(handle, delimiter="\t")
                )
        summary["ms2_count"] = summary["linked_ms2_count"]
        summary["audit_count"] = summary["linked_ms2_count"]
        identification_path = Path(paths["identifications"])
        if identification_path.is_file():
            with identification_path.open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                identification_rows = list(csv.DictReader(handle, delimiter="\t"))
            summary["accepted_identification_count"] = len(identification_rows)
            summary["direct_assay_count"] = sum(
                bool(row.get("assay_id")) for row in identification_rows
            )
        return summary
    features = _read_output_table(feature_path, "features")
    feature_rows = features.to_pylist()
    summary["feature_count"] = len(feature_rows)
    summary["quant_feature_count"] = len(feature_rows)
    linked = 0
    ms2_events_path = Path(paths["ms2_events"])
    if ms2_events_path.is_file():
        linked = _read_output_table(
            ms2_events_path, "ms2_events"
        ).num_rows
    summary["ms2_count"] = linked
    summary["audit_count"] = linked
    summary["linked_ms2_count"] = linked
    identification_path = Path(paths["identifications"])
    if identification_path.is_file():
        try:
            identifications = _read_output_table(
                identification_path, "identifications"
            )
        except Exception:
            identifications = None
        if identifications is not None:
            identification_rows = identifications.to_pylist()
            summary["accepted_identification_count"] = len(identification_rows)
            summary["direct_assay_count"] = sum(
                row.get("assay_id") is not None for row in identification_rows
            )
    if feature_path.suffix.lower() == ".duckdb":
        import duckdb

        with duckdb.connect(str(feature_path), read_only=True) as connection:
            provenance = json.loads(
                connection.execute(
                    "SELECT provenance_json FROM runs LIMIT 1"
                ).fetchone()[0]
            )
        encoded = provenance.get("hybrid_summary_json")
        if encoded:
            summary["hybrid_summary"] = json.loads(encoded)
    elif feature_path.suffix.lower() == ".parquet":
        metadata = pq.ParquetFile(feature_path).metadata.metadata or {}
        encoded = metadata.get(b"biosaur2_hybrid_summary_json")
        if not encoded and metadata.get(b"biosaur2_provenance_json"):
            provenance = json.loads(
                metadata[b"biosaur2_provenance_json"]
            )
            encoded = provenance.get("hybrid_summary_json")
        if encoded:
            summary["hybrid_summary"] = json.loads(encoded)
    return summary


def _write_project_database(
    database, runs, results, options, *, external_stage=None
):
    import duckdb

    temporary = _temporary_neighbor(database)
    connection = duckdb.connect(str(temporary))
    try:
        connection.execute(
            "CREATE TABLE runs (run_order INTEGER, run_id VARCHAR, mzml_path VARCHAR, "
            "psm_path VARCHAR, status VARCHAR, runtime_sec DOUBLE, error VARCHAR, "
            "cpu_user_sec DOUBLE, cpu_system_sec DOUBLE, peak_rss_kib BIGINT, "
            "allocated_workers INTEGER, "
            "output_format VARCHAR, run_output_path VARCHAR, "
            "features_path VARCHAR, ms2_events_path VARCHAR, "
            "identification_path VARCHAR, ms1_path VARCHAR, "
            "external_evidence_path VARCHAR, raw_ms1_cache_path VARCHAR, "
            "input_fingerprint_json VARCHAR, command_json VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE stage_status (run_id VARCHAR, stage VARCHAR, status VARCHAR, detail VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE identification_summary (run_id VARCHAR, accepted_identification_count BIGINT, "
            "direct_assay_count BIGINT)"
        )
        connection.execute(
            "CREATE TABLE qc_metrics (run_id VARCHAR, feature_count BIGINT, ms2_count BIGINT, "
            "audit_count BIGINT, linked_ms2_count BIGINT, quant_feature_count BIGINT)"
        )
        connection.execute(
            "CREATE TABLE hybrid_summary (run_id VARCHAR, strict_feature_count BIGINT, "
            "direct_assay_count BIGINT, recovered_feature_count BIGINT, audit_row_count BIGINT, "
            "direct_linked_count BIGINT, generic_strict_linked_count BIGINT, "
            "generic_local_linked_count BIGINT, generic_local_new_feature_count BIGINT, "
            "generic_decoy_only_count BIGINT, generic_local_decoy_only_count BIGINT, "
            "audit_status_counts_json VARCHAR, generic_summary_json VARCHAR, summary_json VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE project_metadata (key VARCHAR, value_json VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE scheduler_summary (stage VARCHAR, summary_json VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE rt_alignment_models (alignment_group VARCHAR, "
            "reference_run VARCHAR, source_run VARCHAR, target_run VARCHAR, method VARCHAR, "
            "anchor_count INTEGER, inlier_count INTEGER, slope DOUBLE, "
            "intercept DOUBLE, residual_mad_sec DOUBLE, status VARCHAR, "
            "validation_anchor_count INTEGER, validation_median_bias_sec DOUBLE, "
            "validation_mad_sec DOUBLE, validation_q90_abs_error_sec DOUBLE, "
            "x_knots_json VARCHAR, y_knots_json VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE external_summary (run_id VARCHAR, "
            "planned_assay_count BIGINT, evaluated_assay_count BIGINT, "
            "new_external_feature_count BIGINT, new_strict_external_feature_count BIGINT, "
            "new_weak_external_feature_count BIGINT, status_counts_json VARCHAR)"
        )
        for index, run in enumerate(runs):
            result = results[index]
            paths = result["paths"]
            summary = _summary_for_result(result)
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
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
                    json.dumps(_input_fingerprint(run), sort_keys=True),
                    json.dumps(result.get("command", [])),
                ],
            )
            connection.execute(
                "INSERT INTO stage_status VALUES (?, 'run', ?, ?)",
                [run.run_id, result["status"], result.get("error")],
            )
            if options["mode"] == "hybrid":
                cache_status = (
                    "success"
                    if Path(paths["raw_ms1_cache"], "manifest.json").is_file()
                    else "missing"
                )
                connection.execute(
                    "INSERT INTO stage_status VALUES (?, 'raw_ms1_cache', ?, ?)",
                    [run.run_id, cache_status, paths["raw_ms1_cache"]],
                )
                strict_cache_path = paths.get("strict_stage_cache")
                if strict_cache_path:
                    manifest_path = (
                        None
                        if not strict_cache_path
                        else Path(strict_cache_path, "manifest.json")
                    )
                    if manifest_path is not None and manifest_path.is_file():
                        try:
                            manifest = json.loads(
                                manifest_path.read_text(encoding="utf-8")
                            )
                            strict_cache_status = "success"
                            strict_cache_detail = json.dumps(
                                {
                                    "path": strict_cache_path,
                                    "payload_bytes": manifest.get(
                                        "payload_bytes"
                                    ),
                                    "payload_sha256": manifest.get(
                                        "payload_sha256"
                                    ),
                                    "strict_feature_count": manifest.get(
                                        "strict_feature_count"
                                    ),
                                    "context_count": manifest.get(
                                        "context_count"
                                    ),
                                },
                                sort_keys=True,
                            )
                        except (OSError, TypeError, ValueError) as error:
                            strict_cache_status = "invalid"
                            strict_cache_detail = "%s: %s" % (
                                strict_cache_path,
                                error,
                            )
                    else:
                        strict_cache_status = "missing"
                        strict_cache_detail = strict_cache_path
                else:
                    strict_cache_status = "missing"
                    strict_cache_detail = strict_cache_path
                connection.execute(
                    "INSERT INTO stage_status VALUES "
                    "(?, 'strict_stage_cache', ?, ?)",
                    [
                        run.run_id,
                        strict_cache_status,
                        strict_cache_detail,
                    ],
                )
                external_summary = (external_stage or {}).get(
                    "summaries", {}
                ).get(run.run_id)
                external_status = (
                    "disabled"
                    if not options.get("external_id", False)
                    else (
                        "success"
                        if external_summary is not None
                        else "not_run"
                    )
                )
                connection.execute(
                    "INSERT INTO stage_status VALUES (?, 'external_id', ?, ?)",
                    [
                        run.run_id,
                        external_status,
                        paths.get("external_evidence"),
                    ],
                )
            connection.execute(
                "INSERT INTO identification_summary VALUES (?, ?, ?)",
                [run.run_id, summary["accepted_identification_count"], summary["direct_assay_count"]],
            )
            connection.execute(
                "INSERT INTO qc_metrics VALUES (?, ?, ?, ?, ?, ?)",
                [
                    run.run_id,
                    summary["feature_count"],
                    summary["ms2_count"],
                    summary["audit_count"],
                    summary["linked_ms2_count"],
                    summary["quant_feature_count"],
                ],
            )
            hybrid = summary["hybrid_summary"] or {}
            statuses = hybrid.get("audit_status_counts", {})
            generic = hybrid.get("generic_summary") or {}
            direct_linked = sum(
                int(statuses.get(status, 0))
                for status in (
                    "matched_strict_feature",
                    "recovered_direct_feature",
                    "recovered_direct_relaxed_two_point",
                    "recovered_direct_relaxed_partial_envelope",
                    "matched_recovered_feature",
                    "matched_recovered_feature_ambiguous_identity",
                )
            )
            connection.execute(
                "INSERT INTO hybrid_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    run.run_id,
                    hybrid.get("strict_feature_count"),
                    hybrid.get("direct_assay_count"),
                    hybrid.get("recovered_feature_count"),
                    hybrid.get("audit_row_count"),
                    direct_linked,
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
                ],
            )
            external = (external_stage or {}).get("summaries", {}).get(
                run.run_id, {}
            )
            connection.execute(
                "INSERT INTO external_summary VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    run.run_id,
                    external.get("planned_assay_count"),
                    external.get("evaluated_assay_count"),
                    external.get("new_external_feature_count"),
                    external.get("new_strict_external_feature_count"),
                    external.get("new_weak_external_feature_count"),
                    json.dumps(
                        external.get("status_counts", {}), sort_keys=True
                    ),
                ],
            )
        alignment_rows = (external_stage or {}).get("alignment_models", ())
        if alignment_rows:
            connection.executemany(
                "INSERT INTO rt_alignment_models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
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
                    for row in alignment_rows
                ],
            )
        for stage, summary in (
            ("local", options.get("_local_scheduler_summary", {})),
            ("external", (external_stage or {}).get("scheduler_summary", {})),
        ):
            connection.execute(
                "INSERT INTO scheduler_summary VALUES (?, ?)",
                [stage, json.dumps(summary, sort_keys=True)],
            )
        connection.execute(
            "INSERT INTO project_metadata VALUES ('project_schema_version', ?)",
            [PROJECT_SCHEMA_VERSION],
        )
        connection.execute(
            "INSERT INTO project_metadata VALUES ('resolved_options', ?)",
            [json.dumps(options, sort_keys=True, default=str)],
        )
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
        if run.psm_format not in {None, "percolator_tsv"}:
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
            and resume_record["input_fingerprint"] == _input_fingerprint(run)
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
        run = runs[tasks[task_position][0]]
        checkpoint.put_run(
            run.run_id,
            {
                "status": "pending",
                "input_fingerprint": _input_fingerprint(run),
                "command": task["command"],
                "project_option_signature": _local_resume_option_signature(options),
            },
        )

    def checkpoint_local_result(task_position, value):
        if not checkpoint or isinstance(value, WorkerFailure):
            return
        task = tasks[task_position][1]
        run = runs[tasks[task_position][0]]
        checkpoint.put_run(
            run.run_id,
            {
                "status": value.get("status", "failed"),
                "input_fingerprint": _input_fingerprint(run),
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
        database, runs, results, options, external_stage=external_stage
    )
    if any(result["status"] in {"failed", "not_run"} for result in results.values()):
        raise RuntimeError("one or more project runs failed; inspect the project database")
    if checkpoint:
        checkpoint.release()
    return results
