"""Bounded project execution, compact project metadata and validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import resource
import subprocess
import sys
import time

import pyarrow.parquet as pq

from .output import input_stem, publish_staged_files, _temporary_neighbor
from .cache_runtime import run_cache_paths
from .parallel import (
    WorkerFailure,
    effective_worker_budget,
    run_bounded_process_tasks,
    run_budgeted_process_tasks,
)
from .project_manifest import read_manifest
from .raw_ms1 import source_fingerprint
from .external import (
    alignment_model_rows,
    build_alignment_models,
    choose_group_reference_runs,
    plan_external_assays,
    read_external_observations,
    run_external_recipient,
)


logger = logging.getLogger(__name__)


def _scientific_command(command):
    """Remove scheduling/cache-location arguments from a run command."""

    normalized = []
    index = 0
    while index < len(command):
        option = command[index]
        if option in {"--workers", "--cache-dir"}:
            index += 2
            continue
        if option == "--keep-cache":
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
        "cache_dir",
        "keep_cache",
        "_cache_workspace",
    }
    return {
        key: value
        for key, value in options.items()
        if key not in ignored
    }


def _project_worker(task):
    started = time.monotonic()
    command = task.get("execution_command", task["command"])
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    completed = subprocess.run(command, text=True, capture_output=True)
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "run_id": task["run_id"],
        "status": "success" if completed.returncode == 0 else "failed",
        "runtime_sec": time.monotonic() - started,
        "cpu_user_sec": usage_after.ru_utime - usage_before.ru_utime,
        "cpu_system_sec": usage_after.ru_stime - usage_before.ru_stime,
        # Linux reports ru_maxrss in KiB. Project mode currently launches one
        # analysis subprocess per fresh worker, so this is that run's peak RSS.
        "peak_rss_kib": usage_after.ru_maxrss,
        "returncode": completed.returncode,
        "error": None if completed.returncode == 0 else completed.stderr[-8000:],
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "paths": task["paths"],
        "command": command,
    }


def _project_worker_budgeted(task, allocated_workers):
    execution_task = dict(task)
    execution_command = list(task["command"]) + [
        "--workers",
        str(allocated_workers),
    ]
    if task.get("cache_root"):
        execution_command.extend(
            ("--cache-dir", task["cache_root"], "--keep-cache")
        )
    execution_task["execution_command"] = execution_command
    result = _project_worker(execution_task)
    result["allocated_workers"] = allocated_workers
    return result


def _run_paths(run, output_dir, cache_workspace=None):
    directory = output_dir / run.run_id
    stem = input_stem(str(run.mzml_path))
    cache_paths = run_cache_paths(
        cache_workspace or (output_dir / ".biosaur2_cache"),
        run.mzml_path,
    )
    paths = {
        "run_dir": str(directory),
        "features": str(directory / (run.run_id + ".features.parquet")),
        "ms2": str(directory / (stem + ".ms2.parquet")),
        "audit": str(directory / (stem + ".ms2_feature_links.parquet")),
        "feature_quant": str(directory / (stem + ".feature_quant.parquet")),
        "identifications": str(directory / (stem + ".identifications.parquet")),
        "assays": str(directory / (stem + ".id_assays.parquet")),
        "external_evidence": str(
            directory / (stem + ".external_id_evidence.parquet")
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
        paths["features"],
        "--feature-format",
        "parquet",
        "--feature-mode",
        options["mode"],
        "--max-charge",
        str(options.get("max_charge", 7)),
    ]
    if options["overwrite"]:
        command.append("--overwrite")
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
        command.extend(("--feature-baseline", options["feature_baseline"]))
        command.append("--direct-id" if options["direct_id"] else "--no-direct-id")
        command.append("--external-id" if options["external_id"] else "--no-external-id")
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
            option_signature = _resume_option_signature(prior_options)
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
    for key, metric in (
        ("features", "feature_count"),
        ("ms2", "ms2_count"),
        ("audit", "audit_count"),
        ("feature_quant", "quant_feature_count"),
        ("identifications", "accepted_identification_count"),
        ("assays", "direct_assay_count"),
    ):
        path = Path(paths[key])
        if path.is_file():
            summary[metric] = pq.ParquetFile(path).metadata.num_rows
    audit_path = Path(paths["audit"])
    if audit_path.is_file():
        parquet_file = pq.ParquetFile(audit_path)
        table = parquet_file.read(columns=["feature_id"])
        summary["linked_ms2_count"] = sum(value is not None for value in table.column(0).to_pylist())
        metadata = parquet_file.metadata.metadata or {}
        encoded = metadata.get(b"biosaur2_hybrid_summary_json")
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
            "features_path VARCHAR, ms2_path VARCHAR, audit_path VARCHAR, "
            "feature_quant_path VARCHAR, identification_path VARCHAR, assay_path VARCHAR, "
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
            "CREATE TABLE rt_alignment_models (alignment_group VARCHAR, "
            "reference_run VARCHAR, source_run VARCHAR, target_run VARCHAR, method VARCHAR, "
            "anchor_count INTEGER, inlier_count INTEGER, slope DOUBLE, "
            "intercept DOUBLE, residual_mad_sec DOUBLE, status VARCHAR, "
            "x_knots_json VARCHAR, y_knots_json VARCHAR)"
        )
        connection.execute(
            "CREATE TABLE external_summary (run_id VARCHAR, "
            "planned_assay_count BIGINT, evaluated_assay_count BIGINT, "
            "new_external_feature_count BIGINT, status_counts_json VARCHAR)"
        )
        for index, run in enumerate(runs):
            result = results[index]
            paths = result["paths"]
            summary = _summary_for_result(result)
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    paths["features"],
                    paths["ms2"],
                    paths["audit"],
                    paths["feature_quant"],
                    paths["identifications"],
                    paths["assays"],
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
                "INSERT INTO external_summary VALUES (?, ?, ?, ?, ?)",
                [
                    run.run_id,
                    external.get("planned_assay_count"),
                    external.get("evaluated_assay_count"),
                    external.get("new_external_feature_count"),
                    json.dumps(
                        external.get("status_counts", {}), sort_keys=True
                    ),
                ],
            )
        for row in (external_stage or {}).get("alignment_models", ()):
            connection.execute(
                "INSERT INTO rt_alignment_models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    row["x_knots_json"],
                    row["y_knots_json"],
                ],
            )
        connection.execute(
            "INSERT INTO project_metadata VALUES ('project_schema_version', '3')"
        )
        connection.execute(
            "INSERT INTO project_metadata VALUES ('resolved_options', ?)",
            [json.dumps(options, sort_keys=True, default=str)],
        )
    finally:
        connection.close()
    publish_staged_files([(temporary, database)])


def _run_external_stage(runs, results, options):
    successful_runs = [
        run
        for index, run in enumerate(runs)
        if results[index]["status"] in {"success", "skipped_resume"}
    ]
    result_index = {run.run_id: index for index, run in enumerate(runs)}
    observations = {
        run.run_id: read_external_observations(
            run, results[result_index[run.run_id]]["paths"]
        )
        for run in successful_runs
    }
    models = build_alignment_models(
        successful_runs,
        observations,
        min_anchors=int(options.get("external_alignment_min_anchors", 5)),
        max_residual_mad_sec=float(
            options.get("external_alignment_max_mad_sec", 30.0)
        ),
    )
    reference_runs = choose_group_reference_runs(successful_runs, observations)
    plans = plan_external_assays(successful_runs, observations, models)
    tasks = []
    for run in successful_runs:
        paths = results[result_index[run.run_id]]["paths"]
        tasks.append(
            {
                "run": run,
                "paths": paths,
                "plans": plans.get(run.run_id, ()),
                "options": {
                    "ppm": options.get("external_ppm", 8.0),
                    "rt_tolerance_sec": options.get(
                        "ms2_rt_tolerance_sec", 120.0
                    ),
                    "min_isotope_cosine": options.get(
                        "external_min_isotope_cosine", 0.8
                    ),
                    "q_value_max": options.get(
                        "external_q_value_max", 0.01
                    ),
                    "quant_method": options["quant_method"],
                    "baseline": options["feature_baseline"],
                },
            }
        )
    raw, _started = run_bounded_process_tasks(
        run_external_recipient,
        ((task,) for task in tasks),
        min(
            int(options.get("_effective_workers", options.get("workers", 4))),
            max(1, len(tasks)),
        ),
        lambda result: isinstance(result, WorkerFailure),
    )
    summaries = {}
    failures = []
    for position, value in raw.items():
        run_id = tasks[position]["run"].run_id
        if isinstance(value, WorkerFailure):
            failures.append(
                "%s: %s: %s" % (
                    run_id,
                    value.exception_type,
                    value.message,
                )
            )
        else:
            summaries[run_id] = value
    if failures:
        raise RuntimeError("external-ID stage failed: " + "; ".join(failures))
    for run_id in sorted(summaries):
        summary = summaries[run_id]
        logger.info(
            "External-ID run %s: planned=%d evaluated=%d new_features=%d statuses=%s",
            run_id,
            summary["planned_assay_count"],
            summary["evaluated_assay_count"],
            summary["new_external_feature_count"],
            summary["status_counts"],
        )
    return {
        "summaries": summaries,
        "alignment_models": alignment_model_rows(models, reference_runs),
        "reference_runs": reference_runs,
    }


def run_project(manifest, output_dir, project_db, **options):
    runs = read_manifest(manifest)
    output_dir = Path(output_dir).resolve()
    database = Path(project_db).resolve()
    cache_workspace = Path(
        options.get("_cache_workspace")
        or options.get("cache_dir")
        or ".biosaur2_cache"
    ).resolve()
    effective_workers = effective_worker_budget(int(options.get("workers", 4)))
    options["_effective_workers"] = effective_workers
    output_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        if not run.mzml_path.is_file():
            raise FileNotFoundError("mzML input does not exist: %s" % run.mzml_path)
        if run.psm_path is not None and not run.psm_path.is_file():
            raise FileNotFoundError("PSM input does not exist: %s" % run.psm_path)
        if run.psm_format not in {None, "percolator_tsv"}:
            raise ValueError("unsupported psm_format for %s: %s" % (run.run_id, run.psm_format))

    successful = _read_successful_runs(database) if options["resume"] else {}
    tasks = []
    skipped = {}
    for index, run in enumerate(runs):
        paths = _run_paths(run, output_dir, cache_workspace)
        command = _command_for_run(run, paths, options)
        resume_record = successful.get(run.run_id)
        required_paths = [paths["features"]]
        if options["mode"] == "hybrid":
            required_paths.extend(
                [paths["ms2"], paths["audit"], paths["feature_quant"]]
            )
        resume_valid = (
            resume_record is not None
            and resume_record["input_fingerprint"] == _input_fingerprint(run)
            and _scientific_command(resume_record["command"]) == command
            and resume_record.get("project_option_signature")
            == _resume_option_signature(options)
            and all(Path(path).is_file() for path in required_paths)
        )
        if resume_valid:
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
                },
            )
        )

    def task_arguments():
        for _index, task in tasks:
            yield (task,)

    raw, _started, allocations = run_budgeted_process_tasks(
        _project_worker_budgeted,
        task_arguments(),
        effective_workers,
        None if options["continue_on_error"] else lambda result: (
            isinstance(result, WorkerFailure) or result.get("status") == "failed"
        ),
    )
    logger.info(
        "Project worker budget: requested=%d effective=%d allocations=%s",
        int(options.get("workers", 4)),
        effective_workers,
        allocations,
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
        results[manifest_index] = value
    for index, run in enumerate(runs):
        if index not in results:
            results[index] = {
                "run_id": run.run_id,
                "status": "not_run",
                "runtime_sec": None,
                "returncode": None,
                "error": "not scheduled after an earlier failure",
                "paths": _run_paths(run, output_dir, cache_workspace),
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
    return results


def validate_project(project_db):
    import duckdb

    database = Path(project_db).resolve()
    problems = []
    with duckdb.connect(str(database), read_only=True) as connection:
        run_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info('runs')").fetchall()
        }
        external_column = (
            "external_evidence_path"
            if "external_evidence_path" in run_columns
            else "NULL"
        )
        runs = connection.execute(
            "SELECT run_id, status, features_path, ms2_path, audit_path, "
            "feature_quant_path, %s FROM runs ORDER BY run_order"
            % external_column
        ).fetchall()
    for run_id, status, features, ms2, audit, quant, external in runs:
        if status not in {"success", "skipped_resume"}:
            problems.append("%s has status %s" % (run_id, status))
            continue
        for path in (features,):
            if not Path(path).is_file():
                problems.append("%s is missing output %s" % (run_id, path))
        if Path(ms2).is_file() and Path(audit).is_file():
            ms2_count = pq.ParquetFile(ms2).metadata.num_rows
            audit_table = pq.read_table(audit, columns=["ms2_event_id"])
            ids = audit_table.column(0).to_pylist()
            if len(ids) != ms2_count or len(ids) != len(set(ids)):
                problems.append("%s does not have exactly one audit row per MS2" % run_id)
        if Path(quant).is_file():
            ids = pq.read_table(quant, columns=["feature_id"]).column(0).to_pylist()
            if any(value is None or value <= 0 for value in ids) or len(ids) != len(set(ids)):
                problems.append("%s has invalid/duplicate quant feature IDs" % run_id)
            feature_ids = pq.read_table(
                features, columns=["feature_idx"]
            ).column(0).to_pylist()
            if (
                any(value is None or value <= 0 for value in feature_ids)
                or len(feature_ids) != len(set(feature_ids))
            ):
                problems.append("%s has invalid/duplicate feature IDs" % run_id)
            if set(ids) != set(feature_ids):
                problems.append(
                    "%s feature and quant feature-ID sets differ" % run_id
                )
        if external and Path(external).is_file():
            external_rows = pq.read_table(
                external,
                columns=["status", "feature_id", "extraction_q_value"],
            ).to_pylist()
            for row in external_rows:
                if row["status"].startswith("accepted_") and (
                    row["feature_id"] is None
                    or row["feature_id"] <= 0
                    or row["extraction_q_value"] is None
                ):
                    problems.append(
                        "%s has an invalid accepted external-ID evidence row"
                        % run_id
                    )
                    break
    if problems:
        raise ValueError("project validation failed: " + "; ".join(problems))
    return {"run_count": len(runs), "problems": ()}
