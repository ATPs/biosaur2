"""Read-only validation for completed Project outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

from .external import _read_table


def validate_project(project_db):
    import duckdb

    database = Path(project_db).resolve()
    problems = []
    with duckdb.connect(str(database), read_only=True) as connection:
        runs = connection.execute(
            "SELECT run_id, status, output_format, run_output_path, "
            "features_path, identification_path, external_evidence_path "
            "FROM runs ORDER BY run_order"
        ).fetchall()
        metadata = dict(
            connection.execute(
                "SELECT key, value_json FROM project_metadata"
            ).fetchall()
        )
    mode = json.loads(metadata.get("resolved_options", "{}")).get(
        "mode", "legacy"
    )
    for (
        run_id,
        status,
        output_format,
        run_output,
        features,
        identifications,
        external,
    ) in runs:
        if status not in {"success", "skipped_resume"}:
            problems.append("%s has status %s" % (run_id, status))
            continue
        primary = Path(run_output or features)
        if not primary.is_file():
            problems.append("%s is missing output %s" % (run_id, primary))
            continue
        if mode == "legacy" and output_format == "tsv":
            continue
        feature_table = _read_table(features, "features")
        feature_ids = feature_table.column("feature_idx").to_pylist()
        if (
            any(value is None or value <= 0 for value in feature_ids)
            or len(feature_ids) != len(set(feature_ids))
        ):
            problems.append("%s has invalid/duplicate feature IDs" % run_id)
        if mode == "hybrid":
            if not Path(identifications).is_file():
                problems.append(
                    "%s is missing identifications output %s"
                    % (run_id, identifications)
                )
            else:
                try:
                    _read_table(identifications, "identifications")
                except Exception as error:
                    problems.append(
                        "%s has an unreadable identifications table: %s"
                        % (run_id, error)
                    )
        external_rows = []
        if external and Path(external).is_file():
            if output_format == "duckdb":
                try:
                    with duckdb.connect(str(external), read_only=True) as connection:
                        external_rows = connection.execute(
                            "SELECT status, feature_id, acceptance_q_value, extraction_q_value "
                            "FROM external_id_evidence"
                        ).fetch_arrow_table().to_pylist()
                except Exception:
                    external_rows = []
            elif Path(external).suffix.lower() == ".parquet":
                external_rows = pq.read_table(
                    external,
                    columns=[
                        "status", "feature_id", "acceptance_q_value",
                        "extraction_q_value",
                    ],
                ).to_pylist()
            elif Path(external).suffix.lower() == ".tsv":
                with Path(external).open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    external_rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in external_rows:
            accepted_q_value = row.get("acceptance_q_value")
            if accepted_q_value in {None, ""}:
                accepted_q_value = row.get("extraction_q_value")
            if row["status"].startswith("accepted_") and (
                row["feature_id"] in {None, ""}
                or int(row["feature_id"]) <= 0
                or accepted_q_value in {None, ""}
            ):
                problems.append(
                    "%s has an invalid accepted external-ID evidence row"
                    % run_id
                )
                break
    if problems:
        raise ValueError("project validation failed: " + "; ".join(problems))
    return {"run_count": len(runs), "problems": ()}
