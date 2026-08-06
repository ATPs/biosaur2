"""Read-only validation for completed Project outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import (
    LINKED_MS2_EVENT_COLUMNS,
    PROJECT_SCHEMA_VERSION,
    compact_schemas,
)


def _parse_tsv_value(value, data_type):
    if value == "":
        return None
    if pa.types.is_dictionary(data_type):
        data_type = data_type.value_type
    if pa.types.is_list(data_type) or pa.types.is_struct(data_type):
        return json.loads(value)
    if pa.types.is_integer(data_type):
        return int(value)
    if pa.types.is_floating(data_type):
        return float(value)
    if pa.types.is_boolean(data_type):
        return value.lower() in {"1", "true", "yes"}
    return value


def _read_output_table(path, table_name):
    """Read one normal run output table independent of its container format."""

    source = Path(path)
    if source.suffix.lower() == ".duckdb":
        import duckdb

        with duckdb.connect(str(source), read_only=True) as connection:
            return connection.execute(
                'SELECT * FROM "%s"' % table_name.replace('"', '""')
            ).fetch_arrow_table()
    if source.suffix.lower() == ".tsv":
        schema_name = {
            "features": "hybrid_features",
            "ms1": "ms1",
            "ms2_events": "linked_ms2_events",
            "identifications": "merged_identifications",
        }[table_name]
        schema = compact_schemas()[schema_name]
        fields = {field.name: field for field in schema}
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = [
                {
                    name: _parse_tsv_value(value, fields[name].type)
                    for name, value in raw.items()
                }
                for raw in csv.DictReader(handle, delimiter="\t")
            ]
        return pa.Table.from_pylist(rows, schema=schema)
    return pq.read_table(source)


def validate_project(project_db):
    import duckdb

    database = Path(project_db).resolve()
    problems = []
    with duckdb.connect(str(database), read_only=True) as connection:
        metadata = dict(
            connection.execute(
                "SELECT key, value_json FROM project_metadata"
            ).fetchall()
        )
        project_schema = metadata.get("project_schema_version")
        if project_schema != PROJECT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported Project schema %s; expected %s"
                % (project_schema or "missing", PROJECT_SCHEMA_VERSION)
            )
        runs = connection.execute(
            "SELECT run_id, status, output_format, run_output_path, "
            "features_path, ms2_events_path, identification_path, ms1_path, "
            "external_evidence_path "
            "FROM runs ORDER BY run_order"
        ).fetchall()
    resolved_options = json.loads(metadata.get("resolved_options", "{}"))
    mode = resolved_options.get("mode", "legacy")
    external_expected = (
        mode == "hybrid" and resolved_options.get("external_id", True)
    )
    ms1_expected = bool(
        resolved_options.get("write_ms1", mode == "hybrid")
    )
    for (
        run_id,
        status,
        output_format,
        run_output,
        features,
        ms2_events,
        identifications,
        ms1,
        external,
    ) in runs:
        if status not in {"success", "skipped_resume"}:
            problems.append("%s has status %s" % (run_id, status))
            continue
        primary = Path(run_output or features)
        if not primary.is_file():
            problems.append("%s is missing output %s" % (run_id, primary))
            continue
        ms1_ids = None
        if ms1_expected:
            if not ms1 or not Path(ms1).is_file():
                problems.append(
                    "%s is missing MS1 output %s" % (run_id, ms1 or "")
                )
            else:
                try:
                    ms1_table = _read_output_table(ms1, "ms1")
                    expected_schema = compact_schemas()["ms1"]
                    if ms1_table.schema.remove_metadata() != expected_schema:
                        problems.append(
                            "%s has an unexpected MS1 schema" % run_id
                        )
                    else:
                        scan_ids = ms1_table.column("scan_id").to_pylist()
                        if (
                            any(value is None for value in scan_ids)
                            or len(scan_ids) != len(set(scan_ids))
                        ):
                            problems.append(
                                "%s has null/duplicate MS1 scan IDs" % run_id
                            )
                        else:
                            ms1_ids = set(scan_ids)
                except Exception as error:
                    problems.append(
                        "%s has an unreadable MS1 table: %s"
                        % (run_id, error)
                    )
        if mode == "legacy" and output_format == "tsv":
            continue
        feature_table = _read_output_table(features, "features")
        feature_ids = feature_table.column("feature_idx").to_pylist()
        if (
            any(value is None or value <= 0 for value in feature_ids)
            or len(feature_ids) != len(set(feature_ids))
        ):
            problems.append("%s has invalid/duplicate feature IDs" % run_id)
        if mode == "hybrid":
            removed_rt = {
                "rt_start_sec", "rt_apex_sec", "rt_end_sec"
            }.intersection(feature_table.column_names)
            if removed_rt:
                problems.append(
                    "%s still exposes Hybrid RT-second columns" % run_id
                )
            scan_columns = ("scanStart", "scanApex", "scanEnd")
            if any(
                name not in feature_table.column_names
                for name in scan_columns
            ):
                problems.append(
                    "%s is missing Hybrid MS1 scan columns" % run_id
                )
            else:
                scan_values = {}
                for name in scan_columns:
                    scan_values[name] = feature_table.column(name).to_pylist()
                if any(
                    value is None
                    for values in scan_values.values()
                    for value in values
                ):
                    problems.append(
                        "%s has null Hybrid MS1 scan IDs" % run_id
                    )
                if ms1_expected and ms1_ids is not None and any(
                    value is not None and value not in ms1_ids
                    for values in scan_values.values()
                    for value in values
                ):
                    problems.append(
                        "%s has feature scan IDs absent from MS1 output"
                        % run_id
                    )
            if "ms2_events" in feature_table.column_names:
                problems.append(
                    "%s still embeds ms2_events in its feature table" % run_id
                )
            if not Path(ms2_events).is_file():
                problems.append(
                    "%s is missing linked MS2 output %s"
                    % (run_id, ms2_events)
                )
            else:
                try:
                    event_table = _read_output_table(ms2_events, "ms2_events")
                    if event_table.column_names != list(LINKED_MS2_EVENT_COLUMNS):
                        problems.append(
                            "%s has an unexpected linked MS2 schema" % run_id
                        )
                    else:
                        linked_feature_ids = event_table.column(
                            "feature_idx"
                        ).to_pylist()
                        event_ids = event_table.column(
                            "ms2_event_id"
                        ).to_pylist()
                        feature_id_set = set(feature_ids)
                        if any(
                            value is None or value not in feature_id_set
                            for value in linked_feature_ids
                        ):
                            problems.append(
                                "%s has orphan linked MS2 feature IDs" % run_id
                            )
                        if (
                            any(value is None or value < 0 for value in event_ids)
                            or len(event_ids) != len(set(event_ids))
                        ):
                            problems.append(
                                "%s has invalid/duplicate linked MS2 event IDs"
                                % run_id
                            )
                except Exception as error:
                    problems.append(
                        "%s has an unreadable linked MS2 table: %s"
                        % (run_id, error)
                    )
            if not Path(identifications).is_file():
                problems.append(
                    "%s is missing identifications output %s"
                    % (run_id, identifications)
                )
            else:
                try:
                    _read_output_table(identifications, "identifications")
                except Exception as error:
                    problems.append(
                        "%s has an unreadable identifications table: %s"
                        % (run_id, error)
                    )
        external_rows = []
        if external_expected and (
            not external or not Path(external).is_file()
        ):
            problems.append(
                "%s is missing external evidence output %s"
                % (run_id, external or "")
            )
        elif external and Path(external).is_file():
            if output_format == "duckdb":
                try:
                    with duckdb.connect(str(external), read_only=True) as connection:
                        external_rows = connection.execute(
                            "SELECT status, feature_id, acceptance_q_value "
                            "FROM external_id_evidence"
                        ).fetch_arrow_table().to_pylist()
                except Exception as error:
                    if external_expected:
                        problems.append(
                            "%s has an unreadable external evidence table: %s"
                            % (run_id, error)
                        )
            elif Path(external).suffix.lower() == ".parquet":
                external_rows = pq.read_table(
                    external,
                    columns=[
                        "status", "feature_id", "acceptance_q_value",
                    ],
                ).to_pylist()
            elif Path(external).suffix.lower() == ".tsv":
                with Path(external).open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    external_rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in external_rows:
            accepted_q_value = row.get("acceptance_q_value")
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
