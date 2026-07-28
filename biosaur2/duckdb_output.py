"""Optional DuckDB V2 writer for compact outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping

import pyarrow as pa

from .legacy_output import (
    _CompactParquetSink,
    compact_feature,
    compact_hill,
    compact_ms1,
    compact_ms2,
    compact_sort_key,
    row_batches,
)
from .output import (
    _TsvSink,
    _temporary_neighbor,
    build_provenance,
    input_stem,
    ms2_output_path,
    publish_staged_files,
)
from .schema import (
    MS1_COLUMNS,
    MS2_COLUMNS,
    MS2_SCHEMA_VERSION,
    compact_schemas,
    feature_columns,
    hill_columns,
)


def _quoted_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


class DuckDBOutputManager:
    """Write one compact Parquet per requested output or one DuckDB database."""

    def __init__(self, args: Mapping[str, Any]):
        import duckdb

        self.duckdb = duckdb
        self.args = args
        self.overwrite = bool(args.get("overwrite"))
        self.database_mode = bool(args.get("duckdb_output"))
        self.schemas = compact_schemas(
            use64=bool(args.get("use64")),
            include_mono=not args.get("no_mono_hills"),
            extra_details=bool(args.get("write_extra_details")),
            include_hill_lists=not args.get("no_hill_list"),
        )
        self.prefix = self._prefix()
        self.table_names = self._table_names()
        self.tsv_sinks = self._tsv_sinks()
        self.ms2_sink = self._ms2_sink()
        self.final_paths = self._final_paths()
        self._preflight()
        self.staging_path = self._staging_path()
        self.connection = None
        self.temp_outputs: Dict[str, Path] = {}
        self.provenance = None
        self._staged = False

    def _prefix(self):
        explicit = self.args.get("o")
        if explicit:
            path = Path(explicit)
            return (
                path.with_suffix("")
                if path.suffix.lower() in {".parquet", ".tsv"}
                else path
            )
        input_path = Path(self.args["file"])
        return input_path.parent / input_stem(str(input_path))

    def _database_target(self):
        requested = Path(self.args["duckdb_output"])
        if requested.suffix.lower() == ".duckdb":
            return requested
        return requested / (input_stem(self.args["file"]) + ".biosaur2.duckdb")

    def _table_names(self):
        if self.database_mode:
            names = [] if self.args.get("stop_after_hills") else ["features"]
            if self.args.get("write_hills"):
                names.append("hills")
            if self.args.get("write_ms1"):
                names.append("ms1")
            return names
        names = []
        if (
            not self.args.get("stop_after_hills")
            and self.args.get("feature_format", "tsv") == "parquet"
        ):
            names.append("features")
        if self.args.get("write_hills") and self.args.get("hills_format") == "parquet":
            names.append("hills")
        if self.args.get("write_ms1") and self.args.get("ms1_format") == "parquet":
            names.append("ms1")
        if self.args.get("write_ms2"):
            names.append("ms2")
        return names

    def _ms2_sink(self):
        if not (self.database_mode and self.args.get("write_ms2")):
            return None
        return _CompactParquetSink(
            ms2_output_path(self.args),
            self.schemas["ms2"],
            self.args,
            ("run_id", "precursor_resolution", "precursor_mz_source"),
        )

    def _tsv_sinks(self):
        if self.database_mode:
            return {}
        sinks = {}
        if (
            not self.args.get("stop_after_hills")
            and self.args.get("feature_format", "tsv") == "tsv"
        ):
            sinks["features"] = _TsvSink(
                self._target("features", "tsv"),
                feature_columns(
                    not self.args.get("no_mono_hills"),
                    bool(self.args.get("write_extra_details")),
                ),
                self.overwrite,
                decimals=self.args.get("tsv_float_decimals", "roundtrip"),
            )
        if self.args.get("write_hills") and self.args.get("hills_format") == "tsv":
            sinks["hills"] = _TsvSink(
                self._target("hills", "tsv"),
                hill_columns(not self.args.get("no_hill_list")),
                self.overwrite,
                decimals=self.args.get("tsv_float_decimals", "roundtrip"),
            )
        if self.args.get("write_ms1") and self.args.get("ms1_format") == "tsv":
            sinks["ms1"] = _TsvSink(
                self._target("ms1", "tsv"),
                MS1_COLUMNS,
                self.overwrite,
                decimals=self.args.get("tsv_float_decimals", "roundtrip"),
            )
        return sinks

    def _target(self, kind, output_format):
        if kind == "ms2":
            return ms2_output_path(self.args)
        explicit = self.args.get("o")
        if kind == "features" and explicit and str(explicit).lower().endswith(
            "." + output_format
        ):
            return Path(explicit)
        return Path("%s.%s.%s" % (self.prefix, kind, output_format))

    def _final_paths(self):
        if self.database_mode:
            return {"database": self._database_target()}
        return {
            name: self._target(name, "parquet") for name in self.table_names
        }

    def _preflight(self):
        values = list(self.final_paths.values()) + [
            sink.final_path for sink in self.tsv_sinks.values()
        ]
        if self.ms2_sink is not None:
            values.append(self.ms2_sink.final_path)
        if len(values) != len(set(values)):
            raise ValueError("DuckDB output paths collide")
        existing = [path for path in values if path.exists()]
        if existing and not self.overwrite:
            raise FileExistsError(
                "Output already exists; use --overwrite: %s"
                % ", ".join(map(str, existing))
            )

    def _staging_path(self):
        if self.database_mode or not self.args.get("parquet_temp_dir"):
            target = (
                self._database_target()
                if self.database_mode
                else Path(str(self.prefix) + ".staging.duckdb")
            )
            return _temporary_neighbor(target)
        directory = Path(self.args["parquet_temp_dir"])
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=".biosaur2.", suffix=".staging.duckdb", dir=directory
        )
        os.close(descriptor)
        Path(name).unlink()
        return Path(name)

    def _ensure_connection(self):
        if self.connection is not None:
            return
        self.connection = self.duckdb.connect(str(self.staging_path))
        for table_name in self.table_names:
            registration = "_schema_" + table_name
            empty = pa.Table.from_batches([], schema=self.schemas[table_name])
            self.connection.register(registration, empty)
            self.connection.execute(
                'CREATE TABLE "%s" AS SELECT * FROM "%s"'
                % (table_name, registration)
            )
            self.connection.unregister(registration)
        self.connection.execute(
            "CREATE TABLE runs (biosaur2_schema_version VARCHAR, "
            "input_path VARCHAR, input_size BIGINT, "
            "parameters_json VARCHAR, provenance_json VARCHAR)"
        )

    def _append(self, table_name: str, rows: Iterable[Mapping[str, Any]]):
        self._ensure_connection()
        rows = list(rows)
        if not rows:
            return
        registration = "_batch_" + table_name
        table = pa.Table.from_pylist(rows, schema=self.schemas[table_name])
        self.connection.register(registration, table)
        try:
            self.connection.execute(
                'INSERT INTO "%s" SELECT * FROM "%s"'
                % (table_name, registration)
            )
        finally:
            self.connection.unregister(registration)

    def append_features(self, rows):
        converted = [compact_feature(row, self.args) for row in rows]
        if "features" in self.table_names:
            self._append("features", converted)
        if "features" in self.tsv_sinks:
            converted.sort(
                key=lambda row: compact_sort_key(
                    row, "features", self.args.get("parquet_sort", "mz_rt")
                )
            )
            self.tsv_sinks["features"].append(converted)

    def append_hills(self, rows):
        for batch in row_batches(rows):
            converted = [compact_hill(row, self.args) for row in batch]
            if "hills" in self.table_names:
                self._append("hills", converted)
            if "hills" in self.tsv_sinks:
                converted.sort(key=lambda row: compact_sort_key(row, "hills"))
                self.tsv_sinks["hills"].append(converted)

    def append_ms1(self, rows):
        converted = [compact_ms1(row, self.args) for row in rows]
        if "ms1" in self.table_names:
            self._append("ms1", converted)
        if "ms1" in self.tsv_sinks:
            converted.sort(key=lambda row: compact_sort_key(row, "ms1"))
            self.tsv_sinks["ms1"].append(converted)

    def append_ms2(self, rows):
        for batch in row_batches(rows):
            converted = [compact_ms2(row, self.args) for row in batch]
            if "ms2" in self.table_names:
                self._append("ms2", converted)
            if self.ms2_sink is not None:
                self.ms2_sink.append(converted)

    def _order(self, table_name):
        if table_name == "features":
            mode = self.args.get("parquet_sort", "mz_rt")
            if mode == "none":
                return "feature_idx"
            if mode == "rt_mz":
                return "rtApex, mz, charge, feature_idx"
            return "mz, rtApex, charge, feature_idx"
        if table_name == "hills":
            return "mz, rtApex, hill_idx"
        if table_name == "ms2":
            return "ms2_event_id"
        return "scan_id"

    def _copy_parquet(self):
        compression = self.args.get("parquet_compression", "zstd").upper()
        row_group_size = int(self.args.get("parquet_row_group_size", 122880))
        compression_level = int(self.args.get("parquet_compression_level", 6))
        provenance_json = json.dumps(self.provenance, sort_keys=True, default=str)
        kv_metadata = (
            "KV_METADATA {biosaur2_schema_version: %s, biosaur2_provenance_json: %s}"
            % (
                _quoted_sql_string(self.provenance["schema_version"]),
                _quoted_sql_string(provenance_json),
            )
        )
        for table_name in self.table_names:
            final_path = self.final_paths[table_name]
            temp_path = _temporary_neighbor(final_path)
            self.temp_outputs[table_name] = temp_path
            options = [
                "FORMAT PARQUET",
                "COMPRESSION %s" % compression,
                "ROW_GROUP_SIZE %d" % row_group_size,
                "PARQUET_VERSION V2",
                kv_metadata,
            ]
            if compression in {"ZSTD", "BROTLI"}:
                options.append("COMPRESSION_LEVEL %d" % compression_level)
            self.connection.execute(
                'COPY (SELECT * FROM "%s" ORDER BY %s) TO %s (%s)'
                % (
                    table_name,
                    self._order(table_name),
                    _quoted_sql_string(str(temp_path)),
                    ", ".join(options),
                )
            )

    def _write_provenance(self):
        self.provenance = build_provenance(self.args)
        self.provenance["parquet_engine"] = "duckdb"
        self.provenance["duckdb_version"] = self.duckdb.__version__
        self.provenance["duckdb_parquet_version"] = "v2"
        if self.args.get("write_ms2"):
            self.provenance.update(
                {
                    "ms2_schema_version": MS2_SCHEMA_VERSION,
                    "ms2_rt_unit": "second",
                    "ms2_mobility_unit": "1/K0",
                    "ms2_metadata_flags": (
                        "0x0001 missing_precursor_mz; 0x0002 missing_charge; "
                        "0x0004 unresolved_spectrum_ref; "
                        "0x0008 missing_precursor_ms1"
                    ),
                }
            )
        self.connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
            [
                self.provenance["schema_version"],
                self.provenance["input_path"],
                self.provenance["input_size"],
                self.provenance["parameters_json"],
                json.dumps(self.provenance, sort_keys=True, default=str),
            ],
        )

    def stage(self):
        if self._staged:
            return
        self._ensure_connection()
        self._write_provenance()
        if not self.database_mode:
            self._copy_parquet()
        if self.ms2_sink is not None:
            self.ms2_sink.add_provenance(self.provenance)
            self.ms2_sink.close()
        self.connection.close()
        self.connection = None
        for sink in self.tsv_sinks.values():
            sink.close()
        if not self.database_mode:
            self.staging_path.unlink(missing_ok=True)
        self._staged = True

    def staged_files(self):
        self.stage()
        if self.database_mode:
            pairs = [(self.staging_path, self.final_paths["database"])]
        else:
            pairs = [
                (self.temp_outputs[name], self.final_paths[name])
                for name in self.table_names
            ]
        pairs.extend(
            (sink.temp_path, sink.final_path) for sink in self.tsv_sinks.values()
        )
        if self.ms2_sink is not None:
            pairs.append((self.ms2_sink.temp_path, self.ms2_sink.final_path))
        return pairs

    def finalize(self):
        publish_staged_files(self.staged_files())

    def abort(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.staging_path.unlink(missing_ok=True)
        for path in self.temp_outputs.values():
            path.unlink(missing_ok=True)
        for sink in self.tsv_sinks.values():
            sink.abort()
        if self.ms2_sink is not None:
            self.ms2_sink.abort()


def uses_duckdb(args: Mapping[str, Any]) -> bool:
    if args.get("duckdb_output"):
        return True
    if args.get("parquet_engine") != "duckdb":
        return False
    return bool(
        (
            not args.get("stop_after_hills")
            and args.get("feature_format") == "parquet"
        )
        or (args.get("write_hills") and args.get("hills_format") == "parquet")
        or (args.get("write_ms1") and args.get("ms1_format") == "parquet")
        or args.get("write_ms2")
    )
