"""Compact one-table TSV and PyArrow output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from .output import (
    _TsvSink,
    _temporary_neighbor,
    build_provenance,
    input_stem,
    publish_staged_files,
    round_intensity,
)
from .schema import (
    SCHEMA_VERSION,
    compact_schemas,
    feature_columns,
    hill_columns,
    MS1_COLUMNS,
)


def _round_nested(value, decimals):
    if isinstance(value, (list, tuple)):
        return [_round_nested(item, decimals) for item in value]
    return round_intensity(value, decimals)


def _minutes(value):
    return None if value is None else float(value) / 60.0


def compact_feature(row: Mapping[str, Any], args: Mapping[str, Any]):
    decimals = args.get("intensity_decimals", "0")
    result = {
        "massCalib": row.get("massCalib"),
        "rtApex": _minutes(row.get("rtApex")),
        "intensityApex": _round_nested(row.get("intensityApex"), decimals),
        "intensitySum": _round_nested(row.get("intensitySum"), decimals),
        "charge": row.get("charge"),
        "nIsotopes": row.get("nIsotopes"),
        "nScans": row.get("nScans"),
        "mz": row.get("mz"),
        "rtStart": _minutes(row.get("rtStart")),
        "rtEnd": _minutes(row.get("rtEnd")),
        "FAIMS": row.get("FAIMS", row.get("faims_cv")),
        "im": row.get("im", row.get("ion_mobility_1_over_k0")),
        "scanApex": row.get("scanApex", row.get("scan_apex_number")),
        "isoerror": row.get("isoerror"),
        "isoerror2": None if row.get("isoerror2") == -100 else row.get("isoerror2"),
        "feature_idx": row.get("feature_idx"),
        "area_sum": row.get("area_sum"),
    }
    if not args.get("no_mono_hills"):
        result.update(
            {
                "mono_hills_scan_lists": row.get("mono_hills_scan_lists"),
                "mono_hills_intensity_list": _round_nested(
                    row.get("mono_hills_intensity_list"), decimals
                ),
            }
        )
    if args.get("write_extra_details"):
        for name in (
            "isotopes",
            "intensity_array_for_cos_corr",
            "monoisotope hill idx",
            "monoisotope idx",
        ):
            result[name] = row.get(name)
    return result


def compact_hill(row: Mapping[str, Any], args: Mapping[str, Any]):
    decimals = args.get("intensity_decimals", "0")
    result = {
        "rtApex": _minutes(row.get("rtApex")),
        "intensityApex": _round_nested(row.get("intensityApex"), decimals),
        "intensitySum": _round_nested(row.get("intensitySum"), decimals),
        "nScans": row.get("nScans"),
        "mz": row.get("mz"),
        "rtStart": _minutes(row.get("rtStart")),
        "rtEnd": _minutes(row.get("rtEnd")),
        "FAIMS": row.get("FAIMS", row.get("faims_cv")),
        "im": row.get("im", row.get("ion_mobility_1_over_k0")),
        "scanApex": row.get("scanApex", row.get("scan_apex_number")),
        "hill_idx": row.get("hill_idx"),
        "feature_idx": row.get("feature_idx", -1),
    }
    if not args.get("no_hill_list"):
        point_rt = row.get("hills_rt_list")
        if point_rt is None:
            point_rt = [
                point.get("rt_sec") for point in row.get("_hill_points", ())
            ]
        result.update(
            {
                "hills_scan_lists": row.get("hills_scan_lists"),
                "hills_intensity_list": _round_nested(
                    row.get("hills_intensity_list"), decimals
                ),
                "hills_mz_array": row.get("hills_mz_array"),
                "hills_rt_list": point_rt,
            }
        )
    return result


def compact_ms1(row: Mapping[str, Any], args: Mapping[str, Any]):
    scan_id = row.get("scan_number")
    if scan_id is None and row.get("scan_index") is not None:
        scan_id = int(row["scan_index"]) + 1
    return {
        "scan_id": scan_id,
        "RT": row.get("rt_sec"),
        "total_intensity": round_intensity(
            row.get("total_intensity"), args.get("intensity_decimals", "0")
        ),
    }


def compact_sort_key(row, kind, mode="mz_rt"):
    if kind == "features":
        if mode == "none":
            return (row.get("feature_idx") or 0,)
        if mode == "rt_mz":
            return (
                row.get("rtApex") or 0.0,
                row.get("mz") or 0.0,
                row.get("charge") or 0,
                row.get("feature_idx") or 0,
            )
        return (
            row.get("mz") or 0.0,
            row.get("rtApex") or 0.0,
            row.get("charge") or 0,
            row.get("feature_idx") or 0,
        )
    if kind == "hills":
        return (
            row.get("mz") or 0.0,
            row.get("rtApex") or 0.0,
            row.get("hill_idx") or 0,
        )
    return (row.get("scan_id") if row.get("scan_id") is not None else -1,)


def row_batches(rows, batch_size=100000):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class _CompactParquetSink:
    def __init__(self, final_path: Path, schema: pa.Schema, args):
        self.final_path = final_path
        self.schema = schema
        self.args = args
        self.temp_path = _temporary_neighbor(final_path)
        self.writer = None
        self.row_count = 0

    def _open(self):
        if self.writer is not None:
            return
        compression = self.args.get("parquet_compression", "zstd")
        if compression == "uncompressed":
            compression = None
        kwargs = {
            "compression": compression,
            "use_dictionary": False,
            "use_byte_stream_split": [
                field.name
                for field in self.schema
                if pa.types.is_floating(field.type)
            ],
            "data_page_version": "2.0",
        }
        if compression in {"zstd", "brotli"}:
            kwargs["compression_level"] = int(
                self.args.get("parquet_compression_level", 6)
            )
        schema = self.schema.with_metadata(
            {b"biosaur2_schema_version": SCHEMA_VERSION.encode()}
        )
        self.writer = pq.ParquetWriter(self.temp_path, schema, **kwargs)

    def add_provenance(self, provenance):
        self._open()
        self.writer.add_key_value_metadata(
            {
                ("biosaur2_" + key).encode(): str(value).encode()
                for key, value in provenance.items()
            }
        )

    def append(self, rows: Iterable[Mapping[str, Any]]):
        rows = list(rows)
        self._open()
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=self.schema)
        self.writer.write_table(
            table,
            row_group_size=int(self.args.get("parquet_row_group_size", 122880)),
        )
        self.row_count += table.num_rows

    def close(self):
        self._open()
        self.writer.close()
        self.writer = None

    def abort(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.temp_path.unlink(missing_ok=True)


class CompactOutputManager:
    """Own compact feature, hills, and MS1 outputs for one input."""

    def __init__(self, args, compatibility_suffix=False):
        self.args = args
        self.overwrite = bool(args.get("overwrite"))
        self.compatibility_suffix = compatibility_suffix
        self.prefix = self._prefix()
        self.schemas = compact_schemas(
            use64=bool(args.get("use64")),
            include_mono=not args.get("no_mono_hills"),
            extra_details=bool(args.get("write_extra_details")),
            include_hill_lists=not args.get("no_hill_list"),
        )
        self.sinks = {}
        self._staged = False
        self.provenance = {}
        self._build_sinks()
        self._preflight()

    def _prefix(self):
        explicit = self.args.get("o")
        if explicit:
            path = Path(explicit)
            if path.suffix in {".tsv", ".parquet"}:
                return path.with_suffix("")
            return path
        input_path = Path(self.args["file"])
        return input_path.parent / input_stem(str(input_path))

    def _target(self, kind, output_format):
        explicit = self.args.get("o")
        if kind == "features" and explicit and str(explicit).endswith(
            "." + output_format
        ):
            return Path(explicit)
        return Path("%s.%s.%s" % (self.prefix, kind, output_format))

    def _sink(self, kind, output_format, columns):
        target = self._target(kind, output_format)
        if output_format == "tsv":
            return _TsvSink(
                target,
                columns,
                self.overwrite,
                decimals=self.args.get("tsv_float_decimals", "roundtrip"),
            )
        return _CompactParquetSink(target, self.schemas[kind], self.args)

    def _build_sinks(self):
        if not self.args.get("stop_after_hills"):
            columns = feature_columns(
                not self.args.get("no_mono_hills"),
                bool(self.args.get("write_extra_details")),
            )
            self.sinks["features"] = self._sink(
                "features", self.args.get("feature_format", "tsv"), columns
            )
        if self.args.get("write_hills"):
            self.sinks["hills"] = self._sink(
                "hills",
                self.args.get("hills_format", "tsv"),
                hill_columns(not self.args.get("no_hill_list")),
            )
        if self.args.get("write_ms1"):
            self.sinks["ms1"] = self._sink(
                "ms1", self.args.get("ms1_format", "tsv"), MS1_COLUMNS
            )

    def _preflight(self):
        targets = [sink.final_path for sink in self.sinks.values()]
        if len(targets) != len(set(targets)):
            raise ValueError("Output paths collide")
        existing = [path for path in targets if path.exists()]
        if existing and not self.overwrite:
            raise FileExistsError(
                "Output already exists; use --overwrite: %s"
                % ", ".join(map(str, existing))
            )

    def append_features(self, rows):
        if "features" not in self.sinks:
            return
        converted = [compact_feature(row, self.args) for row in rows]
        converted.sort(
            key=lambda row: compact_sort_key(
                row, "features", self.args.get("parquet_sort", "mz_rt")
            )
        )
        self.sinks["features"].append(converted)

    def append_hills(self, rows):
        if "hills" not in self.sinks:
            return
        for batch in row_batches(rows):
            converted = [compact_hill(row, self.args) for row in batch]
            converted.sort(key=lambda row: compact_sort_key(row, "hills"))
            self.sinks["hills"].append(converted)

    def append_ms1(self, rows):
        if "ms1" not in self.sinks:
            return
        converted = [compact_ms1(row, self.args) for row in rows]
        converted.sort(key=lambda row: compact_sort_key(row, "ms1"))
        self.sinks["ms1"].append(converted)

    def stage(self):
        if self._staged:
            return
        parquet_sinks = [
            sink
            for sink in self.sinks.values()
            if isinstance(sink, _CompactParquetSink)
        ]
        if parquet_sinks:
            self.provenance = build_provenance(self.args)
            for sink in parquet_sinks:
                sink.add_provenance(self.provenance)
        for sink in self.sinks.values():
            sink.close()
        self._staged = True

    def staged_files(self):
        self.stage()
        return [(sink.temp_path, sink.final_path) for sink in self.sinks.values()]

    def finalize(self):
        publish_staged_files(self.staged_files())

    def abort(self):
        for sink in self.sinks.values():
            sink.abort()
