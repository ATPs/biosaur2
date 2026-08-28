"""Shared compact-output lifecycle helpers."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Iterable, Mapping

import pyarrow as pa

from .schema import SCHEMA_VERSION


def input_stem(path_value: str) -> str:
    name = Path(path_value).name
    lower = name.lower()
    for suffix in (
        ".mzml.gz",
        ".mzml",
        ".hills.parquet",
        ".hills.tsv",
        ".hills.npz",
    ):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def output_prefix(path_value: str) -> Path:
    """Return the shared prefix for one run's public outputs."""

    path = Path(path_value)
    if path.suffix.lower() in {".tsv", ".parquet"}:
        path = path.with_suffix("")
    if path.name.lower().endswith(".features"):
        path = path.with_name(path.name[: -len(".features")])
    return path


def ms2_output_path(args: Mapping[str, Any]) -> Path:
    """Return the fixed input-stem MS2 sidecar path for one run."""

    stem = input_stem(str(args["file"]))
    explicit_directory = args.get("_ms2_output_directory")
    if explicit_directory:
        extension = args.get("format", "parquet")
        return Path(explicit_directory) / (stem + ".ms2." + extension)
    explicit = args.get("o")
    if explicit:
        path = Path(explicit)
        if args.get("_multiple_inputs"):
            directory = path
        else:
            directory = path.parent
    else:
        directory = Path(args["file"]).parent
    extension = args.get("format", "parquet")
    return directory / (stem + ".ms2." + extension)


def planned_output_paths(args: Mapping[str, Any]):
    """Resolve final outputs without creating writers or staging files."""

    input_path = Path(args["file"])
    explicit = args.get("o")
    if explicit:
        prefix = output_prefix(explicit)
    else:
        prefix = input_path.parent / input_stem(str(input_path))

    paths = []
    database_mode = args.get("format") == "duckdb"
    if database_mode:
        requested = Path(explicit) if explicit else input_path.parent
        paths.append(
            requested
            if requested.suffix.lower() == ".duckdb"
            else requested / (input_stem(str(input_path)) + ".biosaur2.duckdb")
        )
    else:
        if not args.get("stop_after_hills"):
            output_format = args.get("format", "tsv")
            if explicit and str(explicit).lower().endswith("." + output_format):
                paths.append(Path(explicit))
            else:
                paths.append(Path("%s.features.%s" % (prefix, output_format)))
        if args.get("write_hills"):
            paths.append(
                Path("%s.hills.%s" % (prefix, args.get("format", "tsv")))
            )
        if args.get("write_ms1"):
            paths.append(
                Path("%s.ms1.%s" % (prefix, args.get("format", "tsv")))
            )
        if args.get("feature_mode") == "hybrid" and not args.get("stop_after_hills"):
            paths.append(
                Path("%s.ms2_events.%s" % (prefix, args.get("format", "parquet")))
            )
            paths.append(
                Path("%s.identifications.%s" % (prefix, args.get("format", "parquet")))
            )
    if args.get("write_ms2") and not database_mode:
        paths.append(ms2_output_path(args))
    return paths


def _format_tsv(value, decimals="roundtrip"):
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        if any(isinstance(item, dict) for item in value):
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        return "[" + ", ".join(
            "None" if item is None else _format_tsv(item, decimals=decimals)
            for item in value
        ) + "]"
    if isinstance(value, float):
        if decimals != "roundtrip":
            formatted = ("%.*f" % (int(decimals), value)).rstrip("0").rstrip(".")
            return "0" if formatted == "-0" else formatted
        return repr(value)
    return str(value)


def round_intensity(value: Any, decimals: Any):
    if value is None or decimals in (None, "none"):
        return value
    decimal_count = int(decimals)
    if decimal_count < 0:
        raise ValueError("intensity decimals must be nonnegative or 'none'")
    numeric = float(value)
    if not math.isfinite(numeric):
        return numeric
    scale = 10.0**decimal_count
    scaled = numeric * scale
    rounded = math.copysign(math.floor(abs(scaled) + 0.5), scaled)
    return rounded / scale


class _TsvSink:
    def __init__(self, final_path: Path, columns, overwrite: bool, decimals="roundtrip"):
        self.final_path = final_path
        self.columns = tuple(columns)
        self.overwrite = overwrite
        self.decimals = decimals
        self.temp_path = _temporary_neighbor(final_path)
        self.handle = None
        self.writer = None
        self.row_count = 0

    def append(self, rows: Iterable[Mapping[str, Any]]):
        rows = list(rows)
        self._open()
        for row in rows:
            self.writer.writerow(
                [_format_tsv(row.get(column), self.decimals) for column in self.columns]
            )
            self.row_count += 1

    def _open(self):
        if self.handle is not None:
            return
        self.temp_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.temp_path.open("w", newline="", encoding="utf8")
        self.writer = csv.writer(self.handle, delimiter="\t", lineterminator="\n")
        self.writer.writerow(self.columns)

    def close(self):
        self._open()
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.handle = None

    def abort(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.temp_path.unlink(missing_ok=True)


def _temporary_neighbor(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % final_path.name,
        suffix=".tmp",
        dir=final_path.parent,
    )
    os.close(descriptor)
    Path(temporary_name).unlink()
    return Path(temporary_name)


def publish_staged_files(pairs):
    """Publish a staged file set and restore every prior final on failure."""

    pairs = [(Path(temporary), Path(final)) for temporary, final in pairs]
    backups = {}
    published = []
    try:
        for temporary, final in pairs:
            if final.exists():
                backup = _temporary_neighbor(final)
                os.replace(final, backup)
                backups[final] = backup
            os.replace(temporary, final)
            published.append(final)
    except BaseException:
        for final in reversed(published):
            final.unlink(missing_ok=True)
        for final, backup in backups.items():
            if backup.exists():
                os.replace(backup, final)
        raise
    else:
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def build_provenance(args: Mapping[str, Any]):
    input_path = Path(args["file"])
    input_exists = input_path.is_file()
    try:
        package_version = version("biosaur2")
    except PackageNotFoundError:
        package_version = "unknown"
    git_commit = "unknown"
    git_dirty = "unknown"
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        git_dirty = bool(dirty_result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    parameters = {
        key: value
        for key, value in args.items()
        if not key.startswith("_") and key != "file"
    }
    input_fingerprint = None
    if input_exists:
        from .raw_ms1 import source_fingerprint

        input_fingerprint = source_fingerprint(input_path)
    return {
        "version": package_version,
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "command_line": shlex.join(sys.argv),
        "parameters_json": json.dumps(parameters, sort_keys=True, default=str),
        "input_path": str(input_path),
        "input_size": input_path.stat().st_size if input_exists else None,
        "input_fingerprint": input_fingerprint,
        "source_format": "mzML" if ".mzml" in input_path.name.lower() else "hills",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rt_unit": "minute for feature/hill scalar RT; second for MS1 and area",
        "mz_unit": "m/z",
        "mobility_unit": "1/K0 when source units are compatible",
        "intensity_unit": "instrument arbitrary units",
        "area_sum_unit": "instrument intensity * second",
        "area_sum_rt": (
            "approximated_from_hill_anchors"
            if args.get("_area_sum_approximate")
            else "exact_point_rt"
        ),
        "combine_every": args.get("combine_every", 1),
        "requested_workers": args.get("_requested_workers", args.get("workers", 4)),
        "effective_workers": args.get("_effective_workers", args.get("workers", 4)),
        "allocated_workers": args.get("_allocated_workers", args.get("nprocs", 1)),
        "calibration": {
            "hill": args.get("hill_calibration", {"status": "not_requested"}),
            "isotope": args.get("isotope_calibration", {}),
        },
        "quantification": "raw_selected_isotope_area_sum_v1",
        "intensity_decimals": 0,
        "rounding_policy": "half_away_from_zero_output_only",
        "numeric_storage": (
            "64-bit" if args.get("use64") else "compact float32/narrow integers"
        ),
        "parquet_engine": args.get("parquet_engine", "duckdb"),
        "pyarrow_version": pa.__version__,
        "parquet_version": "V2" if args.get("parquet_engine") == "duckdb" else "PyArrow data page V2",
        "parquet_compression": args.get("parquet_compression", "zstd"),
        "parquet_compression_level": args.get("parquet_compression_level", 6),
        "parquet_row_group_size": args.get("parquet_row_group_size", 122880),
        "parquet_sort": args.get("parquet_sort", "mz_rt"),
        "parquet_layout": "single_table_compact",
        "faims_policy": "null_distinct_from_zero",
    }
