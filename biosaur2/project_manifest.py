"""Deterministic project manifests and mzML/PSM auto-pairing."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Optional


MZML_SUFFIXES = (".mzml.gz", ".mzml")
PSM_SUFFIXES = (
    ".percolator.target.psms.tsv.gz",
    ".percolator.target.psms.tsv",
    ".target.psms.tsv.gz",
    ".target.psms.tsv",
    ".psms.tsv.gz",
    ".psms.tsv",
)

MANIFEST_COLUMNS = (
    "run_id",
    "mzml_path",
    "psm_path",
    "psm_format",
    "identification_config",
    "fixed_mods",
    "q_value_max",
    "sample_id",
    "condition",
    "replicate",
    "fraction",
    "batch",
    "alignment_group",
)


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    mzml_path: Path
    psm_path: Optional[Path] = None
    psm_format: Optional[str] = None
    identification_config: Optional[Path] = None
    fixed_mods: Optional[str] = None
    q_value_max: Optional[float] = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PairingReport:
    rows: tuple[dict, ...]
    mzml_without_psm: tuple[Path, ...]
    orphan_psms: tuple[Path, ...]


def _strip_known_suffix(name: str, suffixes: Iterable[str]) -> Optional[str]:
    lower = name.casefold()
    for suffix in suffixes:
        if lower.endswith(suffix.casefold()):
            stem = name[: -len(suffix)]
            return stem or None
    return None


def normalized_mzml_stem(path: Path | str) -> Optional[str]:
    return _strip_known_suffix(Path(path).name, MZML_SUFFIXES)


def normalized_psm_stem(
    path: Path | str, suffixes: Iterable[str] = PSM_SUFFIXES
) -> Optional[str]:
    return _strip_known_suffix(Path(path).name, suffixes)


def _index_known_files(directory: Path, kind: str, suffixes: Iterable[str]):
    if not directory.is_dir():
        raise ValueError("%s directory does not exist: %s" % (kind, directory))
    indexed = {}
    for path in sorted(directory.iterdir(), key=lambda value: value.name.casefold()):
        if not path.is_file():
            continue
        stem = _strip_known_suffix(path.name, suffixes)
        if stem is None:
            continue
        key = stem.casefold()
        if key in indexed:
            raise ValueError(
                "duplicate normalized %s stem %r: %s and %s"
                % (kind, stem, indexed[key], path)
            )
        indexed[key] = path.resolve()
    return indexed


def auto_pair_runs(
    mzml_dir: Path | str,
    psm_dir: Path | str,
    *,
    psm_suffix: Optional[str] = None,
    allow_missing_psm: bool = True,
) -> PairingReport:
    """Pair files by an exact, case-insensitive normalized stem."""

    mzml = _index_known_files(Path(mzml_dir), "mzML", MZML_SUFFIXES)
    psm_suffixes = (psm_suffix,) if psm_suffix else PSM_SUFFIXES
    psms = _index_known_files(Path(psm_dir), "PSM", psm_suffixes)
    if not mzml:
        raise ValueError("no .mzML or .mzML.gz files found in %s" % mzml_dir)

    rows = []
    missing = []
    for key in sorted(mzml):
        mzml_path = mzml[key]
        run_id = normalized_mzml_stem(mzml_path)
        psm_path = psms.get(key)
        if psm_path is None:
            missing.append(mzml_path)
            if not allow_missing_psm:
                continue
        rows.append(
            {
                "run_id": run_id,
                "mzml_path": str(mzml_path),
                "psm_path": "" if psm_path is None else str(psm_path),
                "psm_format": "" if psm_path is None else "percolator_tsv",
            }
        )
    if missing and not allow_missing_psm:
        raise ValueError(
            "%d mzML file(s) have no exact-stem PSM match" % len(missing)
        )
    orphan = tuple(psms[key] for key in sorted(set(psms) - set(mzml)))
    return PairingReport(tuple(rows), tuple(missing), orphan)


def write_manifest(
    path: Path | str,
    rows: Iterable[Mapping[str, object]],
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a stable TSV manifest sorted by run ID."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError("manifest already exists: %s" % target)
    normalized_rows = []
    for source in rows:
        row = {column: source.get(column, "") for column in MANIFEST_COLUMNS}
        row["run_id"] = str(row["run_id"]).strip()
        if not row["run_id"]:
            raise ValueError("manifest run_id must be nonempty")
        normalized_rows.append(row)
    normalized_rows.sort(key=lambda row: (row["run_id"].casefold(), row["run_id"]))
    run_ids = [row["run_id"].casefold() for row in normalized_rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("manifest rows contain duplicate run IDs")

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name, suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(normalized_rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return target


def _resolved_path(value: str, base: Path) -> Optional[Path]:
    value = value.strip()
    if not value:
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    return (base / expanded).resolve() if not expanded.is_absolute() else expanded.resolve()


def read_manifest(path: Path | str) -> tuple[RunSpec, ...]:
    """Read and validate a project TSV without touching its source files."""

    source = Path(path).resolve()
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        headers = tuple((value or "").strip() for value in (reader.fieldnames or ()))
        missing = {"run_id", "mzml_path"} - set(headers)
        if missing:
            raise ValueError(
                "manifest is missing required column(s): %s" % ", ".join(sorted(missing))
            )
        raw_rows = list(reader)

    result = []
    run_ids = set()
    mzml_paths = set()
    for line_number, row in enumerate(raw_rows, start=2):
        run_id = (row.get("run_id") or "").strip()
        if not run_id:
            raise ValueError("manifest line %d has an empty run_id" % line_number)
        run_key = run_id.casefold()
        if run_key in run_ids:
            raise ValueError("duplicate run_id in manifest: %s" % run_id)
        run_ids.add(run_key)

        mzml_path = _resolved_path(row.get("mzml_path") or "", source.parent)
        if mzml_path is None:
            raise ValueError("manifest line %d has an empty mzml_path" % line_number)
        if normalized_mzml_stem(mzml_path) is None:
            raise ValueError("unsupported mzML path in manifest: %s" % mzml_path)
        path_key = os.path.normcase(str(mzml_path))
        if path_key in mzml_paths:
            raise ValueError("duplicate mzML path in manifest: %s" % mzml_path)
        mzml_paths.add(path_key)

        psm_path = _resolved_path(row.get("psm_path") or "", source.parent)
        config_path = _resolved_path(
            row.get("identification_config") or "", source.parent
        )
        q_value_text = (row.get("q_value_max") or "").strip()
        q_value_max = None
        if q_value_text:
            q_value_max = float(q_value_text)
            if not 0.0 <= q_value_max <= 1.0:
                raise ValueError("q_value_max must be in [0, 1] on line %d" % line_number)
        psm_format = (row.get("psm_format") or "").strip() or (
            "percolator_tsv" if psm_path is not None else None
        )
        metadata = {
            column: (row.get(column) or "").strip()
            for column in ("sample_id", "condition", "replicate", "fraction", "batch", "alignment_group")
        }
        result.append(
            RunSpec(
                run_id=run_id,
                mzml_path=mzml_path,
                psm_path=psm_path,
                psm_format=psm_format,
                identification_config=config_path,
                fixed_mods=(row.get("fixed_mods") or "").strip() or None,
                q_value_max=q_value_max,
                metadata=metadata,
            )
        )
    return tuple(result)
