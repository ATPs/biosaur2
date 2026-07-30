"""Robust semantic adapters for identification tables."""

from __future__ import annotations

import bz2
import csv
from dataclasses import asdict, dataclass, field
import gzip
import io
import lzma
import logging
import math
from pathlib import Path
import re
from typing import Mapping, Optional, Sequence


logger = logging.getLogger(__name__)


COLUMN_ALIASES = {
    "psm_id": ("PSMId", "psm_id", "specid", "spectrum_id", "spectrum"),
    "score": ("score", "svm_score", "percolator_score"),
    "q_value": ("q-value", "q_value", "qvalue"),
    "pep": ("posterior_error_prob", "posterior_error_probability", "pep"),
    "peptide": ("peptide", "peptidoform", "modified_peptide", "sequence"),
    "proteins": ("proteinIds", "protein_ids", "proteins"),
    "scan": ("scan", "scan_number", "scannr"),
    "charge": ("charge", "precursor_charge"),
    "rank": ("rank", "hit_rank"),
    "native_id": ("native_id", "spectrum_native_id"),
    "decoy": ("decoy", "is_decoy", "target_decoy", "label"),
}


@dataclass(frozen=True)
class IdentificationRecord:
    source_row: int
    psm_id_raw: str
    score: Optional[float]
    q_value: float
    pep: Optional[float]
    peptide_raw: str
    proteins: Optional[str]
    parsed_run: Optional[str]
    parsed_scan: Optional[int]
    parsed_charge: Optional[int]
    parsed_rank: Optional[int]
    native_id: Optional[str]
    mapping_method: Optional[str]
    mapping_status: str


@dataclass
class IdentificationParserQC:
    path: str
    compression: str
    encoding: str
    delimiter: str
    normalized_headers: tuple[str, ...]
    column_map: dict[str, str]
    row_count: int = 0
    accepted_count: int = 0
    rejected_q_value: int = 0
    rejected_pep: int = 0
    rejected_decoy: int = 0
    failed_rows: int = 0
    unmapped_identity_rows: int = 0
    encoding_fallback: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class IdentificationReadResult:
    records: tuple[IdentificationRecord, ...]
    qc: IdentificationParserQC


@dataclass(frozen=True)
class MappedIdentification:
    identification: IdentificationRecord
    ms2_event_id: Optional[int]
    event: Optional[Mapping]
    mapping_method: Optional[str]
    mapping_status: str
    charge_agreement: Optional[bool]


@dataclass(frozen=True)
class IdentificationMappingResult:
    rows: tuple[MappedIdentification, ...]
    status_counts: Mapping[str, int]
    mapped_count: int
    unmapped_count: int


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _decompress(raw: bytes):
    if raw.startswith(b"\x1f\x8b"):
        return gzip.decompress(raw), "gzip"
    if raw.startswith(b"BZh"):
        return bz2.decompress(raw), "bzip2"
    if raw.startswith(b"\xfd7zXZ\x00"):
        return lzma.decompress(raw), "xz"
    if raw.startswith(b"\x28\xb5\x2f\xfd"):
        try:
            import zstandard
        except ImportError as exc:
            raise ImportError(
                "zstd-compressed PSM input requires the optional zstandard package"
            ) from exc
        return zstandard.ZstdDecompressor().decompress(raw), "zstd"
    return raw, "plain"


def _decode(raw: bytes, explicit: Optional[str]):
    if explicit:
        return raw.decode(explicit, errors="strict"), explicit, False, []
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="strict"), "utf-8-sig", False, []
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="strict"), "utf-16", False, []
    try:
        return raw.decode("utf-8", errors="strict"), "utf-8", False, []
    except UnicodeDecodeError:
        warning = "PSM input is not strict UTF-8; decoded with Windows-1252"
        logger.warning(warning)
        try:
            return raw.decode("cp1252", errors="strict"), "cp1252", True, [warning]
        except UnicodeDecodeError:
            warning = "PSM input required Latin-1 decoding fallback"
            logger.warning(warning)
            return raw.decode("latin-1", errors="strict"), "latin-1", True, [warning]


def _detect_delimiter(text: str, explicit: Optional[str], path: Path) -> str:
    if explicit is not None:
        if explicit not in {"\t", ",", ";", "|"}:
            raise ValueError("delimiter must be tab, comma, semicolon or pipe")
        return explicit
    sample = text[:65536]
    try:
        return csv.Sniffer().sniff(sample, delimiters="\t,;|").delimiter
    except csv.Error:
        return "," if path.name.casefold().endswith(".csv") else "\t"


def _resolve_columns(headers, explicit: Optional[Mapping[str, str]]):
    header_lookup = {}
    for header in headers:
        normalized = _normalize_header(header)
        if normalized in header_lookup:
            raise ValueError("ambiguous duplicate normalized header: %s" % header)
        header_lookup[normalized] = header

    result = {}
    explicit = dict(explicit or {})
    unknown_semantics = set(explicit) - set(COLUMN_ALIASES)
    if unknown_semantics:
        raise ValueError(
            "unknown explicit column semantic(s): %s"
            % ", ".join(sorted(unknown_semantics))
        )
    for semantic, aliases in COLUMN_ALIASES.items():
        if semantic in explicit:
            requested = _normalize_header(explicit[semantic])
            if requested not in header_lookup:
                raise ValueError(
                    "configured %s column was not found: %s"
                    % (semantic, explicit[semantic])
                )
            result[semantic] = header_lookup[requested]
            continue
        matches = {
            header_lookup[_normalize_header(alias)]
            for alias in aliases
            if _normalize_header(alias) in header_lookup
        }
        if len(matches) > 1:
            raise ValueError(
                "ambiguous columns for %s: %s" % (semantic, ", ".join(sorted(matches)))
            )
        if matches:
            result[semantic] = next(iter(matches))
    missing = {"psm_id", "q_value", "peptide"} - set(result)
    if missing:
        raise ValueError(
            "identification table is missing semantic column(s): %s"
            % ", ".join(sorted(missing))
        )
    return result, tuple(header_lookup)


def _finite(value, field_name, *, probability=False, required=False):
    text = "" if value is None else str(value).strip()
    if not text:
        if required:
            raise ValueError("empty %s" % field_name)
        return None
    numeric = float(text)
    if not math.isfinite(numeric):
        raise ValueError("non-finite %s" % field_name)
    if probability and not 0.0 <= numeric <= 1.0:
        raise ValueError("%s is outside [0, 1]" % field_name)
    return numeric


def _integer(value):
    if value is None or not str(value).strip():
        return None
    numeric = float(str(value).strip())
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("expected an integer")
    return int(numeric)


def _is_decoy(value):
    if value is None:
        return False
    normalized = str(value).strip().casefold()
    if normalized in {"", "0", "false", "target", "1"}:
        return False
    if normalized in {"true", "decoy", "-1"}:
        return True
    raise ValueError("unrecognized target/decoy value: %s" % value)


def parse_psm_identity(psm_id: str):
    """Safely parse ``<run>_<scan>_<charge>_<rank>`` from the right."""

    parts = psm_id.rsplit("_", 3)
    if len(parts) != 4 or not parts[0]:
        return None
    try:
        scan, charge, rank = (int(value) for value in parts[1:])
    except ValueError:
        return None
    if scan < 0 or charge <= 0 or rank <= 0:
        return None
    return parts[0], scan, charge, rank


def read_percolator_tsv(
    path: Path | str,
    *,
    q_value_max: float = 0.01,
    pep_max: Optional[float] = None,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    column_map: Optional[Mapping[str, str]] = None,
) -> IdentificationReadResult:
    """Read accepted Percolator PSMs using semantic columns and strict text QC."""

    if not math.isfinite(q_value_max) or not 0.0 <= q_value_max <= 1.0:
        raise ValueError("q_value_max must be finite and in [0, 1]")
    if pep_max is not None and (
        not math.isfinite(pep_max) or not 0.0 <= pep_max <= 1.0
    ):
        raise ValueError("pep_max must be finite and in [0, 1]")

    source = Path(path)
    decoded, compression = _decompress(source.read_bytes())
    text, chosen_encoding, fallback, warnings = _decode(decoded, encoding)
    chosen_delimiter = _detect_delimiter(text, delimiter, source)
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=chosen_delimiter)
    headers = tuple((header or "").strip().lstrip("\ufeff") for header in (reader.fieldnames or ()))
    if not headers:
        raise ValueError("identification table has no header")
    resolved, normalized_headers = _resolve_columns(headers, column_map)
    qc = IdentificationParserQC(
        path=str(source.resolve()),
        compression=compression,
        encoding=chosen_encoding,
        delimiter=chosen_delimiter,
        normalized_headers=normalized_headers,
        column_map=resolved,
        encoding_fallback=fallback,
        warnings=warnings,
    )

    records = []
    for source_row, row in enumerate(reader, start=2):
        qc.row_count += 1
        try:
            if None in row:
                raise ValueError("row has more fields than the header")
            q_value = _finite(
                row.get(resolved["q_value"]), "q-value", probability=True, required=True
            )
            pep = _finite(
                row.get(resolved.get("pep")), "PEP", probability=True
            ) if "pep" in resolved else None
            if "decoy" in resolved and _is_decoy(row.get(resolved["decoy"])):
                qc.rejected_decoy += 1
                continue
            if q_value > q_value_max:
                qc.rejected_q_value += 1
                continue
            if pep_max is not None and (pep is None or pep > pep_max):
                qc.rejected_pep += 1
                continue

            psm_id = (row.get(resolved["psm_id"]) or "").strip()
            peptide = (row.get(resolved["peptide"]) or "").strip()
            if not psm_id or not peptide:
                raise ValueError("empty PSM ID or peptide")
            score = _finite(row.get(resolved.get("score")), "score") if "score" in resolved else None
            native_id = (
                (row.get(resolved["native_id"]) or "").strip() or None
                if "native_id" in resolved else None
            )
            explicit_scan = _integer(row.get(resolved.get("scan"))) if "scan" in resolved else None
            explicit_charge = _integer(row.get(resolved.get("charge"))) if "charge" in resolved else None
            explicit_rank = _integer(row.get(resolved.get("rank"))) if "rank" in resolved else None

            parsed = parse_psm_identity(psm_id)
            parsed_run = parsed_scan = parsed_charge = parsed_rank = None
            if parsed is not None:
                parsed_run, parsed_scan, parsed_charge, parsed_rank = parsed
            if native_id is not None:
                method = "native_id"
                status = "parsed"
            elif explicit_scan is not None:
                method = "scan_column"
                status = "parsed"
                parsed_scan = explicit_scan
                parsed_charge = explicit_charge if explicit_charge is not None else parsed_charge
                parsed_rank = explicit_rank if explicit_rank is not None else parsed_rank
            elif parsed is not None:
                method = "psm_id_right_split"
                status = "parsed"
            else:
                method = None
                status = "unparsed"
                qc.unmapped_identity_rows += 1
            records.append(
                IdentificationRecord(
                    source_row=source_row,
                    psm_id_raw=psm_id,
                    score=score,
                    q_value=q_value,
                    pep=pep,
                    peptide_raw=peptide,
                    proteins=(row.get(resolved["proteins"]) or "").strip() or None
                    if "proteins" in resolved else None,
                    parsed_run=parsed_run,
                    parsed_scan=parsed_scan,
                    parsed_charge=explicit_charge if explicit_charge is not None else parsed_charge,
                    parsed_rank=explicit_rank if explicit_rank is not None else parsed_rank,
                    native_id=native_id,
                    mapping_method=method,
                    mapping_status=status,
                )
            )
            qc.accepted_count += 1
        except (TypeError, ValueError):
            qc.failed_rows += 1
    return IdentificationReadResult(tuple(records), qc)


def map_identifications_to_ms2(
    records: Sequence[IdentificationRecord],
    ms2_rows: Sequence[Mapping],
    *,
    run_id: str,
    max_unmapped_fraction: Optional[float] = 0.05,
) -> IdentificationMappingResult:
    """Map accepted PSM identities to exactly one mzML MS2 event."""

    from collections import Counter, defaultdict

    if max_unmapped_fraction is not None and not 0.0 <= max_unmapped_fraction <= 1.0:
        raise ValueError("max_unmapped_fraction must be in [0, 1]")
    by_native = defaultdict(list)
    by_scan = defaultdict(list)
    for event in ms2_rows:
        native = event.get("native_id")
        scan = event.get("native_scan_number")
        if native is not None:
            by_native[str(native)].append(event)
        if scan is not None:
            by_scan[int(scan)].append(event)

    mapped_rows = []
    status_counts = Counter()
    for record in records:
        candidates = []
        method = None
        if record.native_id is not None:
            candidates = list(by_native.get(record.native_id, ()))
            method = "native_id"
        if not candidates and record.parsed_scan is not None:
            if record.parsed_run is not None and record.parsed_run != run_id:
                status = "run_mismatch"
                status_counts[status] += 1
                mapped_rows.append(
                    MappedIdentification(record, None, None, None, status, None)
                )
                continue
            candidates = list(by_scan.get(int(record.parsed_scan), ()))
            method = "scan_number"
        if not candidates:
            status = "ms2_not_found"
            status_counts[status] += 1
            mapped_rows.append(
                MappedIdentification(record, None, None, method, status, None)
            )
            continue

        charge = record.parsed_charge
        charge_candidates = (
            [event for event in candidates if event.get("charge") == charge]
            if charge is not None
            else candidates
        )
        if len(charge_candidates) == 1:
            selected = charge_candidates[0]
        elif len(candidates) == 1:
            selected = candidates[0]
        else:
            status = "ambiguous_ms2_event"
            status_counts[status] += 1
            mapped_rows.append(
                MappedIdentification(record, None, None, method, status, None)
            )
            continue
        event_charge = selected.get("charge")
        charge_agreement = (
            None if charge is None or event_charge is None else int(charge) == int(event_charge)
        )
        status = "mapped" if charge_agreement is not False else "charge_mismatch"
        status_counts[status] += 1
        mapped_rows.append(
            MappedIdentification(
                record,
                int(selected["ms2_event_id"]),
                selected,
                method,
                status,
                charge_agreement,
            )
        )

    mapped_count = sum(row.ms2_event_id is not None for row in mapped_rows)
    unmapped_count = len(mapped_rows) - mapped_count
    result = IdentificationMappingResult(
        tuple(mapped_rows), dict(status_counts), mapped_count, unmapped_count
    )
    if records and max_unmapped_fraction is not None:
        fraction = unmapped_count / len(records)
        if fraction > max_unmapped_fraction:
            error = ValueError(
                "accepted-PSM mapping appears broken: %d/%d (%.2f%%) are unmapped; "
                "maximum is %.2f%%"
                % (
                    unmapped_count,
                    len(records),
                    100.0 * fraction,
                    100.0 * max_unmapped_fraction,
                )
            )
            error.mapping_result = result
            raise error
    return result
