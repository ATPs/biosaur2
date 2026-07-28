"""Canonical spectrum metadata helpers."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


_SCAN_NUMBER_RE = re.compile(r"(?:^|\s)scan=(\d+)(?:\s|$)")


def extract_scan_number(spectrum: Dict[str, Any]) -> Optional[int]:
    """Return the parsed mzML ``scan=`` number without inventing a fallback."""

    match = _SCAN_NUMBER_RE.search(str(spectrum.get("id", "")))
    if match is None:
        return None
    return int(match.group(1))


def retention_time_seconds(value: Any, default_unit: str = "seconds") -> float:
    """Convert one unit-aware mzML retention time to canonical seconds."""

    unit = getattr(value, "unit_info", None) or default_unit
    normalized_unit = str(unit).strip().lower()
    numeric_value = float(value)
    if normalized_unit in {"minute", "minutes", "min"}:
        return numeric_value * 60.0
    if normalized_unit in {"second", "seconds", "sec", "s"}:
        return numeric_value
    raise ValueError("Unsupported retention-time unit: %s" % unit)


def faims_value(spectrum: Dict[str, Any]) -> Optional[float]:
    value = spectrum.get("FAIMS compensation voltage")
    if value is None:
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def faims_sort_key(value: Optional[float]) -> Tuple[int, float]:
    if value is None:
        return (0, 0.0)
    return (1, float(value))


def group_spectra_by_faims(
    spectra: Iterable[Dict[str, Any]],
) -> List[Tuple[Optional[float], List[Dict[str, Any]]]]:
    """Group spectra exactly once with null distinct from explicit CV=0."""

    groups: Dict[Optional[float], List[Dict[str, Any]]] = {}
    for spectrum in spectra:
        value = faims_value(spectrum)
        groups.setdefault(value, []).append(spectrum)
    return [(value, groups[value]) for value in sorted(groups, key=faims_sort_key)]
