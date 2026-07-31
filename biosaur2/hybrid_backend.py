"""Cython numerical helpers used by hybrid-mode recovery."""

from __future__ import annotations


def configure_backend(requested: str) -> str:
    """Validate a legacy selector and resolve it to the Cython backend."""

    if requested not in {"auto", "cython"}:
        raise ValueError(
            "unsupported hybrid backend: %r; only Cython is available"
            % (requested,)
        )
    return "cython"


def resolved_backend() -> str:
    return "cython"


def extract_trace_values(offsets, mz, intensity, local_indices, target_mz, ppm):
    """Return intensity and weighted-centroid arrays on a selected scan grid."""

    from .cutils import extract_trace_values as cython_extract_trace_values

    return cython_extract_trace_values(
        offsets, mz, intensity, local_indices, target_mz, ppm
    )


def extract_traces_values(offsets, mz, intensity, local_indices, target_mzs, ppm):
    """Return trace matrices for several targets sharing a scan grid."""

    from .cutils import extract_traces_values as cython_extract_traces_values

    return cython_extract_traces_values(
        offsets, mz, intensity, local_indices, target_mzs, ppm
    )
