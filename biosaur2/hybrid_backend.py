"""Runtime selection for optional hybrid numerical accelerators."""

from __future__ import annotations

import importlib
from contextvars import ContextVar


_RESOLVED_BACKEND = ContextVar("biosaur2_hybrid_backend", default="cython")


def _rust_module():
    try:
        return importlib.import_module("biosaur2_rust_core")
    except ImportError:
        return None


def configure_backend(requested: str) -> str:
    """Resolve and retain one numerical backend for the current worker."""

    if requested not in {"auto", "cython", "rust"}:
        raise ValueError("unsupported hybrid backend: %r" % (requested,))
    module = _rust_module()
    if requested == "rust" and module is None:
        raise RuntimeError(
            "--hybrid-backend rust requires the optional biosaur2-rust-core "
            "extension; build it with maturin first"
        )
    resolved = "rust" if requested == "rust" or (
        requested == "auto" and module is not None
    ) else "cython"
    _RESOLVED_BACKEND.set(resolved)
    return resolved


def resolved_backend() -> str:
    return _RESOLVED_BACKEND.get()


def extract_trace_values(offsets, mz, intensity, local_indices, target_mz, ppm):
    """Return intensity and weighted-centroid arrays on a selected scan grid."""

    if resolved_backend() == "rust":
        module = _rust_module()
        if module is None:
            raise RuntimeError("configured Rust hybrid backend is unavailable")
        return module.extract_trace_values(
            offsets, mz, intensity, local_indices, target_mz, ppm
        )
    from .cutils import extract_trace_values as cython_extract_trace_values

    return cython_extract_trace_values(
        offsets, mz, intensity, local_indices, target_mz, ppm
    )


def extract_traces_values(offsets, mz, intensity, local_indices, target_mzs, ppm):
    """Return trace matrices for several targets sharing a scan grid."""

    if resolved_backend() == "rust":
        module = _rust_module()
        if module is None:
            raise RuntimeError("configured Rust hybrid backend is unavailable")
        return module.extract_traces_values(
            offsets, mz, intensity, local_indices, target_mzs, ppm
        )
    from .cutils import extract_traces_values as cython_extract_traces_values

    return cython_extract_traces_values(
        offsets, mz, intensity, local_indices, target_mzs, ppm
    )
