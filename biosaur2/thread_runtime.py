"""Early process-wide limits for implicit native thread pools."""

from __future__ import annotations

import os


NATIVE_THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "ARROW_IO_THREADS",
)


def configure_cli_thread_pools():
    """Make the explicit Biosaur2 worker budget the only CPU parallelism."""

    for name in NATIVE_THREAD_ENVIRONMENT:
        os.environ[name] = "1"
