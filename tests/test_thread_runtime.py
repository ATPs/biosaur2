import json
import os
from pathlib import Path
import subprocess
import sys

from biosaur2.thread_runtime import (
    NATIVE_THREAD_ENVIRONMENT,
    configure_cli_thread_pools,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_configure_cli_thread_pools_forces_one(monkeypatch):
    for index, name in enumerate(NATIVE_THREAD_ENVIRONMENT):
        if index % 2:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, "7")

    configure_cli_thread_pools()
    configure_cli_thread_pools()

    assert {
        name: os.environ.get(name) for name in NATIVE_THREAD_ENVIRONMENT
    } == {name: "1" for name in NATIVE_THREAD_ENVIRONMENT}


def _run_probe(code, environment):
    return subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )


def test_plain_package_import_stays_lazy_and_preserves_environment():
    environment = os.environ.copy()
    for name in NATIVE_THREAD_ENVIRONMENT:
        environment[name] = "7"
    result = _run_probe(
        "import json, os, sys; import biosaur2; "
        "print(json.dumps({'environment': {name: os.environ[name] for name in "
        "('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', "
        "'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', "
        "'ARROW_IO_THREADS')}, 'numpy_loaded': 'numpy' in sys.modules, "
        "'pyarrow_loaded': 'pyarrow' in sys.modules}))",
        environment,
    )
    observed = json.loads(result.stdout)
    assert set(observed["environment"].values()) == {"7"}
    assert observed["numpy_loaded"] is False
    assert observed["pyarrow_loaded"] is False


def test_search_import_caps_environment_before_arrow_initialization():
    environment = os.environ.copy()
    for name in NATIVE_THREAD_ENVIRONMENT:
        environment[name] = "7"
    result = _run_probe(
        "import json, os; import biosaur2.search; import pyarrow as pa; "
        "names = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', "
        "'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', "
        "'ARROW_IO_THREADS'); "
        "print(json.dumps({'environment': {name: os.environ[name] for name in names}, "
        "'arrow_cpu': pa.cpu_count(), 'arrow_io': pa.io_thread_count()}))",
        environment,
    )
    observed = json.loads(result.stdout)
    assert set(observed["environment"].values()) == {"1"}
    assert observed["arrow_cpu"] == 1
    assert observed["arrow_io"] == 1
