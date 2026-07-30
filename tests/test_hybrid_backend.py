import numpy as np
import pytest

from biosaur2.hybrid_backend import configure_backend, resolved_backend
from biosaur2.raw_ms1 import RawMS1StoreBuilder


def _scalar_trace(store, target_mz, ppm, rt_start_sec, rt_end_sec):
    selected = np.flatnonzero(
        (store.rt_sec >= rt_start_sec)
        & (store.rt_sec <= rt_end_sec)
        & np.isnan(store.faims_cv)
    )
    intensity = np.zeros(selected.size, dtype=np.float64)
    observed = np.full(selected.size, np.nan, dtype=np.float64)
    tolerance = target_mz * ppm * 1e-6
    for output_index, local_index in enumerate(selected):
        mz, values = store.scan(int(local_index))
        start = np.searchsorted(mz, target_mz - tolerance, side="left")
        end = np.searchsorted(mz, target_mz + tolerance, side="right")
        values = values[start:end]
        positive = values > 0
        if positive.any():
            mz = mz[start:end][positive]
            values = values[positive]
            intensity[output_index] = np.sum(values, dtype=np.float64)
            observed[output_index] = np.average(mz, weights=values)
    return intensity, observed


def _store():
    builder = RawMS1StoreBuilder()
    builder.append(
        [499.999, 500.0, 500.0005, 501.0],
        [0.0, 10.0, 20.0, 5.0],
        source_scan_index=1,
        scan_number=101,
        rt_sec=1.0,
        faims_cv=None,
    )
    builder.append(
        [499.0, 500.0, 500.0002, 501.0],
        [3.0, 0.0, 30.0, 8.0],
        source_scan_index=2,
        scan_number=102,
        rt_sec=2.0,
        faims_cv=None,
    )
    return builder.finalize()


def test_cython_batch_trace_matches_scalar_reference():
    configure_backend("cython")
    store = _store()
    targets = (500.0, 501.0)
    traces = store.extract_traces(targets, 5.0, 0.0, 3.0)
    for target, trace in zip(targets, traces):
        intensity, observed = _scalar_trace(store, target, 5.0, 0.0, 3.0)
        np.testing.assert_allclose(trace.intensity, intensity)
        np.testing.assert_allclose(trace.observed_mz, observed, equal_nan=True)


def test_auto_backend_falls_back_to_cython_without_optional_extension():
    assert configure_backend("auto") == "cython"
    assert resolved_backend() == "cython"


def test_explicit_missing_rust_backend_is_actionable():
    with pytest.raises(RuntimeError, match="biosaur2-rust-core"):
        configure_backend("rust")
