import numpy as np
import pytest

from biosaur2 import preprocessing, utils
from biosaur2.main import (
    FINAL_RESIDUAL_CALIBRATION_MAX_SIGMA,
    _detect_final_residual_strict,
    _final_residual_calibration_deviation,
    _strict_reference_isotope_calibration,
)
from biosaur2.hybrid import (
    _allocate_strict_feature_population,
    _strict_feature_records,
)
from biosaur2.raw_ms1 import load_raw_ms1_cache, save_raw_ms1_cache
from biosaur2.residual import ResidualMS1Ledger


def _spectrum(native_id, rt, mz, intensity, faims=None):
    result = {
        "id": native_id,
        "ms level": 1,
        "scanList": {"scan": [{"scan start time": rt}]},
        "m/z array": np.asarray(mz, dtype=np.float64),
        "intensity array": np.asarray(intensity, dtype=np.float64),
    }
    if faims is not None:
        result["FAIMS compensation voltage"] = faims
    return result


def _ms2_spectrum(native_id, rt, spectrum_ref):
    return {
        "id": native_id,
        "ms level": 2,
        "scanList": {"scan": [{"scan start time": rt}]},
        "precursorList": {
            "precursor": [
                {
                    "spectrumRef": spectrum_ref,
                    "selectedIonList": {
                        "selectedIon": [
                            {"selected ion m/z": 500.0, "charge state": 2}
                        ]
                    },
                    "isolationWindow": {
                        "isolation window target m/z": 500.0,
                        "isolation window lower offset": 1.0,
                        "isolation window upper offset": 1.0,
                    },
                }
            ]
        },
    }


def test_final_residual_calibration_uses_accepted_strict_reference():
    rng = np.random.default_rng(7)
    candidates = []
    for first, second, third in zip(
        rng.normal(-0.4, 1.1, 1200),
        rng.normal(-0.8, 1.3, 1200),
        rng.normal(-1.2, 1.5, 1200),
    ):
        candidates.append(
            {
                "nScans": 3,
                "isotopes": [
                    {"isotope_number": 1, "mass_diff_ppm": first},
                    {"isotope_number": 2, "mass_diff_ppm": second},
                    {"isotope_number": 3, "mass_diff_ppm": third},
                ],
            }
        )
    contexts = [
        {"faims_cv": None, "candidates": candidates},
        {"faims_cv": -50.0, "candidates": []},
    ]

    calibration, diagnostics = _strict_reference_isotope_calibration(
        contexts, None, {"itol": 8.0}
    )

    assert calibration[1][0] == pytest.approx(-0.4, abs=0.2)
    assert calibration[2][0] == pytest.approx(-0.8, abs=0.2)
    assert calibration[3][0] == pytest.approx(-1.2, abs=0.2)
    assert diagnostics["1"]["source"] == "accepted_input_strict_features"
    assert diagnostics["1"]["sample_count"] == 1200
    assert set(calibration) == set(range(1, 10))


def test_final_residual_calibration_small_run_fallback_stays_within_itol():
    calibration, diagnostics = _strict_reference_isotope_calibration(
        [], None, {"itol": 8.0}
    )

    assert calibration[1] == [0.0, 1.6]
    assert 5 * calibration[1][1] == pytest.approx(8.0)
    assert diagnostics["1"]["source"] == "bounded_original_itol"


def test_final_residual_calibration_guard_rejects_boundary_candidates():
    calibration = {1: [-0.6, 1.7], 2: [-0.8, 1.4]}
    accepted = {
        "isotopes": [
            {
                "isotope_number": 1,
                "mass_diff_ppm": -0.6 + 3.99 * 1.7,
            },
            {
                "isotope_number": 2,
                "mass_diff_ppm": -0.8 - 2.0 * 1.4,
            },
        ]
    }
    rejected = {
        "isotopes": [
            {
                "isotope_number": 1,
                "mass_diff_ppm": -0.6 - 4.01 * 1.7,
            }
        ]
    }

    assert _final_residual_calibration_deviation(
        accepted, calibration
    ) == pytest.approx(3.99)
    assert _final_residual_calibration_deviation(
        rejected, calibration
    ) > FINAL_RESIDUAL_CALIBRATION_MAX_SIGMA
    assert np.isinf(
        _final_residual_calibration_deviation({"isotopes": []}, calibration)
    )


def test_hybrid_ingestion_retains_subthreshold_raw_points_from_one_reader(monkeypatch):
    calls = []
    spectra = [
        _spectrum("scan=1", 10.0, [500.0, 400.0], [5.0, 100.0]),
        _spectrum("scan=2", 11.0, [500.0005], [8.0]),
        _spectrum("scan=3", 12.0, [500.0], [12.0]),
    ]

    def reader(_path):
        calls.append(1)
        return iter(spectra)

    monkeypatch.setattr(utils, "iter_ms1_and_ms2_metadata", reader)
    result = preprocessing.ingest_mzml(
        {
            "file": "run.mzML",
            "combine_every": 1,
            "mini": 10.0,
            "minmz": 350.0,
            "maxmz": 1500.0,
            "input_rt_unit": "seconds",
            "write_ms1": False,
            "write_ms2": True,
            "feature_mode": "hybrid",
        }
    )

    assert calls == [1]
    assert [list(item["intensity array"]) for item in result.spectra] == [
        [100.0],
        [12.0],
    ]
    trace = result.raw_ms1_store.extract_trace(500.0, 2.0, 9.0, 13.0)
    np.testing.assert_array_equal(trace.intensity, [5.0, 8.0, 12.0])
    np.testing.assert_array_equal(trace.point_present, [True, True, True])
    assert result.raw_ms1_store.scan_count == 3


def test_hybrid_ingestion_reads_ms2_metadata_without_legacy_ms2_output(monkeypatch):
    spectra = [
        _spectrum("scan=1", 10.0, [500.0], [100.0]),
        _ms2_spectrum("scan=2", 10.5, "scan=1"),
    ]

    monkeypatch.setattr(
        utils, "iter_ms1_and_ms2_metadata", lambda _path: iter(spectra)
    )
    result = preprocessing.ingest_mzml(
        {
            "file": "run.mzML",
            "combine_every": 1,
            "mini": 10.0,
            "minmz": 350.0,
            "maxmz": 1500.0,
            "input_rt_unit": "seconds",
            "write_ms1": False,
            "write_ms2": False,
            "feature_mode": "hybrid",
        }
    )

    assert len(result.ms2_rows) == 1
    assert result.ms2_rows[0]["precursor_ms1_index"] == 0


def test_trace_extraction_respects_rt_and_faims_and_zero_fills():
    builder = preprocessing.RawMS1StoreBuilder()
    builder.append([500.0], [10.0], source_scan_index=1, scan_number=101, rt_sec=1.0, faims_cv=-45)
    builder.append([600.0], [20.0], source_scan_index=2, scan_number=102, rt_sec=2.0, faims_cv=-45)
    builder.append([500.0], [30.0], source_scan_index=3, scan_number=103, rt_sec=3.0, faims_cv=-60)
    trace = builder.finalize().extract_trace(500.0, 5.0, 0.0, 2.5, faims_cv=-45)
    np.testing.assert_array_equal(trace.scan_number, [101, 102])
    np.testing.assert_array_equal(trace.intensity, [10.0, 0.0])


def test_persisted_and_mmap_raw_stores_are_identical_and_fingerprint_safe(tmp_path):
    source = tmp_path / "run.mzML.gz"
    source.write_bytes(b"mzML-prefix" + b"x" * 100 + b"mzML-suffix")
    builder = preprocessing.RawMS1StoreBuilder()
    builder.append(
        [500.0, 600.0],
        [10.0, 20.0],
        source_scan_index=3,
        scan_number=103,
        rt_sec=12.5,
        faims_cv=None,
    )
    builder.append(
        [500.0001],
        [30.0],
        source_scan_index=8,
        scan_number=108,
        rt_sec=14.0,
        faims_cv=-45.0,
    )
    memory = builder.finalize()
    cache = save_raw_ms1_cache(memory, tmp_path / "raw-cache", source)
    mapped = load_raw_ms1_cache(cache, source, mmap=True)
    copied = load_raw_ms1_cache(cache, source, mmap=False)
    for name in (
        "offsets",
        "mz",
        "intensity",
        "source_scan_index",
        "scan_number",
        "rt_sec",
        "faims_cv",
    ):
        np.testing.assert_array_equal(getattr(mapped, name), getattr(memory, name))
        np.testing.assert_array_equal(getattr(copied, name), getattr(memory, name))
    assert isinstance(mapped.mz, np.memmap)
    assert not isinstance(copied.mz, np.memmap)

    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        load_raw_ms1_cache(cache, source)


def test_final_strict_detector_finds_only_unallocated_residual_feature():
    builder = preprocessing.RawMS1StoreBuilder()
    profile = np.asarray([10.0, 50.0, 100.0, 50.0, 10.0])
    first_mz = (500.0, 500.501675, 501.00335)
    second_mz = (700.0, 700.501675, 701.00335)
    first_ratio = (1.0, 0.476, 0.111)
    second_ratio = (1.0, 0.666, 0.218)
    for scan, scale in enumerate(profile):
        builder.append(
            first_mz + second_mz,
            tuple(scale * value for value in first_ratio + second_ratio),
            source_scan_index=scan,
            scan_number=100 + scan,
            rt_sec=float(scan),
            faims_cv=None,
        )
    ledger = ResidualMS1Ledger(builder.finalize())
    assert ledger.allocate_observed_points(
        "accepted-first-feature",
        [
            (scan, mz, profile[scan] * ratio)
            for scan in range(profile.size)
            for mz, ratio in zip(first_mz, first_ratio)
        ],
    ).accepted
    args = {
        "mini": 1.0,
        "minmz": 350.0,
        "maxmz": 1500.0,
        "htol": 8.0,
        "itol": 8.0,
        "ignore_iso_calib": True,
        "use_hill_calib": False,
        "paseftol": 0.0,
        "nm": 0,
        "hvf": 1.3,
        "ivf": 5.0,
        "minlh": 2,
        "pasefminlh": 1,
        "cmin": 1,
        "cmax": 7,
        "tof": False,
        "profile": False,
        "md_correction": "Orbi",
        "combine_every": 1,
        "iuse": -1,
        "nprocs": 1,
        "feature_mode": "hybrid",
        "no_mono_hills": False,
    }
    result = _detect_final_residual_strict(
        ledger.materialize(),
        strict_contexts=[],
        next_feature_id=10,
        args=args,
    )
    candidates = [
        candidate
        for context in result["contexts"]
        for candidate in context["candidates"]
    ]
    assert result["status"] == "completed"
    assert result["calibration_boundary_guard"] == {
        "status": "applied",
        "reason": "reject_sparse_residual_calibration_boundary",
        "maximum_standard_deviation": 4.0,
        "candidate_count_before_guard": 1,
        "candidate_count_after_guard": 1,
        "rejected_candidate_count": 0,
    }
    assert len(candidates) == 1
    assert candidates[0]["feature_idx"] == 10
    assert candidates[0]["hill_mz_1"] == pytest.approx(700.0, abs=1e-3)
    assert all(
        abs(candidate["hill_mz_1"] - 500.0) > 0.1
        for candidate in candidates
    )
    final_records = _strict_feature_records(result["contexts"])
    ownership = _allocate_strict_feature_population(ledger, final_records)
    assert ownership["status_counts"] == {"accepted": 1}
    assert ownership["failed_feature_count"] == 0
    assert ledger.residual_intensity == pytest.approx(0.0, abs=1e-8)
