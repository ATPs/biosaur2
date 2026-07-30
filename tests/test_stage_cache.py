from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from biosaur2 import utils
from biosaur2.stage_cache import (
    build_strict_stage_payload,
    load_strict_stage_cache,
    save_strict_stage_cache,
    strict_stage_argument_signature,
)
from biosaur2.direct_competitors import DirectProcessedHillCompetitor


def _args(**updates):
    values = {
        "mini": 1.0,
        "minmz": 350.0,
        "maxmz": 1500.0,
        "pasefmini": 100.0,
        "htol": 8.0,
        "itol": 8.0,
        "ignore_iso_calib": False,
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
    }
    values.update(updates)
    return values


def _context():
    hills, _mz_step = utils._build_hills_dict(
        hills_idx_array_unique=[10, 11, 12],
        hills_mz_median=[400.0, 500.0, 500.5],
        hills_im_median=None,
        hills_lengths=[3, 3, 3],
        hills_scan_lists=[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        hills_intensity_array=[[1, 2, 1], [10, 20, 10], [4, 8, 4]],
        rt_start=[0.0, 0.0, 0.0],
        rt_end=[2.0, 2.0, 2.0],
        rt_apex=[1.0, 1.0, 1.0],
        hill_mass_accuracy=8.0,
        paseftol=0.0,
    )
    hills["tmp_mz_array"] = [
        [400.0] * 3,
        [500.0] * 3,
        [500.5] * 3,
    ]
    losing_candidate = {
        "monoisotope idx": 0,
        "monoisotope hill idx": 10,
        "hill_mz_1": 400.0,
        "charge": 2,
        "isotopes": [
            {
                "isotope_idx": 1,
                "isotope_hill_idx": 11,
                "isotope_number": 1,
                "mass_diff_ppm": 0.0,
            }
        ],
        "nIsotopes": 2,
        "nScans": 3,
        "cos_cor_isotopes": 0.95,
        "intensity_array_for_cos_corr": [[1.0, 0.4], [1.0, 0.4]],
    }
    return {
        "hills": hills,
        "rt_by_local": {0: 0.0, 1: 1.0, 2: 2.0},
        "spectra": [
            {"scan_index": value, "scan_number": 100 + value, "rt_sec": float(value)}
            for value in range(3)
        ],
        "faims_cv": None,
        "paseftol": 0.0,
        "candidates": [
            {
                "monoisotope idx": 1,
                "monoisotope hill idx": 11,
                "hill_mz_1": 500.0,
                "charge": 2,
                "isotopes": [
                    {
                        "isotope_idx": 2,
                        "isotope_hill_idx": 12,
                        "isotope_number": 1,
                        "mass_diff_ppm": 0.0,
                    }
                ],
                "nIsotopes": 2,
                "nScans": 3,
                "cos_cor_isotopes": 1.0,
                "feature_idx": 1,
                "intensity_array_for_cos_corr": [[1.0, 0.4], [1.0, 0.4]],
            }
        ],
        "direct_competitors": (
            DirectProcessedHillCompetitor(
                1,
                "run_1_2_1",
                (10, 2, ((1, 11),)),
                losing_candidate,
                0.0,
                0.0,
                0,
                0,
                True,
                True,
                0.9,
            ),
        ),
    }


def test_strict_stage_cache_compacts_hills_and_rejects_stale_parameters(tmp_path):
    source = tmp_path / "run.mzML"
    source.write_bytes(b"source")
    ingestion = SimpleNamespace(
        ms1_rows=[{"scan_index": 0}],
        ms2_rows=[{"ms2_event_id": 0}],
        ms1_metadata={0: {"faims_cv": None}},
    )
    args = _args()
    payload = build_strict_stage_payload(
        ingestion, [_context()], 2, args
    )
    compact = payload["strict_contexts"][0]
    assert len(compact["hills"]["hills_idx_array_unique"]) == 3
    assert compact["candidates"][0]["monoisotope idx"] == 1
    assert compact["candidates"][0]["isotopes"][0]["isotope_idx"] == 2
    assert compact["direct_competitors"][0].candidate[
        "monoisotope idx"
    ] == 0
    assert compact["direct_competitors"][0].candidate["isotopes"][0][
        "isotope_idx"
    ] == 1

    cache = save_strict_stage_cache(
        tmp_path / "strict-cache", source, args, payload
    )
    loaded, manifest = load_strict_stage_cache(cache, source, args)
    assert loaded["next_feature_id"] == 2
    assert manifest["strict_feature_count"] == 1
    assert manifest["direct_competitor_count"] == 1
    assert manifest["payload_bytes"] > 0

    with pytest.raises(ValueError, match="upstream_arguments"):
        load_strict_stage_cache(cache, source, _args(cmax=8))


def test_stage_signature_excludes_downstream_and_scheduling_options():
    first = _args(
        nprocs=1,
        relaxed_ms2_feature=False,
        generic_q_value_max=0.01,
        quant_method="envelope_area",
    )
    second = {
        **first,
        "nprocs": 32,
        "relaxed_ms2_feature": True,
        "generic_q_value_max": 0.02,
        "quant_method": "mono_area",
    }
    assert strict_stage_argument_signature(first) == (
        strict_stage_argument_signature(second)
    )


def test_hybrid_stage_cache_cold_and_replay_are_logically_equal(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "PXD010154_1554451_middle.mzML.gz"
    )
    cold = tmp_path / "cold"
    replay = tmp_path / "replay"
    cold.mkdir()
    replay.mkdir()
    cache_root = tmp_path / "cache"

    def run(output_directory):
        feature_path = output_directory / "features.parquet"
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "biosaur2.search",
                str(source),
                "-o",
                str(feature_path),
                "--feature-format",
                "parquet",
                "--feature-mode",
                "hybrid",
                "--no-generic-ms2-refine",
                "--cache-dir",
                str(cache_root),
                "--keep-cache",
                "--workers",
                "2",
            ],
            text=True,
            capture_output=True,
        )

    cold_result = run(cold)
    assert cold_result.returncode == 0, cold_result.stderr
    replay_result = run(replay)
    assert replay_result.returncode == 0, replay_result.stderr
    assert "Reused strict-stage cache" in replay_result.stderr

    stem = "PXD010154_1554451_middle"
    pairs = [
        (cold / "features.parquet", replay / "features.parquet"),
        (
            cold / (stem + ".feature_quant.parquet"),
            replay / (stem + ".feature_quant.parquet"),
        ),
        (
            cold / (stem + ".ms2.parquet"),
            replay / (stem + ".ms2.parquet"),
        ),
        (
            cold / (stem + ".ms2_feature_links.parquet"),
            replay / (stem + ".ms2_feature_links.parquet"),
        ),
    ]
    for cold_path, replay_path in pairs:
        assert pq.read_table(cold_path).equals(pq.read_table(replay_path))


def test_hybrid_stage_cache_rejects_existing_file_path(tmp_path):
    cache_path = tmp_path / "not-a-directory"
    cache_path.write_text("occupied", encoding="utf-8")
    source = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "PXD010154_1554451_middle.mzML.gz"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "biosaur2.search",
            str(source),
            "-o",
            str(tmp_path / "features.parquet"),
            "--feature-format",
            "parquet",
            "--feature-mode",
            "hybrid",
            "--cache-dir",
            str(cache_path),
            "--keep-cache",
            "--workers",
            "1",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "File exists" in result.stderr
