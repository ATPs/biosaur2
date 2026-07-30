from dataclasses import replace

import pytest

from biosaur2.hybrid import DirectAssay
from biosaur2.main import process_features_iteration
from biosaur2.direct_competitors import (
    capture_direct_processed_hill_competitors,
)
from biosaur2.chemistry import isotope_library, parse_peptidoform


def _assay(event_id=1, rt=2.0, selected_isotope=0):
    peptide = parse_peptidoform("PEPTIDE")
    peaks = isotope_library(peptide.formula, 2, max_isotopes=3)
    return DirectAssay(
        run_id="run",
        ms2_event_id=event_id,
        psm_id="run_%d_2_1" % event_id,
        canonical_peptidoform=peptide.canonical,
        charge=2,
        rt_sec=rt,
        faims_cv=None,
        selected_ion_mz=peaks[selected_isotope].mz,
        selected_isotope_index=selected_isotope,
        selected_mz_error_ppm=0.0,
        peptidoform=peptide,
        isotope_peaks=peaks,
        q_value=0.001,
        pep=0.001,
        score=2.0,
        rank=1,
        precursor_ms1_index=11,
    )


def _candidate(assay, mono_index, isotope_index, cosine, mz_shift_ppm=0.0):
    mono_mz = assay.isotope_peaks[0].mz * (1.0 + mz_shift_ppm * 1e-6)
    return {
        "monoisotope hill idx": 100 + mono_index,
        "monoisotope idx": mono_index,
        "hill_mz_1": mono_mz,
        "charge": 2,
        "FAIMS": None,
        "cos_cor_isotopes": cosine,
        "nIsotopes": 2,
        "isotopes": [
            {
                "isotope_number": 1,
                "isotope_hill_idx": 100 + isotope_index,
                "isotope_idx": isotope_index,
            }
        ],
    }


def _inputs(assay):
    candidates = [
        _candidate(assay, 0, 1, 0.99, 0.2),
        _candidate(assay, 2, 3, 0.96, 1.0),
    ]
    hills = {
        "hills_scan_lists": [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]],
        "tmp_mz_array": [
            [candidates[0]["hill_mz_1"]] * 3,
            [assay.isotope_peaks[1].mz] * 3,
            [candidates[1]["hill_mz_1"]] * 3,
            [assay.isotope_peaks[1].mz] * 3,
        ],
    }
    spectra = [{"scan_index": 10 + value} for value in range(3)]
    rt = {0: 1.0, 1: 2.0, 2: 3.0}
    event = {
        "ms2_event_id": assay.ms2_event_id,
        "precursor_ms1_index": 11,
        "isolation_target_mz": assay.selected_ion_mz,
        "isolation_lower_offset": 0.7,
        "isolation_upper_offset": 0.7,
    }
    return candidates, hills, spectra, rt, {assay.ms2_event_id: event}


def test_capture_is_bounded_ranked_and_deterministic():
    assay = _assay()
    candidates, hills, spectra, rt, events = _inputs(assay)
    first = capture_direct_processed_hill_competitors(
        [assay], candidates, hills, rt, spectra, events,
        ppm=5.0, rt_tolerance_sec=120.0, top_k=1,
    )
    repeated = capture_direct_processed_hill_competitors(
        [assay], list(reversed(candidates)), hills, rt, spectra, events,
        ppm=5.0, rt_tolerance_sec=120.0, top_k=1,
    )
    assert first == repeated
    assert len(first) == 1
    assert first[0].candidate_key == (100, 2, ((1, 101),))
    assert first[0].precursor_scan_distance == 0
    assert first[0].selected_isotope_observed
    assert first[0].isolation_supported


def test_capture_requires_charge_rt_scan_isolation_and_selected_isotope():
    assay = _assay(selected_isotope=1)
    candidates, hills, spectra, rt, events = _inputs(assay)
    assert len(capture_direct_processed_hill_competitors(
        [assay], candidates, hills, rt, spectra, events,
        ppm=5.0, rt_tolerance_sec=120.0,
    )) == 2

    wrong_charge = [{**value, "charge": 3} for value in candidates]
    assert not capture_direct_processed_hill_competitors(
        [assay], wrong_charge, hills, rt, spectra, events,
        ppm=5.0, rt_tolerance_sec=120.0,
    )

    far_rt = replace(assay, rt_sec=500.0)
    assert not capture_direct_processed_hill_competitors(
        [far_rt], candidates, hills, rt, spectra, events,
        ppm=5.0, rt_tolerance_sec=10.0,
    )

    excluded = {
        1: {
            **events[1],
            "isolation_target_mz": assay.selected_ion_mz + 10.0,
            "isolation_lower_offset": 0.1,
            "isolation_upper_offset": 0.1,
        }
    }
    assert not capture_direct_processed_hill_competitors(
        [assay], candidates, hills, rt, spectra, excluded,
        ppm=5.0, rt_tolerance_sec=120.0,
    )

    missing_selected_hill = [
        {**value, "isotopes": []} for value in candidates
    ]
    assert not capture_direct_processed_hill_competitors(
        [assay], missing_selected_hill, hills, rt, spectra, events,
        ppm=5.0, rt_tolerance_sec=120.0,
    )


def test_capture_rejects_invalid_bounds():
    assay = _assay()
    candidates, hills, spectra, rt, events = _inputs(assay)
    with pytest.raises(ValueError, match="top_k"):
        capture_direct_processed_hill_competitors(
            [assay], candidates, hills, rt, spectra, events,
            ppm=5.0, rt_tolerance_sec=120.0, top_k=0,
        )


def test_strict_iteration_retains_only_direct_relevant_losing_candidate(
    monkeypatch,
):
    assay = _assay()
    mono = assay.isotope_peaks[0].mz
    spacing = assay.isotope_peaks[1].mz - mono
    hills = {
        "hills_idx_array_unique": [10, 11, 12],
        "hills_mz_median": [mono, mono + spacing, mono + 2 * spacing],
        "hills_scan_lists": [[0, 1, 2]] * 3,
        "hills_intensity_array": [
            [50.0, 100.0, 50.0],
            [100.0, 200.0, 100.0],
            [40.0, 80.0, 40.0],
        ],
        "hills_intensity_apex": [100.0, 200.0, 80.0],
        "hills_scan_apex": [1, 1, 1],
        "tmp_mz_array": [
            [mono] * 3,
            [mono + spacing] * 3,
            [mono + 2 * spacing] * 3,
        ],
    }
    losing = {
        "monoisotope hill idx": 10,
        "monoisotope idx": 0,
        "hill_mz_1": mono,
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
        "intensity_array_for_cos_corr": [[1.0, 0.5], [1.0, 0.5]],
    }
    winner = {
        "monoisotope hill idx": 11,
        "monoisotope idx": 1,
        "hill_mz_1": mono + spacing,
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
        "cos_cor_isotopes": 0.99,
        "intensity_array_for_cos_corr": [[1.0, 0.4], [1.0, 0.4]],
    }
    monkeypatch.setattr(
        "biosaur2.main.get_initial_isotopes",
        lambda *_args: [dict(winner), dict(losing)],
    )
    args = {
        "itol": 8.0,
        "cmin": 1,
        "cmax": 7,
        "ivf": 5.0,
        "nprocs": 1,
        "md_correction": "Orbi",
        "ignore_iso_calib": True,
        "feature_mode": "hybrid",
        "nm": 0,
        "iuse": -1,
        "no_mono_hills": False,
        "ms2_rt_tolerance_sec": 120.0,
    }
    spectra = [
        {"scan_index": 10 + value, "rt_sec": float(value)}
        for value in range(3)
    ]
    event = {
        "ms2_event_id": 1,
        "precursor_ms1_index": 11,
        "isolation_target_mz": mono,
        "isolation_lower_offset": 0.2,
        "isolation_upper_offset": 0.2,
    }
    sink = []
    _used, _hill_map, _next_id, accepted = process_features_iteration(
        hills,
        None,
        0.01,
        0.0,
        {0: 0.0, 1: 1.0, 2: 2.0},
        0,
        False,
        args,
        next_feature_idx=1,
        data_for_analyse_tmp=spectra,
        direct_assays=[assay],
        direct_events_by_id={1: event},
        direct_competitor_sink=sink,
    )
    assert len(accepted) == 1
    assert accepted[0]["monoisotope hill idx"] == 11
    assert len(sink) == 1
    assert sink[0].candidate["monoisotope hill idx"] == 10
    assert sink[0].psm_id == assay.psm_id
