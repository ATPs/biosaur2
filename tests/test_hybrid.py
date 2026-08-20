from dataclasses import replace
from types import SimpleNamespace

import pytest

from biosaur2.chemistry import isotope_library, parse_peptidoform
from biosaur2.hybrid import (
    DirectAssay,
    QUALITY_FLAG_RAW_BASELINE_FALLBACK,
    QUALITY_FLAG_TWO_POINT_QUANT,
    _apply_generic_strict_associations,
    _calibrate_generic_score_weights,
    _compact_generic_association_summary,
    _compete_generic_local_by_input_family,
    _direct_relaxed_retry_enabled,
    _feature_population_summary,
    _feature_row_as_strict_record,
    _build_final_strict_raw_point_index,
    _final_strict_protection_reason,
    _generic_decoy_rows,
    _generic_local_refinement_events,
    _local_feature_equivalent,
    _ms2_audit_summary,
    _local_candidate_raw_points,
    _protected_local_conflict,
    _processed_hill_retry_parameters,
    _quant_row,
    _rescore_generic_link_rows,
    _candidate_uses_assigned_strict_hill,
    _strict_hill_claim_indexes,
    _strict_record_existing_equivalents,
    _allocate_strict_feature_population,
    _strict_feature_records,
    _update_generic_quant_support,
    build_direct_assays,
    build_strict_feature_index,
    calibrate_direct_run,
    extract_local_feature,
    match_assay_to_strict_feature,
)
from biosaur2.hybrid_strict import _recovered_feature_row
from biosaur2.generic_association import GENERIC_ASSOCIATION_SCORE_WEIGHTS, composite_association_support
from biosaur2.identifications import (
    IdentificationMappingResult,
    IdentificationRecord,
    MappedIdentification,
)
from biosaur2.raw_ms1 import RawMS1StoreBuilder
from biosaur2.residual import ResidualMS1Ledger
from biosaur2.generic_local import evaluate_generic_local_candidate


def _assay(rt=2.0):
    peptide = parse_peptidoform("PEPTIDE")
    peaks = isotope_library(peptide.formula, 2, max_isotopes=3)
    return DirectAssay(
        run_id="run",
        ms2_event_id=1,
        psm_id="run_10_2_1",
        canonical_peptidoform=peptide.canonical,
        charge=2,
        rt_sec=rt,
        faims_cv=None,
        selected_ion_mz=peaks[0].mz,
        selected_isotope_index=0,
        selected_mz_error_ppm=0.0,
        peptidoform=peptide,
        isotope_peaks=peaks,
        q_value=0.001,
        pep=0.001,
        score=2.0,
        rank=1,
    )


def _store(assay, mono, first=None):
    first = [0.0] * len(mono) if first is None else first
    builder = RawMS1StoreBuilder()
    for index, (mono_value, first_value) in enumerate(zip(mono, first)):
        mz = []
        intensity = []
        if mono_value:
            mz.append(assay.isotope_peaks[0].mz)
            intensity.append(mono_value)
        if first_value:
            mz.append(assay.isotope_peaks[1].mz)
            intensity.append(first_value)
        builder.append(
            mz,
            intensity,
            source_scan_index=index,
            scan_number=100 + index,
            rt_sec=float(index),
            faims_cv=None,
        )
    return builder.finalize()


def test_local_exact_assay_recovers_three_point_subthreshold_feature():
    assay = _assay()
    candidate = extract_local_feature(
        _store(assay, [0, 5, 10, 5, 0], [0, 2, 4, 2, 0]),
        assay,
        rt_tolerance_sec=10,
    )
    assert candidate.status == "accepted_local_feature"
    assert candidate.quantitative
    assert candidate.mono_point_count == 3
    assert candidate.rt_start_sec == 1.0
    assert candidate.rt_apex_sec == 2.0
    assert candidate.rt_end_sec == 3.0
    assert candidate.quantification.value == pytest.approx(21.0)
    row = _recovered_feature_row(candidate, 1)
    assert (row["scanStart"], row["scanApex"], row["scanEnd"]) == (
        101, 102, 103
    )


def test_local_exact_assay_requires_selected_isotope_at_precursor_scan():
    assay = replace(_assay(rt=2.0), precursor_ms1_index=2)
    candidate = extract_local_feature(
        _store(assay, [0, 5, 0, 5, 5], [0, 2, 0, 2, 2]),
        assay,
        rt_tolerance_sec=10,
    )
    assert candidate.status == "selected_isotope_absent_at_event_scan"
    assert not candidate.quantitative


def test_single_survey_scan_point_is_never_a_quantitative_feature():
    assay = _assay()
    candidate = extract_local_feature(
        _store(assay, [0, 0, 10, 0, 0]), assay, rt_tolerance_sec=10
    )
    assert candidate.status == "precursor_signal_only"
    assert not candidate.quantitative
    assert candidate.quantification is None


def test_boundary_truncated_exact_feature_remains_quantitative_and_flagged():
    assay = _assay(rt=1.0)
    candidate = extract_local_feature(
        _store(assay, [10, 5, 1, 0], [4, 2, 0.4, 0]),
        assay,
        rt_tolerance_sec=10,
    )
    assert candidate.quantitative
    assert candidate.boundary_truncated
    assert candidate.rt_start_sec == 0.0


def test_two_point_direct_exception_requires_coherent_isotope_support():
    assay = _assay()
    strict = extract_local_feature(
        _store(assay, [0, 5, 5, 0], [0, 2, 2, 0]),
        assay,
        rt_tolerance_sec=10,
    )
    assert strict.status == "precursor_signal_only"
    assert not strict.quantitative

    coherent = extract_local_feature(
        _store(assay, [0, 5, 5, 0], [0, 2, 2, 0]),
        assay,
        rt_tolerance_sec=10,
        allow_two_point_exception=True,
    )
    assert coherent.status == "accepted_local_feature_two_point"
    assert coherent.quantitative

    mono_only = extract_local_feature(
        _store(assay, [0, 5, 5, 0]), assay, rt_tolerance_sec=10
    )
    assert mono_only.status == "precursor_signal_only"
    assert not mono_only.quantitative


def test_direct_partial_envelope_requires_two_multiscan_nonmono_channels():
    base = _assay()
    peptide = parse_peptidoform("PEPTIDE" * 5)
    peaks = isotope_library(peptide.formula, 2, max_isotopes=3)
    assay = DirectAssay(
        **{
            **base.__dict__,
            "canonical_peptidoform": peptide.canonical,
            "peptidoform": peptide,
            "isotope_peaks": peaks,
            "selected_ion_mz": peaks[1].mz,
            "selected_isotope_index": 1,
        }
    )

    def store(include_second):
        builder = RawMS1StoreBuilder()
        for index, value in enumerate([0.0, 5.0, 10.0, 5.0, 0.0]):
            mz = []
            intensity = []
            if value:
                mz.append(assay.isotope_peaks[1].mz)
                intensity.append(value * assay.isotope_peaks[1].probability)
                if include_second:
                    mz.append(assay.isotope_peaks[2].mz)
                    intensity.append(value * assay.isotope_peaks[2].probability)
            builder.append(
                mz,
                intensity,
                source_scan_index=index,
                scan_number=100 + index,
                rt_sec=float(index),
                faims_cv=None,
            )
        return builder.finalize()

    partial = extract_local_feature(store(True), assay, rt_tolerance_sec=10)
    assert partial.status == "precursor_signal_only"
    assert not partial.quantitative

    partial = extract_local_feature(
        store(True),
        assay,
        rt_tolerance_sec=10,
        allow_partial_envelope=True,
    )
    assert partial.status == "accepted_local_feature_partial_envelope"
    assert partial.quantitative
    assert partial.mono_point_count == 0
    assert partial.point_count == 3

    selected_only = extract_local_feature(store(False), assay, rt_tolerance_sec=10)
    assert selected_only.status == "precursor_signal_only"
    assert not selected_only.quantitative


def test_relaxed_direct_retry_is_explicit_and_q_value_boundary_is_exclusive():
    assay = _assay()
    assert not _direct_relaxed_retry_enabled(
        assay, {"relaxed_ms2_feature": False}
    )
    assert _direct_relaxed_retry_enabled(
        assay, {"relaxed_ms2_feature": True}
    )
    assert not _direct_relaxed_retry_enabled(
        replace(assay, q_value=0.01),
        {"relaxed_ms2_feature": True},
    )


def test_direct_run_calibration_is_robust_and_requires_enough_anchors():
    assays = tuple(
        replace(_assay(), ms2_event_id=index, psm_id=str(index))
        for index in range(6)
    )
    theoretical = assays[0].isotope_peaks[0].mz
    matches = tuple(
        (
            {
                "mz": theoretical * (1.0 + value * 1e-6),
                "rt_start": 0.0,
                "rt_apex": 2.5,
                "rt_end": 6.0,
            },
            "matched_strict_feature",
            1,
        )
        for value in (1.8, 1.9, 2.0, 2.0, 2.1, 20.0)
    )
    calibration = calibrate_direct_run(
        assays,
        matches,
        base_ppm=8.0,
        base_rt_tolerance_sec=120.0,
    )
    assert calibration.status == "applied"
    assert calibration.anchor_count == 6
    assert calibration.mass_error_center_ppm == pytest.approx(2.0)
    assert calibration.rt_apex_offset_sec == pytest.approx(0.5)
    assert calibration.width_median_sec == pytest.approx(6.0)
    assert calibration.retry_rt_tolerance_sec == pytest.approx(15.0)

    sparse = calibrate_direct_run(
        assays[:4],
        matches[:4],
        base_ppm=8.0,
        base_rt_tolerance_sec=120.0,
    )
    assert sparse.status == "insufficient_anchors"
    assert sparse.retry_ppm == 8.0


def test_calibrated_mass_center_can_recover_an_unresolved_exact_trace():
    assay = _assay()
    shift_ppm = 5.0
    builder = RawMS1StoreBuilder()
    for index, scale in enumerate([0.0, 1.0, 2.0, 1.0, 0.0]):
        mz = []
        intensity = []
        if scale:
            for peak in assay.isotope_peaks[:2]:
                mz.append(peak.mz * (1.0 + shift_ppm * 1e-6))
                intensity.append(scale * peak.probability * 1000.0)
        builder.append(
            mz,
            intensity,
            source_scan_index=index,
            scan_number=100 + index,
            rt_sec=float(index),
            faims_cv=None,
        )
    store = builder.finalize()
    unresolved = extract_local_feature(
        store, assay, ppm=1.0, rt_tolerance_sec=10.0
    )
    assert not unresolved.quantitative
    recovered = extract_local_feature(
        store,
        assay,
        ppm=1.0,
        rt_tolerance_sec=10.0,
        mz_shift_ppm=shift_ppm,
    )
    assert recovered.quantitative
    assert recovered.status == "accepted_local_feature"


def test_segmentation_selects_component_nearest_direct_ms2():
    assay = _assay(rt=7.0)
    store = _store(assay, [1, 3, 1, 0, 0, 0, 2, 5, 2, 0])
    candidate = extract_local_feature(store, assay, rt_tolerance_sec=10)
    assert candidate.status == "accepted_local_feature"
    assert candidate.rt_start_sec == 6.0
    assert candidate.rt_apex_sec == 7.0
    assert candidate.rt_end_sec == 8.0


def test_joint_envelope_splits_two_peaks_connected_by_a_shallow_trace():
    assay = _assay(rt=2.0)
    mono = [1, 4, 10, 4, 1, 0.5, 1, 4, 9, 4, 1]
    first_isotope = [value * 0.4 for value in mono]
    candidate = extract_local_feature(
        _store(assay, mono, first_isotope), assay, rt_tolerance_sec=20
    )
    assert candidate.quantitative
    assert candidate.rt_start_sec == 0.0
    assert candidate.rt_apex_sec == 2.0
    assert candidate.rt_end_sec == 5.0
    assert candidate.mono_point_count == 6


def test_indistinguishable_direct_identities_share_one_local_component():
    assay = _assay()
    first = extract_local_feature(
        _store(assay, [0, 5, 10, 5, 0], [0, 2, 4, 2, 0]),
        assay,
        rt_tolerance_sec=10,
    )
    alternative_assay = DirectAssay(
        **{
            **assay.__dict__,
            "canonical_peptidoform": "ISOBARIC_ALTERNATIVE",
            "psm_id": "alternative",
        }
    )
    alternative = replace(first, assay=alternative_assay)
    assert _local_feature_equivalent(first, alternative, ppm=8.0)

    shifted_peaks = tuple(
        replace(peak, mz=peak.mz + 0.1) for peak in assay.isotope_peaks
    )
    shifted = replace(
        alternative,
        assay=replace(alternative_assay, isotope_peaks=shifted_peaks),
    )
    assert not _local_feature_equivalent(first, shifted, ppm=8.0)
    assert _local_candidate_raw_points(first)
    assert _protected_local_conflict(first, shifted)

    disjoint = extract_local_feature(
        _store(assay, [0, 0, 0, 0, 5, 10, 5]),
        replace(assay, rt_sec=5.0),
        rt_tolerance_sec=10,
    )
    assert not (
        _local_candidate_raw_points(first)
        & _local_candidate_raw_points(disjoint)
    )
    assert not _protected_local_conflict(first, disjoint)


def test_local_retry_rejects_only_centroids_from_assigned_strict_hills():
    assay = _assay()
    local = extract_local_feature(
        _store(assay, [0, 5, 10, 5, 0], [0, 2, 4, 2, 0]),
        assay,
        rt_tolerance_sec=10,
    )
    hills = {
        "hills_mz_median": [600.0, assay.isotope_peaks[0].mz],
        "hills_scan_lists": [[1, 2, 3], [1, 2, 3]],
        "tmp_mz_array": [
            [600.0, 600.0, 600.0],
            [assay.isotope_peaks[0].mz] * 3,
        ],
    }
    context = {
        "hills": hills,
        "spectra": [{"scan_index": value} for value in range(5)],
        "faims_cv": None,
        "candidates": [
            {"monoisotope idx": 0, "isotopes": []}
        ],
    }
    indexes = _strict_hill_claim_indexes([context])
    assert not _candidate_uses_assigned_strict_hill(local, indexes, 8.0)

    context["candidates"] = [
        {"monoisotope idx": 1, "isotopes": []}
    ]
    indexes = _strict_hill_claim_indexes([context])
    assert _candidate_uses_assigned_strict_hill(local, indexes, 8.0)


def test_strict_population_claims_each_raw_centroid_once():
    builder = RawMS1StoreBuilder()
    for scan, scale in enumerate((1.0, 2.0, 1.0)):
        builder.append(
            [500.0, 500.5],
            [10.0 * scale, 4.0 * scale],
            source_scan_index=10 + scan,
            scan_number=100 + scan,
            rt_sec=float(scan),
            faims_cv=None,
        )
    context = {
        "hills": {
            "hills_mz_median": [500.0, 500.5],
            "hills_scan_lists": [[0, 1, 2], [0, 1, 2]],
            "hills_intensity_array": [[10.0, 20.0, 10.0], [4.0, 8.0, 4.0]],
            "tmp_mz_array": [[500.0] * 3, [500.5] * 3],
        },
        "rt_by_local": {0: 0.0, 1: 1.0, 2: 2.0},
        "spectra": [{"scan_index": 10 + value} for value in range(3)],
        "faims_cv": None,
        "candidates": [
            {
                "feature_idx": 7,
                "monoisotope idx": 0,
                "hill_mz_1": 500.0,
                "charge": 2,
                "isotopes": [{"isotope_idx": 1}],
            }
        ],
    }
    ledger = ResidualMS1Ledger(builder.finalize())
    result = _allocate_strict_feature_population(
        ledger, _strict_feature_records([context])
    )
    assert result["status_counts"] == {"accepted": 1}
    assert result["failed_feature_count"] == 0
    assert ledger.allocation_count == 1
    assert ledger.claimed_point_count == 6
    assert ledger.residual_intensity == pytest.approx(0.0)
    assert ledger.materialize().intensity.sum() == pytest.approx(0.0)


def test_strict_population_parallel_footprints_match_serial_claims():
    builder = RawMS1StoreBuilder()
    for scan, scale in enumerate((1.0, 2.0, 1.0)):
        builder.append(
            [500.0, 500.5, 600.0, 600.5],
            [10.0 * scale, 4.0 * scale, 8.0 * scale, 3.0 * scale],
            source_scan_index=10 + scan,
            scan_number=100 + scan,
            rt_sec=float(scan),
            faims_cv=None,
        )
    context = {
        "hills": {
            "hills_mz_median": [500.0, 500.5, 600.0, 600.5],
            "hills_scan_lists": [[0, 1, 2]] * 4,
            "hills_intensity_array": [
                [10.0, 20.0, 10.0],
                [4.0, 8.0, 4.0],
                [8.0, 16.0, 8.0],
                [3.0, 6.0, 3.0],
            ],
            "tmp_mz_array": [[value] * 3 for value in (500.0, 500.5, 600.0, 600.5)],
        },
        "rt_by_local": {0: 0.0, 1: 1.0, 2: 2.0},
        "spectra": [{"scan_index": 10 + value} for value in range(3)],
        "faims_cv": None,
        "candidates": [
            {
                "feature_idx": 7,
                "monoisotope idx": 0,
                "hill_mz_1": 500.0,
                "charge": 2,
                "isotopes": [{"isotope_idx": 1}],
            },
            {
                "feature_idx": 8,
                "monoisotope idx": 2,
                "hill_mz_1": 600.0,
                "charge": 2,
                "isotopes": [{"isotope_idx": 3}],
            },
        ],
    }
    records = _strict_feature_records([context])
    serial_ledger = ResidualMS1Ledger(builder.finalize())
    parallel_ledger = ResidualMS1Ledger(builder.finalize())
    serial = _allocate_strict_feature_population(serial_ledger, records)
    parallel = _allocate_strict_feature_population(
        parallel_ledger, records, workers=2
    )
    assert parallel == serial
    assert parallel_ledger._allocations == serial_ledger._allocations
    assert parallel_ledger.residual_intensity == pytest.approx(
        serial_ledger.residual_intensity
    )


def test_processed_hill_retry_uses_real_apex_and_bounded_width():
    competitor = SimpleNamespace(
        mono_mz_error_ppm=1.25,
        candidate={"monoisotope idx": 0},
    )
    context = {
        "hills": {
            "hills_scan_lists": [[0, 1, 2]],
            "hills_scan_apex": [1],
            "hills_intensity_array": [[10.0, 30.0, 20.0]],
        },
        "rt_by_local": {0: 100.0, 1: 105.0, 2: 112.0},
    }
    parameters = _processed_hill_retry_parameters(
        competitor, context, 120.0
    )
    assert parameters == {
        "rt_center_sec": 105.0,
        "rt_tolerance_sec": 17.0,
        "mz_shift_ppm": 1.25,
        "rt_start_sec": 100.0,
        "rt_end_sec": 112.0,
    }

    assert _processed_hill_retry_parameters(
        competitor, context, 10.0
    )["rt_tolerance_sec"] == 10.0


def test_final_strict_competitor_protects_superior_and_cross_envelope_signal():
    event = {
        "ms2_event_id": 1,
        "selected_ion_mz": 500.0,
        "isolation_target_mz": 500.0,
        "isolation_lower_offset": 0.7,
        "isolation_upper_offset": 0.7,
        "charge": 2,
        "rt_sec": 2.0,
        "faims_cv": None,
    }
    builder = RawMS1StoreBuilder()
    for scan, scale in enumerate((0.0, 1.0, 1.0, 0.0)):
        mz = []
        intensity = []
        if scale:
            mz = [500.0, 500.5016774]
            intensity = [100.0 * scale, 50.0 * scale]
        builder.append(
            mz,
            intensity,
            source_scan_index=scan,
            scan_number=100 + scan,
            rt_sec=float(scan),
            faims_cv=None,
        )
    challenger = evaluate_generic_local_candidate(
        builder.finalize(),
        event,
        width_limit_sec=20.0,
        rt_tolerance_sec=10.0,
        min_mono_points=2,
        min_channel_points=2,
        min_supported_channels=2,
        min_cosine=0.0,
        relaxed=True,
    )
    assert challenger.quantitative_candidate
    challenger = replace(challenger, isotope_cosine=0.965)
    context = {
        "hills": {
            "hills_scan_lists": [[1, 2]],
            "hills_intensity_array": [[100.0, 100.0]],
            "tmp_mz_array": [[500.0, 500.0]],
        },
        "rt_by_local": {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0},
        "spectra": [{"scan_index": value} for value in range(4)],
        "faims_cv": None,
        "candidates": [
            {
                "feature_idx": 7,
                "monoisotope idx": 0,
                "hill_mz_1": 500.0,
                "charge": 2,
                "cos_cor_isotopes": 0.99,
                "isotopes": [],
            }
        ],
    }
    strict_record = _strict_feature_records([context])[0]
    index = _build_final_strict_raw_point_index([strict_record])
    assert _final_strict_protection_reason(challenger, index, 8.0) == (
        "superior_equivalent_strict_isotope_fit"
    )

    tied_strict = {
        **strict_record,
        "candidate": {
            **strict_record["candidate"],
            "cos_cor_isotopes": 0.96,
        },
    }
    tied_index = _build_final_strict_raw_point_index([tied_strict])
    assert _final_strict_protection_reason(challenger, tied_index, 8.0) is None

    cross_envelope = {
        **strict_record,
        "charge": 3,
        "candidate": {**strict_record["candidate"], "charge": 3},
    }
    cross_index = _build_final_strict_raw_point_index([cross_envelope])
    assert _final_strict_protection_reason(challenger, cross_index, 8.0) == (
        "unidentifiable_cross_candidate_overlap"
    )


def _mapped_psm(psm_id, peptide, q_value=0.001, event_id=1):
    parsed_run, scan, charge, rank = psm_id.rsplit("_", 3)
    record = IdentificationRecord(
        source_row=2,
        psm_id_raw=psm_id,
        score=2.0,
        q_value=q_value,
        pep=0.001,
        peptide_raw=peptide,
        proteins=None,
        parsed_run=parsed_run,
        parsed_scan=int(scan),
        parsed_charge=int(charge),
        parsed_rank=int(rank),
        native_id=None,
        mapping_method="psm_id_right_split",
        mapping_status="parsed",
    )
    peptidoform = parse_peptidoform(peptide, fixed_modifications=["C=UNIMOD:4"])
    selected = isotope_library(peptidoform.formula, int(charge), max_isotopes=2)[0].mz
    event = {
        "ms2_event_id": event_id,
        "rt_sec": 10.0,
        "faims_cv": None,
        "selected_ion_mz": selected,
        "charge": int(charge),
    }
    return MappedIdentification(record, event_id, event, "scan_number", "mapped", True)


def test_direct_assays_collapse_exact_duplicates_but_preserve_audit_rows():
    rows = (
        _mapped_psm("run_10_2_1", "ACDC", q_value=0.001),
        _mapped_psm("run_10_2_2", "AC[UNIMOD:4]DC[UNIMOD:4]", q_value=0.002),
    )
    mapping = IdentificationMappingResult(rows, {"mapped": 2}, 2, 0)
    result = build_direct_assays(
        mapping, run_id="run", fixed_modifications=["C=UNIMOD:4"]
    )
    assert len(result.assays) == 1
    assert len(result.audit) == 2
    assert {row["assay_status"] for row in result.audit} == {
        "accepted_direct_assay",
        "duplicate_identification_collapsed",
    }


def test_distinct_peptidoforms_on_one_ms2_remain_conflicting_assays():
    rows = (
        _mapped_psm("run_10_2_1", "PEPTIDE"),
        _mapped_psm("run_10_2_2", "PEPTIDER"),
    )
    mapping = IdentificationMappingResult(rows, {"mapped": 2}, 2, 0)
    result = build_direct_assays(mapping, run_id="run")
    assert len(result.assays) == 2
    assert all(
        assay.conflict_status == "conflicting_identifications"
        for assay in result.assays
    )


def test_strict_feature_mz_index_preserves_bounded_matching():
    assay = _assay()
    records = [
        {
            "feature_id": 1,
            "mz": assay.isotope_peaks[0].mz,
            "charge": assay.charge,
            "faims_cv": None,
            "rt_start": 1.0,
            "rt_end": 3.0,
        },
        {
            "feature_id": 2,
            "mz": 900.0,
            "charge": assay.charge,
            "faims_cv": None,
            "rt_start": 1.0,
            "rt_end": 3.0,
        },
    ]
    matched, status, count = match_assay_to_strict_feature(
        assay, build_strict_feature_index(records)
    )
    assert matched["feature_id"] == 1
    assert status == "matched_strict_feature"
    assert count == 1


def test_strict_feature_index_preserves_faims_tolerance_at_rounding_boundary():
    assay = DirectAssay(**{**_assay().__dict__, "faims_cv": 1.4999996})
    records = [
        {
            "feature_id": 1,
            "mz": assay.isotope_peaks[0].mz,
            "charge": assay.charge,
            "faims_cv": 1.5000004,
            "rt_start": 1.0,
            "rt_end": 3.0,
        }
    ]
    linear = match_assay_to_strict_feature(assay, records)
    indexed = match_assay_to_strict_feature(
        assay, build_strict_feature_index(records)
    )
    assert indexed == linear
    assert indexed[0]["feature_id"] == 1


def _generic_link(event_id, *, status, support=None, feature_id=None):
    return {
        "ms2_event_id": event_id,
        "feature_id": feature_id,
        "status": status,
        "selected_ion_isotope_offset": 1 if feature_id is not None else None,
        "mz_error_ppm": 0.5 if feature_id is not None else None,
        "rt_distance_sec": 0.0 if feature_id is not None else None,
        "association_support": support,
        "reason_flags": 1,
    }


def _score_components(**updates):
    values = {name: 0.5 for name in GENERIC_ASSOCIATION_SCORE_WEIGHTS}
    values.update(updates)
    return values


def test_generic_score_calibration_uses_direct_anchors_and_held_out_decoys():
    audits = {}
    targets = []
    decoys = []
    for event_id in range(80):
        feature_id = event_id + 100
        audits[event_id] = {
            "status": "matched_strict_feature",
            "feature_id": feature_id,
        }
        target = _generic_link(
            event_id,
            status="matched_existing_feature",
            support=0.5,
            feature_id=feature_id,
        )
        target["_score_components"] = _score_components(
            mz_support=0.9,
            selected_intensity_support=0.9,
            isolation_support=0.5,
        )
        decoy = _generic_link(
            event_id,
            status="matched_existing_feature",
            support=0.5,
            feature_id=feature_id + 1000,
        )
        decoy["_score_components"] = _score_components(
            mz_support=0.3,
            selected_intensity_support=0.1,
            isolation_support=0.5,
        )
        targets.append(target)
        decoys.append(decoy)

    weights, report = _calibrate_generic_score_weights(audits, targets, decoys)

    assert report["status"] == "applied"
    assert report["paired_anchor_count"] == 80
    assert report["training_pair_count"] == 40
    assert report["validation_pair_count"] == 40
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["selected_intensity_support"] > weights["isolation_support"]
    assert (
        report["selected_validation"]["median_margin"]
        > report["base_validation"]["median_margin"]
    )

    target_before = targets[0]["association_support"]
    decoy_before = decoys[0]["association_support"]
    assert _rescore_generic_link_rows(targets, weights) == 80
    assert _rescore_generic_link_rows(decoys, weights) == 80
    assert targets[0]["association_support"] > target_before
    assert decoys[0]["association_support"] < decoy_before
    assert targets[0]["association_support"] == pytest.approx(
        composite_association_support(targets[0]["_score_components"], weights)
    )


def test_generic_score_calibration_retains_base_weights_when_anchors_are_sparse():
    audits = {
        event_id: {"status": "matched_strict_feature", "feature_id": event_id}
        for event_id in range(10)
    }
    targets = []
    decoys = []
    for event_id in range(10):
        target = _generic_link(
            event_id,
            status="matched_existing_feature",
            support=0.8,
            feature_id=event_id,
        )
        target["_score_components"] = _score_components(mz_support=0.9)
        decoy = _generic_link(
            event_id,
            status="matched_existing_feature",
            support=0.2,
            feature_id=event_id + 100,
        )
        decoy["_score_components"] = _score_components(mz_support=0.1)
        targets.append(target)
        decoys.append(decoy)

    weights, report = _calibrate_generic_score_weights(audits, targets, decoys)

    assert report["status"] == "base_weights_insufficient_paired_anchors"
    assert weights == GENERIC_ASSOCIATION_SCORE_WEIGHTS


def test_generic_score_calibration_rejects_weights_that_reduce_q_accepted_targets():
    audits = {}
    targets = []
    decoys = []
    for event_id in range(80):
        feature_id = event_id + 100
        audits[event_id] = {
            "status": "matched_strict_feature",
            "feature_id": feature_id,
        }
        target = _generic_link(
            event_id,
            status="matched_existing_feature",
            support=0.5,
            feature_id=feature_id,
        )
        target["_score_components"] = _score_components(
            mz_support=0.9, selected_intensity_support=0.9
        )
        decoy = _generic_link(
            event_id,
            status="matched_existing_feature",
            support=0.5,
            feature_id=feature_id + 1000,
        )
        decoy["_score_components"] = _score_components(
            mz_support=0.1, selected_intensity_support=0.1
        )
        targets.append(target)
        decoys.append(decoy)

    for event_id in range(1000, 1200):
        audits[event_id] = {
            "status": "unresolved_no_direct_identification",
            "feature_id": None,
        }
        target = _generic_link(
            event_id,
            status="matched_existing_feature",
            support=0.6,
            feature_id=event_id,
        )
        target["_score_components"] = {
            name: (
                0.0
                if name in {"mz_support", "selected_intensity_support"}
                else 0.675
            )
            for name in GENERIC_ASSOCIATION_SCORE_WEIGHTS
        }
        decoy = _generic_link(
            event_id,
            status="matched_existing_feature",
            support=0.4,
            feature_id=event_id + 1000,
        )
        decoy["_score_components"] = {
            name: (1.0 if name in {"mz_support", "selected_intensity_support"} else 0.0)
            for name in GENERIC_ASSOCIATION_SCORE_WEIGHTS
        }
        targets.append(target)
        decoys.append(decoy)

    weights, report = _calibrate_generic_score_weights(audits, targets, decoys)

    assert report["base_generic_q_metrics"]["accepted_target_count"] == 200
    assert report["status"] == "base_weights_retained_by_dual_validation"
    assert weights == GENERIC_ASSOCIATION_SCORE_WEIGHTS
    assert all(
        not item["generic_q_acceptance_preserved"]
        for item in report["candidate_evaluations"]
    )


def test_generic_decoy_shifts_selected_ion_and_isolation_window_together():
    source = {
        "ms2_event_id": 7,
        "charge": 2,
        "selected_ion_mz": 500.0,
        "isolation_target_mz": 500.1,
        "isolation_lower_offset": 0.7,
        "isolation_upper_offset": 0.8,
    }
    decoy = _generic_decoy_rows("run", [source])[0]
    assert decoy["selected_ion_mz"] != source["selected_ion_mz"]
    assert decoy["selected_ion_mz"] - source["selected_ion_mz"] == pytest.approx(
        decoy["isolation_target_mz"] - source["isolation_target_mz"]
    )
    assert decoy["isolation_lower_offset"] == source["isolation_lower_offset"]
    assert decoy["isolation_upper_offset"] == source["isolation_upper_offset"]
    assert source["selected_ion_mz"] == 500.0


def test_generic_association_summary_persists_histogram_not_per_event_array():
    compact = _compact_generic_association_summary(
        {
            "eligible_event_count": 3,
            "association_local_hill_count": 4,
            "local_candidate_counts": [0, 2, 2],
            "status_counts": {"matched_existing_feature": 1},
        }
    )
    assert compact["local_candidate_count_histogram"] == {"0": 1, "2": 2}
    assert compact["local_candidate_count_mean"] == pytest.approx(4 / 3)
    assert "local_candidate_counts" not in compact


def test_generic_strict_links_require_target_decoy_q_value_and_preserve_direct():
    audits = {
        event_id: {
            "status": "unresolved_no_direct_identification",
            "feature_id": None,
            "association_tier": "none",
        }
        for event_id in range(202)
    }
    audits[201].update(
        {"status": "precursor_signal_only", "association_tier": "direct_id"}
    )
    targets = [
        _generic_link(
            event_id,
            status="matched_existing_feature",
            support=1.0,
            feature_id=event_id + 1,
        )
        for event_id in range(200)
    ]
    targets.extend(
        [
            _generic_link(200, status="no_standard_candidate"),
            _generic_link(
                201,
                status="matched_existing_feature",
                support=1.0,
                feature_id=999,
            ),
        ]
    )
    decoys = [
        _generic_link(event_id, status="no_standard_candidate")
        for event_id in range(202)
    ]
    counts, competition_counts = _apply_generic_strict_associations(
        audits, targets, decoys, q_value_max=0.01
    )
    assert counts == {
        "generic_matched_strict_feature": 200,
        "generic_no_standard_candidate": 1,
    }
    assert audits[0]["feature_id"] == 1
    assert audits[0]["extraction_q_value"] == pytest.approx(0.005)
    assert audits[200]["feature_id"] is None
    assert audits[201]["status"] == "precursor_signal_only"
    assert audits[201]["feature_id"] is None
    assert competition_counts == {
        "competition_count": 201,
        "target_candidate_count": 200,
        "decoy_candidate_count": 0,
        "both_candidate_count": 0,
        "target_only_candidate_count": 200,
        "decoy_only_candidate_count": 0,
        "target_winner_count": 200,
        "decoy_winner_count": 0,
        "no_winner_count": 1,
    }


def test_generic_strict_links_can_recheck_explicit_unlinked_events_only():
    audits = {
        event_id: {
            "status": "generic_insufficient_mono_points",
            "feature_id": None,
            "association_tier": "generic_ms2",
        }
        for event_id in range(100)
    }
    audits[100] = {
        "status": "precursor_signal_only",
        "feature_id": None,
        "association_tier": "direct_id",
    }
    targets = [
        _generic_link(
            event_id,
            status="matched_existing_feature",
            support=1.0,
            feature_id=1000 + event_id,
        )
        for event_id in range(101)
    ]
    decoys = [
        _generic_link(event_id, status="no_standard_candidate")
        for event_id in range(101)
    ]

    counts, competition_counts = _apply_generic_strict_associations(
        audits,
        targets,
        decoys,
        q_value_max=0.01,
        eligible_event_ids=range(100),
    )

    assert counts == {"generic_matched_strict_feature": 100}
    assert competition_counts["competition_count"] == 100
    assert audits[0]["feature_id"] == 1000
    assert audits[0]["extraction_q_value"] == pytest.approx(0.01)
    assert audits[100] == {
        "status": "precursor_signal_only",
        "feature_id": None,
        "association_tier": "direct_id",
    }


def test_final_generic_recheck_preserves_specific_failure_when_not_accepted():
    audits = {
        1: {
            "status": "generic_local_insufficient_mono_points",
            "feature_id": None,
            "association_tier": "generic_ms2",
            "reason_flags": 17,
        }
    }

    counts, competition_counts = _apply_generic_strict_associations(
        audits,
        [
            _generic_link(
                1,
                status="matched_existing_feature",
                support=1.0,
                feature_id=100,
            )
        ],
        [_generic_link(1, status="no_standard_candidate")],
        q_value_max=0.05,
        eligible_event_ids={1},
        preserve_failed_audit=True,
    )

    assert counts == {"generic_local_insufficient_mono_points": 1}
    assert audits[1] == {
        "status": "generic_local_insufficient_mono_points",
        "feature_id": None,
        "association_tier": "generic_ms2",
        "reason_flags": 17,
    }
    assert competition_counts["target_winner_count"] == 1


def test_quant_method_all_reports_three_values_with_envelope_primary():
    row = _quant_row(
        "run",
        1,
        "strict_untargeted",
        "strict",
        [0.0, 1.0, 2.0],
        [[0.0, 4.0, 0.0], [0.0, 2.0, 0.0]],
        method="all",
        baseline="none",
        quality_score=1.0,
        isotope_cosine=1.0,
        mass_error=0.0,
        supporting_psm_count=0,
        supporting_ms2_count=0,
    )
    assert row["quant_method"] == "all"
    assert row["quant_value"] == pytest.approx(6.0)
    assert row["quant_envelope_area"] == pytest.approx(6.0)
    assert row["quant_mono_area"] == pytest.approx(4.0)
    assert row["quant_envelope_apex"] == pytest.approx(6.0)


def test_generic_support_updates_metadata_without_duplicating_quant_rows():
    quant_rows = [
        {
            "feature_id": 10,
            "confidence_tier": "strict",
            "supporting_ms2_count": 0,
            "quant_value": 123.0,
        },
        {
            "feature_id": 20,
            "confidence_tier": "direct_id",
            "supporting_ms2_count": 1,
            "quant_value": 456.0,
        },
    ]
    audit = {
        1: {"status": "generic_matched_strict_feature", "feature_id": 10},
        2: {"status": "generic_matched_strict_feature", "feature_id": 10},
        3: {"status": "generic_matched_strict_feature", "feature_id": 20},
        4: {"status": "generic_q_value_rejected", "feature_id": None},
    }
    assert _update_generic_quant_support(quant_rows, audit) == {10: 2, 20: 1}
    assert len(quant_rows) == 2
    assert quant_rows[0] == {
        "feature_id": 10,
        "confidence_tier": "generic_ms2",
        "supporting_ms2_count": 2,
        "quant_value": 123.0,
    }
    assert quant_rows[1] == {
        "feature_id": 20,
        "confidence_tier": "direct_id",
        "supporting_ms2_count": 2,
        "quant_value": 456.0,
    }


def test_feature_population_summary_keeps_feature_and_ms2_denominators_separate():
    quant_rows = [
        {
            "feature_id": 1,
            "feature_origin": "strict_untargeted",
            "quant_status": "baseline_corrected",
            "quant_value": 10.0,
            "supporting_psm_count": 1,
            "supporting_ms2_count": 2,
        },
        {
            "feature_id": 2,
            "feature_origin": "strict_untargeted",
            "quant_status": "raw_fallback",
            "quant_value": 20.0,
            "supporting_psm_count": 0,
            "supporting_ms2_count": 0,
        },
        {
            "feature_id": 3,
            "feature_origin": "direct_identified",
            "quant_status": "raw_fallback",
            "quant_value": 30.0,
            "supporting_psm_count": 1,
            "supporting_ms2_count": 1,
        },
    ]
    audit = {
        10: {"feature_id": 1},
        11: {"feature_id": 1},
        12: {"feature_id": None},
    }
    summary = _feature_population_summary(quant_rows, audit)
    assert summary == {
        "feature_count": 3,
        "quantified_feature_count": 3,
        "null_or_nonpositive_quant_count": 0,
        "feature_origin_counts": {
            "direct_identified": 1,
            "strict_untargeted": 2,
        },
        "quant_status_counts": {
            "baseline_corrected": 1,
            "raw_fallback": 2,
        },
        "features_with_psm_support": 2,
        "features_with_ms2_support": 2,
        "features_linked_from_ms2_audit": 1,
        "features_without_ms2_audit_link": 2,
        "linked_ms2_event_count": 2,
        "unlinked_ms2_event_count": 1,
    }


def test_ms2_audit_summary_reports_required_coverage_and_mutually_exclusive_outcomes():
    quant_rows = [
        {"feature_id": 1, "quant_value": 10.0},
        {"feature_id": 2, "quant_value": 20.0},
    ]
    audit = {
        1: {
            "feature_id": 1,
            "association_tier": "direct_id",
            "status": "matched_strict_feature",
        },
        2: {
            "feature_id": 2,
            "association_tier": "generic_ms2",
            "status": "generic_recovered_local_feature",
        },
        3: {"feature_id": None, "status": "precursor_signal_only"},
        4: {"feature_id": None, "status": "generic_local_raw_point_conflict"},
        5: {"feature_id": None, "status": "no_signal"},
        6: {"feature_id": None, "status": "generic_local_q_value_rejected"},
        7: {
            "feature_id": None,
            "status": "generic_local_insufficient_mono_points",
        },
        8: {
            "feature_id": None,
            "status": "unresolved_no_direct_identification",
        },
        9: {
            "feature_id": None,
            "status": "generic_local_low_averagine_cosine",
        },
    }

    summary = _ms2_audit_summary(quant_rows, audit)

    assert summary["audit_coverage_fraction"] == 1.0
    assert summary["any_ms1_signal_association_count"] == 7
    assert summary["quantitative_feature_count"] == 2
    assert summary["direct_psm_quantitative_feature_count"] == 1
    assert summary["generic_ms2_quantitative_feature_count"] == 1
    assert summary["outcome_counts"] == {
        "ambiguous": 1,
        "insufficient_chromatographic_evidence": 2,
        "metadata_or_assay_unavailable": 1,
        "no_ms1_signal": 1,
        "precursor_signal_only": 1,
        "quantitative_feature": 2,
        "statistical_rejection": 1,
    }
    assert summary["outcomes_cover_all_ms2"]


def test_final_residual_dedup_includes_previously_accepted_local_features():
    local = _feature_row_as_strict_record(
        {
            "feature_idx": 20,
            "mz": 500.0,
            "charge": 2,
            "rtStart": 100.0,
            "rtApex": 105.0,
            "rtEnd": 110.0,
            "FAIMS": None,
        },
        "ms2_guided_full",
    )
    residual = {
        "feature_id": 30,
        "mz": 500.002,
        "charge": 2,
        "rt_start": 104.0,
        "rt_apex": 106.0,
        "rt_end": 112.0,
        "faims_cv": None,
    }

    matches = _strict_record_existing_equivalents(
        residual, build_strict_feature_index([local]), 8.0
    )

    assert [row["feature_id"] for row in matches] == [20]
    assert matches[0]["feature_origin"] == "ms2_guided_full"


def test_quant_row_flags_two_point_and_raw_baseline_fallback():
    row = _quant_row(
        "run",
        1,
        "strict_untargeted",
        "strict",
        [1.0, 2.0],
        [[10.0, 20.0], [5.0, 10.0]],
        method="envelope_area",
        baseline="edge_linear",
        quality_score=1.0,
        isotope_cosine=1.0,
        mass_error=0.0,
        supporting_psm_count=0,
        supporting_ms2_count=0,
    )
    assert row["quant_status"] == "raw_fallback"
    assert row["points_across_peak"] == 2
    assert row["quality_flags"] & QUALITY_FLAG_TWO_POINT_QUANT
    assert row["quality_flags"] & QUALITY_FLAG_RAW_BASELINE_FALLBACK


def test_generic_decoy_only_has_distinct_audit_status():
    audits = {
        1: {
            "status": "unresolved_no_direct_identification",
            "feature_id": None,
            "association_tier": "none",
        }
    }
    targets = [_generic_link(1, status="no_standard_candidate")]
    decoys = [
        _generic_link(
            1,
            status="matched_existing_feature",
            support=0.8,
            feature_id=99,
        )
    ]
    counts, competition_counts = _apply_generic_strict_associations(
        audits, targets, decoys, q_value_max=0.01
    )
    assert counts == {"generic_decoy_only": 1}
    assert audits[1]["status"] == "generic_decoy_only"
    assert audits[1]["feature_id"] is None
    assert competition_counts["decoy_candidate_count"] == 1
    assert competition_counts["decoy_only_candidate_count"] == 1
    assert competition_counts["decoy_winner_count"] == 1


def test_generic_local_refinement_retries_all_unresolved_competition_classes():
    rows = [{"ms2_event_id": event_id} for event_id in range(6)]
    audits = {
        0: {"status": "generic_no_standard_candidate"},
        1: {"status": "generic_q_value_rejected"},
        2: {"status": "generic_decoy_won"},
        3: {"status": "generic_decoy_only"},
        4: {"status": "generic_matched_strict_feature"},
        5: {"status": "matched_strict_feature"},
    }

    selected = _generic_local_refinement_events(rows, audits)

    assert [row["ms2_event_id"] for row in selected] == [0, 1, 2, 3]


def test_generic_local_competition_keeps_retry_q_value_families_separate():
    def candidate(event_id, score):
        return SimpleNamespace(
            event={"ms2_event_id": event_id},
            score=score,
            quantitative_candidate=score is not None,
        )

    targets = [candidate(1, 1.0), candidate(2, 0.8), candidate(3, 0.7)]
    decoys = [candidate(1, None), candidate(2, None), candidate(3, 0.9)]
    statuses = {
        1: "generic_no_standard_candidate",
        2: "generic_q_value_rejected",
        3: "generic_decoy_won",
    }

    competitions, counts = _compete_generic_local_by_input_family(
        targets, decoys, statuses
    )

    assert [value.event_id for value in competitions] == [1, 2, 3]
    assert counts == {"no_standard_candidate": 1, "strict_recheck": 2}
    # Each family has its own conservative +1 correction.
    assert competitions[0].q_value == 1.0
    assert competitions[1].winner == "target"
    assert competitions[2].winner == "decoy"
