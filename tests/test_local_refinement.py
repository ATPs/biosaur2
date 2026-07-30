from dataclasses import replace

import numpy as np

from biosaur2.chemistry import isotope_library, parse_peptidoform
from biosaur2.hybrid import (
    DirectAssay,
    _local_feature_equivalent,
    _protected_local_conflict,
    extract_local_feature,
)
from biosaur2.local_refinement import (
    refine_local_isotope_components,
    repair_local_trace_segments,
)
from biosaur2.raw_ms1 import RawMS1StoreBuilder


def test_reversible_repair_proposals_cover_new_extend_merge_and_split():
    theoretical = np.asarray([1.0, 0.4, 0.2])

    gap_profile = np.asarray([0, 2, 5, 0, 4, 2, 0], dtype=float)
    gap_matrix = theoretical[:, None] * gap_profile
    segments, edits, initial = repair_local_trace_segments(
        gap_matrix, theoretical
    )
    assert initial == ((1, 3), (4, 6))
    assert segments == ((1, 6),)
    merge = next(edit for edit in edits if edit.action == "merge")
    assert merge.accepted and merge.objective_delta > 0
    assert merge.revert() == initial

    relink_profile = np.asarray([0, 2, 5, 0, 0, 4, 2, 0], dtype=float)
    segments, edits, initial = repair_local_trace_segments(
        theoretical[:, None] * relink_profile, theoretical
    )
    assert initial == ((1, 3), (5, 7))
    assert segments == ((1, 7),)
    relink = next(edit for edit in edits if edit.action == "relink")
    assert relink.accepted and relink.objective_delta > 0
    assert relink.revert() == initial

    extend_matrix = np.zeros((3, 8), dtype=float)
    extend_matrix[0, 2:6] = [1, 3, 3, 1]
    extend_matrix[1, 1:7] = [0.2, 1, 2, 2, 1, 0.2]
    extend_matrix[2, 1:7] = [0.1, 0.5, 1, 1, 0.5, 0.1]
    segments, edits, _initial = repair_local_trace_segments(
        extend_matrix, theoretical
    )
    assert segments == ((1, 7),)
    assert any(edit.action == "extend" and edit.accepted for edit in edits)

    no_mono = np.zeros((3, 5), dtype=float)
    no_mono[1, 1:4] = [1, 3, 1]
    no_mono[2, 1:4] = [0.5, 1.5, 0.5]
    segments, edits, initial = repair_local_trace_segments(
        no_mono, theoretical
    )
    assert initial == ()
    assert segments == ((1, 4),)
    assert any(edit.action == "new_trace" and edit.accepted for edit in edits)

    bimodal = np.asarray([1, 4, 10, 4, 1, 0.2, 1, 4, 9, 4, 1], dtype=float)
    segments, edits, _initial = repair_local_trace_segments(
        theoretical[:, None] * bimodal, theoretical
    )
    assert len(segments) == 2
    assert any(edit.action == "split" and edit.accepted for edit in edits)


def test_identifiable_overlap_nnls_improves_truth_area_and_conserves_intensity():
    theoretical = np.asarray([1.0, 0.4, 0.2])
    scans = np.arange(31, dtype=float)
    left_truth = 100.0 * np.exp(-0.5 * ((scans - 9.0) / 3.0) ** 2)
    right_truth = 70.0 * np.exp(-0.5 * ((scans - 18.0) / 3.0) ** 2)
    matrix = theoretical[:, None] * (left_truth + right_truth)[None, :]

    result = refine_local_isotope_components(matrix, theoretical)
    assert len(result.components) == 2
    assert all(
        component.source == "identifiable_nnls"
        and component.intensity_conserved
        for component in result.components
    )
    allocated = sum(
        (component.allocated_matrix for component in result.components),
        start=np.zeros_like(matrix),
    )
    np.testing.assert_allclose(allocated, matrix, rtol=1e-12, atol=1e-12)

    truth_areas = np.asarray(
        [
            np.trapezoid(theoretical.sum() * left_truth, scans),
            np.trapezoid(theoretical.sum() * right_truth, scans),
        ]
    )
    fitted_areas = np.asarray(
        [
            np.trapezoid(
                component.allocated_matrix.sum(axis=0), scans
            )
            for component in result.components
        ]
    )
    fitted_relative_error = np.abs(fitted_areas - truth_areas) / truth_areas
    naive_full_area = np.trapezoid(matrix.sum(axis=0), scans)
    naive_relative_error = np.abs(naive_full_area - truth_areas) / truth_areas
    assert np.max(fitted_relative_error) < 0.01
    assert np.mean(fitted_relative_error) < np.mean(naive_relative_error)


def _overlap_assay(rt):
    peptide = parse_peptidoform("PEPTIDE")
    peaks = isotope_library(peptide.formula, 2, max_isotopes=6)
    return DirectAssay(
        run_id="run",
        ms2_event_id=int(rt),
        psm_id="psm-%s" % rt,
        canonical_peptidoform=peptide.canonical,
        charge=2,
        rt_sec=float(rt),
        faims_cv=None,
        selected_ion_mz=peaks[0].mz,
        selected_isotope_index=0,
        selected_mz_error_ppm=0.0,
        peptidoform=peptide,
        isotope_peaks=peaks,
        q_value=0.001,
        pep=0.001,
        score=10.0,
        rank=1,
    )


def _overlap_store(assay):
    selected = tuple(
        peak
        for peak in assay.isotope_peaks
        if peak.isotope_index == 0 or peak.relative_abundance >= 0.01
    )
    scans = np.arange(31, dtype=float)
    left = 100.0 * np.exp(-0.5 * ((scans - 9.0) / 3.0) ** 2)
    right = 70.0 * np.exp(-0.5 * ((scans - 18.0) / 3.0) ** 2)
    builder = RawMS1StoreBuilder()
    for index, abundance in enumerate(left + right):
        builder.append(
            [peak.mz for peak in selected],
            [abundance * peak.probability for peak in selected],
            source_scan_index=index,
            scan_number=1000 + index,
            rt_sec=float(index),
            faims_cv=None,
        )
    return builder.finalize()


def test_production_exact_extraction_selects_distinct_conserved_components():
    left_assay = _overlap_assay(9.0)
    right_assay = replace(
        left_assay, rt_sec=18.0, ms2_event_id=18, psm_id="right"
    )
    store = _overlap_store(left_assay)
    left = extract_local_feature(
        store, left_assay, rt_tolerance_sec=30, baseline="none"
    )
    right = extract_local_feature(
        store, right_assay, rt_tolerance_sec=30, baseline="none"
    )
    assert left.status == right.status == "accepted_local_feature_deconvolved"
    assert left.allocation_group_key == right.allocation_group_key
    assert left.allocation_component_index == 0
    assert right.allocation_component_index == 1
    assert left.intensity_conserved and right.intensity_conserved
    assert not _local_feature_equivalent(left, right, ppm=8.0)
    assert not _protected_local_conflict(left, right)

    observed = sum(
        np.trapezoid(trace.intensity, left.traces[0].rt_sec)
        for trace in left.traces
    )
    allocated = left.quantification.value + right.quantification.value
    np.testing.assert_allclose(
        allocated, observed, rtol=1e-10, atol=1e-10
    )
