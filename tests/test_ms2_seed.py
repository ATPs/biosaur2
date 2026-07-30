from pathlib import Path

import pyarrow.parquet as pq

from biosaur2.legacy_output import CompactOutputManager
from biosaur2.ms2_seed import (
    C13_C12_MASS_DIFF,
    COMPOSITE_SCORE_WEIGHTS,
    annotate_candidate_support,
    build_link_rows,
    composite_seed_support,
    partition_mono_indices,
    prepare_seed_context,
    precursor_joint_support,
)
from biosaur2.schema import compact_schemas


def _hills():
    return {
        "hills_mz_median": [500.0, 501.003354835],
        "hills_idx_array_unique": [10, 11],
        "hills_scan_lists": [[0, 1], [0, 1]],
        "hills_scan_sets": [{0, 1}, {0, 1}],
        "hills_intensity_array": [[10.0, 20.0], [4.0, 8.0]],
        "tmp_mz_array": [[500.0, 500.0], [501.003354835, 501.003354835]],
    }


def _spectra():
    return [
        {"scan_index": 100, "rt_sec": 50.0},
        {"scan_index": 101, "rt_sec": 60.0},
    ]


def _args(**values):
    result = {
        "cmin": 1, "cmax": 6, "itol": 8.0, "ms2_seed_ppm": 10.0,
        "ms2_seed_rt_tolerance_sec": 120.0,
        "ms2_seed_isotope_errors": (-1, 0, 1, 2, 3),
    }
    result.update(values)
    return result


def _event(**values):
    result = {
        "run_id": "run", "ms2_event_id": 7, "selected_ion_mz": 500.0,
        "charge": 2, "rt_sec": 55.0, "precursor_ms1_index": 100,
        "faims_cv": None, "isolation_target_mz": 500.0,
        "isolation_lower_offset": 1.0, "isolation_upper_offset": 1.0,
        "selected_ion_intensity": None,
    }
    result.update(values)
    return result


def _candidate():
    return {
        "monoisotope idx": 0, "monoisotope hill idx": 10, "hill_mz_1": 500.0,
        "charge": 2, "isotopes": [{"isotope_idx": 1, "isotope_hill_idx": 11}],
        "nIsotopes": 2, "cos_cor_isotopes": 0.9, "feature_idx": 1,
    }


def test_seed_partition_and_joint_offsets():
    hills = _hills()
    event = _event(selected_ion_mz=500.0 + C13_C12_MASS_DIFF / 2.0)
    context = prepare_seed_context(
        hills, _spectra(), [event], {100: {"faims_cv": None}}, None,
        {0: 50.0, 1: 60.0}, _args(), 1,
    )
    seeded, remaining = partition_mono_indices([0, 1], context)
    assert 0 in seeded
    assert set(seeded).isdisjoint(remaining)
    assert sorted(seeded + remaining) == [0, 1]


def test_seed_support_uses_source_scan_and_isolation():
    hills = _hills()
    context = prepare_seed_context(
        hills, _spectra(), [_event()], {100: {"faims_cv": None}}, None,
        {0: 50.0, 1: 60.0}, _args(), 1,
    )
    candidate = _candidate()
    annotate_candidate_support(candidate, hills, context, _args())
    assert candidate["_ms2_seed_support"] > 0.9
    edge = candidate["_ms2_seed_edges"][0]
    assert edge["offset"] == 0
    assert edge["scan_distance"] == 0
    assert edge["isolated_index"] == 0


def test_hybrid_composite_support_rewards_feature_quality_and_event_apex():
    hills = _hills()
    shoulder_event = _event(ms2_event_id=7, precursor_ms1_index=100)
    apex_event = _event(ms2_event_id=8, precursor_ms1_index=101, rt_sec=60.0)
    args = _args(ms2_seed_composite_score=True)
    context = prepare_seed_context(
        hills,
        _spectra(),
        [shoulder_event, apex_event],
        {100: {"faims_cv": None}, 101: {"faims_cv": None}},
        None,
        {0: 50.0, 1: 60.0},
        args,
        1,
    )
    candidate = _candidate()
    candidate["isotopes"][0]["mass_diff_ppm"] = 0.0
    annotate_candidate_support(candidate, hills, context, args)
    edges = {edge["event_id"]: edge for edge in candidate["_ms2_seed_edges"]}

    assert edges[8]["support"] > edges[7]["support"]
    assert edges[8]["score_components"]["event_apex_support"] == 1.0
    assert edges[7]["score_components"]["event_apex_support"] == 0.5
    assert edges[8]["score_components"]["isotope_cosine_support"] > 0.7


def test_hybrid_composite_support_uses_exact_selected_isotope_intensity():
    hills = _hills()
    matched = _event(
        ms2_event_id=7,
        precursor_ms1_index=101,
        rt_sec=60.0,
        selected_ion_intensity=20.0,
    )
    mismatched = _event(
        ms2_event_id=8,
        precursor_ms1_index=101,
        rt_sec=60.0,
        selected_ion_intensity=2000.0,
    )
    args = _args(ms2_seed_composite_score=True)
    context = prepare_seed_context(
        hills,
        _spectra(),
        [matched, mismatched],
        {100: {"faims_cv": None}, 101: {"faims_cv": None}},
        None,
        {0: 50.0, 1: 60.0},
        args,
        1,
    )
    candidate = _candidate()
    candidate["isotopes"][0]["mass_diff_ppm"] = 0.0
    annotate_candidate_support(candidate, hills, context, args)
    edges = {edge["event_id"]: edge for edge in candidate["_ms2_seed_edges"]}

    assert edges[7]["score_components"]["selected_intensity_support"] == 1.0
    assert edges[8]["score_components"]["selected_intensity_support"] < 0.2
    assert edges[7]["support"] > edges[8]["support"]


def test_composite_seed_support_weights_are_normalized_and_bounded():
    assert sum(COMPOSITE_SCORE_WEIGHTS.values()) == 1.0
    assert composite_seed_support({}) == 0.0
    assert composite_seed_support(
        {name: 1.0 for name in COMPOSITE_SCORE_WEIGHTS}
    ) == 1.0
    assert composite_seed_support(
        {name: 2.0 for name in COMPOSITE_SCORE_WEIGHTS}
    ) == 1.0


def test_precursor_joint_support_requires_all_localization_evidence():
    components = {
        "mz_support": 1.0,
        "selected_intensity_support": 0.81,
        "event_apex_support": 1.0,
        "isotope_cosine_support": 1.0,
    }
    assert precursor_joint_support(components) == 0.81 ** 0.25
    components["event_apex_support"] = 0.0
    assert precursor_joint_support(components) == 0.0


def test_support_can_use_isotope_scan_when_mono_is_not_seed_local():
    hills = _hills()
    hills["hills_scan_lists"] = [[0, 1], [1, 2]]
    hills["hills_scan_sets"] = [{0, 1}, {1, 2}]
    hills["hills_intensity_array"] = [[10.0, 20.0], [4.0, 8.0]]
    hills["tmp_mz_array"] = [
        [500.0, 500.0], [501.003354835, 501.003354835]
    ]
    event = _event(
        rt_sec=60.0,
        precursor_ms1_index=103,
        isolation_target_mz=501.003354835,
        isolation_lower_offset=0.1,
        isolation_upper_offset=0.1,
    )
    context = prepare_seed_context(
        hills,
        [{"scan_index": 100, "rt_sec": 50.0},
         {"scan_index": 101, "rt_sec": 55.0},
         {"scan_index": 103, "rt_sec": 60.0}],
        [event], {103: {"faims_cv": None}}, None,
        {0: 50.0, 1: 55.0, 2: 60.0}, _args(), 1,
    )

    seeded, remaining = partition_mono_indices([0, 1], context)
    assert 0 not in seeded
    assert 0 in remaining

    candidate = _candidate()
    annotate_candidate_support(candidate, hills, context, _args())
    assert candidate["_ms2_seed_support"] > 0
    edge = candidate["_ms2_seed_edges"][0]
    assert edge["scan_distance"] == 0
    assert edge["isolated_index"] == 1


def test_incompatible_isolation_does_not_create_support():
    hills = _hills()
    event = _event(
        isolation_target_mz=700.0,
        isolation_lower_offset=0.1,
        isolation_upper_offset=0.1,
    )
    context = prepare_seed_context(
        hills, _spectra(), [event], {100: {"faims_cv": None}}, None,
        {0: 50.0, 1: 60.0}, _args(), 1,
    )
    candidate = _candidate()
    annotate_candidate_support(candidate, hills, context, _args())
    row = build_link_rows([event], context, [candidate])[0]

    assert row["status"] == "no_standard_candidate"
    assert row["feature_id"] is None
    assert candidate["_ms2_seed_edges"] == []


def test_missing_charge_is_ineligible_and_keeps_null_link():
    event = _event(charge=None)
    context = prepare_seed_context(
        _hills(), _spectra(), [event], {100: {"faims_cv": None}}, None,
        {0: 50.0, 1: 60.0}, _args(), 1,
    )
    row = build_link_rows([event], context, [])[0]
    assert row["status"] == "ineligible_seed"
    assert row["feature_id"] is None
    assert row["seed_eligible"] is False


def test_reported_charge_seven_is_eligible_when_max_charge_is_seven():
    event = _event(charge=7)
    excluded = prepare_seed_context(
        _hills(), _spectra(), [event], {100: {"faims_cv": None}}, None,
        {0: 50.0, 1: 60.0}, _args(cmax=6), 1,
    )
    included = prepare_seed_context(
        _hills(), _spectra(), [event], {100: {"faims_cv": None}}, None,
        {0: 50.0, 1: 60.0}, _args(cmax=7), 1,
    )
    assert not excluded["events"][7]["eligible"]
    assert included["events"][7]["eligible"]


def test_link_sidecar_schema_and_atomic_publication(tmp_path):
    source = tmp_path / "sample.mzML.gz"
    source.write_bytes(b"input")
    args = {
        "file": str(source), "o": "", "ms2_seed": True, "write_ms2": True,
        "stop_after_hills": False, "write_hills": False, "write_ms1": False,
        "feature_format": "tsv", "hills_format": "tsv", "ms1_format": "tsv",
        "no_mono_hills": True, "no_hill_list": True, "write_extra_details": False,
        "overwrite": False, "use64": False, "intensity_decimals": "0",
        "tsv_float_decimals": "roundtrip", "parquet_compression": "zstd",
        "parquet_compression_level": 6, "parquet_row_group_size": 100,
        "parquet_sort": "mz_rt", "parquet_engine": "pyarrow", "combine_every": 1,
    }
    manager = CompactOutputManager(args)
    target = Path(str(source).removesuffix(".mzML.gz") + ".ms2_feature_links.parquet")
    manager.append_ms2_feature_links([{
        "run_id": "sample", "ms2_event_id": 0, "status": "ineligible_seed",
        "seed_eligible": False, "seed_used_in_selection": False, "reason_flags": 0,
    }])
    assert not target.exists()
    manager.finalize()
    schema = pq.read_schema(target)
    assert schema == compact_schemas()["ms2_feature_links"].with_metadata(schema.metadata)
