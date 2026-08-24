from . import utils
from copy import deepcopy
import logging
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from scipy.stats import binom

from .cutils import get_initial_isotopes, checking_cos_correlation_for_carbon, detect_hills, process_hills
from .parallel import balanced_ranges, run_process_tasks
from .calibration import fit_mass_calibration
from .spectra import faims_sort_key, group_spectra_by_faims
from .hills import assign_deterministic_hill_ids, normalize_hills_dataframe
from .preprocessing import (
    MzMLIngestion,
    centroid_pasef_data,
    ingest_mzml,
    process_profile,
    process_tof,
)
from .raw_ms1 import load_raw_ms1_cache
from .stage_cache import invalidate_stale_strict_stage_cache, load_strict_stage_cache
from .hybrid import AssayBuildResult, build_direct_assays, run_hybrid_postprocessing
from .direct_competitors import capture_direct_processed_hill_competitors
from .peak_splitting import split_peaks_multi
from .candidate_selection import (
    candidate_hill_ids as _candidate_hill_ids,
    select_nonconflicting_isotope_candidates as _select_nonconflicting_isotope_candidates,
    select_nonconflicting_isotope_candidates_parallel as _select_nonconflicting_isotope_candidates_parallel,
)
from .strict_cache_writer import (
    cancel_strict_cache_writer as _cancel_strict_cache_writer,
    finish_strict_cache_writer as _finish_strict_cache_writer,
    start_strict_cache_writer as _start_strict_cache_writer,
)
from .identifications import (
    map_identifications_to_ms2,
    psm_column_map_from_args,
    read_identification_table,
)
from .output import input_stem
from .external_weak import publish_detector_outcomes

logger = logging.getLogger(__name__)

FINAL_RESIDUAL_CALIBRATION_MAX_SIGMA = 4.0

def _debug_stage_start(name, **details):
    logger.debug('Stage start: %s details=%s', name, details)
    return time.monotonic()

def _debug_stage_complete(name, started, **details):
    logger.debug(
        'Stage complete: %s runtime_sec=%.3f details=%s',
        name,
        time.monotonic() - started,
        details,
    )


def _final_residual_detector_supported(strict_contexts, args):
    if int(args.get("combine_every", 1)) != 1:
        return False, "combined_ms1_scans"
    if bool(args.get("profile", False)):
        return False, "profile_centroid_provenance_unavailable"
    if bool(args.get("tof", False)):
        return False, "tof_filtered_provenance_unavailable"
    if bool(args.get("use_hill_calib", False)):
        return False, "dynamic_hill_calibration_not_replayable"
    if any(float(context.get("paseftol", 0.0)) > 0 for context in strict_contexts):
        return False, "ion_mobility_provenance_unavailable"
    return True, "supported"

def _strict_reference_isotope_calibration(strict_contexts, faims_cv, args):
    """Estimate residual-detector isotope calibration from accepted strict features."""

    values = {ordinal: [] for ordinal in range(1, 10)}
    for context in strict_contexts:
        if context.get("faims_cv") != faims_cv:
            continue
        for candidate in context["candidates"]:
            if int(candidate.get("nScans", 0)) < 3:
                continue
            for isotope in candidate["isotopes"]:
                ordinal = int(isotope["isotope_number"])
                if ordinal in values:
                    values[ordinal].append(float(isotope["mass_diff_ppm"]))

    calibration = {}
    diagnostics = {}
    itol = float(args["itol"])
    for ordinal in range(1, 4):
        samples = values[ordinal]
        fitted = (
            fit_mass_calibration(samples, bin_width=0.05)
            if len(samples) >= 1000
            else None
        )
        if fitted is not None and fitted.status == "applied":
            calibration[ordinal] = [float(fitted.shift), float(fitted.sigma)]
            diagnostics[str(ordinal)] = {
                **fitted.to_dict(),
                "source": "accepted_input_strict_features",
            }
        elif ordinal == 1:
            # The final detector may still operate on small synthetic/real runs,
            # but its fallback cannot widen beyond the original candidate window.
            calibration[ordinal] = [0.0, itol / 5.0]
            diagnostics[str(ordinal)] = {
                "status": "fallback",
                "reason": "insufficient_strict_reference",
                "shift": 0.0,
                "sigma": itol / 5.0,
                "sample_count": len(samples),
                "source": "bounded_original_itol",
            }
        else:
            previous = calibration[ordinal - 1]
            prior = calibration.get(ordinal - 2, [0.0, previous[1]])
            calibration[ordinal] = [
                previous[0] + previous[0] - prior[0],
                previous[1] * previous[1] / prior[1],
            ]
            diagnostics[str(ordinal)] = {
                "status": "extrapolated",
                "reason": "insufficient_strict_reference",
                "shift": calibration[ordinal][0],
                "sigma": calibration[ordinal][1],
                "sample_count": len(samples),
                "source": "lower_isotope_reference",
            }
    for ordinal in range(4, 10):
        previous = calibration[ordinal - 1]
        prior = calibration[ordinal - 2]
        calibration[ordinal] = [
            previous[0] + previous[0] - prior[0],
            previous[1] * previous[1] / prior[1],
        ]
        diagnostics[str(ordinal)] = {
            "status": "extrapolated",
            "reason": "higher_isotope_reference",
            "shift": calibration[ordinal][0],
            "sigma": calibration[ordinal][1],
            "sample_count": len(values[ordinal]),
            "source": "lower_isotope_reference",
        }
    return calibration, diagnostics

def _final_residual_calibration_deviation(candidate, calibration):
    """Return the largest calibrated isotope error for a residual candidate."""

    isotopes = candidate.get("isotopes", ())
    if not isotopes:
        return float("inf")
    maximum = 0.0
    for isotope in isotopes:
        reference = calibration.get(int(isotope["isotope_number"]))
        if reference is None:
            return float("inf")
        shift, sigma = (float(reference[0]), float(reference[1]))
        if not np.isfinite(sigma) or sigma <= 0:
            return float("inf")
        deviation = abs(float(isotope["mass_diff_ppm"]) - shift) / sigma
        maximum = max(maximum, deviation)
    return maximum

def _detect_final_residual_strict(
    residual_store,
    *,
    strict_contexts,
    next_feature_id,
    args,
):
    """Run the strict detector plus a residual-only calibration guard."""

    supported, reason = _final_residual_detector_supported(
        strict_contexts, args
    )
    if not supported:
        return {
            "status": "not_run",
            "reason": reason,
            "contexts": (),
            "next_feature_id": int(next_feature_id),
        }
    spectra = residual_store.detector_spectra(
        min_intensity=float(args["mini"]),
        min_mz=float(args["minmz"]),
        max_mz=float(args["maxmz"]),
    )
    if not spectra:
        return {
            "status": "completed",
            "reason": "no_residual_points_above_strict_threshold",
            "contexts": (),
            "next_feature_id": int(next_feature_id),
        }

    detector_args = dict(args)
    detector_args["use_hill_calib"] = False
    detector_args["_parallel_final_residual_candidates"] = True
    detector_args["_feature_detection_phase"] = "final_residual"
    contexts = []
    next_hill_id = 1
    md_correction_int = {"Orbi": 1, "Tof": 2, "Icr": 3}.get(
        detector_args.get("md_correction"), 1
    )
    groups = group_spectra_by_faims(spectra)
    calibration_diagnostics = {}
    calibration_boundary_guard = {
        "status": "applied",
        "reason": "reject_sparse_residual_calibration_boundary",
        "maximum_standard_deviation": FINAL_RESIDUAL_CALIBRATION_MAX_SIGMA,
        "candidate_count_before_guard": 0,
        "candidate_count_after_guard": 0,
        "rejected_candidate_count": 0,
    }
    for faims_cv, group in groups:
        group_started = time.monotonic()
        calibration_started = time.monotonic()
        isotope_calibration, group_diagnostics = (
            _strict_reference_isotope_calibration(
                strict_contexts, faims_cv, detector_args
            )
        )
        calibration_seconds = time.monotonic() - calibration_started
        detector_args["_isotope_calibration_override"] = isotope_calibration
        calibration_diagnostics[
            "null" if faims_cv is None else str(faims_cv)
        ] = group_diagnostics
        rt_by_local = {
            local: spectrum["rt_sec"]
            for local, spectrum in enumerate(group)
        }
        max_mz = max(
            float(np.max(spectrum["m/z array"])) for spectrum in group
        )
        mz_step = float(detector_args["htol"]) * 1e-6 * max_mz
        hills_started = time.monotonic()
        hills, _mass_diff = detect_hills(
            group,
            detector_args,
            mz_step,
            0.0,
            md_correction_int=md_correction_int,
        )
        hills_seconds = time.monotonic() - hills_started
        if not len(hills.get("hills_idx_array", ())):
            logger.debug(
                "Hybrid final residual detector group complete: faims=%s "
                "calibration_sec=%.3f detect_hills_sec=%.3f runtime_sec=%.3f "
                "reason=no_hills",
                faims_cv, calibration_seconds, hills_seconds,
                time.monotonic() - group_started,
            )
            continue
        feature_started = time.monotonic()
        hills = split_peaks_multi(hills, group, detector_args)
        if "hills_idx_array_unique" in hills and not len(
            hills["hills_idx_array_unique"]
        ):
            logger.debug(
                "Hybrid final residual detector group complete: faims=%s "
                "calibration_sec=%.3f detect_hills_sec=%.3f runtime_sec=%.3f "
                "reason=no_split_hills",
                faims_cv, calibration_seconds, hills_seconds,
                time.monotonic() - group_started,
            )
            continue
        hills = process_hills(
            hills, group, mz_step, 0.0, detector_args
        )
        if not len(hills.get("hills_idx_array_unique", ())):
            logger.debug(
                "Hybrid final residual detector group complete: faims=%s "
                "calibration_sec=%.3f detect_hills_sec=%.3f process_sec=%.3f "
                "runtime_sec=%.3f reason=no_processed_hills",
                faims_cv, calibration_seconds, hills_seconds,
                time.monotonic() - feature_started,
                time.monotonic() - group_started,
            )
            continue
        next_hill_id = assign_deterministic_hill_ids(
            hills, next_hill_id
        )
        _ready, _hill_map, next_feature_id, candidates = (
            process_features_iteration(
                hills,
                faims_cv,
                mz_step,
                0.0,
                rt_by_local,
                0,
                False,
                detector_args,
                next_feature_idx=int(next_feature_id),
                data_for_analyse_tmp=group,
            )
        )
        calibration_boundary_guard["candidate_count_before_guard"] += len(
            candidates
        )
        candidates = [
            candidate
            for candidate in candidates
            if _final_residual_calibration_deviation(
                candidate, isotope_calibration
            )
            <= FINAL_RESIDUAL_CALIBRATION_MAX_SIGMA
        ]
        calibration_boundary_guard["candidate_count_after_guard"] += len(
            candidates
        )
        if candidates:
            contexts.append(
                {
                    "hills": hills,
                    "rt_by_local": rt_by_local,
                    "spectra": group,
                    "faims_cv": faims_cv,
                    "candidates": candidates,
                    "paseftol": 0.0,
                }
            )
        logger.debug(
            "Hybrid final residual detector group complete: faims=%s "
            "calibration_sec=%.3f detect_hills_sec=%.3f process_features_sec=%.3f "
            "candidates=%d runtime_sec=%.3f",
            faims_cv,
            calibration_seconds,
            hills_seconds,
            time.monotonic() - feature_started,
            len(candidates),
            time.monotonic() - group_started,
        )
    calibration_boundary_guard["rejected_candidate_count"] = (
        calibration_boundary_guard["candidate_count_before_guard"]
        - calibration_boundary_guard["candidate_count_after_guard"]
    )
    return {
        "status": "completed",
        "reason": "strict_thresholds_unchanged",
        "contexts": tuple(contexts),
        "next_feature_id": int(next_feature_id),
        "isotope_calibration_reference": calibration_diagnostics,
        "calibration_boundary_guard": calibration_boundary_guard,
    }

def _generate_initial_isotope_candidates(
    hills_dict, faims_val, mz_step, paseftol, args
):
    isotopes_mass_accuracy = args['itol']
    min_charge = args['cmin']
    max_charge = args['cmax']
    ivf = args['ivf']
    isotopes_list = list(range(10))
    averagine_mass = 111.1254
    averagine_C = 4.9384
    a = dict()

    for i in range(0, 20000 * max_charge, 100):
        int_arr = binom.pmf(
            isotopes_list,
            round(float(i) / averagine_mass * averagine_C),
            0.0107
        )
        max_pos = np.argmax(int_arr)
        int_arr_norm = int_arr / int_arr.sum()
        a[i] = (int_arr_norm, max_pos)

    n_procs = args['nprocs']

    md_correction = args['md_correction']
    if md_correction == 'Orbi':
        md_correction_int = 1
    elif md_correction == 'Tof':
        md_correction_int = 2
    elif md_correction == 'Icr':
        md_correction_int = 3
    else:
        logger.warning('md_correction parameter MUST BE Orbi,Tof or ICR. Using Orbi now')
        md_correction_int = 1

    ready = []
    sorted_idx_full = [
        idx_1
        for (idx_1, _hill_idx), _hill_mz in sorted(
            zip(enumerate(hills_dict['hills_idx_array_unique']), hills_dict['hills_mz_median']),
            key=lambda value: value[-1],
        )
    ]
    algorithm_faims_value = 0.0 if faims_val is None else faims_val
    common_args = (
        hills_dict,
        isotopes_mass_accuracy,
        isotopes_list,
        a,
        min_charge,
        max_charge,
        mz_step,
        paseftol,
        algorithm_faims_value,
        ivf,
    )
    def generate(indices):
        generated = []
        ranges = balanced_ranges(len(indices), n_procs)
        if len(ranges) == 1:
            start, end = ranges[0]
            generated.extend(get_initial_isotopes(
                *common_args, list(indices[start:end]), md_correction_int
            ))
        elif ranges:
            task_args = [
                common_args + (list(indices[start:end]), md_correction_int)
                for start, end in ranges
            ]
            for worker_result in run_process_tasks(get_initial_isotopes, task_args):
                generated.extend(worker_result)
        return generated

    ready = generate(sorted_idx_full)

    for candidate in ready:
        candidate['FAIMS'] = faims_val
    logger.info('Number of potential isotope clusters: %d', len(ready))
    return ready

def _calibrate_and_filter_isotope_candidates(ready, faims_val, args, rejected_sink=None):
    isotope_calibration_override = args.get(
        '_isotope_calibration_override'
    )
    if isotope_calibration_override is not None:
        isotopes_mass_error_map = {
            int(ordinal): [float(value[0]), float(value[1])]
            for ordinal, value in isotope_calibration_override.items()
        }
        isotope_calibration_status = 'strict_reference'
    elif args['ignore_iso_calib']:
        isotopes_mass_error_map = {}
        for ic in range(1, 10, 1):
            isotopes_mass_error_map[ic] = [0, args['itol']]
        isotope_calibration_status = 'not_applied'
    else:

        isotopes_mass_error_map = {}
        for ic in range(1, 10, 1):
            isotopes_mass_error_map[ic] = []

        for i in range(9):
            tmp = []
            for pf in ready:
                isotopes = pf['isotopes']
                scans = pf['nScans']
                if len(isotopes) >= i + 1 and scans >= 3:
                    tmp.append(isotopes[i]['mass_diff_ppm'])
            isotopes_mass_error_map[i+1] = tmp

        for ic in range(1, 10, 1):
            if ic <= 3:

                if len(isotopes_mass_error_map[ic]) >= 1000:

                    calibration = fit_mass_calibration(
                        isotopes_mass_error_map[ic], bin_width=0.05
                    )
                    if calibration.status == 'applied':
                        isotopes_mass_error_map[ic] = [
                            calibration.shift,
                            calibration.sigma,
                        ]
                    else:
                        logger.warning(
                            'Isotope %d calibration not applied: %s',
                            ic,
                            calibration.reason,
                        )
                        isotopes_mass_error_map[ic] = [0, args['itol']]

                else:
                    if ic -1 in isotopes_mass_error_map:
                        isotopes_mass_error_map[ic] = deepcopy(isotopes_mass_error_map[ic-1])
                        isotopes_mass_error_map[ic][0] += isotopes_mass_error_map[ic-1][0] - isotopes_mass_error_map.get(ic-2, [0, ])[0]
                        isotopes_mass_error_map[ic][1] *= isotopes_mass_error_map[ic-1][1] / isotopes_mass_error_map.get(ic-2, isotopes_mass_error_map[ic-1])[1]

                    else:
                        isotopes_mass_error_map[ic] = [0, 10]

            else:
                isotopes_mass_error_map[ic] = deepcopy(isotopes_mass_error_map[ic-1])
                isotopes_mass_error_map[ic][0] += isotopes_mass_error_map[ic-1][0] - isotopes_mass_error_map.get(ic-2, [0, ])[0]
                isotopes_mass_error_map[ic][1] *= isotopes_mass_error_map[ic-1][1] / isotopes_mass_error_map.get(ic-2, isotopes_mass_error_map[ic-1])[1]
        isotope_calibration_status = 'applied_or_fallback'

    args.setdefault('isotope_calibration', {})[
        'null' if faims_val is None else str(faims_val)
    ] = {
        str(ordinal): {
            'shift': float(values[0]),
            'sigma': float(values[1]),
            'status': isotope_calibration_status,
        }
        for ordinal, values in isotopes_mass_error_map.items()
    }
    logger.info('Average mass shift between monoisotopic and first 13C isotope: %.3f ppm', isotopes_mass_error_map[1][0])
    logger.info('Average mass std between monoisotopic and first 13C isotope: %.3f ppm', isotopes_mass_error_map[1][1])

    logger.debug(isotopes_mass_error_map)
    workers = int(args.get('nprocs', 1))
    if (
        args.get('_parallel_final_residual_candidates', False)
        and rejected_sink is None
        and workers > 1
        and len(ready) > 1
    ):
        filtered = []
        for accepted, _rejected in run_process_tasks(
            _filter_isotope_candidate_batch,
            [
                (ready[start:end], isotopes_mass_error_map)
                for start, end in balanced_ranges(len(ready), workers)
            ],
        ):
            filtered.extend(accepted)
        ready[:] = filtered
    else:
        _filter_isotope_candidates_in_place(
            ready, isotopes_mass_error_map, rejected_sink
        )
    logger.info('Number of potential isotope clusters after smart mass accuracy for isotopes: %d', len(ready))
    return ready


def _filter_isotope_candidate_batch(ready, isotopes_mass_error_map):
    """Filter a candidate slice after the parent has fixed calibration."""

    accepted = []
    rejected = []
    for pep_feature in ready:
        if _filter_one_isotope_candidate(pep_feature, isotopes_mass_error_map):
            accepted.append(pep_feature)
        else:
            rejected.append(pep_feature)
    return accepted, rejected


def _filter_one_isotope_candidate(pep_feature, isotopes_mass_error_map):
    tmp = []
    for cand in pep_feature['isotopes']:
        map_val = isotopes_mass_error_map[cand['isotope_number']]
        if abs(cand['mass_diff_ppm'] - map_val[0]) <= 5 * map_val[1]:
            tmp.append(cand)
        else:
            break
    tmp_n_isotopes = len(tmp)
    if not tmp_n_isotopes:
        return False
    all_theoretical_int, all_exp_intensity = pep_feature['intensity_array_for_cos_corr']
    all_theoretical_int = all_theoretical_int[:tmp_n_isotopes+1]
    all_exp_intensity = all_exp_intensity[:tmp_n_isotopes+1]
    cos_corr, _number_of_passed_isotopes = checking_cos_correlation_for_carbon(
        all_theoretical_int, all_exp_intensity, 0.6
    )
    if not cos_corr:
        return False
    pep_feature['cos_cor_isotopes'] = cos_corr
    pep_feature['isotopes'] = tmp
    pep_feature['nIsotopes'] = tmp_n_isotopes + 1
    pep_feature['intensity_array_for_cos_corr'] = [
        all_theoretical_int, all_exp_intensity
    ]
    return True


def _filter_isotope_candidates_in_place(
    ready, isotopes_mass_error_map, rejected_sink=None
):
    """Preserve legacy in-place filtering and reject ordering."""

    max_l = len(ready)
    cur_l = 0
    while cur_l < max_l:
        pep_feature = ready[cur_l]
        accepted = _filter_one_isotope_candidate(
            pep_feature, isotopes_mass_error_map
        )
        if not accepted:
            if rejected_sink is not None:
                rejected_sink.append(pep_feature)
            del ready[cur_l]
            max_l -= 1
            cur_l -= 1
        cur_l += 1

def _capture_direct_competitors(
    ready, hills_dict, RT_dict, data_for_analyse_tmp, args, direct_assays,
    direct_events_by_id, direct_competitor_sink,
):
    captured_direct_competitors = ()
    if direct_competitor_sink is not None and direct_assays:
        captured_direct_competitors = (
            capture_direct_processed_hill_competitors(
                direct_assays,
                ready,
                hills_dict,
                RT_dict,
                data_for_analyse_tmp,
                direct_events_by_id or {},
                ppm=float(args['itol']),
                rt_tolerance_sec=float(
                    args.get('ms2_rt_tolerance_sec', 120.0)
                ),
                # Accepted representations are removed after the greedy pass;
                # retain a small oversampled set so they cannot consume all
                # final losing-competitor slots.
                top_k=6,
            )
        )
    return captured_direct_competitors

def _record_losing_direct_competitors(
    ready_final, captured_direct_competitors, direct_competitor_sink
):
    if direct_competitor_sink is not None:
        accepted_candidate_keys = {
            _candidate_hill_ids(candidate) + (int(candidate['charge']),)
            for candidate in ready_final
        }
        losing_competitors = [
            competitor
            for competitor in captured_direct_competitors
            if tuple(
                sorted(
                    [
                        int(
                            competitor.candidate[
                                'monoisotope hill idx'
                            ]
                        )
                    ]
                    + [
                        int(value['isotope_hill_idx'])
                        for value in competitor.candidate['isotopes']
                    ]
                )
            )
            + (int(competitor.candidate['charge']),)
            not in accepted_candidate_keys
        ]
        losing_per_psm = {}
        for competitor in losing_competitors:
            psm_id = str(competitor.psm_id)
            count = losing_per_psm.get(psm_id, 0)
            if count >= 3:
                continue
            direct_competitor_sink.append(competitor)
            losing_per_psm[psm_id] = count + 1

def _assign_feature_indices_and_write(
    ready_final, hills_dict, faims_val, RT_dict, data_start_id, write_header,
    args, next_feature_idx, data_for_analyse_tmp,
):
    hill_to_feature_idx = {}
    for offset, pep_feature in enumerate(ready_final):
        feature_idx = next_feature_idx + offset
        pep_feature['feature_idx'] = feature_idx
        hill_to_feature_idx[pep_feature['monoisotope hill idx']] = feature_idx
        for cand in pep_feature['isotopes']:
            hill_to_feature_idx[cand['isotope_hill_idx']] = feature_idx
    next_feature_idx += len(ready_final)

    # Hybrid retains strict candidates/hills until targeted and residual
    # conflict decisions are complete.  Final strict rows are emitted by
    # run_hybrid_postprocessing; legacy streaming is unchanged.
    if args.get('feature_mode') != 'hybrid':
        negative_mode = args['nm']
        isotopes_for_intensity = args['iuse']
        peptide_features = utils.calc_peptide_features(
            hills_dict,
            ready_final,
            negative_mode,
            faims_val,
            RT_dict,
            data_start_id,
            isotopes_for_intensity,
            include_mono_hills=not args.get('no_mono_hills', False),
            quantification_args=args,
            spectra=data_for_analyse_tmp,
        )

        utils.write_output(peptide_features, args, write_header)
    return hill_to_feature_idx, next_feature_idx

def process_features_iteration(hills_dict, faims_val, mz_step, paseftol, RT_dict, data_start_id, write_header, args, next_feature_idx=1, data_for_analyse_tmp=None, direct_assays=(), direct_events_by_id=None, direct_competitor_sink=None):
    if not len(hills_dict.get('hills_idx_array_unique', ())):
        utils.write_output([], args, write_header)
        return set(), {}, next_feature_idx, []

    candidate_started = time.monotonic()
    ready = _generate_initial_isotope_candidates(
        hills_dict, faims_val, mz_step, paseftol, args
    )
    collect_weak = args.get('feature_mode') == 'hybrid' and args.get('external_id')
    initial_candidate_count = len(ready)
    smart_rejects = [] if collect_weak else None
    filter_started = time.monotonic()
    ready = _calibrate_and_filter_isotope_candidates(ready, faims_val, args, smart_rejects)
    smart_accepted_count = len(ready)
    captured_direct_competitors = _capture_direct_competitors(
        ready, hills_dict, RT_dict, data_for_analyse_tmp, args, direct_assays,
        direct_events_by_id, direct_competitor_sink,
    )
    greedy_rejects = [] if collect_weak else None
    selection_started = time.monotonic()
    if (
        args.get('_parallel_final_residual_candidates', False)
        and greedy_rejects is None
        and int(args.get('nprocs', 1)) > 1
        and len(ready) > 1
    ):
        ready_set, ready_final = _select_nonconflicting_isotope_candidates_parallel(
            ready, hills_dict, RT_dict, data_start_id, args['nprocs']
        )
    else:
        ready_set, ready_final = _select_nonconflicting_isotope_candidates(
            ready, hills_dict, RT_dict, data_start_id, greedy_rejects
        )
    selection_seconds = time.monotonic() - selection_started
    if collect_weak:
        publish_detector_outcomes(
            hills_dict, smart_rejects, greedy_rejects, initial_count=initial_candidate_count,
            smart_accepted_count=smart_accepted_count, strict_selected_count=len(ready_final)
        )
    _record_losing_direct_competitors(
        ready_final, captured_direct_competitors, direct_competitor_sink
    )
    hill_to_feature_idx, next_feature_idx = _assign_feature_indices_and_write(
        ready_final, hills_dict, faims_val, RT_dict, data_start_id, write_header,
        args, next_feature_idx, data_for_analyse_tmp,
    )
    logger.debug(
        'Feature candidate timing: phase=%s faims=%s generated=%d selected=%d '
        'generation_sec=%.3f filter_sec=%.3f selection_sec=%.3f write_sec=%.3f',
        args.get('_feature_detection_phase', 'strict'), faims_val,
        initial_candidate_count, len(ready_final),
        filter_started - candidate_started, selection_started - filter_started,
        selection_seconds, time.monotonic() - selection_started - selection_seconds,
    )

    return ready_set, hill_to_feature_idx, next_feature_idx, ready_final

def process_file(args):

    input_file_path = args['file']
    process_started = _debug_stage_start(
        'process_file',
        input=input_file_path,
        feature_mode=args.get('feature_mode'),
        output=args.get('o'),
    )
    logger.debug(
        'Processing input: path=%s mode=%s format=%s workers=%s output=%s',
        input_file_path,
        args.get('feature_mode'),
        args.get('format'),
        args.get('nprocs'),
        args.get('o'),
    )
    # Some detector settings are resolved from the data below (for example,
    # paseftol becomes zero when ion mobility is absent, and hill calibration
    # may tighten htol).  Cache validity must nevertheless be keyed by the
    # user-supplied upstream configuration so that an identical command can
    # reuse what it created.  Keep this snapshot separate from the mutable
    # processing dictionary.
    strict_stage_cache_args = dict(args)
    stop_after_hills = bool(args.get('stop_after_hills'))
    stop_after_logged = False
    next_feature_idx = 1
    next_hill_idx = 1
    identification_result = None
    if (
        args.get('feature_mode') == 'hybrid'
        and args.get('direct_id', True)
        and args.get('psm_path')
    ):
        identification_started = _debug_stage_start(
            'read_identification_table', path=args['psm_path']
        )
        identification_result = read_identification_table(
            args['psm_path'],
            q_value_max=float(args.get('psm_q_value_max', 0.01)),
            pep_max=args.get('psm_pep_max'),
            column_map=psm_column_map_from_args(args),
            run_id=input_stem(input_file_path),
        )
        args['_identification_parser_qc'] = identification_result.qc.to_dict()
        _debug_stage_complete(
            'read_identification_table',
            identification_started,
            qc=args['_identification_parser_qc'],
        )
        logger.debug('Parsed direct-identification input: qc=%s', args['_identification_parser_qc'])

    md_correction = args['md_correction']
    if md_correction == 'Orbi':
        md_correction_int = 1
    elif md_correction == 'Tof':
        md_correction_int = 2
    elif md_correction == 'Icr':
        md_correction_int = 3
    else:
        logger.warning('md_correction parameter MUST BE Orbi,Tof or ICR. Using Orbi now')
        md_correction_int = 1

    if input_file_path.lower().endswith('.mzml') or input_file_path.lower().endswith('.mzml.gz'):
        return _process_mzml_file(
            args, input_file_path, strict_stage_cache_args, identification_result,
            md_correction_int, process_started,
        )
    if (
        input_file_path.lower().endswith('.hills.tsv')
        or input_file_path.lower().endswith('.hills.parquet')
        or input_file_path.lower().endswith('.hills.npz')
    ):
        return _process_hills_file(
            args, input_file_path, stop_after_hills, stop_after_logged,
            next_feature_idx, next_hill_idx, process_started,
        )

def _prepare_mzml_processing(
    args, input_file_path, strict_stage_cache_args, identification_result
):
    strict_stage_payload = None
    strict_stage_manifest = None
    strict_stage_cache = args.get('hybrid_stage_cache_dir')
    if strict_stage_cache and Path(strict_stage_cache).exists() and not Path(
        strict_stage_cache
    ).is_dir():
        raise ValueError(
            'strict stage cache path exists but is not a directory: %s'
            % strict_stage_cache
        )
    invalidate_stale_strict_stage_cache(strict_stage_cache, input_file_path, strict_stage_cache_args)
    if strict_stage_cache and Path(strict_stage_cache).is_dir():
        logger.debug('Checking strict-stage cache: %s', strict_stage_cache)
        strict_cache_started = _debug_stage_start(
            'load_strict_stage_cache', path=strict_stage_cache
        )
        strict_stage_payload, strict_stage_manifest = load_strict_stage_cache(
            strict_stage_cache, input_file_path, strict_stage_cache_args
        )
        raw_store = load_raw_ms1_cache(
            args['raw_ms1_cache_dir'], input_file_path, mmap=True
        )
        _debug_stage_complete(
            'load_strict_stage_cache',
            strict_cache_started,
            strict_feature_count=strict_stage_manifest['strict_feature_count'],
        )
        ingestion = MzMLIngestion(
            [],
            strict_stage_payload['ms1_rows'],
            strict_stage_payload['ms2_rows'],
            strict_stage_payload['ms1_metadata'],
            raw_store,
        )
        logger.info(
            'Reused strict-stage cache %s: %d strict features in %d contexts',
            strict_stage_cache,
            strict_stage_manifest['strict_feature_count'],
            strict_stage_manifest['context_count'],
        )
        args['_strict_stage_cache'] = {
            'status': 'reused',
            'path': str(Path(strict_stage_cache).resolve()),
            'payload_bytes': strict_stage_manifest['payload_bytes'],
            'strict_feature_count': strict_stage_manifest[
                'strict_feature_count'
            ],
        }
    else:
        logger.debug('Ingesting mzML: raw_cache=%s', args.get('raw_ms1_cache_dir'))
        ingestion_started = _debug_stage_start(
            'ingest_mzml', raw_cache=args.get('raw_ms1_cache_dir')
        )
        ingestion = ingest_mzml(args)
        _debug_stage_complete(
            'ingest_mzml',
            ingestion_started,
            spectra=len(ingestion.spectra),
            ms1_rows=len(ingestion.ms1_rows),
            ms2_rows=len(ingestion.ms2_rows),
        )
    logger.debug(
        'Ingestion complete: spectra=%d ms1_rows=%d ms2_rows=%d',
        len(ingestion.spectra),
        len(ingestion.ms1_rows),
        len(ingestion.ms2_rows),
    )
    if args.get('write_ms1', False):
        utils.write_ms1_output(ingestion.ms1_rows, args)
    if args.get('write_ms2', False):
        utils.write_ms2_output(ingestion.ms2_rows, args)
    assay_result = AssayBuildResult((), (), {})
    if args.get('feature_mode') == 'hybrid' and identification_result is not None:
        direct_assay_started = _debug_stage_start(
            'build_direct_assays', identification_count=len(identification_result.records)
        )
        mapping = map_identifications_to_ms2(
            identification_result.records,
            ingestion.ms2_rows,
            run_id=input_stem(input_file_path),
            max_unmapped_fraction=float(args.get('max_unmapped_psm_fraction', 0.05)),
        )
        assay_result = build_direct_assays(
            mapping,
            run_id=input_stem(input_file_path),
            fixed_modifications=tuple(args.get('fixed_mod', ())),
            precursor_ppm=float(args.get('direct_id_precursor_ppm', 5.0)),
        )
        args['_identification_mapping_summary'] = {
            'mapped_count': mapping.mapped_count,
            'unmapped_count': mapping.unmapped_count,
            'status_counts': mapping.status_counts,
        }
        args['_direct_assay_summary'] = assay_result.status_counts
        logger.debug(
            'Direct-assay build complete: mapping=%s assays=%d audit_rows=%d statuses=%s',
            args['_identification_mapping_summary'],
            len(assay_result.assays),
            len(assay_result.audit),
            assay_result.status_counts,
        )
        _debug_stage_complete(
            'build_direct_assays',
            direct_assay_started,
            assays=len(assay_result.assays),
            audit_rows=len(assay_result.audit),
        )
    direct_events_by_id = {
        int(event['ms2_event_id']): event
        for event in ingestion.ms2_rows
    }

    return ingestion, assay_result, direct_events_by_id, strict_stage_payload, strict_stage_cache

def _run_cached_mzml_hybrid(
    args, input_file_path, process_started, ingestion, assay_result,
    strict_stage_payload,
):
    if strict_stage_payload is not None:
        strict_contexts = list(strict_stage_payload['strict_contexts'])
        manager = args.get('_output_manager')
        if manager is None:
            raise RuntimeError('Output manager is required.')
        hybrid_started = _debug_stage_start(
            'hybrid_postprocessing_from_cache', contexts=len(strict_contexts)
        )
        run_hybrid_postprocessing(
            run_id=input_stem(input_file_path),
            ingestion=ingestion,
            assay_result=assay_result,
            strict_contexts=strict_contexts,
            manager=manager,
            next_feature_id=int(
                strict_stage_payload['next_feature_id']
            ),
            args=args,
            final_strict_detector=_detect_final_residual_strict,
        )
        _debug_stage_complete(
            'hybrid_postprocessing_from_cache', hybrid_started
        )
        _debug_stage_complete('process_file', process_started)
        return

def _process_mzml_faims_contexts(
    args, ingestion, assay_result, direct_events_by_id, md_correction_int,
    next_feature_idx, next_hill_idx, write_header, stop_after_hills,
    stop_after_logged,
):
    data_for_analyse = ingestion.spectra
    #Process faims

    faims_groups = group_spectra_by_faims(data_for_analyse)
    if len(faims_groups) > 1 or faims_groups[0][0] is not None:
        logger.info('Detected FAIMS values: %s', [value for value, _ in faims_groups])

    data_start_id = 0
    strict_contexts = []

    for faims_val, data_for_analyse_tmp in faims_groups:

        context_started = _debug_stage_start(
            'faims_context', faims=faims_val, spectra=len(data_for_analyse_tmp)
        )

        if len(faims_groups) > 1:
            logger.info('Spectra analysis for CV = %s', faims_val)

        RT_dict = {
            local_index: spectrum['rt_sec']
            for local_index, spectrum in enumerate(data_for_analyse_tmp)
        }

        hill_mass_accuracy = args['htol']
        max_mz_value = 0
        for z in data_for_analyse_tmp:
            max_mz_value = max(max_mz_value, z['m/z array'].max())

        mz_step = hill_mass_accuracy * 1e-6 * max_mz_value
        logger.debug(
            'Context start: faims=%s spectra=%d max_mz=%.6f mz_step=%.9f paseftol=%s',
            faims_val,
            len(data_for_analyse_tmp),
            max_mz_value,
            mz_step,
            args['paseftol'],
        )

        #Process TOF
        if args['tof']:
            data_for_analyse_tmp = process_tof(data_for_analyse_tmp)

        #Process profile
        if args['profile']:
            data_for_analyse_tmp = process_profile(data_for_analyse_tmp)

        #Process ion mobility

        if all('ignore_ion_mobility' not in z for z in data_for_analyse_tmp):
            centroid_pasef_data(data_for_analyse_tmp, args, mz_step)
        else:
            args['paseftol'] = 0

        paseftol = args['paseftol']

        if args['use_hill_calib']:

            l_data = len(data_for_analyse_tmp)

            if l_data <= 1000:

                hills_dict, total_mass_diff = detect_hills(data_for_analyse_tmp, args, mz_step, paseftol, md_correction_int=md_correction_int)
            else:

                hills_dict, total_mass_diff = detect_hills(data_for_analyse_tmp[int(l_data/2)-500:int(l_data/2)+500], args, mz_step, paseftol, md_correction_int=md_correction_int)

            total_mass_diff = np.array(total_mass_diff)
            counter_hills_idx = Counter(hills_dict['hills_idx_array'])
            min_length_hill = args['minlh']

            tmp_hill_length = np.array([counter_hills_idx[hill_idx] for hill_idx in hills_dict['hills_idx_array']])
            idx_minl = tmp_hill_length >= min_length_hill
            total_mass_diff = total_mass_diff[idx_minl]

            calibration = fit_mass_calibration(total_mass_diff, bin_width=0.05)
            args['hill_calibration'] = calibration.to_dict()
            if calibration.status == 'applied':
                args['htol'] = min(args['htol'], 3 * calibration.sigma)
                logger.info(
                    'Automatically optimized htol parameter: %.3f ppm',
                    args['htol'],
                )
            else:
                logger.warning(
                    'Hill calibration not applied; keeping htol %.3f ppm: %s',
                    args['htol'],
                    calibration.reason,
                )

        hills_started = _debug_stage_start(
            'detect_hills', faims=faims_val, spectra=len(data_for_analyse_tmp)
        )
        hills_dict, total_mass_diff = detect_hills(data_for_analyse_tmp, args, mz_step, paseftol, md_correction_int=md_correction_int)
        _debug_stage_complete(
            'detect_hills',
            hills_started,
            hill_count=len(set(hills_dict['hills_idx_array'])),
        )

        logger.info('Detected number of hills before splitting: %d', len(set(hills_dict['hills_idx_array'])))

        hill_processing_started = _debug_stage_start(
            'split_and_process_hills', faims=faims_val
        )
        hills_dict = split_peaks_multi(hills_dict, data_for_analyse_tmp, args)
        logger.info('Starting hills processing')
        hills_dict = process_hills(hills_dict, data_for_analyse_tmp, mz_step, paseftol, args)
        next_hill_idx = assign_deterministic_hill_ids(
            hills_dict, next_hill_idx
        )
        _debug_stage_complete(
            'split_and_process_hills',
            hill_processing_started,
            hill_count=len(set(hills_dict['hills_idx_array'])),
        )

        logger.info('Detected number of hills: %d', len(set(hills_dict['hills_idx_array'])))
        if stop_after_hills:
            if args['write_hills']:
                hills_features = utils.iter_hills_extra(
                    hills_dict,
                    RT_dict,
                    faims_val,
                    data_start_id,
                    mz_step,
                    paseftol,
                    data_for_analyse_tmp=data_for_analyse_tmp,
                    include_point_lists=not args.get('no_hill_list', False),
                )
                utils.write_output(hills_features, args, write_header, hills=True)
                if not stop_after_logged:
                    logger.info('--stop_after_hills flag set, skipping feature detection after writing hills.')
                    stop_after_logged = True
            else:
                if not stop_after_logged:
                    logger.warning('--stop_after_hills flag set but hills output is disabled; skipping feature detection anyway.')
                    stop_after_logged = True
            write_header = False
            _debug_stage_complete(
                'faims_context', context_started, faims=faims_val,
                stopped_after_hills=True,
            )
            continue

        direct_competitors = []
        feature_detection_started = _debug_stage_start(
            'process_features', faims=faims_val, hill_count=len(set(hills_dict['hills_idx_array']))
        )
        _, hill_to_feature_idx, next_feature_idx, ready_final = process_features_iteration(
            hills_dict,
            faims_val,
            mz_step,
            paseftol,
            RT_dict,
            data_start_id,
            write_header,
            args,
            next_feature_idx=next_feature_idx,
            data_for_analyse_tmp=data_for_analyse_tmp,
            direct_assays=assay_result.assays,
            direct_events_by_id=direct_events_by_id,
            direct_competitor_sink=direct_competitors,
        )
        logger.debug(
            'Context feature detection complete: faims=%s strict_features=%d direct_competitors=%d next_feature_id=%d',
            faims_val,
            len(ready_final),
            len(direct_competitors),
            next_feature_idx,
        )
        _debug_stage_complete(
            'process_features',
            feature_detection_started,
            strict_features=len(ready_final),
            direct_competitors=len(direct_competitors),
        )
        if args.get('feature_mode') == 'hybrid':
            strict_contexts.append({
                'hills': hills_dict,
                'rt_by_local': RT_dict,
                'spectra': data_for_analyse_tmp,
                'faims_cv': faims_val,
                'candidates': ready_final,
                'direct_competitors': tuple(direct_competitors),
                'paseftol': paseftol,
            })
        if args['write_hills']:
            hills_features = utils.iter_hills_extra(
                hills_dict,
                RT_dict,
                faims_val,
                data_start_id,
                mz_step,
                paseftol,
                data_for_analyse_tmp=data_for_analyse_tmp,
                include_point_lists=not args.get('no_hill_list', False),
                feature_idx_by_hill=hill_to_feature_idx,
            )
            utils.write_output(hills_features, args, write_header, hills=True)

        write_header = False
        _debug_stage_complete(
            'faims_context', context_started, faims=faims_val
        )

    return strict_contexts, next_feature_idx, next_hill_idx, write_header, stop_after_logged

def _finalize_mzml_processing(
    args, input_file_path, strict_stage_cache_args, strict_stage_cache,
    ingestion, assay_result, strict_contexts, next_feature_idx, stop_after_hills,
):
    strict_cache_writer = None
    strict_cache_started = None
    if (
        args.get('feature_mode') == 'hybrid'
        and not stop_after_hills
        and strict_stage_cache
        and not Path(strict_stage_cache).exists()
    ):
        strict_cache_started = _debug_stage_start(
            'publish_strict_stage_cache', path=strict_stage_cache
        )
        strict_cache_writer = _start_strict_cache_writer(
            strict_stage_cache,
            input_file_path,
            strict_stage_cache_args,
            ingestion,
            strict_contexts,
            next_feature_idx,
            args,
        )

    if args.get('feature_mode') == 'hybrid' and not stop_after_hills:
        manager = args.get('_output_manager')
        if manager is None:
            raise RuntimeError('Output manager is required.')
        logger.debug(
            'Starting hybrid postprocessing: contexts=%d next_feature_id=%d candidate_cache=%s',
            len(strict_contexts),
            next_feature_idx,
            args.get('hybrid_candidate_cache_dir'),
        )
        hybrid_started = _debug_stage_start(
            'hybrid_postprocessing', contexts=len(strict_contexts)
        )
        try:
            next_feature_idx = run_hybrid_postprocessing(
                run_id=input_stem(input_file_path),
                ingestion=ingestion,
                assay_result=assay_result,
                strict_contexts=strict_contexts,
                manager=manager,
                next_feature_id=next_feature_idx,
                args=args,
                final_strict_detector=_detect_final_residual_strict,
            )
        except BaseException:
            if strict_cache_writer is not None:
                _cancel_strict_cache_writer(*strict_cache_writer)
            raise
        if strict_cache_writer is not None:
            cache_info = _finish_strict_cache_writer(*strict_cache_writer)
            logger.info(
                'Published strict-stage cache: %s (%d features, %d bytes)',
                cache_info['path'],
                cache_info['strict_feature_count'],
                cache_info['payload_bytes'],
            )
            args['_strict_stage_cache'] = {
                'status': 'created',
                **cache_info,
            }
            _debug_stage_complete(
                'publish_strict_stage_cache',
                strict_cache_started,
                strict_feature_count=cache_info['strict_feature_count'],
                payload_bytes=cache_info['payload_bytes'],
            )
        _debug_stage_complete(
            'hybrid_postprocessing', hybrid_started,
            next_feature_id=next_feature_idx,
        )

def _process_mzml_file(
    args, input_file_path, strict_stage_cache_args, identification_result,
    md_correction_int, process_started,
):
    (
        ingestion, assay_result, direct_events_by_id, strict_stage_payload,
        strict_stage_cache,
    ) = _prepare_mzml_processing(
        args, input_file_path, strict_stage_cache_args, identification_result
    )
    if strict_stage_payload is not None:
        return _run_cached_mzml_hybrid(
            args, input_file_path, process_started, ingestion, assay_result,
            strict_stage_payload,
        )
    (
        strict_contexts, next_feature_idx, _next_hill_idx, _write_header,
        _stop_after_logged,
    ) = _process_mzml_faims_contexts(
        args, ingestion, assay_result, direct_events_by_id, md_correction_int,
        1, 1, True, bool(args.get('stop_after_hills')), False,
    )
    _finalize_mzml_processing(
        args, input_file_path, strict_stage_cache_args, strict_stage_cache,
        ingestion, assay_result, strict_contexts, next_feature_idx,
        bool(args.get('stop_after_hills')),
    )
    _debug_stage_complete('process_file', process_started)

def _process_hills_file(
    args, input_file_path, stop_after_hills, stop_after_logged,
    next_feature_idx, next_hill_idx, process_started,
):
    hills_load_started = _debug_stage_start('load_hills_input', path=input_file_path)
    if args.get('write_ms1', False):
        logger.warning('--write_ms1 requires mzML input and is ignored for hills input: %s', input_file_path)
    if stop_after_hills and not stop_after_logged:
        logger.info('--stop_after_hills flag has no effect when reading hills input; proceeding with feature detection.')
        stop_after_logged = True
    if input_file_path.lower().endswith('.hills.tsv'):
        hills_features = pd.read_table(input_file_path)
    elif input_file_path.lower().endswith('.hills.parquet'):
        hills_features = pd.read_parquet(input_file_path, engine='pyarrow')
    else:
        hills_features = pd.DataFrame(utils.get_hills_features_from_hills_npz(input_file_path))
    hills_features = normalize_hills_dataframe(
        hills_features, args.get('input_rt_unit', 'seconds')
    )
    _debug_stage_complete(
        'load_hills_input', hills_load_started, hill_rows=len(hills_features)
    )
    RT_dict = False
    write_header = True
    data_start_id = 0

    faims_values = [
        None if pd.isna(value) else float(value)
        for value in hills_features['FAIMS']
    ]
    has_faims = any(value is not None for value in faims_values)
    if has_faims:
        paseftol = 0
    else:
        im_values = pd.to_numeric(hills_features['im'], errors='coerce')
        if im_values.notna().any() and np.any(im_values.fillna(0).values):
            paseftol = args['paseftol']
        else:
            paseftol = 0

    if paseftol == 0:
        faims_set = sorted(set(faims_values), key=faims_sort_key)
    else:
        faims_set = [None]

    if len(faims_set) > 1 or faims_set[0] is not None:
        logger.info('Detected FAIMS values: %s', faims_set)

    for faims_val in faims_set:

        context_started = _debug_stage_start(
            'hills_faims_context', faims=faims_val
        )

        if len(faims_set) > 1:
            logger.info('Spectra analysis for CV = %s', faims_val)

        if paseftol == 0:
            if faims_val is None:
                hills_features_local = hills_features[hills_features['FAIMS'].isna()]
            else:
                hills_features_local = hills_features[hills_features['FAIMS'] == faims_val]
        else:
            hills_features_local = hills_features

        hill_mass_accuracy = args['htol']
        hill_conversion_started = _debug_stage_start(
            'convert_hills_input', faims=faims_val,
            hill_rows=len(hills_features_local),
        )
        hills_dict, mz_step = utils.get_hills_dict_from_hills_features(hills_features_local, hill_mass_accuracy, paseftol)
        next_hill_idx = assign_deterministic_hill_ids(
            hills_dict, next_hill_idx
        )
        _debug_stage_complete(
            'convert_hills_input', hill_conversion_started,
            hill_count=len(set(hills_dict['hills_idx_array_unique'])),
        )

        logger.info('Detected number of hills: %d', len(set(hills_dict['hills_idx_array_unique'])))

        feature_detection_started = _debug_stage_start(
            'process_features', faims=faims_val,
            hill_count=len(set(hills_dict['hills_idx_array_unique'])),
        )
        _, _, next_feature_idx, ready_final = process_features_iteration(
            hills_dict,
            faims_val,
            mz_step,
            paseftol,
            RT_dict,
            data_start_id,
            write_header,
            args,
            next_feature_idx=next_feature_idx,
        )
        _debug_stage_complete(
            'process_features',
            feature_detection_started,
            strict_features=len(ready_final),
        )

        write_header = False
        _debug_stage_complete(
            'hills_faims_context', context_started, faims=faims_val
        )
    _debug_stage_complete('process_file', process_started)
