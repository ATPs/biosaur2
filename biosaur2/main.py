from . import utils
from collections import Counter
from copy import deepcopy
import logging

import numpy as np
import pandas as pd
from scipy.stats import binom

from .cutils import get_initial_isotopes, checking_cos_correlation_for_carbon, split_peaks, detect_hills, process_hills, get_and_calc_apex_intensity_and_scan
from .parallel import balanced_ranges, run_process_tasks
from .calibration import fit_mass_calibration
from .spectra import faims_sort_key, group_spectra_by_faims
from .hills import assign_deterministic_hill_ids, normalize_hills_dataframe
from .preprocessing import (
    centroid_pasef_data,
    ingest_mzml,
    process_profile,
    process_tof,
)
from .ms2_seed import (
    annotate_candidate_support,
    build_link_rows,
    candidate_bonus,
    partition_mono_indices,
    prepare_seed_context,
)

logger = logging.getLogger(__name__)

def _split_peaks_task(*args):
    return list(split_peaks(*args))


def _candidate_hill_ids(candidate):
    return tuple(
        sorted(
            [int(candidate['monoisotope hill idx'])]
            + [int(value['isotope_hill_idx']) for value in candidate['isotopes']]
        )
    )


def _candidate_conflict_key(candidate, hills_dict, seed_enabled=False):
    mono_index = int(candidate['monoisotope idx'])
    _, apex_intensity, _ = get_and_calc_apex_intensity_and_scan(
        hills_dict, mono_index
    )
    mass_errors = [abs(float(value['mass_diff_ppm'])) for value in candidate['isotopes']]
    mean_absolute_error = float(np.mean(mass_errors)) if mass_errors else float('inf')
    isotope_count = int(candidate['nIsotopes'])
    isotope_cosine = float(candidate['cos_cor_isotopes'])
    bonus = candidate_bonus(candidate) if seed_enabled else 0.0
    return (
        -(isotope_count + isotope_cosine + bonus),
        -isotope_count,
        -isotope_cosine,
        -float(apex_intensity),
        mean_absolute_error,
        float(candidate['hill_mz_1']),
        int(candidate['charge']),
        _candidate_hill_ids(candidate),
    )


def _final_feature_key(candidate, hills_dict, RT_dict, data_start_id):
    mono_index = int(candidate['monoisotope idx'])
    _, _, apex_scan = get_and_calc_apex_intensity_and_scan(hills_dict, mono_index)
    if RT_dict is False:
        apex_rt = float(hills_dict['rtApex'][mono_index])
    else:
        apex_rt = float(RT_dict[apex_scan + data_start_id])
    return (
        float(candidate['hill_mz_1']),
        apex_rt,
        int(candidate['charge']),
        int(candidate['monoisotope hill idx']),
        _candidate_hill_ids(candidate),
    )

def process_features_iteration(hills_dict, faims_val, mz_step, paseftol, RT_dict, data_start_id, write_header, args, next_feature_idx=1, data_for_analyse_tmp=None, seed_context=None):
    if not len(hills_dict.get('hills_idx_array_unique', ())):
        utils.write_output([], args, write_header)
        return set(), {}, next_feature_idx, []

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

    if seed_context is None:
        ready = generate(sorted_idx_full)
    else:
        seeded_indices, remaining_indices = partition_mono_indices(
            sorted_idx_full, seed_context
        )
        ready = generate(seeded_indices) + generate(remaining_indices)
        rank = {index: position for position, index in enumerate(sorted_idx_full)}
        ready.sort(key=lambda candidate: rank[int(candidate['monoisotope idx'])])
        unique_ready = []
        seen_candidate_keys = set()
        for candidate in ready:
            key = (
                int(candidate['monoisotope hill idx']), int(candidate['charge']),
                tuple((int(item['isotope_number']), int(item['isotope_hill_idx']))
                      for item in candidate['isotopes']),
            )
            if key not in seen_candidate_keys:
                seen_candidate_keys.add(key)
                unique_ready.append(candidate)
        ready = unique_ready

    for candidate in ready:
        candidate['FAIMS'] = faims_val


    logger.info('Number of potential isotope clusters: %d', len(ready))


    if args['ignore_iso_calib']:
        isotopes_mass_error_map = {}
        for ic in range(1, 10, 1):
            isotopes_mass_error_map[ic] = [0, args['itol']]
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

    args.setdefault('isotope_calibration', {})[
        'null' if faims_val is None else str(faims_val)
    ] = {
        str(ordinal): {
            'shift': float(values[0]),
            'sigma': float(values[1]),
            'status': 'not_applied' if args['ignore_iso_calib'] else 'applied_or_fallback',
        }
        for ordinal, values in isotopes_mass_error_map.items()
    }
    logger.info('Average mass shift between monoisotopic and first 13C isotope: %.3f ppm', isotopes_mass_error_map[1][0])
    logger.info('Average mass std between monoisotopic and first 13C isotope: %.3f ppm', isotopes_mass_error_map[1][1])

    logger.debug(isotopes_mass_error_map)

    max_l = len(ready)
    cur_l = 0

    while cur_l < max_l:
        pep_feature = ready[cur_l]

        tmp = []

        for cand in pep_feature['isotopes']:
            map_val = isotopes_mass_error_map[cand['isotope_number']]

            if abs(cand['mass_diff_ppm'] - map_val[0]) <= 5 * map_val[1]:
                tmp.append(cand)
            else:
                break

        tmp_n_isotopes = len(tmp)

        if tmp_n_isotopes:
            all_theoretical_int, all_exp_intensity = pep_feature['intensity_array_for_cos_corr']
            all_theoretical_int = all_theoretical_int[:tmp_n_isotopes+1]
            all_exp_intensity = all_exp_intensity[:tmp_n_isotopes+1]
            cos_corr, number_of_passed_isotopes = checking_cos_correlation_for_carbon(all_theoretical_int, all_exp_intensity, 0.6)
            if cos_corr:

                ready[cur_l]['cos_cor_isotopes'] = cos_corr
                ready[cur_l]['isotopes'] = tmp
                ready[cur_l]['nIsotopes'] = tmp_n_isotopes + 1
                ready[cur_l]['intensity_array_for_cos_corr'] = [all_theoretical_int, all_exp_intensity]


            else:
                del ready[cur_l]
                max_l -= 1
                cur_l -= 1


        else:
            del ready[cur_l]
            max_l -= 1
            cur_l -= 1

        cur_l += 1

    logger.info('Number of potential isotope clusters after smart mass accuracy for isotopes: %d', len(ready))

    max_l = len(ready)
    cur_l = 0

    if seed_context is not None:
        for candidate in ready:
            annotate_candidate_support(candidate, hills_dict, seed_context, args)
    func_for_sort = lambda candidate: _candidate_conflict_key(
        candidate, hills_dict, seed_enabled=seed_context is not None
    )

    ready_final = []
    ready_set = set()
    if not ready:
        logger.info('No isotope clusters remained after smart mass accuracy filtering.')
    else:
        ready = sorted(ready, key=func_for_sort)
        cur_isotopes = ready[0]['nIsotopes']

        while cur_l < max_l:
            pep_feature = ready[cur_l]
            n_iso = pep_feature['nIsotopes']
            if n_iso < cur_isotopes:
                ready = sorted(ready, key=func_for_sort)
                cur_isotopes = n_iso
                cur_l = 0
                pep_feature = ready[cur_l]

            if pep_feature['monoisotope hill idx'] not in ready_set:
                if not any(cand['isotope_hill_idx'] in ready_set for cand in pep_feature['isotopes']):
                    ready_final.append(pep_feature)
                    ready_set.add(pep_feature['monoisotope hill idx'])
                    for cand in pep_feature['isotopes']:
                        ready_set.add(cand['isotope_hill_idx'])
                    del ready[cur_l]
                    max_l -= 1
                    cur_l -= 1

                else:
                    tmp = []

                    for cand in pep_feature['isotopes']:
                        if cand['isotope_hill_idx'] not in ready_set:
                            tmp.append(cand)
                        else:
                            break

                    tmp_n_isotopes = len(tmp)

                    if tmp_n_isotopes:

                        all_theoretical_int, all_exp_intensity = pep_feature['intensity_array_for_cos_corr']
                        all_theoretical_int = all_theoretical_int[:tmp_n_isotopes+1]
                        all_exp_intensity = all_exp_intensity[:tmp_n_isotopes+1]
                        cos_corr, number_of_passed_isotopes = checking_cos_correlation_for_carbon(all_theoretical_int, all_exp_intensity, 0.6)
                        if cos_corr:
                            ready[cur_l]['cos_cor_isotopes'] = cos_corr
                            ready[cur_l]['isotopes'] = tmp
                            ready[cur_l]['nIsotopes'] = tmp_n_isotopes + 1
                            ready[cur_l]['intensity_array_for_cos_corr'] = [all_theoretical_int, all_exp_intensity]
                            if seed_context is not None:
                                annotate_candidate_support(
                                    ready[cur_l], hills_dict, seed_context, args
                                )
                            cur_l -= 1
                        else:
                            del ready[cur_l]
                            max_l -= 1
                            cur_l -= 1


                    else:
                        del ready[cur_l]
                        max_l -= 1
                        cur_l -= 1
            else:
                del ready[cur_l]
                max_l -= 1
                cur_l -= 1

            cur_l += 1

    logger.info('Number of detected isotope clusters: %d', len(ready_final))

    ready_final.sort(
        key=lambda candidate: _final_feature_key(
            candidate, hills_dict, RT_dict, data_start_id
        )
    )

    hill_to_feature_idx = {}
    for offset, pep_feature in enumerate(ready_final):
        feature_idx = next_feature_idx + offset
        pep_feature['feature_idx'] = feature_idx
        hill_to_feature_idx[pep_feature['monoisotope hill idx']] = feature_idx
        for cand in pep_feature['isotopes']:
            hill_to_feature_idx[cand['isotope_hill_idx']] = feature_idx
    next_feature_idx += len(ready_final)

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

    return ready_set, hill_to_feature_idx, next_feature_idx, ready_final

def split_peaks_multi(hills_dict, data_for_analyse_tmp, args):
    min_length_hill = args['minlh']

    hills_dict['orig_idx_array'] = np.array(hills_dict['orig_idx_array'])
    hills_dict['scan_idx_array'] = np.array(hills_dict['scan_idx_array'])
    hills_dict['hills_idx_array'] = np.array(hills_dict['hills_idx_array'])

    counter_hills_idx = Counter(hills_dict['hills_idx_array'])
    counter_hills_idx_2 = dict()
    for k,v in counter_hills_idx.items():
        counter_hills_idx_2[k] = v
    counter_hills_idx = counter_hills_idx_2

    tmp_hill_length = np.array([counter_hills_idx[hill_idx] for hill_idx in hills_dict['hills_idx_array']])
    idx_minl = tmp_hill_length >= min_length_hill
    hills_dict['hills_idx_array'] = hills_dict['hills_idx_array'][idx_minl]
    hills_dict['scan_idx_array'] = hills_dict['scan_idx_array'][idx_minl]
    hills_dict['orig_idx_array'] = hills_dict['orig_idx_array'][idx_minl]

    if not len(hills_dict['orig_idx_array']):
        hills_dict['hills_idx_array_unique'] = []
        return hills_dict

    if len(hills_dict['orig_idx_array']):

        idx_sort = np.argsort(hills_dict['hills_idx_array'] + ((hills_dict['scan_idx_array'] + 1) / (hills_dict['scan_idx_array'].max()+2)))
        hills_dict['hills_idx_array'] = hills_dict['hills_idx_array'][idx_sort]
        hills_dict['scan_idx_array'] = hills_dict['scan_idx_array'][idx_sort]
        hills_dict['orig_idx_array'] = hills_dict['orig_idx_array'][idx_sort]

        hills_dict['hills_idx_array_unique'] = sorted(list(set(hills_dict['hills_idx_array'])))

        data_for_analyse_tmp_intensity = [z['intensity array'] for z in data_for_analyse_tmp]

        requested_procs = args['nprocs']
        len_full = len(hills_dict['hills_idx_array_unique'])
        if len_full <= 1000 * requested_procs:
            requested_procs = 1
        ranges = balanced_ranges(len_full, requested_procs)
        task_args = []
        checked_id = 0
        for worker_id, (start, end) in enumerate(ranges):
            sorted_idx_child_process = list(
                hills_dict['hills_idx_array_unique'][start:end]
            )
            idx_unique_set = set(sorted_idx_child_process)
            local_idx = np.array(
                [value in idx_unique_set for value in hills_dict['hills_idx_array']]
            )
            sorted_idx_array_child_process = hills_dict['hills_idx_array'][local_idx]
            task_args.append(
                (
                    hills_dict,
                    data_for_analyse_tmp_intensity,
                    args,
                    counter_hills_idx,
                    sorted_idx_child_process,
                    sorted_idx_array_child_process,
                    worker_id,
                    checked_id,
                )
            )
            checked_id += len(sorted_idx_array_child_process)

        if len(task_args) == 1:
            worker_results = [_split_peaks_task(*task_args[0])]
        else:
            worker_results = run_process_tasks(_split_peaks_task, task_args)
        new_idx_res = dict(enumerate(worker_results))
        n_procs = len(worker_results)

    final_idx_array = []
    last_id = 1
    for i in range(n_procs):
        added_idx_map = {}
        for ii, idx_val in enumerate(new_idx_res[i]):
            if idx_val not in added_idx_map:
                added_idx_map[idx_val] = int(last_id)
                last_id += 1
            final_idx_array.append(added_idx_map[idx_val])

    # hills_dict['scan_idx_array'] = np.concatenate([hills_dict['scan_idx_array'][all_sorted_idx[i]] for i in range(n_procs)])
    # hills_dict['orig_idx_array'] = np.concatenate([hills_dict['orig_idx_array'][all_sorted_idx[i]] for i in range(n_procs)])

    hills_dict['hills_idx_array'] = list(final_idx_array)
    del hills_dict['hills_idx_array_unique']

    return hills_dict

def process_file(args):

    input_file_path = args['file']
    stop_after_hills = bool(args.get('stop_after_hills'))
    stop_after_logged = False
    next_feature_idx = 1
    next_hill_idx = 1

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
        write_header = True
        ingestion = ingest_mzml(args)
        if args.get('write_ms1', False):
            utils.write_ms1_output(ingestion.ms1_rows, args)
        if args.get('write_ms2', False):
            utils.write_ms2_output(ingestion.ms2_rows, args)
        data_for_analyse = ingestion.spectra

        #Process faims

        faims_groups = group_spectra_by_faims(data_for_analyse)
        if len(faims_groups) > 1 or faims_groups[0][0] is not None:
            logger.info('Detected FAIMS values: %s', [value for value, _ in faims_groups])

        data_start_id = 0
        seed_contexts = []
        seed_final_candidates = []

        for faims_val, data_for_analyse_tmp in faims_groups:

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

            hills_dict, total_mass_diff = detect_hills(data_for_analyse_tmp, args, mz_step, paseftol, md_correction_int=md_correction_int)



            logger.info('Detected number of hills before splitting: %d', len(set(hills_dict['hills_idx_array'])))

            hills_dict = split_peaks_multi(hills_dict, data_for_analyse_tmp, args)
            logger.info('Starting hills processing')
            hills_dict = process_hills(hills_dict, data_for_analyse_tmp, mz_step, paseftol, args)
            next_hill_idx = assign_deterministic_hill_ids(
                hills_dict, next_hill_idx
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
                continue

            seed_context = None
            if args.get('ms2_seed'):
                seed_context = prepare_seed_context(
                    hills_dict,
                    data_for_analyse_tmp,
                    ingestion.ms2_rows,
                    ingestion.ms1_metadata,
                    faims_val,
                    RT_dict,
                    args,
                    len(faims_groups),
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
                seed_context=seed_context,
            )
            if seed_context is not None:
                seed_contexts.append(seed_context)
                seed_final_candidates.extend(ready_final)
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

        if args.get('ms2_seed'):
            aggregate = {
                'events': {}, 'event_edges': {},
                'summary': {'eligible_seed_count': 0, 'seed_local_hill_count': 0,
                            'local_candidate_counts': []},
            }
            for context in seed_contexts:
                for event_id, event in context['events'].items():
                    if event.get('eligible') or event_id not in aggregate['events']:
                        aggregate['events'][event_id] = event
                for event_id, edges in context['event_edges'].items():
                    aggregate['event_edges'].setdefault(event_id, []).extend(edges)
                for key in ('eligible_seed_count', 'seed_local_hill_count'):
                    aggregate['summary'][key] += context['summary'][key]
                aggregate['summary']['local_candidate_counts'].extend(
                    context['summary']['local_candidate_counts']
                )
            manager = args.get('_output_manager')
            if manager is None:
                raise RuntimeError('Output manager is required.')
            link_rows = build_link_rows(
                ingestion.ms2_rows, aggregate, seed_final_candidates
            )
            args['_ms2_seed_summary'] = aggregate['summary']
            manager.append_ms2_feature_links(link_rows)

    elif (
        input_file_path.lower().endswith('.hills.tsv')
        or input_file_path.lower().endswith('.hills.parquet')
        or input_file_path.lower().endswith('.hills.npz')
    ):
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
            hills_dict, mz_step = utils.get_hills_dict_from_hills_features(hills_features_local, hill_mass_accuracy, paseftol)
            next_hill_idx = assign_deterministic_hill_ids(
                hills_dict, next_hill_idx
            )


            logger.info('Detected number of hills: %d', len(set(hills_dict['hills_idx_array_unique'])))

            _, _, next_feature_idx, _ = process_features_iteration(
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


            write_header = False
