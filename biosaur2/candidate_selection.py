"""Deterministic isotope-candidate conflict selection helpers."""

from __future__ import annotations

import logging

import numpy as np

from .cutils import checking_cos_correlation_for_carbon, get_and_calc_apex_intensity_and_scan
from .external_weak import append_rejected_candidate, remember_reject_snapshot
from .parallel import run_process_tasks

logger = logging.getLogger(__name__)


def candidate_hill_ids(candidate):
    return tuple(sorted(
        [int(candidate['monoisotope hill idx'])]
        + [int(value['isotope_hill_idx']) for value in candidate['isotopes']]
    ))


def candidate_conflict_key(candidate, hills_dict):
    mono_index = int(candidate['monoisotope idx'])
    _, apex_intensity, _ = get_and_calc_apex_intensity_and_scan(
        hills_dict, mono_index
    )
    mass_errors = [abs(float(value['mass_diff_ppm'])) for value in candidate['isotopes']]
    mean_absolute_error = float(np.mean(mass_errors)) if mass_errors else float('inf')
    isotope_count = int(candidate['nIsotopes'])
    isotope_cosine = float(candidate['cos_cor_isotopes'])
    return (
        -(isotope_count + isotope_cosine), -isotope_count, -isotope_cosine,
        -float(apex_intensity), mean_absolute_error, float(candidate['hill_mz_1']),
        int(candidate['charge']), candidate_hill_ids(candidate),
    )


def final_feature_key(candidate, hills_dict, rt_dict, data_start_id):
    mono_index = int(candidate['monoisotope idx'])
    _, _, apex_scan = get_and_calc_apex_intensity_and_scan(hills_dict, mono_index)
    apex_rt = (
        float(hills_dict['rtApex'][mono_index])
        if rt_dict is False else float(rt_dict[apex_scan + data_start_id])
    )
    return (
        float(candidate['hill_mz_1']), apex_rt, int(candidate['charge']),
        int(candidate['monoisotope hill idx']), candidate_hill_ids(candidate),
    )


def select_nonconflicting_isotope_candidates(
    ready, hills_dict, rt_dict, data_start_id, rejected_sink=None
):
    max_l = len(ready)
    cur_l = 0
    ready_final = []
    ready_set = set()
    original_by_candidate = {}
    if not ready:
        logger.info('No isotope clusters remained after smart mass accuracy filtering.')
    else:
        ready = sorted(ready, key=lambda candidate: candidate_conflict_key(candidate, hills_dict))
        cur_isotopes = ready[0]['nIsotopes']
        while cur_l < max_l:
            pep_feature = ready[cur_l]
            n_iso = pep_feature['nIsotopes']
            if n_iso < cur_isotopes:
                ready = sorted(ready, key=lambda candidate: candidate_conflict_key(candidate, hills_dict))
                cur_isotopes = n_iso
                cur_l = 0
                pep_feature = ready[cur_l]
            if pep_feature['monoisotope hill idx'] not in ready_set:
                if not any(cand['isotope_hill_idx'] in ready_set for cand in pep_feature['isotopes']):
                    ready_final.append(pep_feature)
                    original_by_candidate.pop(id(pep_feature), None)
                    ready_set.add(pep_feature['monoisotope hill idx'])
                    for cand in pep_feature['isotopes']:
                        ready_set.add(cand['isotope_hill_idx'])
                    del ready[cur_l]
                    max_l -= 1
                    cur_l -= 1
                else:
                    remaining = []
                    for cand in pep_feature['isotopes']:
                        if cand['isotope_hill_idx'] not in ready_set:
                            remaining.append(cand)
                        else:
                            break
                    if remaining:
                        if rejected_sink is not None:
                            remember_reject_snapshot(original_by_candidate, pep_feature)
                        all_theoretical, all_experimental = pep_feature['intensity_array_for_cos_corr']
                        all_theoretical = all_theoretical[:len(remaining) + 1]
                        all_experimental = all_experimental[:len(remaining) + 1]
                        cos_corr, _passed = checking_cos_correlation_for_carbon(
                            all_theoretical, all_experimental, 0.6
                        )
                        if cos_corr:
                            ready[cur_l]['cos_cor_isotopes'] = cos_corr
                            ready[cur_l]['isotopes'] = remaining
                            ready[cur_l]['nIsotopes'] = len(remaining) + 1
                            ready[cur_l]['intensity_array_for_cos_corr'] = [all_theoretical, all_experimental]
                            cur_l -= 1
                        else:
                            append_rejected_candidate(rejected_sink, original_by_candidate, pep_feature)
                            del ready[cur_l]
                            max_l -= 1
                            cur_l -= 1
                    else:
                        append_rejected_candidate(rejected_sink, original_by_candidate, pep_feature)
                        del ready[cur_l]
                        max_l -= 1
                        cur_l -= 1
            else:
                append_rejected_candidate(rejected_sink, original_by_candidate, pep_feature)
                del ready[cur_l]
                max_l -= 1
                cur_l -= 1
            cur_l += 1
    ready_final.sort(key=lambda candidate: final_feature_key(candidate, hills_dict, rt_dict, data_start_id))
    logger.info('Number of detected isotope clusters: %d', len(ready_final))
    return ready_set, ready_final


def _candidate_component_indexes(ready):
    parent = list(range(len(ready)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    first_by_hill = {}
    for index, candidate in enumerate(ready):
        for hill_id in candidate_hill_ids(candidate):
            union(first_by_hill.setdefault(hill_id, index), index)
    components = {}
    for index in range(len(ready)):
        components.setdefault(find(index), []).append(index)
    return tuple(sorted(components.values(), key=lambda values: values[0]))


def _select_component_task(candidates, hills_dict, rt_dict, data_start_id):
    return select_nonconflicting_isotope_candidates(
        candidates, hills_dict, rt_dict, data_start_id
    )


def select_nonconflicting_isotope_candidates_parallel(
    ready, hills_dict, rt_dict, data_start_id, workers
):
    """Select independent hill-ID components then deterministically merge."""
    components = _candidate_component_indexes(ready)
    if len(components) <= 1:
        return select_nonconflicting_isotope_candidates(
            ready, hills_dict, rt_dict, data_start_id
        )
    batches = [[] for _ in range(min(int(workers), len(components)))]
    loads = [0] * len(batches)
    for component in sorted(components, key=len, reverse=True):
        target = min(range(len(batches)), key=lambda index: loads[index])
        batches[target].append(component)
        loads[target] += len(component)
    tasks = [(
        [ready[index] for component in sorted(batch, key=lambda values: values[0]) for index in component],
        hills_dict, rt_dict, data_start_id,
    ) for batch in batches]
    ready_set = set()
    ready_final = []
    for component_set, component_final in run_process_tasks(_select_component_task, tasks):
        ready_set.update(component_set)
        ready_final.extend(component_final)
    ready_final.sort(key=lambda candidate: final_feature_key(candidate, hills_dict, rt_dict, data_start_id))
    logger.info('Number of detected isotope clusters: %d', len(ready_final))
    return ready_set, ready_final
