"""Parallel hill splitting with disk-backed worker result transport."""

from __future__ import annotations

from collections import Counter
import logging
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from .cutils import split_peaks
from .parallel import balanced_ranges, run_process_tasks

logger = logging.getLogger(__name__)


def _split_peaks_artifact_task(
    hills_dict, intensities, args, counters, hill_ids, hill_index_array,
    worker_id, checked_id, artifact_dir,
):
    values = np.asarray(list(split_peaks(
        hills_dict, intensities, args, counters, hill_ids, hill_index_array,
        worker_id, checked_id,
    )), dtype=np.int64)
    path = Path(artifact_dir) / ("worker-%03d.npy" % worker_id)
    np.save(path, values, allow_pickle=False)
    return worker_id, str(path), len(values)


def _split_artifact_directory(args):
    raw_cache = args.get('raw_ms1_cache_dir')
    parent = Path(raw_cache) if raw_cache else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='biosaur2-split-', dir=parent))


def _remap_worker_ids(values, next_id):
    """Map labels by first encounter, preserving the legacy parent order."""
    labels, first_positions, inverse = np.unique(
        values, return_index=True, return_inverse=True
    )
    label_to_global = np.empty(len(labels), dtype=np.int64)
    label_to_global[np.argsort(first_positions)] = np.arange(
        next_id, next_id + len(labels), dtype=np.int64
    )
    return label_to_global[inverse], next_id + len(labels)


def split_peaks_multi(hills_dict, data_for_analyse_tmp, args):
    started = time.monotonic()
    min_length_hill = args['minlh']
    hills_dict['orig_idx_array'] = np.asarray(hills_dict['orig_idx_array'])
    hills_dict['scan_idx_array'] = np.asarray(hills_dict['scan_idx_array'])
    hills_dict['hills_idx_array'] = np.asarray(hills_dict['hills_idx_array'])

    counts = dict(Counter(hills_dict['hills_idx_array']))
    lengths = np.asarray([counts[hill_id] for hill_id in hills_dict['hills_idx_array']])
    keep = lengths >= min_length_hill
    for key in ('hills_idx_array', 'scan_idx_array', 'orig_idx_array'):
        hills_dict[key] = hills_dict[key][keep]
    if not len(hills_dict['orig_idx_array']):
        hills_dict['hills_idx_array_unique'] = []
        return hills_dict

    sort_index = np.argsort(
        hills_dict['hills_idx_array']
        + ((hills_dict['scan_idx_array'] + 1)
           / (hills_dict['scan_idx_array'].max() + 2))
    )
    for key in ('hills_idx_array', 'scan_idx_array', 'orig_idx_array'):
        hills_dict[key] = hills_dict[key][sort_index]
    hills_dict['hills_idx_array_unique'] = sorted(set(hills_dict['hills_idx_array']))
    intensities = [spectrum['intensity array'] for spectrum in data_for_analyse_tmp]

    requested_workers = int(args['nprocs'])
    total_hills = len(hills_dict['hills_idx_array_unique'])
    if total_hills <= 1000 * requested_workers:
        requested_workers = 1
    ranges = balanced_ranges(total_hills, requested_workers)
    preparation_seconds = time.monotonic() - started
    artifact_dir = _split_artifact_directory(args)
    try:
        tasks = []
        hill_index_array = hills_dict['hills_idx_array']
        for worker_id, (start, end) in enumerate(ranges):
            hill_ids = list(hills_dict['hills_idx_array_unique'][start:end])
            point_start = int(np.searchsorted(hill_index_array, hill_ids[0]))
            point_end = int(np.searchsorted(hill_index_array, hill_ids[-1], side='right'))
            tasks.append((
                hills_dict, intensities, args, counts, hill_ids,
                hill_index_array[point_start:point_end], worker_id, point_start,
                str(artifact_dir),
            ))
        if len(tasks) == 1:
            artifacts = [_split_peaks_artifact_task(*tasks[0])]
        else:
            artifacts = list(run_process_tasks(_split_peaks_artifact_task, tasks))
        worker_seconds = time.monotonic() - started - preparation_seconds
        merge_started = time.monotonic()
        remapped = []
        next_id = 1
        for _worker_id, path, count in sorted(artifacts):
            values = np.load(path, mmap_mode='r', allow_pickle=False)
            if len(values) != count:
                raise RuntimeError('split-peaks artifact length changed before merge')
            mapped, next_id = _remap_worker_ids(values, next_id)
            remapped.append(mapped)
        hills_dict['hills_idx_array'] = np.concatenate(remapped).tolist()
        logger.debug(
            'Split-peaks timing: hills=%d workers=%d preparation_sec=%.3f '
            'worker_sec=%.3f mmap_merge_sec=%.3f',
            total_hills, len(tasks), preparation_seconds, worker_seconds,
            time.monotonic() - merge_started,
        )
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)
    del hills_dict['hills_idx_array_unique']
    return hills_dict
