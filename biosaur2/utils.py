from pyteomics import mzml, xml
import ast
from collections import defaultdict
from contextlib import contextmanager
import gzip
import logging

import numpy as np
import pandas as pd

from .cutils import get_and_calc_apex_intensity_and_scan
from .calibration import fit_mass_calibration
from .spectra import extract_scan_number
from .quantification import raw_area_sum

logger = logging.getLogger(__name__)

_SCAN_ID_MISMATCH_WARNED = False

HILLS_NPZ_SCHEMA_VERSION = 1
HILLS_NPZ_REQUIRED_KEYS = (
    'schema_version',
    'mz',
    'rtApex',
    'rtStart',
    'rtEnd',
    'FAIMS',
    'im',
    'point_offsets',
    'hills_scan_flat',
    'hills_intensity_flat',
)
HILLS_NPZ_FIXED_KEYS = (
    'mz',
    'rtApex',
    'rtStart',
    'rtEnd',
    'FAIMS',
    'im',
)
def _is_mzml_gzip_path(input_file_path):
    return str(input_file_path).lower().endswith('.mzml.gz')


@contextmanager
def open_mzml_source(input_file_path):
    if _is_mzml_gzip_path(input_file_path):
        with gzip.open(input_file_path, 'rb') as source:
            yield source
    else:
        yield input_file_path


def iter_ms1_spectra(input_file_path):
    with open_mzml_source(input_file_path) as mzml_source:
        for spec in MS1OnlyMzML(source=mzml_source):
            yield spec


def iter_all_spectra(input_file_path):
    with open_mzml_source(input_file_path) as mzml_source:
        for spec in mzml.read(mzml_source):
            yield spec


def iter_ms1_and_ms2_metadata(input_file_path):
    """Yield decoded MS1 spectra and metadata-only non-MS1 spectra once."""

    with open_mzml_source(input_file_path) as mzml_source:
        for spec in MS1AndMetadataMzML(source=mzml_source):
            yield spec


def _build_hills_dict(
    hills_idx_array_unique,
    hills_mz_median,
    hills_im_median,
    hills_lengths,
    hills_scan_lists,
    hills_intensity_array,
    rt_start,
    rt_end,
    rt_apex,
    hill_mass_accuracy,
    paseftol,
    hills_point_rt=None,
):
    hills_dict = dict()
    hills_dict['hills_idx_array_unique'] = np.asarray(hills_idx_array_unique)
    hills_dict['hills_mz_median'] = np.asarray(hills_mz_median)

    max_mz_value = float(np.max(hills_dict['hills_mz_median'])) if hills_dict['hills_mz_median'].size else 1.0
    mz_step = hill_mass_accuracy * 1e-6 * max_mz_value
    if mz_step == 0:
        mz_step = hill_mass_accuracy * 1e-6 if hill_mass_accuracy else 1e-6

    has_im = hills_im_median is not None and len(hills_im_median) and np.any(hills_im_median)
    if has_im:
        hills_dict['hills_im_median'] = np.asarray(hills_im_median)

    hills_dict['hills_lengths'] = np.asarray(hills_lengths)
    hills_dict['hills_scan_lists'] = [list(map(int, slist)) for slist in hills_scan_lists]
    hills_dict['hills_scan_sets'] = [set(slist) for slist in hills_dict['hills_scan_lists']]
    hills_dict['hills_intensity_array'] = [list(map(float, ilist)) for ilist in hills_intensity_array]
    if hills_point_rt is not None:
        hills_dict['hills_point_rt_array'] = [
            list(map(float, values)) for values in hills_point_rt
        ]

    hills_dict['hills_mz_median_fast_dict'] = defaultdict(list)
    if paseftol > 0 and has_im:
        hills_dict['hills_im_median_fast_dict'] = defaultdict(set)

    for idx_1, mz_val in enumerate(hills_dict['hills_mz_median']):
        mz_median_int = int(mz_val / mz_step)
        tmp_scans_list = hills_dict['hills_scan_lists'][idx_1]
        tmp_val = (idx_1, tmp_scans_list[0], tmp_scans_list[-1])
        hills_dict['hills_mz_median_fast_dict'][mz_median_int-1].append(tmp_val)
        hills_dict['hills_mz_median_fast_dict'][mz_median_int].append(tmp_val)
        hills_dict['hills_mz_median_fast_dict'][mz_median_int+1].append(tmp_val)

        if paseftol > 0 and has_im:
            im_median_int = int(hills_dict['hills_im_median'][idx_1] / paseftol)
            hills_dict['hills_im_median_fast_dict'][im_median_int-1].add(idx_1)
            hills_dict['hills_im_median_fast_dict'][im_median_int].add(idx_1)
            hills_dict['hills_im_median_fast_dict'][im_median_int+1].add(idx_1)

    hills_dict['hills_idict'] = [None] * len(hills_dict['hills_idx_array_unique'])
    hills_dict['hill_sqrt_of_i'] = [None] * len(hills_dict['hills_idx_array_unique'])
    hills_dict['hills_intensity_apex'] = [None] * len(hills_dict['hills_idx_array_unique'])
    hills_dict['hills_scan_apex'] = [None] * len(hills_dict['hills_idx_array_unique'])

    hills_dict['rtStart'] = np.asarray(rt_start)
    hills_dict['rtEnd'] = np.asarray(rt_end)
    hills_dict['rtApex'] = np.asarray(rt_apex)

    return hills_dict, mz_step


def _parse_ragged_column(column):
    if len(column) == 0:
        return np.array([], dtype=object)
    first_value = column.iloc[0]
    if isinstance(first_value, str):
        return column.apply(ast.literal_eval).values
    return column.values


def _validate_hills_npz_payload(payload, source_path):
    missing_keys = [key for key in HILLS_NPZ_REQUIRED_KEYS if key not in payload]
    if missing_keys:
        raise ValueError('Invalid hills NPZ file %s: missing keys: %s' % (source_path, ', '.join(missing_keys)))

    schema_version = int(np.asarray(payload['schema_version']).reshape(-1)[0])
    if schema_version != HILLS_NPZ_SCHEMA_VERSION:
        raise ValueError(
            'Unsupported hills NPZ schema version in %s: %s (expected %s)'
            % (source_path, schema_version, HILLS_NPZ_SCHEMA_VERSION)
        )

    row_count = int(np.asarray(payload['mz']).shape[0])
    for key in HILLS_NPZ_FIXED_KEYS:
        if int(np.asarray(payload[key]).shape[0]) != row_count:
            raise ValueError('Invalid hills NPZ file %s: key %s has inconsistent row count.' % (source_path, key))

    for optional_key in ('hill_idx', 'nScans', 'intensityApex', 'intensitySum'):
        if optional_key in payload and int(np.asarray(payload[optional_key]).shape[0]) != row_count:
            raise ValueError('Invalid hills NPZ file %s: key %s has inconsistent row count.' % (source_path, optional_key))

    point_offsets = np.asarray(payload['point_offsets'])
    if point_offsets.ndim != 1:
        raise ValueError('Invalid hills NPZ file %s: point_offsets must be one-dimensional.' % (source_path, ))
    if point_offsets.size != row_count + 1:
        raise ValueError('Invalid hills NPZ file %s: point_offsets size mismatch.' % (source_path, ))
    if point_offsets[0] != 0:
        raise ValueError('Invalid hills NPZ file %s: point_offsets must start at 0.' % (source_path, ))
    if np.any(np.diff(point_offsets) < 0):
        raise ValueError('Invalid hills NPZ file %s: point_offsets must be nondecreasing.' % (source_path, ))

    flat_scan = np.asarray(payload['hills_scan_flat'])
    flat_intensity = np.asarray(payload['hills_intensity_flat'])
    point_count = int(flat_scan.shape[0])

    if int(flat_intensity.shape[0]) != point_count:
        raise ValueError('Invalid hills NPZ file %s: flattened point arrays must have identical size.' % (source_path, ))
    if int(point_offsets[-1]) != point_count:
        raise ValueError('Invalid hills NPZ file %s: point_offsets end does not match flattened arrays.' % (source_path, ))

    if 'hills_mz_flat' in payload:
        flat_mz = np.asarray(payload['hills_mz_flat'])
        if int(flat_mz.shape[0]) != point_count:
            raise ValueError('Invalid hills NPZ file %s: flattened point arrays must have identical size.' % (source_path, ))

    if 'nScans' in payload:
        nscans = np.asarray(payload['nScans'], dtype=np.int64)
        if not np.array_equal(np.diff(point_offsets).astype(np.int64), nscans):
            raise ValueError('Invalid hills NPZ file %s: nScans does not match point_offsets.' % (source_path, ))

    if 'hills_float' in payload:
        hills_float = str(np.asarray(payload['hills_float']).reshape(-1)[0])
        if hills_float not in ('float32', 'float64'):
            raise ValueError('Invalid hills NPZ file %s: hills_float must be float32 or float64.' % (source_path, ))


def _load_hills_npz_payload(npz_path):
    with np.load(npz_path, allow_pickle=False) as npz_data:
        payload = {key: npz_data[key] for key in npz_data.files}
    _validate_hills_npz_payload(payload, npz_path)
    return payload


def _extract_ms1_scan_id(spec, fallback_idx):
    global _SCAN_ID_MISMATCH_WARNED

    scan_from_id = extract_scan_number(spec)

    scan_from_index = None
    if 'index' in spec:
        try:
            scan_from_index = int(spec['index']) + 1
        except (TypeError, ValueError):
            scan_from_index = None

    if scan_from_id is not None:
        if (
            scan_from_index is not None
            and scan_from_id != scan_from_index
            and not _SCAN_ID_MISMATCH_WARNED
        ):
            logger.warning(
                'Mismatch between spectrum id scan= value (%d) and index+1 (%d). '
                'Using scan= value for scan_id/scanApex.',
                scan_from_id,
                scan_from_index,
            )
            _SCAN_ID_MISMATCH_WARNED = True
        return scan_from_id

    if scan_from_index is not None:
        return scan_from_index

    return int(fallback_idx) + 1


def write_ms1_output(ms1_rows, args):
    manager = args.get('_output_manager')
    if manager is None:
        raise RuntimeError('Output manager is required.')
    manager.append_ms1(ms1_rows)


def write_ms2_output(ms2_rows, args):
    manager = args.get('_output_manager')
    if manager is None:
        raise RuntimeError('Output manager is required.')
    manager.append_ms2(ms2_rows)


def get_hills_features_from_hills_npz(npz_path):
    payload = _load_hills_npz_payload(npz_path)
    row_count = int(np.asarray(payload['mz']).shape[0])
    point_offsets = np.asarray(payload['point_offsets'], dtype=np.int64)
    hills_scan_flat = np.asarray(payload['hills_scan_flat'])
    hills_intensity_flat = np.asarray(payload['hills_intensity_flat'])
    has_flat_mz = 'hills_mz_flat' in payload
    if has_flat_mz:
        hills_mz_flat = np.asarray(payload['hills_mz_flat'])

    hills_scan_lists = []
    hills_intensity_list = []
    hills_mz_array = []
    for idx_1 in range(row_count):
        idx_start = point_offsets[idx_1]
        idx_end = point_offsets[idx_1+1]
        hills_scan_lists.append(hills_scan_flat[idx_start:idx_end].astype(np.int64).tolist())
        hills_intensity_list.append(hills_intensity_flat[idx_start:idx_end].astype(float).tolist())
        if has_flat_mz:
            hills_mz_array.append(hills_mz_flat[idx_start:idx_end].astype(float).tolist())
        else:
            hills_mz_array.append(np.full(idx_end - idx_start, payload['mz'][idx_1], dtype=float).tolist())

    return {
        'rtApex': np.asarray(payload['rtApex']),
        'nScans': np.diff(point_offsets).astype(np.int32),
        'mz': np.asarray(payload['mz']),
        'rtStart': np.asarray(payload['rtStart']),
        'rtEnd': np.asarray(payload['rtEnd']),
        'FAIMS': np.asarray(payload['FAIMS']),
        'im': np.asarray(payload['im']),
        'hill_idx': np.asarray(payload['hill_idx']) if 'hill_idx' in payload else np.arange(row_count, dtype=np.int64),
        'hills_scan_lists': hills_scan_lists,
        'hills_intensity_list': hills_intensity_list,
        'hills_mz_array': hills_mz_array,
    }





class MS1OnlyMzML(mzml.MzML): 
     _default_iter_path = '//spectrum[./*[local-name()="cvParam" and @name="ms level" and @value="1"]]' 
     _use_index = False 
     _iterative = False


class MS1AndMetadataMzML(mzml.MzML):
    """Decode peak arrays only for MS1 while retaining MS2 XML metadata."""

    _default_iter_path = "//spectrum"
    _use_index = False

    def _get_info_smart(self, element, **kwargs):
        if xml._local_name(element) == "spectrum":
            ms_level = None
            for child in element.iterchildren():
                if (
                    xml._local_name(child) == "cvParam"
                    and child.get("name") == "ms level"
                ):
                    try:
                        ms_level = int(child.get("value"))
                    except (TypeError, ValueError):
                        pass
                    break
            if ms_level != 1:
                for child in list(element.iterchildren()):
                    if xml._local_name(child) == "binaryDataArrayList":
                        element.remove(child)
        return super()._get_info_smart(element, **kwargs)

def calibrate_mass(bwidth, mass_left, mass_right, true_md):
    result = fit_mass_calibration(true_md, bin_width=bwidth)
    if result.status != 'applied':
        raise ValueError('Mass calibration was not applied: %s' % result.reason)
    return result.shift, result.sigma, result.covariance

def get_hills_dict_from_hills_features(hills_features, hill_mass_accuracy, paseftol):
    missing_cols = [col for col in ('hills_scan_lists', 'hills_intensity_list') if col not in hills_features.columns]
    if missing_cols:
        raise ValueError(
            'Hills input is missing required columns: %s. '
            'If this hills file was generated with --no_hill_list, it cannot be used for feature detection.'
            % (', '.join(missing_cols), )
        )

    hills_scan_lists = _parse_ragged_column(hills_features['hills_scan_lists'])
    hills_intensity_array = _parse_ragged_column(hills_features['hills_intensity_list'])
    hills_point_rt = (
        _parse_ragged_column(hills_features['hills_rt_list'])
        if 'hills_rt_list' in hills_features
        else None
    )
    hills_point_mz = (
        _parse_ragged_column(hills_features['hills_mz_array'])
        if 'hills_mz_array' in hills_features
        else None
    )
    hills_scan_numbers = (
        _parse_ragged_column(hills_features['hills_scan_number_list'])
        if 'hills_scan_number_list' in hills_features
        else None
    )
    for row_index, scans in enumerate(hills_scan_lists):
        expected = len(scans)
        arrays = [("intensity", hills_intensity_array[row_index])]
        if hills_point_rt is not None:
            arrays.append(("RT", hills_point_rt[row_index]))
        if hills_point_mz is not None:
            arrays.append(("m/z", hills_point_mz[row_index]))
        if hills_scan_numbers is not None:
            arrays.append(("scan-number", hills_scan_numbers[row_index]))
        for label, values in arrays:
            if len(values) != expected:
                raise ValueError(
                    "Hills row %d has %d scan indexes but %d %s values."
                    % (row_index, expected, len(values), label)
                )

    hills_im_median = None
    if 'im' in hills_features:
        im_values = pd.to_numeric(hills_features['im'], errors='coerce')
        if im_values.notna().any():
            hills_im_median = im_values.fillna(0).values
    hills_dict, mz_step = _build_hills_dict(
        hills_idx_array_unique=hills_features['hill_idx'].values,
        hills_mz_median=hills_features['mz'].values,
        hills_im_median=hills_im_median,
        hills_lengths=hills_features['nScans'].values,
        hills_scan_lists=hills_scan_lists,
        hills_intensity_array=hills_intensity_array,
        rt_start=hills_features['rtStart'].values,
        rt_end=hills_features['rtEnd'].values,
        rt_apex=hills_features['rtApex'].values,
        hill_mass_accuracy=hill_mass_accuracy,
        paseftol=paseftol,
        hills_point_rt=hills_point_rt,
    )
    if hills_point_mz is not None:
        hills_dict['tmp_mz_array'] = [
            [float(value) for value in values] for values in hills_point_mz
        ]
    if hills_scan_numbers is not None:
        hills_dict['hills_scan_number_array'] = [
            [None if value is None else int(value) for value in values]
            for values in hills_scan_numbers
        ]
    return hills_dict, mz_step


def iter_hills_extra(
    hills_dict, RT_dict, faims_val, data_start_id, mz_step, paseftol,
    data_for_analyse_tmp=None,
    include_point_lists=True,
    feature_idx_by_hill=None,
):
    for idx_1 in range(len(hills_dict['hills_idx_array_unique'])):
        hill_feature = {}
        hills_dict, hill_intensity_apex_1, hill_scan_apex_1 = get_and_calc_apex_intensity_and_scan(hills_dict, idx_1)
        hill_feature['mz'] = hills_dict['hills_mz_median'][idx_1]
        hill_feature['nScans'] = hills_dict['hills_lengths'][idx_1]
        hill_feature['rtApex'] = RT_dict[hill_scan_apex_1+data_start_id]
        hill_feature['intensityApex'] = hill_intensity_apex_1
        hill_feature['intensitySum'] = sum(hills_dict['hills_intensity_array'][idx_1])
        hill_feature['rtStart'] = RT_dict[hills_dict['hills_scan_lists'][idx_1][0]+data_start_id]
        hill_feature['rtEnd'] = RT_dict[hills_dict['hills_scan_lists'][idx_1][-1]+data_start_id]
        hill_feature['FAIMS'] = None if faims_val is None else float(faims_val)
        if 'hills_im_median' in hills_dict:
            hill_feature['im'] = float(hills_dict['hills_im_median'][idx_1])
        else:
            hill_feature['im'] = None
        hill_feature['hill_idx'] = hills_dict['hills_idx_array_unique'][idx_1]
        if feature_idx_by_hill is not None:
            hill_feature['feature_idx'] = int(
                feature_idx_by_hill.get(hill_feature['hill_idx'], -1)
            )
        local_scans = [int(v) for v in hills_dict['hills_scan_lists'][idx_1]]
        local_apex = int(hill_scan_apex_1)
        sources = data_for_analyse_tmp
        apex_source = sources[local_apex] if sources is not None else None
        hill_feature['scanApex'] = (
            apex_source.get('scan_number') if apex_source else None
        )
        if include_point_lists:
            hill_feature['hills_scan_lists'] = local_scans
            hill_feature['hills_intensity_list'] = [
                float(v) for v in hills_dict['hills_intensity_array'][idx_1]
            ]
            hill_feature['hills_mz_array'] = [
                float(v) for v in hills_dict['tmp_mz_array'][idx_1]
            ]
            hill_feature['hills_rt_list'] = [
                float(RT_dict[scan + data_start_id]) for scan in local_scans
            ]
        yield hill_feature


def calc_peptide_features(
    hills_dict,
    peptide_features,
    negative_mode,
    faims_val,
    RT_dict,
    data_start_id,
    isotopes_for_intensity,
    include_mono_hills=True,
    quantification_args=None,
    spectra=None,
):

    for pep_feature in peptide_features:

        pep_feature['mz'] = pep_feature['hill_mz_1']
        pep_feature['isoerror'] = pep_feature['isotopes'][0]['mass_diff_ppm']
        pep_feature['isoerror2'] = pep_feature['isotopes'][1]['mass_diff_ppm'] if len(pep_feature['isotopes']) > 1 else -100
        pep_feature['nScans'] = hills_dict['hills_lengths'][pep_feature['monoisotope idx']]

        pep_feature['massCalib'] = pep_feature['mz'] * pep_feature['charge'] - 1.0072765 * pep_feature['charge'] * (-1 if negative_mode else 1)

        hills_dict, _, _ = get_and_calc_apex_intensity_and_scan(hills_dict, pep_feature['monoisotope idx'])
        pep_feature['intensityApex'] = hills_dict['hills_intensity_apex'][pep_feature['monoisotope idx']]
        pep_feature['intensitySum'] = sum(hills_dict['hills_intensity_array'][pep_feature['monoisotope idx']])

        if isotopes_for_intensity != 0:
            idx_cur = 0
            for cand in pep_feature['isotopes']:
                idx_cur += 1
                if idx_cur == isotopes_for_intensity + 1:
                    break
                else:
                    iso_idx = cand['isotope_idx']
                    hills_dict, _, _ = get_and_calc_apex_intensity_and_scan(hills_dict, iso_idx)
                    pep_feature['intensityApex'] += hills_dict['hills_intensity_apex'][iso_idx]
                    pep_feature['intensitySum'] += sum(hills_dict['hills_intensity_array'][iso_idx])
                

        pep_feature['scanApex'] = hills_dict['hills_scan_apex'][pep_feature['monoisotope idx']]
        if RT_dict is not False:
            pep_feature['rtApex'] = RT_dict[hills_dict['hills_scan_apex'][pep_feature['monoisotope idx']]+data_start_id]
            pep_feature['rtStart'] = RT_dict[hills_dict['hills_scan_lists'][pep_feature['monoisotope idx']][0]+data_start_id]
            pep_feature['rtEnd'] = RT_dict[hills_dict['hills_scan_lists'][pep_feature['monoisotope idx']][-1]+data_start_id]
        else:
            pep_feature['rtApex'] = hills_dict['rtApex'][pep_feature['monoisotope idx']]
            pep_feature['rtStart'] = hills_dict['rtStart'][pep_feature['monoisotope idx']]
            pep_feature['rtEnd'] = hills_dict['rtEnd'][pep_feature['monoisotope idx']]

        mono_scans = [int(v) for v in hills_dict['hills_scan_lists'][pep_feature['monoisotope idx']]]
        local_apex = int(hills_dict['hills_scan_apex'][pep_feature['monoisotope idx']])
        apex_source = spectra[local_apex] if spectra is not None else None
        stored_scan_numbers = hills_dict.get('hills_scan_number_array')
        if stored_scan_numbers is not None:
            local_numbers = stored_scan_numbers[pep_feature['monoisotope idx']]
            apex_position = mono_scans.index(local_apex)
        else:
            local_numbers = None
            apex_position = None
        pep_feature['scanApex'] = (
            apex_source.get('scan_number')
            if apex_source
            else (local_numbers[apex_position] if local_numbers else None)
        )
        pep_feature['FAIMS'] = None if faims_val is None else float(faims_val)
        pep_feature['im'] = None
        if 'hills_im_median' in hills_dict:
            pep_feature['im'] = float(
                hills_dict['hills_im_median'][pep_feature['monoisotope idx']]
            )

        if include_mono_hills:
            pep_feature['mono_hills_scan_lists'] = [int(v) for v in hills_dict['hills_scan_lists'][pep_feature['monoisotope idx']]]
            pep_feature['mono_hills_intensity_list'] = [float(v) for v in hills_dict['hills_intensity_array'][pep_feature['monoisotope idx']]]

        area_sum, approximate_area = raw_area_sum(
            hills_dict,
            pep_feature,
            RT_dict if RT_dict is not False else None,
            isotopes_for_intensity,
        )
        pep_feature['area_sum'] = area_sum
        if approximate_area and quantification_args is not None:
            quantification_args['_area_sum_approximate'] = True

    return peptide_features


def write_output(peptide_features, args, write_header=True, hills=False):
    manager = args.get('_output_manager')
    if manager is None:
        raise RuntimeError('Output manager is required.')
    if hills:
        manager.append_hills(peptide_features)
    else:
        manager.append_features(peptide_features)
