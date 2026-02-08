from pyteomics import mzml
import numpy as np
from collections import defaultdict, Counter
from os import path
import math
from scipy.optimize import curve_fit
import logging
logger = logging.getLogger(__name__)
from .cutils import get_fast_dict, get_and_calc_apex_intensity_and_scan, centroid_pasef_scan
import ast

HILLS_NPZ_SCHEMA_VERSION = 1
HILLS_NPZ_REQUIRED_KEYS = (
    'schema_version',
    'hills_float',
    'hill_idx',
    'nScans',
    'mz',
    'rtApex',
    'intensityApex',
    'intensitySum',
    'rtStart',
    'rtEnd',
    'FAIMS',
    'im',
    'point_offsets',
    'hills_scan_flat',
    'hills_intensity_flat',
    'hills_mz_flat',
)
HILLS_NPZ_FIXED_KEYS = (
    'hill_idx',
    'nScans',
    'mz',
    'rtApex',
    'intensityApex',
    'intensitySum',
    'rtStart',
    'rtEnd',
    'FAIMS',
    'im',
)


def _get_hills_float_name(args):
    return args.get('hills_float', 'float32')


def _get_hills_float_dtype(args):
    float_name = _get_hills_float_name(args)
    if float_name == 'float32':
        return np.float32
    if float_name == 'float64':
        return np.float64
    raise ValueError('Unsupported hills float type: %s' % (float_name, ))


def _get_output_file(args, hills=False):
    input_mzml_path = args['file']
    if hills:
        hills_ext = 'hills.tsv'
        if args.get('hills_format', 'tsv') == 'npz':
            hills_ext = 'hills.npz'
        if args['o']:
            return path.splitext(args['o'])[0] + path.extsep + hills_ext
        return path.splitext(input_mzml_path)[0] + path.extsep + hills_ext
    if args['o']:
        return args['o']
    return path.splitext(input_mzml_path)[0] + path.extsep + 'features.tsv'


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


def _build_hills_npz_payload(hills_features, float_dtype, float_name):
    row_count = len(hills_features)

    point_offsets = np.zeros(row_count + 1, dtype=np.int64)
    for idx_1, hill_feature in enumerate(hills_features):
        point_offsets[idx_1+1] = point_offsets[idx_1] + len(hill_feature['hills_scan_lists'])

    total_points = int(point_offsets[-1])
    hills_scan_flat = np.empty(total_points, dtype=np.int32)
    hills_intensity_flat = np.empty(total_points, dtype=float_dtype)
    hills_mz_flat = np.empty(total_points, dtype=float_dtype)

    for idx_1, hill_feature in enumerate(hills_features):
        idx_start = point_offsets[idx_1]
        idx_end = point_offsets[idx_1+1]

        tmp_scans = np.asarray(hill_feature['hills_scan_lists'], dtype=np.int32)
        tmp_intensity = np.asarray(hill_feature['hills_intensity_list'], dtype=float_dtype)
        tmp_mz = np.asarray(hill_feature['hills_mz_array'], dtype=float_dtype)

        if not (tmp_scans.size == tmp_intensity.size == tmp_mz.size):
            raise ValueError('Inconsistent hills list lengths for hill index %s' % (hill_feature.get('hill_idx'), ))

        hills_scan_flat[idx_start:idx_end] = tmp_scans
        hills_intensity_flat[idx_start:idx_end] = tmp_intensity
        hills_mz_flat[idx_start:idx_end] = tmp_mz

    def as_float_array(field_name):
        return np.asarray([hill_feature[field_name] for hill_feature in hills_features], dtype=float_dtype)

    payload = {
        'schema_version': np.array(HILLS_NPZ_SCHEMA_VERSION, dtype=np.int32),
        'hills_float': np.array(float_name),
        'hill_idx': np.asarray([hill_feature['hill_idx'] for hill_feature in hills_features], dtype=np.int64),
        'nScans': np.asarray([hill_feature['nScans'] for hill_feature in hills_features], dtype=np.int32),
        'mz': as_float_array('mz'),
        'rtApex': as_float_array('rtApex'),
        'intensityApex': as_float_array('intensityApex'),
        'intensitySum': as_float_array('intensitySum'),
        'rtStart': as_float_array('rtStart'),
        'rtEnd': as_float_array('rtEnd'),
        'FAIMS': as_float_array('FAIMS'),
        'im': as_float_array('im'),
        'point_offsets': point_offsets,
        'hills_scan_flat': hills_scan_flat,
        'hills_intensity_flat': hills_intensity_flat,
        'hills_mz_flat': hills_mz_flat,
    }

    return payload


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

    row_count = int(np.asarray(payload['hill_idx']).shape[0])
    for key in HILLS_NPZ_FIXED_KEYS:
        if int(np.asarray(payload[key]).shape[0]) != row_count:
            raise ValueError('Invalid hills NPZ file %s: key %s has inconsistent row count.' % (source_path, key))

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
    flat_mz = np.asarray(payload['hills_mz_flat'])
    point_count = int(flat_scan.shape[0])

    if int(flat_intensity.shape[0]) != point_count or int(flat_mz.shape[0]) != point_count:
        raise ValueError('Invalid hills NPZ file %s: flattened point arrays must have identical size.' % (source_path, ))
    if int(point_offsets[-1]) != point_count:
        raise ValueError('Invalid hills NPZ file %s: point_offsets end does not match flattened arrays.' % (source_path, ))

    nscans = np.asarray(payload['nScans'], dtype=np.int64)
    if not np.array_equal(np.diff(point_offsets).astype(np.int64), nscans):
        raise ValueError('Invalid hills NPZ file %s: nScans does not match point_offsets.' % (source_path, ))

    hills_float = str(np.asarray(payload['hills_float']).reshape(-1)[0])
    if hills_float not in ('float32', 'float64'):
        raise ValueError('Invalid hills NPZ file %s: hills_float must be float32 or float64.' % (source_path, ))


def _load_hills_npz_payload(npz_path):
    with np.load(npz_path, allow_pickle=False) as npz_data:
        payload = {key: npz_data[key] for key in npz_data.files}
    _validate_hills_npz_payload(payload, npz_path)
    return payload


def _merge_hills_npz_payload(existing_payload, new_payload):
    merged_payload = {
        'schema_version': existing_payload['schema_version'],
        'hills_float': existing_payload['hills_float'],
    }

    for key in HILLS_NPZ_FIXED_KEYS:
        merged_payload[key] = np.concatenate((np.asarray(existing_payload[key]), np.asarray(new_payload[key])))

    merged_payload['hills_scan_flat'] = np.concatenate(
        (np.asarray(existing_payload['hills_scan_flat']), np.asarray(new_payload['hills_scan_flat']))
    )
    merged_payload['hills_intensity_flat'] = np.concatenate(
        (np.asarray(existing_payload['hills_intensity_flat']), np.asarray(new_payload['hills_intensity_flat']))
    )
    merged_payload['hills_mz_flat'] = np.concatenate(
        (np.asarray(existing_payload['hills_mz_flat']), np.asarray(new_payload['hills_mz_flat']))
    )

    existing_offsets = np.asarray(existing_payload['point_offsets'])
    new_offsets = np.asarray(new_payload['point_offsets'])
    if new_offsets.size > 1:
        shifted_new_offsets = new_offsets[1:] + existing_offsets[-1]
        merged_payload['point_offsets'] = np.concatenate((existing_offsets, shifted_new_offsets))
    else:
        merged_payload['point_offsets'] = existing_offsets.copy()

    return merged_payload


def write_hills_npz(hills_features, output_file, write_header, args):
    float_dtype = _get_hills_float_dtype(args)
    float_name = _get_hills_float_name(args)

    new_payload = _build_hills_npz_payload(hills_features, float_dtype, float_name)
    if not write_header and path.exists(output_file):
        existing_payload = _load_hills_npz_payload(output_file)
        existing_float = str(np.asarray(existing_payload['hills_float']).reshape(-1)[0])
        if existing_float != float_name:
            raise ValueError(
                'Existing hills NPZ file uses %s precision, but current run requested %s.'
                % (existing_float, float_name)
            )
        payload = _merge_hills_npz_payload(existing_payload, new_payload)
    else:
        payload = new_payload

    np.savez_compressed(output_file, **payload)


def get_hills_features_from_hills_npz(npz_path):
    payload = _load_hills_npz_payload(npz_path)
    row_count = int(np.asarray(payload['hill_idx']).shape[0])
    point_offsets = np.asarray(payload['point_offsets'], dtype=np.int64)
    hills_scan_flat = np.asarray(payload['hills_scan_flat'])
    hills_intensity_flat = np.asarray(payload['hills_intensity_flat'])
    hills_mz_flat = np.asarray(payload['hills_mz_flat'])

    hills_scan_lists = []
    hills_intensity_list = []
    hills_mz_array = []
    for idx_1 in range(row_count):
        idx_start = point_offsets[idx_1]
        idx_end = point_offsets[idx_1+1]
        hills_scan_lists.append(hills_scan_flat[idx_start:idx_end].astype(np.int64).tolist())
        hills_intensity_list.append(hills_intensity_flat[idx_start:idx_end].astype(float).tolist())
        hills_mz_array.append(hills_mz_flat[idx_start:idx_end].astype(float).tolist())

    return {
        'rtApex': np.asarray(payload['rtApex']),
        'intensityApex': np.asarray(payload['intensityApex']),
        'intensitySum': np.asarray(payload['intensitySum']),
        'nScans': np.asarray(payload['nScans']),
        'mz': np.asarray(payload['mz']),
        'rtStart': np.asarray(payload['rtStart']),
        'rtEnd': np.asarray(payload['rtEnd']),
        'FAIMS': np.asarray(payload['FAIMS']),
        'im': np.asarray(payload['im']),
        'hill_idx': np.asarray(payload['hill_idx']),
        'hills_scan_lists': hills_scan_lists,
        'hills_intensity_list': hills_intensity_list,
        'hills_mz_array': hills_mz_array,
    }





class MS1OnlyMzML(mzml.MzML): 
     _default_iter_path = '//spectrum[./*[local-name()="cvParam" and @name="ms level" and @value="1"]]' 
     _use_index = False 
     _iterative = False

def noisygaus(x, a, x0, sigma, b):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + b

def calibrate_mass(bwidth, mass_left, mass_right, true_md):

    bbins = np.arange(-mass_left, mass_right, bwidth)
    H1, b1 = np.histogram(true_md, bins=bbins)
    b1 = b1 + bwidth
    b1 = b1[:-1]

    popt, pcov = curve_fit(noisygaus, b1, H1, p0=[1, np.median(true_md), 1, 1])
    mass_shift, mass_sigma = popt[1], abs(popt[2])
    return mass_shift, mass_sigma, pcov[0][0]

def masklist(ar1, mask1):
    return [a for a,b in zip(ar1, mask1) if b]

def filter_hills(hills_dict, ready_set, hill_mass_accuracy, paseftol):
    idx_to_keep = [hid not in ready_set for hid in hills_dict['hills_idx_array']]
    hills_dict2 = dict()

    hills_dict2['hills_idx_array'] = masklist(list(hills_dict['hills_idx_array']), idx_to_keep)
    hills_dict2['orig_idx_array'] = masklist(list(hills_dict['orig_idx_array']), idx_to_keep)
    hills_dict2['scan_idx_array'] = masklist(list(hills_dict['scan_idx_array']), idx_to_keep)
    hills_dict2['mzs_array'] = masklist(list(hills_dict['mzs_array']), idx_to_keep)
    hills_dict2['intensity_array'] = masklist(list(hills_dict['intensity_array']), idx_to_keep)
    if 'im_array' in hills_dict:
        hills_dict2['im_array'] = masklist(list(hills_dict['im_array']), idx_to_keep)
    return hills_dict2

def get_hills_dict_from_hills_features(hills_features, hill_mass_accuracy, paseftol):
    hills_scan_lists = _parse_ragged_column(hills_features['hills_scan_lists'])
    hills_intensity_array = _parse_ragged_column(hills_features['hills_intensity_list'])

    hills_im_median = hills_features['im'].values if 'im' in hills_features else None
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
    )
    return hills_dict, mz_step


def process_hills_extra(hills_dict, RT_dict, faims_val, data_start_id, mz_step, paseftol):

    hills_features = []
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
        hill_feature['FAIMS'] = faims_val
        if 'hills_im_median' in hills_dict:
            hill_feature['im'] = hills_dict['hills_im_median'][idx_1]
        else:
            hill_feature['im'] = 0
        hill_feature['hill_idx'] = hills_dict['hills_idx_array_unique'][idx_1]
        hill_feature['hills_scan_lists'] = hills_dict['hills_scan_lists'][idx_1]
        hill_feature['hills_intensity_list'] = hills_dict['hills_intensity_array'][idx_1]
        hill_feature['hills_mz_array'] = hills_dict['tmp_mz_array'][idx_1]
        hills_features.append(hill_feature)

    return hills_dict, hills_features


def calc_peptide_features(hills_dict, peptide_features, negative_mode, faims_val, RT_dict, data_start_id, isotopes_for_intensity):

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

        pep_feature['mono_hills_scan_lists'] = hills_dict['hills_scan_lists'][pep_feature['monoisotope idx']]
        pep_feature['mono_hills_intensity_list'] =  hills_dict['hills_intensity_array'][pep_feature['monoisotope idx']]

    return peptide_features


def write_output(peptide_features, args, write_header=True, hills=False):
    output_file = _get_output_file(args, hills=hills)

    if hills and args.get('hills_format', 'tsv') == 'npz':
        write_hills_npz(peptide_features, output_file, write_header, args)
        return

    if hills:

        columns_for_output = [
            'rtApex',
            'intensityApex',
            'intensitySum',
            'nScans',
            'mz',
            'rtStart',
            'rtEnd',
            'FAIMS',
            'im',
        ]
        # if args['write_extra_details']:
        columns_for_output += ['hill_idx', 'hills_scan_lists', 'hills_intensity_list', 'hills_mz_array']
    else:
        columns_for_output = [
            'massCalib',
            'rtApex',
            'intensityApex',
            'intensitySum',
            'charge',
            'nIsotopes',
            'nScans',
            'mz',
            'rtStart',
            'rtEnd',
            'FAIMS',
            'im',
            'mono_hills_scan_lists',
            'mono_hills_intensity_list',
            'scanApex',
            'isoerror',
            'isoerror2',
        ]
        if args['write_extra_details']:
            columns_for_output += ['isoerror','isotopes','intensity_array_for_cos_corr','monoisotope hill idx','monoisotope idx']

    if write_header:

        out_file = open(output_file, 'w')
        out_file.write('\t'.join(columns_for_output) + '\n')
        out_file.close()

    out_file = open(output_file, 'a')
    for pep_feature in peptide_features:
        out_file.write('\t'.join([str(pep_feature[col]) for col in columns_for_output]) + '\n')

    out_file.close()


def centroid_pasef_data(data_for_analyse_tmp, args, mz_step):

    cnt_ms1_scans = len(data_for_analyse_tmp)

    ion_mobility_accuracy = args['paseftol']
    hill_mz_accuracy = args['htol']
    pasefmini = args['pasefmini']
    pasefminlh = args['pasefminlh']
    for spec_idx, z in enumerate(data_for_analyse_tmp):

        logger.debug('PASEF scans analysis: %d/%d', spec_idx+1, cnt_ms1_scans)
        logger.debug('number of m/z peaks in scan: %d', len(z['m/z array']))

        if 'ignore_ion_mobility' not in z:

            # mz_ar_new = []
            # intensity_ar_new = []
            # ion_mobility_ar_new = []

            # mz_ar = z['m/z array']
            # intensity_ar = z['intensity array']
            # ion_mobility_ar = z['mean inverse reduced ion mobility array']

            # ion_mobility_step = max(ion_mobility_ar) * ion_mobility_accuracy

            # ion_mobility_ar_fast = (ion_mobility_ar/ion_mobility_step).astype(int)
            # mz_ar_fast = (mz_ar/mz_step).astype(int)

            # idx = np.argsort(mz_ar_fast)
            # mz_ar_fast = mz_ar_fast[idx]
            # ion_mobility_ar_fast = ion_mobility_ar_fast[idx]

            # mz_ar = mz_ar[idx]
            # intensity_ar = intensity_ar[idx]
            # ion_mobility_ar = ion_mobility_ar[idx]

            # max_peak_idx = len(mz_ar)

            # banned_idx = set()

            # peak_idx = 0
            # while peak_idx < max_peak_idx:

            #     if peak_idx not in banned_idx:

            #         mass_accuracy_cur = mz_ar[peak_idx] * 1e-6 * hill_mz_accuracy

            #         mz_val_int = mz_ar_fast[peak_idx]
            #         ion_mob_val_int = ion_mobility_ar_fast[peak_idx]

            #         tmp = [peak_idx, ]

            #         peak_idx_2 = peak_idx + 1

            #         while peak_idx_2 < max_peak_idx:


            #             if peak_idx_2 not in banned_idx:

            #                 mz_val_int_2 = mz_ar_fast[peak_idx_2]
            #                 if mz_val_int_2 - mz_val_int > 1:
            #                     break
            #                 elif abs(mz_ar[peak_idx]-mz_ar[peak_idx_2]) <= mass_accuracy_cur:
            #                     ion_mob_val_int_2 = ion_mobility_ar_fast[peak_idx_2]
            #                     if abs(ion_mob_val_int - ion_mob_val_int_2) <= 1:
            #                         if abs(ion_mobility_ar[peak_idx] - ion_mobility_ar[peak_idx_2]) <= ion_mobility_accuracy:
            #                             tmp.append(peak_idx_2)
            #                             peak_idx = peak_idx_2
            #             peak_idx_2 += 1

            #     all_intensity = [intensity_ar[p_id] for p_id in tmp]
            #     i_val_new = sum(all_intensity)

            #     if i_val_new >= pasefmini and len(all_intensity) >= pasefminlh:

            #         all_mz = [mz_ar[p_id] for p_id in tmp]
            #         all_ion_mob = [ion_mobility_ar[p_id] for p_id in tmp]

            #         mz_val_new = np.average(all_mz, weights=all_intensity)
            #         ion_mob_new = np.average(all_ion_mob, weights=all_intensity)

            #         intensity_ar_new.append(i_val_new)
            #         mz_ar_new.append(mz_val_new)
            #         ion_mobility_ar_new.append(ion_mob_new)

            #         banned_idx.update(tmp)

            #     peak_idx += 1

            mz_ar_new, intensity_ar_new, ion_mobility_ar_new = centroid_pasef_scan(z, mz_step, hill_mz_accuracy, ion_mobility_accuracy, pasefmini, pasefminlh)

            data_for_analyse_tmp[spec_idx]['m/z array'] = np.array(mz_ar_new)
            data_for_analyse_tmp[spec_idx]['intensity array'] = np.array(intensity_ar_new)
            data_for_analyse_tmp[spec_idx]['mean inverse reduced ion mobility array'] = np.array(ion_mobility_ar_new)

        logger.debug('number of m/z peaks in scan after centroiding: %d', len(data_for_analyse_tmp[spec_idx]['m/z array']))

    data_for_analyse_tmp = [z for z in data_for_analyse_tmp if len(z['m/z array'] > 0)]
    logger.info('Number of MS1 scans after combining ion mobility peaks: %d', len(data_for_analyse_tmp))

    return data_for_analyse_tmp

def process_profile(data_for_analyse_tmp):

    data_for_analyse_tmp_out = []

    for z in data_for_analyse_tmp:

        best_mz = 0
        best_int = 0
        best_im = 0
        prev_mz = False
        prev_int = False

        threshold = 0.05

        ar1 = []
        ar2 = []
        ar3 = []
        for mzv, intv, imv in zip(z['m/z array'], z['intensity array'], z['mean inverse reduced ion mobility array']):
            if prev_mz is False:
                best_mz = mzv
                best_int = intv
                best_im = imv
            elif mzv - prev_mz > threshold:
                ar1.append(best_mz)
                ar2.append(best_int)
                ar3.append(best_im)
                best_mz = mzv
                best_int = intv
                best_im = imv
            elif best_int > prev_int and intv > prev_int:
                ar1.append(best_mz)
                ar2.append(best_int)
                ar3.append(best_im)
                best_mz = mzv
                best_int = intv
                best_im = imv
            elif intv > best_int:
                best_mz = mzv
                best_int = intv
                best_im = imv
            prev_mz = mzv
            prev_int = intv

        ar1.append(best_mz)
        ar2.append(best_int)
        ar3.append(best_im)

        z['m/z array'] = np.array(ar1)
        z['intensity array'] = np.array(ar2)
        z['mean inverse reduced ion mobility array'] = np.array(ar3)

        data_for_analyse_tmp_out.append(z)
    return data_for_analyse_tmp_out



def process_tof(data_for_analyse_tmp):

            # print(len(z['m/z array']))
    universal_dict = {}
    cnt = 0


    temp_i = defaultdict(list)
    for z in data_for_analyse_tmp:
        cnt += 1
        fast_set = z['m/z array'] // 50

        if cnt <= 25:



            for l in set(fast_set):

                if l not in universal_dict:

                    idxt = fast_set == l
                    true_i = np.log10(z['intensity array'])[idxt]
                    temp_i[l].extend(true_i)

                    if len(temp_i[l]) > 150:

                        temp_i[l] = np.array(temp_i[l])
                        i_left = temp_i[l].min()
                        i_right = temp_i[l].max()

                        i_shift, i_sigma, covvalue = calibrate_mass(0.05, i_left, i_right, temp_i[l])
                        # median_val = 
                        print(i_shift, i_sigma, covvalue)
                        universal_dict[l] = 10**(i_shift + 2 * i_sigma)#10**(np.median(true_i[idxt]) * 2)
            

    cnt = 0

    for z in data_for_analyse_tmp:

        fast_set = z['m/z array'] // 50
        while cnt <= 50:

            cnt += 1

            temp_i = []

            for l in set(fast_set):
                idxt = fast_set == l
                true_i = np.log10(z['intensity array'])[idxt]
                temp_i.extend(true_i)

                if len(true_i) > 150:

                    i_left = true_i.min()
                    i_right = true_i.max()

                    i_shift, i_sigma, covvalue = calibrate_mass(0.05, i_left, i_right, true_i)
                    # median_val = 
                    print(i_shift, i_sigma, covvalue)
                    universal_dict[l] = 10**(i_shift + 3 * i_sigma)#10**(np.median(true_i[idxt]) * 2)
            

            
        thresholds = [universal_dict.get(zz, 150) for zz in list(fast_set)]
        idxt2 = z['intensity array'] <= thresholds
        z['intensity array'][idxt2] = -1


        idx = z['intensity array'] > 0
        z['intensity array'] = z['intensity array'][idx]
        z['m/z array'] = z['m/z array'][idx]
        z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]



        cnt += 1

        data_for_analyse_tmp = [z for z in data_for_analyse_tmp if len(z['m/z array'])]

    return data_for_analyse_tmp


def process_mzml(args):

    input_mzml_path = args['file']
    min_intensity = args['mini']
    min_mz = args['minmz']
    max_mz = args['maxmz']

    skipped = 0
    data_for_analyse = []

    cnt = 0
    combine_every = args["combine_every"]
    assert (
        isinstance(combine_every, int) and combine_every > 0
    ), "combine_every must be a positive integer"
    if combine_every > 1:
        logger.info("Combining every %s MS1 scans.", combine_every)
    buffer = []  # temporary storage for z's to be merged

    for z in MS1OnlyMzML(source=input_mzml_path):
        if z['ms level'] == 1:

            if 'raw ion mobility array' in z:
                z['mean inverse reduced ion mobility array'] = z['raw ion mobility array']

            if 'mean inverse reduced ion mobility array' not in z:
                z['ignore_ion_mobility'] = True
                z['mean inverse reduced ion mobility array'] = np.zeros(len(z['m/z array']))

            # intensity filter
            idx = z['intensity array'] >= min_intensity
            z['intensity array'] = z['intensity array'][idx]
            z['m/z array'] = z['m/z array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            # min_mz filter
            idx = z['m/z array'] >= min_mz
            z['m/z array'] = z['m/z array'][idx]
            z['intensity array'] = z['intensity array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            # max_mz filter
            idx = z['m/z array'] <= max_mz
            z['m/z array'] = z['m/z array'][idx]
            z['intensity array'] = z['intensity array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            # sort by m/z
            idx = np.argsort(z['m/z array'])
            z['m/z array'] = z['m/z array'][idx]
            z['intensity array'] = z['intensity array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            cnt += 1

            if combine_every == 1:
                # just append z directly
                if len(z['m/z array']):
                    data_for_analyse.append(z)
                else:
                    skipped += 1
            else:
                # store in buffer and only merge when reaching combine_every
                buffer.append(z)
                if len(buffer) == combine_every:
                    merged = {
                        "m/z array": np.concatenate([b["m/z array"] for b in buffer]),
                        "intensity array": np.concatenate(
                            [b["intensity array"] for b in buffer]
                        ),
                        "mean inverse reduced ion mobility array": np.concatenate(
                            [
                                b["mean inverse reduced ion mobility array"]
                                for b in buffer
                            ]
                        ),
                    }
                    merged.update(
                        {k: buffer[0][k] for k in buffer[0] if k not in merged}
                    )

                    if len(merged["m/z array"]):
                        data_for_analyse.append(merged)
                        if cnt % 5000 == 0 and logger.level == logging.DEBUG:
                            logger.debug(
                                "m/z array start and end for scan %s after merged: %s - %s",
                                cnt,
                                merged["m/z array"][0],
                                merged["m/z array"][-1],
                            )
                    else:
                        skipped += 1
                    buffer = []

    # handle leftover spectra if not divisible by combine_every
    if buffer:
        logger.info("Combining %s leftover MS1 scans..", len(buffer))
        merged = {
            "m/z array": np.concatenate([b["m/z array"] for b in buffer]),
            "intensity array": np.concatenate([b["intensity array"] for b in buffer]),
            "mean inverse reduced ion mobility array": np.concatenate(
                [b["mean inverse reduced ion mobility array"] for b in buffer]
            ),
        }
        merged.update({k: buffer[0][k] for k in buffer[0] if k not in merged})

        if len(merged["m/z array"]):
            data_for_analyse.append(merged)
        else:
            skipped += 1


    logger.info('Number of MS1 scans: %d', len(data_for_analyse))
    logger.info('Number of skipped MS1 scans: %d', skipped)

    if len(data_for_analyse) == 0:
        raise Exception('no MS1 scans in input file')

    return data_for_analyse



def process_mzml_dia(args):

    input_mzml_path = args['file']
    # min_intensity = args['mini']
    # min_mz = args['minmz']
    # max_mz = args['maxmz']
    min_intensity = 0
    min_mz = 1
    max_mz = 1e6

    skipped = 0
    data_for_analyse = []

    cnt = 0
    ms1_scans = 0

    for z in mzml.read(input_mzml_path):
        if z['ms level'] == 1:
            ms1_scans += 1
        elif z['ms level'] == 2:

            if 'mean inverse reduced ion mobility array' not in z:
                z['ignore_ion_mobility'] = True
                z['mean inverse reduced ion mobility array'] = np.zeros(len(z['m/z array']))

            idx = z['intensity array'] >= min_intensity
            z['intensity array'] = z['intensity array'][idx]
            z['m/z array'] = z['m/z array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            idx = z['m/z array'] >= min_mz
            z['m/z array'] = z['m/z array'][idx]
            z['intensity array'] = z['intensity array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            idx = z['m/z array'] <= max_mz
            z['m/z array'] = z['m/z array'][idx]
            z['intensity array'] = z['intensity array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            idx = np.argsort(z['m/z array'])
            z['m/z array'] = z['m/z array'][idx]
            z['intensity array'] = z['intensity array'][idx]
            z['mean inverse reduced ion mobility array'] = z['mean inverse reduced ion mobility array'][idx]

            cnt += 1

            # if len(data_for_analyse) > 5000:
            #     break

            if len(z['m/z array']):
                data_for_analyse.append(z)
            else:
                skipped += 1


    logger.info('Number of MS2 scans: %d', len(data_for_analyse))
    logger.info('Number of skipped MS2 scans: %d', skipped)

    return data_for_analyse, ms1_scans, cnt
