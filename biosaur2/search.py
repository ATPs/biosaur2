from . import main, main_dia, main_dia2
import argparse
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version as pkg_version
import logging
import os


def _get_biosaur2_version():
    try:
        return pkg_version('biosaur2')
    except PackageNotFoundError:
        return 'unknown'


def run():
    parser = argparse.ArgumentParser(
        description='A feature detection LC-MS1 spectra',
        epilog='''

    Example usage
    -------------
    $ biosaur2 input.mzML
    $ biosaur2 input.mzML.gz
    -------------
    ''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s {}'.format(_get_biosaur2_version()),
    )
    parser.add_argument(
        'files',
        help='input files: mzML (.mzML/.mzML.gz) or hills (Experimental) (.hills.tsv/.hills.parquet/.hills.npz)',
        nargs='+',
    )
    parser.add_argument('-mini', help='min intensity', default=1, type=float)
    parser.add_argument('-minmz', help='min mz', default=350, type=float)
    parser.add_argument('-maxmz', help='max mz', default=1500, type=float)
    parser.add_argument('-pasefmini', help='min intensity after combining hills in PASEF analysis', default=100, type=float)
    parser.add_argument('-htol', help='mass accuracy for hills in ppm', default=8, type=float)
    parser.add_argument('-itol', help='mass accuracy for isotopes in ppm', default=8, type=float)
    parser.add_argument('-ignore_iso_calib', help='Turn off accurate isotope error estimation', action='store_true')
    parser.add_argument('-use_hill_calib', help='Experimental. Turn on accurate hills error estimation', action='store_true')
    parser.add_argument('-paseftol', help='ion mobility accuracy for hills', default=0.05, type=float)
    parser.add_argument('-nm', help='negative mode. 1-true, 0-false', default=0, type=int)
    parser.add_argument('-o', help='path to output features file', default='')
    parser.add_argument('-iuse', help='Number of isotopes used for intensity calucation. 0 - only mono, -1 - use all, 1 - use mono and first isotope, etc.', default=0, type=int)
    parser.add_argument('-hvf', help='Threshold to split hills into multiple if local minimum intensity multiplied by hvf is less than both surrounding local maximums', default=1.3, type=float)
    parser.add_argument('-ivf', help='Threshold to split isotope pattern into multiple features if local minimum intensity multiplied by ivf is less right local maximum', default=5.0, type=float)
    parser.add_argument('-minlh', help='minimum length for hill', default=2, type=int)
    parser.add_argument('-pasefminlh', help='minimum length for pasef hill', default=1, type=int)
    parser.add_argument('-cmin', help='min charge', default=1, type=int)
    parser.add_argument('-cmax', help='max charge', default=6, type=int)
    parser.add_argument('-nprocs', help='number of processes', default=4, type=int)
    parser.add_argument('-dia',  help='create mgf file for DIA MS/MS. Experimental', action='store_true')
    parser.add_argument('-dia2',  help='create mgf file for DIA MS/MS with no look at MS1 spectra. Experimental', action='store_true')
    parser.add_argument('-diahtol', help='mass accuracy for DIA hills in ppm', default=25, type=float)
    parser.add_argument('-diaminlh', help='minimum length for dia hill', default=1, type=int)
    parser.add_argument('-diadynrange', help='diadynrange', default=1000, type=int)
    parser.add_argument('-min_ms2_peaks', help='min_ms2_peaks', default=5, type=int)
    parser.add_argument('-mgf', help='path to output mgf file', default='')
    parser.add_argument('-debug', help='log debugging information', action='store_true')
    parser.add_argument('-tof', help='smart tof processing. Experimental', action='store_true')
    parser.add_argument('-profile', help='profile processing. Experimental', action='store_true')
    parser.add_argument('-write_hills', help='write detected hills output file (format is controlled by --hills_format)', action='store_true')
    parser.add_argument(
        '--hills_format',
        help='hills output format used by -write_hills',
        default='tsv',
        choices=['tsv', 'parquet'],
    )
    parser.add_argument(
        '--no_hill_list',
        help='for -write_hills output, do not include hills_scan_lists/hills_intensity_list/hills_mz_array (output cannot be reused for feature detection)',
        action='store_true',
    )
    parser.add_argument(
        '--write_ms1',
        help='write MS1 summary output (scan_id, RT in seconds, total_intensity)',
        action='store_true',
    )
    parser.add_argument(
        '--ms1_format',
        help='MS1 summary output format used by --write_ms1',
        default='tsv',
        choices=['tsv', 'parquet'],
    )
    parser.add_argument(
        '--feature_format',
        help='feature output format',
        default='tsv',
        choices=['tsv', 'parquet'],
    )
    parser.add_argument(
        '--no-mono-hills',
        help='do not include mono_hills_scan_lists and mono_hills_intensity_list in feature output',
        action='store_true',
    )
    parser.add_argument(
        '--64',
        dest='use64',
        help=(
            'for parquet output, store rtApex/mz/rtStart/rtEnd/FAIMS/im/hill_idx and list elements '
            'in hills_scan_lists/hills_intensity_list/hills_mz_array (and mono_hills_* lists), '
            'plus MS1 columns scan_id/RT/total_intensity, as 64-bit types; default parquet mode uses 32-bit types.'
        ),
        action='store_true',
    )
    parser.add_argument('--stop_after_hills', help='stop processing after writing hills output', action='store_true')
    parser.add_argument(
        '-write_extra_details',
        help=(
            'write additional per-feature diagnostic columns to feature output '
            '(including isotope candidate details such as isotopes, '
            'intensity_array_for_cos_corr, monoisotope hill/index IDs). '
            'This option is intended for debugging/inspection and increases output size.'
        ),
        action='store_true',
    )
    parser.add_argument('-md_correction', help='EXPERIMENTAL. Can be Orbi, Icr or Tof. Sqrt, Linear or Uniform mass error normalization, respectively.', default='Orbi')
    parser.add_argument(
        "-combine_every",
        help="combine every n ms1 scans, useful for e.g. gas phase fractionation data",
        default=1,
        type=int,
    )
    args = vars(parser.parse_args())
    if args['no_mono_hills'] and args['dia']:
        parser.error('--no-mono-hills cannot be used with -dia because DIA processing requires mono_hills_* columns.')
    forced_write_hills = args['stop_after_hills'] and not args['write_hills']
    if forced_write_hills:
        args['write_hills'] = True
    logging.basicConfig(format='%(levelname)9s: %(asctime)s %(message)s',
            datefmt='[%H:%M:%S]', level=[logging.INFO, logging.DEBUG][args['debug']])
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    if forced_write_hills:
        logger.info('--stop_after_hills requested; turning on --write_hills automatically.')
    logger.debug('Starting with args: %s', args)

    if os.name == 'nt':
        # logger.info('Turning off multiprocessing for Windows system')
        args['nprocs'] = 1

    for filename in args['files']:
        logger.info('Starting file: %s', filename)
        if 1:
            args['file'] = filename

            if args['dia2']:
                main_dia2.process_file(deepcopy(args))
            else:

                main.process_file(deepcopy(args))
                if args['stop_after_hills']:
                    logger.info('Hills extraction is finished for file: %s', filename)
                else:
                    logger.info('Feature detection is finished for file: %s', filename)
                if args['dia'] and not args['stop_after_hills']:
                    main_dia.process_file(deepcopy(args))
        
        # except Exception as e:
        #     logger.error(e)
        #     logger.error('Feature detection failed for file: %s', filename)

if __name__ == '__main__':
    run()
