from . import main, main_dia, main_dia2
import argparse
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version as pkg_version
import logging
import os
from pathlib import Path
import sys
from .output import input_stem
from .duckdb_output import DuckDBOutputManager, uses_duckdb
from .legacy_output import CompactOutputManager


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


def _get_biosaur2_version():
    try:
        return pkg_version('biosaur2')
    except PackageNotFoundError:
        return 'unknown'


def _nonnegative_or(keyword):
    def parse(value):
        if value == keyword:
            return value
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "must be %r or a nonnegative integer" % keyword
            ) from exc
        if parsed < 0:
            raise argparse.ArgumentTypeError(
                "must be %r or a nonnegative integer" % keyword
            )
        return str(parsed)

    return parse


def _positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _option_was_supplied(*spellings):
    return any(value in sys.argv[1:] for value in spellings)


def _is_hills_input(filename):
    lower = str(filename).lower()
    return lower.endswith((".hills.tsv", ".hills.parquet", ".hills.npz"))


def _create_output_manager(run_args):
    if run_args.get('dia2'):
        return None
    if run_args.get('duckdb_output'):
        try:
            return DuckDBOutputManager(run_args)
        except ImportError as exc:
            raise ImportError(
                '--duckdb-output requires DuckDB; install duckdb or biosaur2[duckdb].'
            ) from exc
    if uses_duckdb(run_args):
        try:
            return DuckDBOutputManager(run_args)
        except ImportError:
            logging.getLogger(__name__).warning(
                'DuckDB is unavailable; falling back to PyArrow Parquet. '
                'Install duckdb or biosaur2[duckdb] to use DuckDB V2.'
            )
            run_args['parquet_engine'] = 'pyarrow'
            run_args['_parquet_engine_fallback'] = 'duckdb_to_pyarrow'
    return CompactOutputManager(run_args)


def run():
    parser = argparse.ArgumentParser(
        description='Detect isotope features in centroided LC-MS1 mzML data.',
        epilog='''
Output contract:
  Plain invocation writes one compact features.tsv file. Every requested
  Parquet product uses DuckDB V2 with ZSTD level 6, including
  hills/MS1 Parquet when features remain TSV. If DuckDB is unavailable
  Biosaur2 warns and falls back to optimized PyArrow. No feature sidecars or
  manifest are produced; feature Parquet is one compact features.parquet file.

  Feature/hill rtStart, rtApex, and rtEnd are in minutes. MS1 RT is in seconds;
  area_sum is raw trapezoidal intensity * seconds across the -iuse isotope
  subset. Missing FAIMS, im, native scanApex, errors, and area_sum are null.

  Default Parquet storage uses float32, int8 charge/nIsotopes, int16 nScans,
  and int32 scan/feature_idx/list IDs. --64 widens structured numeric storage.
  Intensities are rounded half-away-from-zero to 0 decimals only at output.

  Default feature columns:
    massCalib rtApex intensityApex intensitySum charge nIsotopes nScans mz
    rtStart rtEnd FAIMS im mono_hills_scan_lists mono_hills_intensity_list
    scanApex isoerror isoerror2 feature_idx area_sum
  --no-mono-hills removes the two mono_hills arrays. --write-extra-details
  adds the original four nested diagnostic columns to the same feature file.

Examples:
  biosaur2 input.mzML.gz
  biosaur2 input.mzML.gz --feature-format parquet
  biosaur2 input.mzML.gz --feature-format parquet --no-mono-hills
  biosaur2 input.mzML.gz --duckdb-output output.biosaur2.duckdb
    ''',
        formatter_class=_HelpFormatter)

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
    parser.add_argument('-iuse', help='isotopes used for intensity and area_sum: -1=all assigned, 0=mono only, N=mono plus N isotopes', default=-1, type=int)
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
    parser.add_argument('-write_hills', '--write-hills', dest='write_hills', help='write detected hills output file (format is controlled by --hills-format)', action='store_true')
    parser.add_argument(
        '--hills_format', '--hills-format',
        dest='hills_format',
        help='hills output format used by -write_hills',
        default='tsv',
        choices=['tsv', 'parquet'],
    )
    parser.add_argument(
        '--no_hill_list', '--no-hill-list',
        dest='no_hill_list',
        help='for -write_hills output, omit all point arrays including scan, intensity, m/z, and RT (output cannot be reused for feature detection)',
        action='store_true',
    )
    parser.add_argument(
        '--write_ms1', '--write-ms1',
        dest='write_ms1',
        help='write MS1 summary output (scan_id, RT in seconds, total_intensity)',
        action='store_true',
    )
    parser.add_argument(
        '--ms1_format', '--ms1-format',
        dest='ms1_format',
        help='MS1 summary output format used by --write_ms1',
        default='tsv',
        choices=['tsv', 'parquet'],
    )
    parser.add_argument(
        '--feature_format', '--feature-format',
        dest='feature_format',
        help='feature output format; parquet writes one compact features.parquet file',
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
        help='store Parquet/DuckDB numeric payloads as 64-bit instead of compact float32 and narrow integers',
        action='store_true',
    )
    parser.add_argument('--stop_after_hills', '--stop-after-hills', dest='stop_after_hills', help='stop processing after writing hills output', action='store_true')
    parser.add_argument(
        '-write_extra_details', '--write-extra-details',
        dest='write_extra_details',
        help=(
            'write additional per-feature diagnostic columns to feature output '
            '(including isotope candidate details such as isotopes, '
            'intensity_array_for_cos_corr, monoisotope hill/index IDs). '
            'This option is intended for debugging/inspection and increases output size.'
        ),
        action='store_true',
    )
    parser.add_argument('-md_correction', help='EXPERIMENTAL. Can be Orbi, Icr or Tof. Sqrt, Linear or Uniform mass error normalization, respectively.', default='Orbi', choices=['Orbi', 'Icr', 'Tof'])
    parser.add_argument(
        "-combine_every",
        "--combine-every",
        dest="combine_every",
        help="combine every n ms1 scans, useful for e.g. gas phase fractionation data",
        default=1,
        type=int,
    )
    parser.add_argument('--input-rt-unit', '--input_rt_unit', dest='input_rt_unit', choices=['seconds', 'minutes'], default='seconds', help='fallback unit for metadata-free mzML/hills RT; mzML metadata takes precedence')
    parser.add_argument('--tsv-float-decimals', '--tsv_float_decimals', dest='tsv_float_decimals', type=_nonnegative_or('roundtrip'), default='roundtrip', help='TSV float text: shortest round-trip representation or a nonnegative decimal count')
    parser.add_argument('--parquet-engine', '--parquet_engine', dest='parquet_engine', choices=['pyarrow', 'duckdb'], default='duckdb', help='Parquet writer; DuckDB V2 is preferred, with a visible PyArrow fallback when DuckDB is unavailable')
    parser.add_argument('--parquet-compression', '--parquet_compression', dest='parquet_compression', choices=['zstd', 'snappy', 'lz4', 'brotli', 'uncompressed'], default='zstd', help='Parquet compression codec')
    parser.add_argument('--parquet-compression-level', '--parquet_compression_level', dest='parquet_compression_level', type=int, default=6, help='zstd/brotli compression level')
    parser.add_argument('--parquet-row-group-size', '--parquet_row_group_size', dest='parquet_row_group_size', type=_positive_integer, default=122880, help='positive Parquet row-group size')
    parser.add_argument('--parquet-sort', '--parquet_sort', dest='parquet_sort', choices=['none', 'mz_rt', 'rt_mz'], default='mz_rt', help='deterministic physical feature order')
    parser.add_argument('--parquet-temp-dir', '--parquet_temp_dir', dest='parquet_temp_dir', default='', help='DuckDB staging workspace; final files remain staged beside their targets')
    parser.add_argument('--intensity-decimals', '--intensity_decimals', dest='intensity_decimals', type=_nonnegative_or('none'), default='0', help='output-only half-away-from-zero intensity rounding; use none to preserve fractional values')
    parser.add_argument('--duckdb-output', '--duckdb_output', dest='duckdb_output', default='', help='write one compact .duckdb database instead of ordinary feature output')
    parser.add_argument('--overwrite', action='store_true', help='atomically replace existing output targets')
    args = vars(parser.parse_args())
    if args['nprocs'] < 1:
        parser.error('-nprocs must be a positive integer.')
    if args['combine_every'] < 1:
        parser.error('-combine_every must be a positive integer.')
    if args['combine_every'] > 1:
        parser.error('-combine_every greater than 1 is incompatible with area_sum output.')
    if args['iuse'] < -1:
        parser.error('-iuse must be -1 or a nonnegative integer.')
    if args['parquet_compression'] not in {'zstd', 'brotli'} and _option_was_supplied(
        '--parquet-compression-level', '--parquet_compression_level'
    ):
        parser.error('--parquet-compression-level is supported only by zstd and brotli.')
    parquet_requested = (
        (not args['stop_after_hills'] and args['feature_format'] == 'parquet')
        or ((args['write_hills'] or args['stop_after_hills']) and args['hills_format'] == 'parquet')
        or (args['write_ms1'] and args['ms1_format'] == 'parquet')
    )
    if _option_was_supplied('--parquet-engine', '--parquet_engine') and (
        not parquet_requested and not args['duckdb_output']
    ):
        parser.error('--parquet-engine requires at least one Parquet output.')
    if (args['dia'] or args['dia2']) and args['duckdb_output']:
        parser.error('DIA modes do not support --duckdb-output.')
    if args['no_mono_hills'] and args['dia']:
        parser.error('--no-mono-hills cannot be used with -dia because DIA processing requires mono_hills_* columns.')
    for filename in args['files']:
        if not _is_hills_input(filename):
            continue
        invalid_options = [
            spelling
            for enabled, spelling in (
                (args['stop_after_hills'], '--stop-after-hills'),
                (args['write_hills'], '--write-hills'),
                (args['write_ms1'], '--write-ms1'),
            )
            if enabled
        ]
        if invalid_options:
            parser.error(
                '%s cannot be used with hills input: %s'
                % (', '.join(invalid_options), filename)
            )
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

    multiple_inputs = len(args['files']) > 1
    if multiple_inputs:
        stems = [input_stem(filename) for filename in args['files']]
        if len(stems) != len(set(stems)):
            parser.error('Multiple inputs resolve to duplicate output stems.')
        if args['o']:
            output_directory = Path(args['o'])
            if output_directory.exists() and not output_directory.is_dir():
                parser.error('-o must be a directory when multiple inputs are supplied.')
            output_directory.mkdir(parents=True, exist_ok=True)
        if args['duckdb_output']:
            duckdb_directory = Path(args['duckdb_output'])
            if duckdb_directory.suffix.lower() == '.duckdb':
                parser.error('--duckdb-output must be a directory for multiple inputs.')
            if duckdb_directory.exists() and not duckdb_directory.is_dir():
                parser.error('--duckdb-output must be a directory for multiple inputs.')
            duckdb_directory.mkdir(parents=True, exist_ok=True)

    jobs = []
    try:
        for filename in args['files']:
            run_args = deepcopy(args)
            run_args['file'] = filename
            if multiple_inputs and run_args['o']:
                extension = (
                    'parquet'
                    if run_args.get('feature_format') == 'parquet'
                    else 'tsv'
                )
                run_args['o'] = str(
                    Path(run_args['o'])
                    / ('%s.features.%s' % (input_stem(filename), extension))
                )
            manager = _create_output_manager(run_args)
            if manager is not None:
                run_args['_output_manager'] = manager
            jobs.append((filename, run_args, manager))
    except BaseException:
        for _filename, _run_args, manager in jobs:
            if manager is not None:
                manager.abort()
        raise

    for filename, run_args, manager in jobs:
        logger.info('Starting file: %s', filename)
        if 1:
            try:
                if run_args['dia2']:
                    main_dia2.process_file(run_args)
                else:
                    main.process_file(run_args)
                if manager is not None:
                    manager.finalize()
            except BaseException:
                if manager is not None:
                    manager.abort()
                raise
            if not run_args['dia2']:
                if args['stop_after_hills']:
                    logger.info('Hills extraction is finished for file: %s', filename)
                else:
                    logger.info('Feature detection is finished for file: %s', filename)
                if args['dia'] and not args['stop_after_hills']:
                    dia_args = {
                        key: deepcopy(value)
                        for key, value in run_args.items()
                        if key != '_output_manager'
                    }
                    main_dia.process_file(dia_args)
        
        # except Exception as e:
        #     logger.error(e)
        #     logger.error('Feature detection failed for file: %s', filename)

if __name__ == '__main__':
    run()
