from . import main, main_dia, main_dia2
import argparse
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version as pkg_version
import logging
import math
import os
from pathlib import Path
import sys
import time
import traceback
from .output import input_stem
from .output import planned_output_paths
from .duckdb_output import DuckDBOutputManager, uses_duckdb
from .legacy_output import CompactOutputManager
from .parallel import WorkerFailure, run_bounded_process_tasks


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    def _get_help_string(self, action):
        help_text = action.help or ''
        if (
            '%(default)' in help_text
            or '(default:' in help_text
            or action.default is argparse.SUPPRESS
        ):
            return help_text
        if not action.option_strings and action.nargs not in ('?', '*'):
            return help_text
        default = action.default
        if default == '':
            default = 'empty'
        elif default is None or default == []:
            default = 'none'
        return '%s (default: %s)' % (help_text, default)


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


def _is_mzml_input(filename):
    lower = str(filename).lower()
    return lower.endswith((".mzml", ".mzml.gz"))


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


def _run_file_worker(run_args):
    """Process one file and return only a compact parent-safe status record."""

    started = time.monotonic()
    manager = None
    filename = run_args["file"]
    try:
        manager = _create_output_manager(run_args)
        if manager is not None:
            run_args["_output_manager"] = manager
        if run_args["dia2"]:
            main_dia2.process_file(run_args)
        else:
            main.process_file(run_args)
        if manager is not None:
            manager.finalize()
            manager = None
        if run_args["dia"] and not run_args["stop_after_hills"]:
            dia_args = {
                key: deepcopy(value)
                for key, value in run_args.items()
                if key != "_output_manager"
            }
            main_dia.process_file(dia_args)
        return {
            "file": filename,
            "status": "success",
            "runtime_sec": time.monotonic() - started,
            "error": None,
            "traceback": None,
        }
    except BaseException as exc:
        if manager is not None:
            manager.abort()
        return {
            "file": filename,
            "status": "failed",
            "runtime_sec": time.monotonic() - started,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }


def _run_args_for_file(args, filename, multiple_inputs):
    run_args = {
        key: value
        for key, value in args.items()
        if key not in {"files", "continue_on_error"}
    }
    run_args["file"] = filename
    if multiple_inputs and run_args["o"]:
        extension = "parquet" if run_args.get("feature_format") == "parquet" else "tsv"
        output_directory = Path(run_args["o"])
        run_args["o"] = str(
            output_directory / ("%s.features.%s" % (input_stem(filename), extension))
        )
        if not run_args.get("duckdb_output"):
            run_args["_ms2_output_directory"] = str(output_directory)
    run_args["_multiple_inputs"] = multiple_inputs
    run_args["_requested_nprocs"] = args["nprocs"]
    if args["run_workers"] > 1:
        run_args["nprocs"] = 1
    return run_args


def _log_batch_report(logger, files, results):
    logger.info("Batch report: file\tstatus\truntime_sec\terror")
    for index, filename in enumerate(files):
        result = results.get(index)
        if result is None:
            result = {
                "status": "not_run",
                "runtime_sec": None,
                "error": None,
                "traceback": None,
            }
        runtime = result["runtime_sec"]
        logger.info(
            "%s\t%s\t%s\t%s",
            filename,
            result["status"],
            "" if runtime is None else "%.3f" % runtime,
            result["error"] or "",
        )
        if result.get("traceback"):
            logger.error("File failed: %s\n%s", filename, result["traceback"])


def run():
    if len(sys.argv) > 1 and sys.argv[1] == 'project':
        from .project_cli import run_project_cli

        return run_project_cli(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'build-manifest':
        from .project_cli import run_build_manifest_alias

        return run_build_manifest_alias(sys.argv[2:])
    parser = argparse.ArgumentParser(
        description=(
            'Detect and quantify isotope features in centroided LC-MS1 mzML '
            'data. Legacy detection is the default; use --feature-mode hybrid '
            'for identification/MS2-guided residual feature detection.'
        ),
        epilog='''
Input notes:
  mzML/mzML.gz input should contain centroided MS1 spectra. Hybrid mode also
  reads DDA MS2 precursor metadata and optionally a same-run Percolator target
  PSM table. Profile, DIA/DIA2, and reusable hills paths are experimental.

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
  # Ordinary untargeted LC-MS1 feature detection (default legacy mode)
  biosaur2 input.mzML.gz

  # Compact Parquet feature output for downstream analysis
  biosaur2 input.mzML.gz --feature-format parquet

  # Hybrid DDA without PSMs: strict features plus generic MS2 evidence
  biosaur2 input.mzML.gz --feature-mode hybrid --feature-format parquet

  # Hybrid DDA with q-filtered same-run Percolator PSMs
  biosaur2 input.mzML.gz --feature-mode hybrid \\
    --psm-path input.percolator.target.psms.tsv \\
    --psm-q-value-max 0.01 --fixed-mod C=UNIMOD:4 \\
    --quant-method envelope_area --feature-format parquet

  # Repeated hybrid tuning: raw cache is required by strict-stage cache
  biosaur2 input.mzML.gz --feature-mode hybrid \\
    --raw-ms1-cache-dir cache/input.raw \\
    --hybrid-stage-cache-dir cache/input.strict \\
    --hybrid-candidate-cache-dir cache/input.candidates \\
    --feature-format parquet

  # Smaller feature payload when point arrays are not needed
  biosaur2 input.mzML.gz --feature-format parquet --no-mono-hills

  # Put requested ordinary products in one DuckDB database
  biosaur2 input.mzML.gz --duckdb-output output.biosaur2.duckdb

Use `biosaur2 project --help` for manifest-driven multi-run processing.
See README.md for input examples and option guidance, design.md for the
algorithm, and updates/2026-07-30.md for validation results.
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
    parser.add_argument('-mini', help='minimum centroid intensity considered during hill detection', default=1, type=float)
    parser.add_argument('-minmz', help='lower m/z boundary for MS1 feature detection', default=350, type=float)
    parser.add_argument('-maxmz', help='upper m/z boundary for MS1 feature detection', default=1500, type=float)
    parser.add_argument('-pasefmini', help='minimum intensity after combining PASEF/ion-mobility hills', default=100, type=float)
    parser.add_argument('-htol', help='maximum point-to-hill m/z deviation in ppm', default=8, type=float)
    parser.add_argument('-itol', help='maximum isotope-envelope mass deviation in ppm', default=8, type=float)
    parser.add_argument('-ignore_iso_calib', help='disable run-specific isotope mass-error calibration and use configured tolerances', action='store_true')
    parser.add_argument('-use_hill_calib', help='enable experimental run-specific hill mass-error calibration', action='store_true')
    parser.add_argument('-paseftol', help='1/K0 ion-mobility tolerance used to connect PASEF hill points', default=0.05, type=float)
    parser.add_argument('-nm', help='ionization polarity selector: 0=positive, 1=negative', default=0, type=int)
    parser.add_argument('-o', help='single-input feature path; with multiple inputs this must be an output directory; empty writes beside the input', default='')
    parser.add_argument('-iuse', help='isotopes used for intensity and area_sum: -1=all assigned, 0=mono only, N=mono plus N isotopes', default=-1, type=int)
    parser.add_argument('-hvf', help='Threshold to split hills into multiple if local minimum intensity multiplied by hvf is less than both surrounding local maximums', default=1.3, type=float)
    parser.add_argument('-ivf', help='Threshold to split isotope pattern into multiple features if local minimum intensity multiplied by ivf is less right local maximum', default=5.0, type=float)
    parser.add_argument('-minlh', help='minimum number of MS1 points in an ordinary hill', default=2, type=int)
    parser.add_argument('-pasefminlh', help='minimum number of points in a PASEF hill', default=1, type=int)
    parser.add_argument('-cmin', help='minimum feature/precursor charge hypothesis', default=1, type=int)
    parser.add_argument(
        '-cmax', '--max-charge', '--max_charge', dest='cmax',
        help='maximum feature/precursor charge hypothesis',
        default=7,
        type=int,
    )
    parser.add_argument('-nprocs', help='internal worker-process count for one file', default=4, type=int)
    parser.add_argument(
        '--run-workers',
        help='bounded file-level worker count; each file uses one detection worker when greater than one',
        default=1,
        type=_positive_integer,
    )
    parser.add_argument(
        '--continue-on-error',
        help='continue scheduling independent input files after a failed file',
        action='store_true',
    )
    parser.add_argument('-dia',  help='create mgf file for DIA MS/MS. Experimental', action='store_true')
    parser.add_argument('-dia2',  help='create mgf file for DIA MS/MS with no look at MS1 spectra. Experimental', action='store_true')
    parser.add_argument('-diahtol', help='experimental DIA hill mass tolerance in ppm', default=25, type=float)
    parser.add_argument('-diaminlh', help='experimental minimum DIA hill length', default=1, type=int)
    parser.add_argument('-diadynrange', help='experimental DIA dynamic-range limit', default=1000, type=int)
    parser.add_argument('-min_ms2_peaks', help='minimum fragment-peak count for an experimental MGF entry', default=5, type=int)
    parser.add_argument('-mgf', help='experimental DIA/DIA2 MGF output path; empty derives it from the input', default='')
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
        '--write-ms2',
        help='write compact precursor-only <input-stem>.ms2.parquet sidecar',
        action='store_true',
    )
    parser.add_argument(
        '--feature-mode',
        choices=('legacy', 'hybrid'),
        default='legacy',
        help='legacy=strict untargeted; hybrid=direct/generic MS2 residual workflow',
    )
    parser.add_argument('--psm-path', default='', help='same-run Percolator target PSM TSV (optionally compressed); empty runs hybrid without direct PSM assays')
    parser.add_argument('--psm-q-value-max', type=float, default=0.01, help='maximum Percolator PSM q-value accepted before direct-assay construction')
    parser.add_argument('--psm-pep-max', type=float, default=None, help='optional maximum PSM posterior error probability; none disables the additional PEP filter')
    parser.add_argument('--fixed-mod', action='append', default=[], help='explicit hybrid fixed modification SITE=MOD; repeatable')
    parser.add_argument(
        '--quant-method',
        choices=('envelope_area', 'mono_area', 'envelope_apex'),
        default='envelope_area',
        help='hybrid feature abundance: assigned envelope area, monoisotopic area, or common-scan envelope apex',
    )
    parser.add_argument(
        '--feature-baseline', choices=('none', 'edge_linear'), default=None,
        help=(
            'hybrid quantification baseline preprocessing '
            '(default: automatic; edge_linear in hybrid mode, none otherwise)'
        ),
    )
    parser.add_argument(
        '--direct-id', action=argparse.BooleanOptionalAction, default=True,
        help='enable/disable q-filtered same-run PSM assay association and local recovery in hybrid mode',
    )
    parser.add_argument(
        '--external-id', action=argparse.BooleanOptionalAction, default=True,
        help='project compatibility switch for aligned external assays; single-run commands have no donor runs',
    )
    parser.add_argument(
        '--generic-ms2-refine', action=argparse.BooleanOptionalAction, default=True,
        help='enable/disable unidentified-MS2 charge/C13 hypotheses, target-decoy association, and local recovery in hybrid mode',
    )
    parser.add_argument(
        '--generic-q-value-max', type=float, default=0.01,
        help='maximum extraction q-value for generic MS2 target/decoy families; independent of the PSM q-value',
    )
    parser.add_argument(
        '--relaxed-ms2-feature',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'retry otherwise unresolved MS2 once with conservative multi-scan '
            'criteria; direct retries require same-run PSM q-value < 0.01 and '
            'generic retries retain paired target-decoy control'
        ),
    )
    parser.add_argument(
        '--raw-ms1-cache-dir',
        default='',
        help='optional persisted mmap-compatible raw MS1 cache directory for hybrid mode',
    )
    parser.add_argument(
        '--hybrid-stage-cache-dir',
        default='',
        help=(
            'optional fingerprinted strict-stage cache; a valid existing '
            'cache skips mzML ingestion, hill detection and strict candidate '
            'generation, otherwise the completed strict stage is saved here'
        ),
    )
    parser.add_argument(
        '--hybrid-candidate-cache-dir',
        default='',
        help=(
            'optional fingerprinted cache root for expensive hybrid local '
            'target/decoy candidate extraction on a specific residual state'
        ),
    )
    parser.add_argument(
        '--generic-ms2-ppm', dest='generic_ms2_ppm', type=float,
        default=10.0,
        help='selected-ion precursor tolerance in ppm for generic MS2 hypotheses',
    )
    parser.add_argument(
        '--ms2-rt-tolerance-sec', dest='ms2_rt_tolerance_sec',
        type=float, default=120.0,
        help='initial RT search tolerance in seconds around an MS2 event; calibrated retries may tighten it',
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
    if args['cmin'] < 1 or args['cmax'] < args['cmin']:
        parser.error('-cmin must be positive and -cmax/--max-charge must be at least -cmin.')
    if args['combine_every'] < 1:
        parser.error('-combine_every must be a positive integer.')
    if args['combine_every'] > 1:
        parser.error('-combine_every greater than 1 is incompatible with area_sum output.')
    if args['iuse'] < -1:
        parser.error('-iuse must be -1 or a nonnegative integer.')
    if not math.isfinite(args['generic_ms2_ppm']) or args['generic_ms2_ppm'] <= 0:
        parser.error('--generic-ms2-ppm must be a finite positive number.')
    if not math.isfinite(args['ms2_rt_tolerance_sec']) or args['ms2_rt_tolerance_sec'] < 0:
        parser.error('--ms2-rt-tolerance-sec must be a finite nonnegative number.')
    if not math.isfinite(args['psm_q_value_max']) or not 0 <= args['psm_q_value_max'] <= 1:
        parser.error('--psm-q-value-max must be finite and in [0, 1].')
    if args['psm_pep_max'] is not None and (
        not math.isfinite(args['psm_pep_max']) or not 0 <= args['psm_pep_max'] <= 1
    ):
        parser.error('--psm-pep-max must be finite and in [0, 1].')
    if not math.isfinite(args['generic_q_value_max']) or not 0 <= args['generic_q_value_max'] <= 1:
        parser.error('--generic-q-value-max must be finite and in [0, 1].')
    if args['feature_mode'] == 'hybrid':
        args['write_ms2'] = True
        args['feature_baseline'] = args['feature_baseline'] or 'edge_linear'
    else:
        args['feature_baseline'] = args['feature_baseline'] or 'none'
    if args['raw_ms1_cache_dir'] and args['feature_mode'] != 'hybrid':
        parser.error('--raw-ms1-cache-dir requires --feature-mode hybrid.')
    if args['hybrid_stage_cache_dir'] and args['feature_mode'] != 'hybrid':
        parser.error('--hybrid-stage-cache-dir requires --feature-mode hybrid.')
    if args['hybrid_candidate_cache_dir'] and args['feature_mode'] != 'hybrid':
        parser.error('--hybrid-candidate-cache-dir requires --feature-mode hybrid.')
    if args['hybrid_stage_cache_dir'] and not args['raw_ms1_cache_dir']:
        parser.error(
            '--hybrid-stage-cache-dir requires --raw-ms1-cache-dir so local '
            'post-processing can reuse the fingerprinted raw MS1 store.'
        )
    if args['parquet_compression'] not in {'zstd', 'brotli'} and _option_was_supplied(
        '--parquet-compression-level', '--parquet_compression_level'
    ):
        parser.error('--parquet-compression-level is supported only by zstd and brotli.')
    parquet_requested = (
        (not args['stop_after_hills'] and args['feature_format'] == 'parquet')
        or ((args['write_hills'] or args['stop_after_hills']) and args['hills_format'] == 'parquet')
        or (args['write_ms1'] and args['ms1_format'] == 'parquet')
        or args['write_ms2']
    )
    if _option_was_supplied('--parquet-engine', '--parquet_engine') and (
        not parquet_requested and not args['duckdb_output']
    ):
        parser.error('--parquet-engine requires at least one Parquet output.')
    if (args['dia'] or args['dia2']) and args['duckdb_output']:
        parser.error('DIA modes do not support --duckdb-output.')
    if args['no_mono_hills'] and args['dia']:
        parser.error('--no-mono-hills cannot be used with -dia because DIA processing requires mono_hills_* columns.')
    if args['feature_mode'] == 'hybrid' and (
        args['dia'] or args['dia2'] or any(not _is_mzml_input(path) for path in args['files'])
    ):
        parser.error('--feature-mode hybrid is supported only for the normal mzML feature workflow.')
    for filename in args['files']:
        if not _is_hills_input(filename):
            continue
        invalid_options = [
            spelling
            for enabled, spelling in (
                (args['stop_after_hills'], '--stop-after-hills'),
                (args['write_hills'], '--write-hills'),
                (args['write_ms1'], '--write-ms1'),
                (args['write_ms2'], '--write-ms2'),
            )
            if enabled
        ]
        if invalid_options:
            parser.error(
                '%s cannot be used with hills input: %s'
                % (', '.join(invalid_options), filename)
            )
    if args['write_ms2'] and (
        args['dia'] or args['dia2'] or any(not _is_mzml_input(path) for path in args['files'])
    ):
        parser.error('--write-ms2 is supported only for the normal mzML feature workflow.')
    if args['run_workers'] > 1 and (
        args['dia'] or args['dia2'] or any(not _is_mzml_input(path) for path in args['files'])
    ):
        parser.error('--run-workers greater than 1 is supported only for normal mzML inputs.')
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

    planned_paths = []
    for filename in args['files']:
        run_args = _run_args_for_file(args, filename, multiple_inputs)
        planned_paths.extend(planned_output_paths(run_args))
    if len(planned_paths) != len(set(planned_paths)):
        parser.error('Output paths collide.')
    existing = [path for path in planned_paths if path.exists()]
    if existing and not args['overwrite']:
        parser.error(
            'Output already exists; use --overwrite: %s'
            % ', '.join(map(str, existing))
        )

    requested_nprocs = args['nprocs']
    effective_nprocs = 1 if args['run_workers'] > 1 else requested_nprocs
    logger.info(
        'Effective parallel configuration: run_workers=%d, nprocs=%d (requested nprocs=%d)',
        args['run_workers'],
        effective_nprocs,
        requested_nprocs,
    )
    results = {}
    if args['run_workers'] == 1:
        for index, filename in enumerate(args['files']):
            logger.info('Starting file: %s', filename)
            result = _run_file_worker(
                _run_args_for_file(args, filename, multiple_inputs)
            )
            results[index] = result
            if result['status'] == 'failed' and not args['continue_on_error']:
                break
    else:
        def task_arguments():
            for filename in args['files']:
                yield (_run_args_for_file(args, filename, multiple_inputs),)

        raw_results, _started = run_bounded_process_tasks(
            _run_file_worker,
            task_arguments(),
            args['run_workers'],
            (
                None
                if args['continue_on_error']
                else lambda result: (
                    isinstance(result, WorkerFailure)
                    or result.get('status') == 'failed'
                )
            ),
        )
        for index, result in raw_results.items():
            if isinstance(result, WorkerFailure):
                results[index] = {
                    'file': args['files'][index],
                    'status': 'failed',
                    'runtime_sec': None,
                    'error': '%s: %s' % (result.exception_type, result.message),
                    'traceback': result.traceback_text,
                }
            else:
                results[index] = result

    if multiple_inputs or any(
        result['status'] == 'failed' for result in results.values()
    ):
        _log_batch_report(logger, args['files'], results)
    failed = [result for result in results.values() if result['status'] == 'failed']
    if failed:
        raise RuntimeError('%d input file(s) failed; see batch report for details.' % len(failed))

if __name__ == '__main__':
    run()
