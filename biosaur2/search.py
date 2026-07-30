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
from .cache_runtime import CacheWorkspace, default_cache_dir
from .duckdb_output import DuckDBOutputManager, uses_duckdb
from .legacy_output import CompactOutputManager
from .parallel import (
    WorkerFailure,
    effective_worker_budget,
    run_budgeted_process_tasks,
    worker_slot_allocations,
)
from .hybrid_backend import configure_backend


_LOG_LEVELS = {
    'quiet': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG,
}


def _add_log_level_argument(parser, show_legacy_debug=False):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--log-level',
        choices=tuple(_LOG_LEVELS),
        default='info',
        help='logging verbosity: quiet keeps errors only; warning, info or debug',
    )
    group.add_argument(
        '-debug',
        dest='log_level',
        action='store_const',
        const='debug',
        help=_advanced_help(show_legacy_debug, 'legacy alias for --log-level debug'),
    )


def _configure_logging(log_level, run_label=None):
    level = _LOG_LEVELS[log_level]
    run_label = run_label or os.environ.get('BIOSAUR2_LOG_RUN_ID')
    if run_label:
        safe_label = str(run_label).replace('\n', ' ').replace('\r', ' ').replace('%', '%%')
        log_format = (
            '%(levelname)9s: %(asctime)s [run='
            + safe_label
            + ' pid=%(process)d] %(message)s'
        )
    elif log_level == 'debug':
        log_format = '%(levelname)9s: %(asctime)s [pid=%(process)d] %(message)s'
    else:
        log_format = '%(levelname)9s: %(asctime)s %(message)s'
    logging.basicConfig(
        format=log_format,
        datefmt='[%H:%M:%S]',
        level=level,
        force=True,
    )
    logging.getLogger('matplotlib').setLevel(max(logging.WARNING, level))


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


def _comma_separated_integers(value):
    try:
        parsed = tuple(int(item.strip()) for item in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'must be a comma-separated integer list, for example 0,1,2,3'
        ) from exc
    if not parsed:
        raise argparse.ArgumentTypeError('must contain at least one integer')
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError('must not contain duplicate values')
    if any(item < -8 or item > 8 for item in parsed):
        raise argparse.ArgumentTypeError('values must be between -8 and 8')
    return tuple(sorted(parsed))


def _auto_or_positive_float(value):
    if value == 'auto':
        return value
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be 'auto' or a positive number of seconds"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "must be 'auto' or a finite positive number of seconds"
        )
    return parsed


def _advanced_help(show_all, text):
    return text if show_all else argparse.SUPPRESS


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
    if run_args.get('format') == 'duckdb':
        try:
            return DuckDBOutputManager(run_args)
        except ImportError as exc:
            raise ImportError(
                '--format duckdb requires DuckDB; install duckdb or biosaur2[duckdb].'
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
    _configure_logging(
        run_args['log_level'],
        os.environ.get('BIOSAUR2_LOG_RUN_ID') or input_stem(filename),
    )
    logger = logging.getLogger(__name__)
    logger.debug(
        'Run start: file=%s mode=%s format=%s workers=%d output=%s cache=%s',
        filename,
        run_args.get('feature_mode'),
        run_args.get('format'),
        run_args.get('nprocs'),
        run_args.get('o'),
        run_args.get('raw_ms1_cache_dir'),
    )
    try:
        if run_args.get('feature_mode') == 'hybrid':
            resolved_backend = configure_backend(run_args['hybrid_backend'])
            run_args['_resolved_hybrid_backend'] = resolved_backend
            logger.debug(
                'Hybrid numerical backend: requested=%s resolved=%s',
                run_args['hybrid_backend'],
                resolved_backend,
            )
        manager = _create_output_manager(run_args)
        if manager is not None:
            run_args["_output_manager"] = manager
            logger.debug('Output manager: %s', type(manager).__name__)
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
        logger.debug('Run complete: file=%s runtime_sec=%.3f', filename, time.monotonic() - started)
        return {
            "file": filename,
            "status": "success",
            "runtime_sec": time.monotonic() - started,
            "error": None,
            "traceback": None,
        }
    except BaseException as exc:
        logger.debug('Run failed: file=%s error=%s', filename, exc, exc_info=True)
        if manager is not None:
            manager.abort()
        return {
            "file": filename,
            "status": "failed",
            "runtime_sec": time.monotonic() - started,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        }


def _run_args_for_file(args, filename, multiple_inputs, allocated_workers=None):
    run_args = {
        key: value
        for key, value in args.items()
        if key not in {"files", "continue_on_error"}
    }
    run_args["file"] = filename
    if multiple_inputs and run_args["o"] and run_args.get("format") != "duckdb":
        extension = run_args["format"]
        output_directory = Path(run_args["o"])
        run_args["o"] = str(
            output_directory / ("%s.features.%s" % (input_stem(filename), extension))
        )
        run_args["_ms2_output_directory"] = str(output_directory)
    run_args["_multiple_inputs"] = multiple_inputs
    workers = args["workers"] if allocated_workers is None else allocated_workers
    run_args["nprocs"] = workers
    run_args["_requested_workers"] = args["workers"]
    run_args["_allocated_workers"] = workers
    cache_workspace = args.get("_cache_workspace")
    if run_args.get("feature_mode") == "hybrid" and cache_workspace:
        cache_paths = CacheWorkspace(
            root=Path(args["cache_dir"]).resolve(),
            workspace=Path(cache_workspace),
            keep=bool(args.get("keep_cache")),
        ).paths_for(filename)
        run_args["raw_ms1_cache_dir"] = cache_paths["raw_ms1_cache"]
        run_args["hybrid_stage_cache_dir"] = cache_paths["strict_stage_cache"]
        run_args["hybrid_candidate_cache_dir"] = cache_paths["candidate_cache"]
    return run_args


def _run_file_worker_budgeted(run_args, allocated_workers):
    run_args["nprocs"] = allocated_workers
    run_args["_allocated_workers"] = allocated_workers
    return _run_file_worker(run_args)


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


def _execute_inputs(args, parser, logger):
    multiple_inputs = len(args['files']) > 1
    if multiple_inputs:
        if args['o']:
            output_directory = Path(args['o'])
            if output_directory.exists() and not output_directory.is_dir():
                parser.error('-o must be a directory when multiple inputs are supplied.')
            output_directory.mkdir(parents=True, exist_ok=True)

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
    logger.debug(
        'Input planning: files=%s planned_outputs=%s overwrite=%s',
        args['files'],
        [str(path) for path in planned_paths],
        args['overwrite'],
    )

    effective_workers = effective_worker_budget(args['workers'])
    args['_effective_workers'] = effective_workers
    normal_parallel = not (args['dia'] or args['dia2']) and all(
        _is_mzml_input(path) for path in args['files']
    )
    allocations = (
        worker_slot_allocations(effective_workers, len(args['files']))
        if multiple_inputs and normal_parallel
        else [effective_workers]
    )
    logger.debug(
        'Execution mode: multiple_inputs=%s normal_parallel=%s allocations=%s',
        multiple_inputs,
        normal_parallel,
        allocations,
    )
    logger.info(
        'Effective worker budget: requested=%d effective=%d run_slots=%d allocations=%s',
        args['workers'],
        effective_workers,
        len(allocations),
        allocations,
    )

    results = {}
    if not multiple_inputs or not normal_parallel:
        for index, filename in enumerate(args['files']):
            logger.info('Starting file with %d workers: %s', effective_workers, filename)
            result = _run_file_worker(
                _run_args_for_file(
                    args, filename, multiple_inputs, effective_workers
                )
            )
            results[index] = result
            if result['status'] == 'failed' and not args['continue_on_error']:
                break
    else:
        def task_arguments():
            for filename in args['files']:
                yield (_run_args_for_file(args, filename, multiple_inputs),)

        raw_results, _started, _allocations = run_budgeted_process_tasks(
            _run_file_worker_budgeted,
            task_arguments(),
            effective_workers,
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


def _build_parser(show_all):
    help_epilog = '''
Input notes:
  mzML/mzML.gz input should contain centroided MS1 spectra. Hybrid mode also
  reads DDA MS2 precursor metadata and can use a same-run Percolator target
  PSM table.

Outputs:
  Legacy mode defaults to <stem>.features.tsv. Hybrid mode defaults to two
  Parquet files: <stem>.features.parquet contains feature coordinates,
  quantification and linked MS2 events; <stem>.identifications.parquet contains
  accepted PSM fields and direct-assay fields. --format duckdb stores the same
  tables in one <stem>.biosaur2.duckdb database per input.

Examples:
  biosaur2 input.mzML.gz
  biosaur2 input.mzML.gz --format parquet
  biosaur2 input.mzML.gz --feature-mode hybrid
  biosaur2 input.mzML.gz --feature-mode hybrid \\
    --psm-path input.percolator.target.psms.tsv \\
    --psm-q-value-max 0.01 --fixed-mod C=UNIMOD:4 --quant-method all
  biosaur2 input.mzML.gz --feature-mode hybrid \\
    --cache-dir .biosaur2_cache --keep-cache

See README.md for a short introduction and docs/ for inputs, parameters,
outputs, quantification and multi-run project workflows.
    '''
    if show_all:
        help_epilog += '''

Advanced output notes:
  Feature rtStart, rtApex and rtEnd are minutes; Hybrid rt_*_sec and MS2 RT are
  seconds. Parquet defaults to DuckDB V2, ZSTD level 6 and compact numeric
  types. Diagnostic hills, MS1, MS2 and extra feature columns are opt-in.
        '''
    parser = argparse.ArgumentParser(
        description=(
            'Detect and quantify isotope features in centroided LC-MS1 mzML '
            'data. Legacy detection is the default; use --feature-mode hybrid '
            'for identification/MS2-guided residual feature detection.'
        ),
        epilog=help_epilog,
        formatter_class=_HelpFormatter)

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s {}'.format(_get_biosaur2_version()),
    )
    parser.add_argument(
        '--help-all',
        action='help',
        help='show everyday and advanced options, then exit',
    )
    parser.add_argument(
        'files',
        help='input files: mzML (.mzML/.mzML.gz) or hills (Experimental) (.hills.tsv/.hills.parquet/.hills.npz)',
        nargs='+',
    )
    parser.add_argument('-mini', help=_advanced_help(show_all, 'minimum centroid intensity considered during hill detection'), default=1, type=float)
    parser.add_argument('-minmz', help=_advanced_help(show_all, 'lower m/z boundary for MS1 feature detection'), default=350, type=float)
    parser.add_argument('-maxmz', help=_advanced_help(show_all, 'upper m/z boundary for MS1 feature detection'), default=1500, type=float)
    parser.add_argument('-pasefmini', help=_advanced_help(show_all, 'minimum intensity after combining PASEF/ion-mobility hills'), default=100, type=float)
    parser.add_argument('-htol', help=_advanced_help(show_all, 'maximum point-to-hill m/z deviation in ppm'), default=8, type=float)
    parser.add_argument('-itol', help=_advanced_help(show_all, 'maximum isotope-envelope mass deviation in ppm'), default=8, type=float)
    parser.add_argument('-ignore_iso_calib', help=_advanced_help(show_all, 'disable run-specific isotope mass-error calibration and use configured tolerances'), action='store_true')
    parser.add_argument('-use_hill_calib', help=_advanced_help(show_all, 'enable experimental run-specific hill mass-error calibration'), action='store_true')
    parser.add_argument('-paseftol', help=_advanced_help(show_all, '1/K0 ion-mobility tolerance used to connect PASEF hill points'), default=0.05, type=float)
    parser.add_argument('-nm', help=_advanced_help(show_all, 'ionization polarity selector: 0=positive, 1=negative'), default=0, type=int)
    parser.add_argument('-o', help='single-input feature path; with multiple inputs this must be an output directory; empty writes beside the input', default='')
    parser.add_argument('-iuse', help=_advanced_help(show_all, 'isotopes used for intensity and area_sum: -1=all assigned, 0=mono only, N=mono plus N isotopes'), default=-1, type=int)
    parser.add_argument('-hvf', help=_advanced_help(show_all, 'hill splitting ratio at an internal local minimum'), default=1.3, type=float)
    parser.add_argument('-ivf', help=_advanced_help(show_all, 'isotope-pattern splitting ratio at an internal local minimum'), default=5.0, type=float)
    parser.add_argument('-minlh', help=_advanced_help(show_all, 'minimum number of MS1 points in an ordinary hill'), default=2, type=int)
    parser.add_argument('-pasefminlh', help=_advanced_help(show_all, 'minimum number of points in a PASEF hill'), default=1, type=int)
    parser.add_argument('-cmin', help=_advanced_help(show_all, 'minimum feature/precursor charge hypothesis'), default=1, type=int)
    parser.add_argument(
        '-cmax', '--max-charge', '--max_charge', dest='cmax',
        help='maximum feature/precursor charge hypothesis',
        default=7,
        type=int,
    )
    parser.add_argument(
        '--workers',
        help=(
            'total CPU worker-process budget; multiple files share this budget '
            'with a target of about four workers per active run'
        ),
        default=4,
        type=_positive_integer,
    )
    parser.add_argument(
        '--continue-on-error',
        help='continue scheduling independent input files after a failed file',
        action='store_true',
    )
    parser.add_argument('-dia',  help=_advanced_help(show_all, 'create MGF for experimental DIA processing'), action='store_true')
    parser.add_argument('-dia2',  help=_advanced_help(show_all, 'create MGF for experimental DIA processing without MS1'), action='store_true')
    parser.add_argument('-diahtol', help=_advanced_help(show_all, 'experimental DIA hill mass tolerance in ppm'), default=25, type=float)
    parser.add_argument('-diaminlh', help=_advanced_help(show_all, 'experimental minimum DIA hill length'), default=1, type=int)
    parser.add_argument('-diadynrange', help=_advanced_help(show_all, 'experimental DIA dynamic-range limit'), default=1000, type=int)
    parser.add_argument('-min_ms2_peaks', help=_advanced_help(show_all, 'minimum fragment-peak count for an experimental MGF entry'), default=5, type=int)
    parser.add_argument('-mgf', help=_advanced_help(show_all, 'experimental DIA/DIA2 MGF output path'), default='')
    _add_log_level_argument(parser, show_legacy_debug=show_all)
    parser.add_argument('-tof', help=_advanced_help(show_all, 'enable experimental TOF processing'), action='store_true')
    parser.add_argument('-profile', help=_advanced_help(show_all, 'enable experimental profile processing'), action='store_true')
    parser.add_argument('-write_hills', '--write-hills', dest='write_hills', help=_advanced_help(show_all, 'write detected hills using --format'), action='store_true')
    parser.add_argument(
        '--no_hill_list', '--no-hill-list',
        dest='no_hill_list',
        help=_advanced_help(show_all, 'for --write-hills, omit point arrays; the result cannot be reused for feature detection'),
        action='store_true',
    )
    parser.add_argument(
        '--write_ms1', '--write-ms1',
        dest='write_ms1',
        help=_advanced_help(show_all, 'write MS1 scan_id, RT seconds and total intensity using --format'),
        action='store_true',
    )
    parser.add_argument(
        '--write-ms2',
        help=_advanced_help(show_all, 'legacy-only diagnostic export of every normalized precursor event using --format'),
        action='store_true',
    )
    parser.add_argument(
        '--feature-mode',
        choices=('legacy', 'hybrid'),
        default='legacy',
        help='legacy=strict untargeted; hybrid=direct/generic MS2 residual workflow',
    )
    parser.add_argument(
        '--hybrid-backend',
        choices=('auto', 'cython', 'rust'),
        default='auto',
        help='hybrid numerical accelerator; auto uses Rust when installed and otherwise Cython',
    )
    parser.add_argument('--psm-path', default='', help='same-run Percolator target PSM TSV (optionally compressed); empty runs hybrid without direct PSM assays')
    parser.add_argument('--psm-q-value-max', type=float, default=0.01, help='maximum Percolator PSM q-value accepted before direct-assay construction')
    parser.add_argument('--psm-pep-max', type=float, default=None, help=_advanced_help(show_all, 'optional maximum PSM posterior error probability; none disables the additional PEP filter'))
    parser.add_argument('--fixed-mod', action='append', default=[], help='repeatable fixed modification SITE=MOD, for example C=UNIMOD:4 or peptide_n_term=UNIMOD:1')
    parser.add_argument(
        '--quant-method',
        choices=('all', 'envelope_area', 'mono_area', 'envelope_apex'),
        default='all',
        help='hybrid abundance columns to report; all keeps envelope area as the primary quant_value',
    )
    parser.add_argument(
        '--feature-baseline', choices=('none', 'edge_linear'), default=None,
        help=_advanced_help(show_all, (
            'hybrid quantification baseline preprocessing '
            '(default: automatic; edge_linear in hybrid mode, none otherwise)'
        )),
    )
    parser.add_argument(
        '--direct-id', action=argparse.BooleanOptionalAction, default=True,
        help=_advanced_help(show_all, 'enable/disable q-filtered same-run PSM assay association and local recovery in hybrid mode'),
    )
    parser.add_argument(
        '--external-id', action=argparse.BooleanOptionalAction, default=True,
        help=_advanced_help(show_all, 'project compatibility switch for aligned external assays; single-run commands have no donor runs'),
    )
    parser.add_argument(
        '--generic-ms2-refine', action=argparse.BooleanOptionalAction, default=True,
        help='enable/disable unidentified-MS2 charge/C13 hypotheses, target-decoy association, and local recovery in hybrid mode',
    )
    parser.add_argument(
        '--generic-q-value-max', type=float, default=0.01,
        help=(
            'estimated false-discovery limit for unidentified-MS2 associations '
            'from target/decoy (real-versus-shifted precursor) competition; '
            'not the PSM q-value'
        ),
    )
    parser.add_argument(
        '--relaxed-ms2-feature',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=_advanced_help(show_all, (
            'retry otherwise unresolved MS2 once with conservative multi-scan '
            'criteria; direct retries require same-run PSM q-value < 0.01 and '
            'generic retries retain paired target-decoy control'
        )),
    )
    parser.add_argument(
        '--cache-dir',
        default=str(default_cache_dir()),
        help='root for all raw, strict-stage, candidate, and project caches',
    )
    parser.add_argument(
        '--keep-cache',
        action='store_true',
        help='retain fingerprinted caches for reuse; otherwise remove this job cache when it finishes',
    )
    parser.add_argument(
        '--generic-ms2-ppm', dest='generic_ms2_ppm', type=float,
        default=10.0,
        help='selected-ion precursor tolerance in ppm for generic MS2 hypotheses',
    )
    parser.add_argument(
        '--generic-ms2-isotope-errors',
        type=_comma_separated_integers,
        default=(0, 1, 2, 3),
        help=_advanced_help(
            show_all,
            'candidate selected-isotope indices. For error N, mono m/z is '
            'selected-ion m/z - N*1.003354835/charge. Default 0,1,2,3 tests M '
            'through M+3; negative values infer a mono peak above the selected '
            'peak and should be used only for validated unusual metadata',
        ),
    )
    parser.add_argument('--generic-local-isotope-count', type=_positive_integer, default=5, help=_advanced_help(show_all, 'number of isotope channels evaluated for each generic local envelope'))
    parser.add_argument('--generic-local-min-mono-points', type=_positive_integer, default=3, help=_advanced_help(show_all, 'minimum nonzero MS1 scans required in the monoisotopic channel for standard generic local recovery'))
    parser.add_argument('--generic-local-min-channel-points', type=_positive_integer, default=3, help=_advanced_help(show_all, 'minimum nonzero MS1 scans required for an isotope channel to count as supported in standard generic local recovery'))
    parser.add_argument('--generic-local-min-supported-channels', type=_positive_integer, default=2, help=_advanced_help(show_all, 'minimum isotope channels meeting --generic-local-min-channel-points in standard generic local recovery'))
    parser.add_argument('--generic-local-min-isotope-cosine', type=float, default=0.90, help=_advanced_help(show_all, 'minimum cosine similarity between the observed integrated envelope and the averagine envelope in standard generic local recovery'))
    parser.add_argument(
        '--generic-local-max-width-sec',
        type=_auto_or_positive_float,
        default='auto',
        help=_advanced_help(
            show_all,
            "maximum recovered component width in seconds. 'auto' uses the "
            '99th percentile of strict-feature widths, clamped to 15-60 s; '
            'if no strict widths exist it uses 30 s. This rejects overly broad '
            'components and is separate from --ms2-rt-tolerance-sec',
        ),
    )
    parser.add_argument('--generic-relaxed-min-mono-points', type=_positive_integer, default=2, help=_advanced_help(show_all, 'minimum monoisotopic MS1 points for the optional relaxed generic retry'))
    parser.add_argument('--generic-relaxed-min-channel-points', type=_positive_integer, default=2, help=_advanced_help(show_all, 'minimum points for a supported isotope channel in the optional relaxed generic retry'))
    parser.add_argument('--generic-relaxed-min-supported-channels', type=_positive_integer, default=2, help=_advanced_help(show_all, 'minimum supported isotope channels in the optional relaxed generic retry'))
    parser.add_argument('--generic-relaxed-min-isotope-cosine', type=float, default=0.95, help=_advanced_help(show_all, 'minimum averagine cosine in the optional relaxed retry; its higher default offsets the two-point allowance'))
    parser.add_argument(
        '--ms2-rt-tolerance-sec', dest='ms2_rt_tolerance_sec',
        type=float, default=120.0,
        help=(
            'initial same-run MS1 search distance before/after an MS2 event in '
            'seconds; this is not cross-run RT alignment'
        ),
    )
    parser.add_argument(
        '--format',
        choices=('tsv', 'parquet', 'duckdb'),
        default=None,
        help='output format (default: automatic: tsv in legacy mode, parquet in hybrid mode)',
    )
    parser.add_argument(
        '--no-mono-hills',
        help=_advanced_help(show_all, 'omit monoisotopic hill point arrays from feature output'),
        action='store_true',
    )
    parser.add_argument(
        '--64',
        dest='use64',
        help=_advanced_help(show_all, 'store Parquet/DuckDB numeric payloads as 64-bit instead of compact types'),
        action='store_true',
    )
    parser.add_argument('--stop_after_hills', '--stop-after-hills', dest='stop_after_hills', help=_advanced_help(show_all, 'stop processing after writing hills output'), action='store_true')
    parser.add_argument(
        '-write_extra_details', '--write-extra-details',
        dest='write_extra_details',
        help=_advanced_help(show_all, (
            'write additional per-feature diagnostic columns to feature output '
            '(including isotope candidate details such as isotopes, '
            'intensity_array_for_cos_corr, monoisotope hill/index IDs). '
            'This option is intended for debugging/inspection and increases output size.'
        )),
        action='store_true',
    )
    parser.add_argument('-md_correction', help=_advanced_help(show_all, 'experimental mass-error model: Orbi, Icr or Tof'), default='Orbi', choices=['Orbi', 'Icr', 'Tof'])
    parser.add_argument(
        "-combine_every",
        "--combine-every",
        dest="combine_every",
        help=_advanced_help(show_all, "combine every N MS1 scans for experimental fractionation data"),
        default=1,
        type=int,
    )
    parser.add_argument('--input-rt-unit', '--input_rt_unit', dest='input_rt_unit', choices=['seconds', 'minutes'], default='seconds', help=_advanced_help(show_all, 'fallback unit for metadata-free mzML/hills RT; mzML metadata takes precedence'))
    parser.add_argument('--tsv-float-decimals', '--tsv_float_decimals', dest='tsv_float_decimals', type=_nonnegative_or('roundtrip'), default='roundtrip', help=_advanced_help(show_all, 'TSV float text: shortest round-trip representation or a decimal count'))
    parser.add_argument('--parquet-engine', choices=['pyarrow', 'duckdb'], default='duckdb', help=_advanced_help(show_all, 'Parquet writer; DuckDB V2 is preferred, with a visible PyArrow fallback when DuckDB is unavailable'))
    parser.add_argument('--parquet-compression', choices=['zstd', 'snappy', 'lz4', 'brotli', 'uncompressed'], default='zstd', help=_advanced_help(show_all, 'Parquet compression codec'))
    parser.add_argument('--parquet-compression-level', type=int, default=6, help=_advanced_help(show_all, 'zstd/brotli compression level'))
    parser.add_argument('--parquet-row-group-size', type=_positive_integer, default=122880, help=_advanced_help(show_all, 'positive Parquet row-group size'))
    parser.add_argument('--parquet-sort', choices=['none', 'mz_rt', 'rt_mz'], default='mz_rt', help=_advanced_help(show_all, 'deterministic physical feature order'))
    parser.add_argument('--overwrite', action='store_true', help='atomically replace existing output targets')
    return parser


def _run_with_parser(parser):
    args = vars(parser.parse_args())
    args['format'] = args['format'] or (
        'parquet' if args['feature_mode'] == 'hybrid' else 'tsv'
    )
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
        if args['write_ms2']:
            parser.error('--write-ms2 is a legacy-only diagnostic option.')
        args['feature_baseline'] = args['feature_baseline'] or 'edge_linear'
    else:
        args['feature_baseline'] = args['feature_baseline'] or 'none'
    if args['generic_local_isotope_count'] > 10:
        parser.error('--generic-local-isotope-count must be at most 10.')
    for name in (
        'generic_local_min_isotope_cosine',
        'generic_relaxed_min_isotope_cosine',
    ):
        if not math.isfinite(args[name]) or not 0 <= args[name] <= 1:
            parser.error('--%s must be finite and in [0, 1].' % name.replace('_', '-'))
    if args['generic_local_min_supported_channels'] > args['generic_local_isotope_count']:
        parser.error('--generic-local-min-supported-channels cannot exceed --generic-local-isotope-count.')
    if args['generic_relaxed_min_supported_channels'] > args['generic_local_isotope_count']:
        parser.error('--generic-relaxed-min-supported-channels cannot exceed --generic-local-isotope-count.')
    if args['parquet_compression'] not in {'zstd', 'brotli'} and _option_was_supplied(
        '--parquet-compression-level', '--parquet_compression_level'
    ):
        parser.error('--parquet-compression-level is supported only by zstd and brotli.')
    parquet_requested = (
        args['format'] in {'parquet', 'duckdb'}
        and (
            not args['stop_after_hills']
            or args['write_hills']
            or args['write_ms1']
            or args['write_ms2']
        )
    )
    if _option_was_supplied('--parquet-engine', '--parquet_engine') and (
        not parquet_requested and args['format'] != 'duckdb'
    ):
        parser.error('--parquet-engine requires at least one Parquet output.')
    if (args['dia'] or args['dia2']) and args['format'] == 'duckdb':
        parser.error('DIA modes do not support --format duckdb.')
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
    forced_write_hills = args['stop_after_hills'] and not args['write_hills']
    if forced_write_hills:
        args['write_hills'] = True
    _configure_logging(args['log_level'])
    logger = logging.getLogger(__name__)
    if forced_write_hills:
        logger.info('--stop_after_hills requested; turning on --write_hills automatically.')
    logger.debug('Starting with args: %s', args)

    if os.name == 'nt':
        logger.info('Using one worker on Windows.')
        args['workers'] = 1

    cache_workspace = None
    if args['feature_mode'] == 'hybrid':
        cache_workspace = CacheWorkspace.create(
            args['cache_dir'], keep=args['keep_cache']
        )
        args['_cache_workspace'] = str(cache_workspace.workspace)
        logger.info(
            'Hybrid cache workspace: %s (%s)',
            cache_workspace.workspace,
            'retained' if cache_workspace.keep else 'temporary',
        )
    try:
        _execute_inputs(args, parser, logger)
    finally:
        if cache_workspace is not None:
            cache_workspace.cleanup()


def run():
    if len(sys.argv) > 1 and sys.argv[1] == 'project':
        from .project_cli import run_project_cli

        return run_project_cli(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'build-manifest':
        from .project_cli import run_build_manifest_alias

        return run_build_manifest_alias(sys.argv[2:])
    show_all = '--help-all' in sys.argv[1:]
    if show_all:
        sys.argv = [value for value in sys.argv if value != '--help-all'] + ['--help']
    return _run_with_parser(_build_parser(show_all))

if __name__ == '__main__':
    run()
