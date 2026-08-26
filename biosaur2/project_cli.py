"""Command-line entry points for project manifests and hybrid workflows."""

from __future__ import annotations

import argparse
import logging
import math

from .cache_runtime import ProjectCacheWorkspace, default_cache_dir
from .parallel import physical_memory_bytes
from .project_manifest import auto_pair_runs, write_manifest
from .project import run_project, validate_project
from .search import (
    _add_log_level_argument,
    _advanced_help,
    _auto_or_positive_float,
    _comma_separated_integers,
    _configure_logging,
    _positive_integer,
)
from .identifications import PSM_COLUMN_OPTIONS


logger = logging.getLogger(__name__)


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    def _get_help_string(self, action):
        if action.required:
            return (action.help or "") + " (required)"
        if "(default:" in (action.help or ""):
            return action.help
        return super()._get_help_string(action)


def _manifest_parser(prog):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Build a deterministic Biosaur2 run manifest by exact stem.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--mzml-dir", required=True, help="directory containing .mzML or .mzML.gz files")
    parser.add_argument("--psm-dir", required=True, help="directory containing PSM tables paired by exact normalized stem")
    parser.add_argument("--psm-suffix", default=None, help="optional exact PSM suffix override; none recognizes built-in Percolator/PSM suffixes")
    parser.add_argument("--output", required=True, help="output manifest TSV path")
    parser.add_argument(
        "--require-psm",
        action="store_true",
        help="fail if any mzML has no exact-stem PSM file",
    )
    parser.add_argument("--overwrite", action="store_true", help="atomically replace an existing manifest")
    _add_log_level_argument(parser)
    return parser


def _run_make_manifest(arguments, prog):
    args = _manifest_parser(prog).parse_args(arguments)
    _configure_logging(args.log_level)
    report = auto_pair_runs(
        args.mzml_dir,
        args.psm_dir,
        psm_suffix=args.psm_suffix,
        allow_missing_psm=not args.require_psm,
    )
    target = write_manifest(args.output, report.rows, overwrite=args.overwrite)
    logger.info("Wrote %d manifest rows to %s", len(report.rows), target)
    if report.mzml_without_psm:
        logger.warning(
            "%d mzML file(s) have no exact-stem PSM match: %s",
            len(report.mzml_without_psm),
            ", ".join(path.name for path in report.mzml_without_psm),
        )
    if report.orphan_psms:
        logger.warning(
            "%d orphan PSM file(s) have no exact-stem mzML match: %s",
            len(report.orphan_psms),
            ", ".join(path.name for path in report.orphan_psms),
        )
    return 0


def run_project_cli(arguments):
    if arguments and arguments[0] == "make-manifest":
        return _run_make_manifest(
            arguments[1:], "biosaur2 project make-manifest"
        )
    if arguments and arguments[0] == "run":
        show_all = "--help-all" in arguments[1:]
        if show_all:
            arguments = [
                value for value in arguments if value != "--help-all"
            ] + ["--help"]
        parser = argparse.ArgumentParser(
            prog="biosaur2 project run",
            description=(
                "Run a deterministic multi-file Biosaur2 project. Hybrid mode "
                "can add direct/generic MS2 processing and a post-run aligned "
                "strong-to-weak feature match-between-runs stage."
            ),
            epilog='''
Examples:
  biosaur2 project run --manifest runs.tsv --output-dir results \\
    --project-db results/project.duckdb --mode hybrid \\
    --workers 16 --cache-dir .biosaur2_cache --keep-cache

  biosaur2 project validate --project-db results/project.duckdb

The manifest requires run_id and mzml_path. Optional psm_path, fixed_mods,
q_value_max, alignment_group, sample and batch columns are documented in
README.md and examples/hybrid_project_manifest.tsv.
            ''',
            formatter_class=_HelpFormatter,
        )
        parser.add_argument("--manifest", required=True, help="input project manifest TSV")
        parser.add_argument("--help-all", action="help", help="show everyday and advanced project options, then exit")
        _add_log_level_argument(parser)
        parser.add_argument("--output-dir", required=True, help="root directory for per-run atomic outputs")
        parser.add_argument("--project-db", required=True, help="DuckDB path for run/stage status, resolved options, alignment and validation metadata")
        parser.add_argument(
            "--mode",
            choices=("legacy", "hybrid"),
            default="legacy",
            help=(
                "legacy=strict untargeted; hybrid=direct/generic residual workflow"
            ),
        )
        parser.add_argument(
            "--format", choices=("tsv", "parquet", "duckdb"), default=None,
            help="per-run output format (default: automatic: tsv in legacy mode, parquet in hybrid mode)",
        )
        parser.add_argument("--workers", type=int, default=4, help="target busy CPU cores; Project manager may use bounded soft overcommit across runs")
        parser.add_argument(
            "--max-memory",
            type=_positive_integer,
            default=max(1, physical_memory_bytes() // (1024 ** 3)),
            help=(
                "maximum host-use ceiling in auto mode or Project PSS admission "
                "limit in detailed mode, in integer GiB; swap is excluded"
            ),
        )
        parser.add_argument(
            "--scheduler-heartbeat-seconds",
            type=_positive_integer,
            default=60,
            help="seconds between owned-process resource heartbeats",
        )
        parser.add_argument(
            "--scheduler-resource-mode",
            choices=("auto", "detailed"),
            default="auto",
            help=(
                "auto=host memory every 5 seconds plus owned-tree RSS every "
                "heartbeat; detailed=full Project PSS accounting"
            ),
        )
        parser.add_argument("--cache-dir", default=str(default_cache_dir()), help="root for all raw, strict-stage, candidate, and project caches")
        parser.add_argument("--keep-cache", action="store_true", help="retain fingerprinted caches for later reuse")
        parser.add_argument(
            "--max-charge", "--max_charge", type=int, default=7,
            help="maximum feature/precursor charge hypothesis passed to each run",
        )
        parser.add_argument("--continue-on-error", action="store_true", help="finish independent runs after a failure but retain a failed project status")
        parser.add_argument(
            "--resume",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="reuse compatible completed work by default; --no-resume starts a fresh project",
        )
        parser.add_argument("--overwrite", action="store_true", help="atomically replace existing per-run/project outputs instead of refusing collisions")
        parser.add_argument("--psm-q-value-max", type=float, default=0.01, help="default maximum Percolator q-value; manifest q_value_max may override it per run")
        parser.add_argument("--psm-pep-max", type=float, default=None, help=_advanced_help(show_all, "optional maximum PSM posterior error probability; none disables this filter"))
        for _semantic, option, description in PSM_COLUMN_OPTIONS:
            parser.add_argument(
                option,
                default=None,
                help=_advanced_help(
                    show_all,
                    "override automatic PSM-column detection for %s" % description,
                ),
            )
        parser.add_argument("--fixed-mod", action="append", default=[], help="explicit repeatable fixed modification SITE=MOD, for example C=UNIMOD:4")
        parser.add_argument("--quant-method", choices=("all", "envelope_area", "mono_area", "envelope_apex"), default="all", help="quantification output; all reports every metric and uses envelope area as quant_value")
        parser.add_argument("--write-mono-hills", action="store_true", help=_advanced_help(show_all, "include monoisotopic hill point arrays in Hybrid feature output"))
        parser.add_argument("--write-quant-details", action="store_true", help=_advanced_help(show_all, "include raw and baseline-corrected Hybrid area columns"))
        parser.add_argument(
            "--write-ms1",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="write the per-run MS1 scan_id, RT seconds and total intensity table; enabled by default in hybrid mode",
        )
        parser.add_argument("--feature-baseline", choices=("none", "edge_linear"), default="edge_linear", help=_advanced_help(show_all, "baseline preprocessing before hybrid feature quantification"))
        parser.add_argument("--direct-id", action=argparse.BooleanOptionalAction, default=True, help=_advanced_help(show_all, "enable/disable q-filtered same-run direct PSM assays in hybrid mode"))
        parser.add_argument("--external-id", action=argparse.BooleanOptionalAction, default=True, help="enable/disable weak-candidate generation and cross-run strong-feature support inside compatible alignment groups")
        parser.add_argument("--external-q-value-max", type=float, default=0.10, help=_advanced_help(show_all, "maximum target/decoy q-value for feature match-between-runs rescue"))
        parser.add_argument("--external-ppm", type=float, default=8.0, help=_advanced_help(show_all, "m/z tolerance in ppm for strong-feature RT anchors and weak-to-strong support matches"))
        parser.add_argument(
            "--external-rt-tolerance-sec", type=float, default=120.0,
            help=_advanced_help(show_all, "maximum recipient apex distance from aligned donor RT in seconds"),
        )
        parser.add_argument(
            "--external-alignment-min-anchors", type=int, default=20,
            help=_advanced_help(show_all, "minimum feature-only mutual-nearest fit anchors required after held-out RT validation anchors are reserved"),
        )
        parser.add_argument(
            "--external-alignment-max-mad-sec", type=float, default=30.0,
            help=_advanced_help(show_all, "maximum held-out RT-alignment absolute median bias and residual MAD in seconds"),
        )
        parser.add_argument(
            "--external-alignment-max-anchors", type=int, default=256,
            help=_advanced_help(show_all, "maximum deterministic RT anchors fitted per reference edge"),
        )
        parser.add_argument("--external-weak-min-mono-points", type=_positive_integer, default=2, help=_advanced_help(show_all, "minimum monoisotopic points for a weak Project candidate"))
        parser.add_argument("--external-weak-min-secondary-points", type=_positive_integer, default=2, help=_advanced_help(show_all, "minimum raw points in one secondary isotope"))
        parser.add_argument("--external-weak-min-isotope-cosine", type=float, default=0.6, help=_advanced_help(show_all, "minimum isotope cosine for a weak Project candidate"))
        parser.add_argument("--external-weak-max-strong-overlap", type=float, default=0.30, help=_advanced_help(show_all, "maximum fraction of weak-candidate raw hill intensity already owned by final same-run strong features"))
        parser.add_argument("--external-min-support-runs", type=_positive_integer, default=1, help=_advanced_help(show_all, "minimum distinct source runs required for one target or decoy support score"))
        parser.add_argument("--external-max-support-runs", type=_positive_integer, default=4, help=_advanced_help(show_all, "maximum distinct source-run supports combined as empirical log-likelihood evidence and reported per weak candidate (1-16)"))
        parser.add_argument("--generic-ms2-refine", action=argparse.BooleanOptionalAction, default=True, help="enable/disable unidentified-MS2 hypotheses and residual local recovery")
        parser.add_argument("--generic-q-value-max", type=float, default=0.05, help="estimated false-discovery limit for unidentified-MS2 associations from target/decoy (real-versus-shifted precursor) competition; not the PSM q-value")
        parser.add_argument("--generic-ms2-ppm", type=float, default=10.0, help=_advanced_help(show_all, "selected-ion precursor tolerance in ppm for generic MS2 hypotheses"))
        parser.add_argument("--generic-ms2-isotope-errors", type=_comma_separated_integers, default=(0, 1, 2, 3), help=_advanced_help(show_all, "selected-isotope indices; mono m/z = selected-ion m/z - error*1.003354835/charge. Default 0,1,2,3 tests M through M+3; negative values require validation"))
        parser.add_argument("--generic-local-isotope-count", type=_positive_integer, default=5, help=_advanced_help(show_all, "number of isotope channels evaluated for generic local envelopes"))
        parser.add_argument("--generic-local-min-mono-points", type=_positive_integer, default=3, help=_advanced_help(show_all, "minimum monoisotopic points in standard generic local recovery"))
        parser.add_argument("--generic-local-min-channel-points", type=_positive_integer, default=3, help=_advanced_help(show_all, "minimum points for an isotope channel to count as supported"))
        parser.add_argument("--generic-local-min-supported-channels", type=_positive_integer, default=2, help=_advanced_help(show_all, "minimum supported isotope channels in standard recovery"))
        parser.add_argument("--generic-local-min-isotope-cosine", type=float, default=0.90, help=_advanced_help(show_all, "minimum observed-versus-averagine envelope cosine in standard recovery"))
        parser.add_argument("--generic-local-max-width-sec", type=_auto_or_positive_float, default="auto", help=_advanced_help(show_all, "maximum local component width. auto uses strict-feature (rt_end_sec - rt_start_sec) q99 clamped to 15-60 s, or 30 s when no strict widths exist; an explicit positive value disables adaptation. This rejects candidate width and is not the MS2 search window"))
        parser.add_argument("--generic-relaxed-min-mono-points", type=_positive_integer, default=2, help=_advanced_help(show_all, "minimum monoisotopic points in the optional relaxed retry"))
        parser.add_argument("--generic-relaxed-min-channel-points", type=_positive_integer, default=2, help=_advanced_help(show_all, "minimum supported-channel points in the optional relaxed retry"))
        parser.add_argument("--generic-relaxed-min-supported-channels", type=_positive_integer, default=2, help=_advanced_help(show_all, "minimum supported channels in the optional relaxed retry"))
        parser.add_argument("--generic-relaxed-min-isotope-cosine", type=float, default=0.95, help=_advanced_help(show_all, "minimum averagine cosine in the relaxed retry; the higher default offsets its two-point allowance"))
        parser.add_argument(
            "--relaxed-ms2-feature",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=_advanced_help(show_all, (
                "retry unresolved direct/generic MS2 once with conservative "
                "multi-scan criteria and the applicable confidence control"
            )),
        )
        parser.add_argument(
            "--ms2-rt-tolerance-sec", type=float, default=120.0,
            help="initial same-run MS1 search distance around each MS2 event; not cross-run RT alignment",
        )
        args = parser.parse_args(arguments[1:])
        _configure_logging(args.log_level)
        args.format = args.format or (
            "parquet" if args.mode == "hybrid" else "tsv"
        )
        if args.write_ms1 is None:
            args.write_ms1 = args.mode == "hybrid"
        if args.mode != "hybrid" and (
            args.write_mono_hills or args.write_quant_details
        ):
            parser.error(
                "--write-mono-hills and --write-quant-details require --mode hybrid"
            )
        if args.workers < 1:
            parser.error("--workers must be positive")
        if args.max_charge < 1:
            parser.error("--max-charge must be positive")
        if (
            not math.isfinite(args.psm_q_value_max)
            or not 0 <= args.psm_q_value_max <= 1
        ):
            parser.error("--psm-q-value-max must be finite and in [0, 1]")
        if args.psm_pep_max is not None and (
            not math.isfinite(args.psm_pep_max)
            or not 0 <= args.psm_pep_max <= 1
        ):
            parser.error("--psm-pep-max must be finite and in [0, 1]")
        if not math.isfinite(args.generic_q_value_max) or not 0 <= args.generic_q_value_max <= 1:
            parser.error("--generic-q-value-max must be finite and in [0, 1]")
        if not math.isfinite(args.generic_ms2_ppm) or args.generic_ms2_ppm <= 0:
            parser.error("--generic-ms2-ppm must be finite and positive")
        if args.generic_local_isotope_count > 10:
            parser.error("--generic-local-isotope-count must be at most 10")
        if args.generic_local_min_supported_channels > args.generic_local_isotope_count:
            parser.error("--generic-local-min-supported-channels cannot exceed --generic-local-isotope-count")
        if args.generic_relaxed_min_supported_channels > args.generic_local_isotope_count:
            parser.error("--generic-relaxed-min-supported-channels cannot exceed --generic-local-isotope-count")
        for name in (
            "generic_local_min_isotope_cosine",
            "generic_relaxed_min_isotope_cosine",
        ):
            value = getattr(args, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                parser.error("--%s must be finite and in [0, 1]" % name.replace("_", "-"))
        if not math.isfinite(args.external_q_value_max) or not 0 <= args.external_q_value_max <= 1:
            parser.error("--external-q-value-max must be finite and in [0, 1]")
        if not math.isfinite(args.external_ppm) or args.external_ppm <= 0:
            parser.error("--external-ppm must be finite and positive")
        if not math.isfinite(args.external_rt_tolerance_sec) or args.external_rt_tolerance_sec < 0:
            parser.error("--external-rt-tolerance-sec must be finite and nonnegative")
        if args.external_alignment_min_anchors < 1:
            parser.error("--external-alignment-min-anchors must be positive")
        if args.external_alignment_max_anchors < 1:
            parser.error("--external-alignment-max-anchors must be positive")
        if (
            not math.isfinite(args.external_alignment_max_mad_sec)
            or args.external_alignment_max_mad_sec < 0
        ):
            parser.error(
                "--external-alignment-max-mad-sec must be finite and nonnegative"
            )
        if not math.isfinite(args.external_weak_min_isotope_cosine) or not 0 <= args.external_weak_min_isotope_cosine <= 1:
            parser.error("--external-weak-min-isotope-cosine must be finite and in [0, 1]")
        if not math.isfinite(args.external_weak_max_strong_overlap) or not 0 <= args.external_weak_max_strong_overlap <= 1:
            parser.error("--external-weak-max-strong-overlap must be finite and in [0, 1]")
        if args.external_min_support_runs > 16:
            parser.error("--external-min-support-runs must be at most 16")
        if args.external_max_support_runs > 16:
            parser.error("--external-max-support-runs must be at most 16")
        if args.external_min_support_runs > args.external_max_support_runs:
            parser.error("--external-min-support-runs cannot exceed --external-max-support-runs")
        if (
            not math.isfinite(args.ms2_rt_tolerance_sec)
            or args.ms2_rt_tolerance_sec < 0
        ):
            parser.error(
                "--ms2-rt-tolerance-sec must be a finite nonnegative number"
            )
        options = vars(args)
        manifest = options.pop("manifest")
        output_dir = options.pop("output_dir")
        project_db = options.pop("project_db")
        cache_workspace = ProjectCacheWorkspace.create(
            options["cache_dir"], project_db, keep=options["keep_cache"]
        )
        options["_cache_workspace"] = str(cache_workspace.workspace)
        options["_project_checkpoint_path"] = str(cache_workspace.checkpoint_path)
        completed = False
        try:
            run_project(manifest, output_dir, project_db, **options)
            completed = True
        finally:
            cache_workspace.cleanup(success=completed)
        return 0
    if arguments and arguments[0] == "validate":
        parser = argparse.ArgumentParser(
            prog="biosaur2 project validate",
            description="Validate project paths, run states and output contracts.",
            formatter_class=_HelpFormatter,
        )
        parser.add_argument("--project-db", required=True, help="completed project DuckDB path")
        parser.add_argument(
            "--workers",
            type=_positive_integer,
            default=None,
            help="parallel per-run validation readers; default: Project worker budget",
        )
        _add_log_level_argument(parser)
        args = parser.parse_args(arguments[1:])
        _configure_logging(args.log_level)
        result = validate_project(args.project_db, workers=args.workers)
        logger.info("Validated %d project runs", result["run_count"])
        return 0
    parser = argparse.ArgumentParser(
        prog="biosaur2 project",
        description="Project-level Biosaur2 commands.",
        epilog=(
            "Use `biosaur2 project <command> --help` for detailed options.\n"
            "Typical order: make-manifest, run, validate."
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("command", choices=("make-manifest", "run", "validate"), help="project operation to execute")
    parser.parse_args(arguments)
    raise AssertionError("unreachable project command")


def run_build_manifest_alias(arguments):
    return _run_make_manifest(arguments, "biosaur2 build-manifest")
