"""Command-line entry points for project manifests and hybrid workflows."""

from __future__ import annotations

import argparse
import logging
import math

from .cache_runtime import CacheWorkspace, default_cache_dir
from .project_manifest import auto_pair_runs, write_manifest
from .project import run_project, validate_project


logger = logging.getLogger(__name__)


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    def _get_help_string(self, action):
        if action.required:
            return (action.help or "") + " (required)"
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
    return parser


def _run_make_manifest(arguments, prog):
    args = _manifest_parser(prog).parse_args(arguments)
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
        parser = argparse.ArgumentParser(
            prog="biosaur2 project run",
            description=(
                "Run a deterministic multi-file Biosaur2 project. Hybrid mode "
                "can add direct/generic MS2 processing and a post-run aligned "
                "external-assay stage."
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
        parser.add_argument("--workers", type=int, default=4, help="total CPU worker-process budget shared dynamically across runs")
        parser.add_argument("--cache-dir", default=str(default_cache_dir()), help="root for all raw, strict-stage, candidate, and project caches")
        parser.add_argument("--keep-cache", action="store_true", help="retain fingerprinted caches for later reuse")
        parser.add_argument(
            "--max-charge", "--max_charge", type=int, default=7,
            help="maximum feature/precursor charge hypothesis passed to each run",
        )
        parser.add_argument("--continue-on-error", action="store_true", help="finish independent runs after a failure but retain a failed project status")
        parser.add_argument("--resume", action="store_true", help="reuse successful runs only when input and resolved option signatures still match")
        parser.add_argument("--overwrite", action="store_true", help="atomically replace existing per-run/project outputs instead of refusing collisions")
        parser.add_argument("--psm-q-value-max", type=float, default=0.01, help="default maximum Percolator q-value; manifest q_value_max may override it per run")
        parser.add_argument("--psm-pep-max", type=float, default=None, help="optional maximum PSM posterior error probability; none disables this filter")
        parser.add_argument("--fixed-mod", action="append", default=[], help="explicit repeatable fixed modification SITE=MOD, for example C=UNIMOD:4")
        parser.add_argument("--quant-method", choices=("all", "envelope_area", "mono_area", "envelope_apex"), default="all", help="quantification output; all reports every metric and uses envelope area as quant_value")
        parser.add_argument("--feature-baseline", choices=("none", "edge_linear"), default="edge_linear", help="baseline preprocessing before hybrid feature quantification")
        parser.add_argument("--direct-id", action=argparse.BooleanOptionalAction, default=True, help="enable/disable q-filtered same-run direct PSM assays in hybrid mode")
        parser.add_argument("--external-id", action=argparse.BooleanOptionalAction, default=True, help="enable/disable aligned external assays inside compatible alignment groups")
        parser.add_argument("--external-q-value-max", type=float, default=0.01, help="maximum target/decoy q-value for aligned recipient extraction")
        parser.add_argument("--external-ppm", type=float, default=8.0, help="recipient-run m/z tolerance in ppm for aligned external assays")
        parser.add_argument(
            "--external-alignment-min-anchors", type=int, default=5,
            help="minimum shared direct peptide/charge anchors required for RT alignment",
        )
        parser.add_argument(
            "--external-alignment-max-mad-sec", type=float, default=30.0,
            help="maximum robust RT-alignment residual MAD in seconds",
        )
        parser.add_argument(
            "--external-min-isotope-cosine", type=float, default=0.8,
            help="minimum theoretical/observed isotope cosine for an external candidate",
        )
        parser.add_argument("--generic-ms2-refine", action=argparse.BooleanOptionalAction, default=True, help="enable/disable unidentified-MS2 hypotheses and residual local recovery")
        parser.add_argument("--generic-q-value-max", type=float, default=0.01, help="estimated false-discovery limit for unidentified-MS2 associations from target/decoy (real-versus-shifted precursor) competition; not the PSM q-value")
        parser.add_argument(
            "--relaxed-ms2-feature",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "retry unresolved direct/generic MS2 once with conservative "
                "multi-scan criteria and the applicable confidence control"
            ),
        )
        parser.add_argument(
            "--ms2-rt-tolerance-sec", type=float, default=120.0,
            help="initial same-run MS1 search distance around each MS2 event; not cross-run RT alignment",
        )
        args = parser.parse_args(arguments[1:])
        if args.workers < 1:
            parser.error("--workers must be positive")
        if args.max_charge < 1:
            parser.error("--max-charge must be positive")
        if not math.isfinite(args.generic_q_value_max) or not 0 <= args.generic_q_value_max <= 1:
            parser.error("--generic-q-value-max must be finite and in [0, 1]")
        if not math.isfinite(args.external_q_value_max) or not 0 <= args.external_q_value_max <= 1:
            parser.error("--external-q-value-max must be finite and in [0, 1]")
        if not math.isfinite(args.external_ppm) or args.external_ppm <= 0:
            parser.error("--external-ppm must be finite and positive")
        if args.external_alignment_min_anchors < 1:
            parser.error("--external-alignment-min-anchors must be positive")
        if (
            not math.isfinite(args.external_alignment_max_mad_sec)
            or args.external_alignment_max_mad_sec < 0
        ):
            parser.error(
                "--external-alignment-max-mad-sec must be finite and nonnegative"
            )
        if (
            not math.isfinite(args.external_min_isotope_cosine)
            or not 0 <= args.external_min_isotope_cosine <= 1
        ):
            parser.error(
                "--external-min-isotope-cosine must be finite and in [0, 1]"
            )
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
        cache_workspace = CacheWorkspace.create(
            options["cache_dir"], keep=options["keep_cache"]
        )
        options["_cache_workspace"] = str(cache_workspace.workspace)
        try:
            run_project(manifest, output_dir, project_db, **options)
        finally:
            cache_workspace.cleanup()
        return 0
    if arguments and arguments[0] == "validate":
        parser = argparse.ArgumentParser(
            prog="biosaur2 project validate",
            description="Validate project paths, run states and output contracts.",
            formatter_class=_HelpFormatter,
        )
        parser.add_argument("--project-db", required=True, help="completed project DuckDB path")
        args = parser.parse_args(arguments[1:])
        result = validate_project(args.project_db)
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
