"""Command-line entry points for project manifests and hybrid workflows."""

from __future__ import annotations

import argparse
import logging
import math

from .project_manifest import auto_pair_runs, write_manifest
from .project import run_project, validate_project


logger = logging.getLogger(__name__)


def _manifest_parser(prog):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Build a deterministic Biosaur2 run manifest by exact stem.",
    )
    parser.add_argument("--mzml-dir", required=True)
    parser.add_argument("--psm-dir", required=True)
    parser.add_argument("--psm-suffix", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--require-psm",
        action="store_true",
        help="fail if any mzML has no exact-stem PSM file",
    )
    parser.add_argument("--overwrite", action="store_true")
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
    if arguments and arguments[0] == "run":
        parser = argparse.ArgumentParser(prog="biosaur2 project run")
        parser.add_argument("--manifest", required=True)
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--project-db", required=True)
        parser.add_argument(
            "--mode",
            choices=("legacy", "weak-ms2", "hybrid"),
            default="legacy",
            help=(
                "feature evidence mode (default: legacy); hybrid residual "
                "detection must be enabled explicitly"
            ),
        )
        parser.add_argument("--run-workers", type=int, default=1)
        parser.add_argument("--nprocs", type=int, default=4)
        parser.add_argument(
            "--max-charge", "--max_charge", type=int, default=7
        )
        parser.add_argument(
            "--allow-nested-parallelism",
            action="store_true",
            help=(
                "allow each file worker to use --nprocs internal workers; "
                "total process budget is run-workers times nprocs"
            ),
        )
        parser.add_argument("--continue-on-error", action="store_true")
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--overwrite", action="store_true")
        parser.add_argument("--psm-q-value-max", type=float, default=0.01)
        parser.add_argument("--psm-pep-max", type=float, default=None)
        parser.add_argument("--fixed-mod", action="append", default=[])
        parser.add_argument("--quant-method", choices=("envelope_area", "mono_area", "envelope_apex"), default="envelope_area")
        parser.add_argument("--feature-baseline", choices=("none", "edge_linear"), default="edge_linear")
        parser.add_argument("--direct-id", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--external-id", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--external-q-value-max", type=float, default=0.01)
        parser.add_argument("--external-ppm", type=float, default=8.0)
        parser.add_argument(
            "--external-alignment-min-anchors", type=int, default=5
        )
        parser.add_argument(
            "--external-alignment-max-mad-sec", type=float, default=30.0
        )
        parser.add_argument(
            "--external-min-isotope-cosine", type=float, default=0.8
        )
        parser.add_argument("--generic-ms2-refine", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--generic-q-value-max", type=float, default=0.01)
        parser.add_argument(
            "--hybrid-stage-cache",
            action=argparse.BooleanOptionalAction,
            default=False,
            help=(
                "persist/reuse each run's fingerprinted strict-stage cache "
                "for downstream hybrid parameter iterations"
            ),
        )
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
            "--ms2-seed-rt-tolerance-sec", type=float, default=120.0
        )
        args = parser.parse_args(arguments[1:])
        if args.run_workers < 1 or args.nprocs < 1:
            parser.error("--run-workers and --nprocs must be positive")
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
            not math.isfinite(args.ms2_seed_rt_tolerance_sec)
            or args.ms2_seed_rt_tolerance_sec < 0
        ):
            parser.error(
                "--ms2-seed-rt-tolerance-sec must be a finite nonnegative number"
            )
        options = vars(args)
        manifest = options.pop("manifest")
        output_dir = options.pop("output_dir")
        project_db = options.pop("project_db")
        run_project(manifest, output_dir, project_db, **options)
        return 0
    if arguments and arguments[0] == "validate":
        parser = argparse.ArgumentParser(prog="biosaur2 project validate")
        parser.add_argument("--project-db", required=True)
        args = parser.parse_args(arguments[1:])
        result = validate_project(args.project_db)
        logger.info("Validated %d project runs", result["run_count"])
        return 0
    parser = argparse.ArgumentParser(
        prog="biosaur2 project",
        description="Project-level Biosaur2 commands.",
    )
    parser.add_argument("command", choices=("make-manifest", "run", "validate"))
    namespace, remaining = parser.parse_known_args(arguments)
    if namespace.command == "make-manifest":
        return _run_make_manifest(remaining, "biosaur2 project make-manifest")
    raise AssertionError("unreachable project command")


def run_build_manifest_alias(arguments):
    return _run_make_manifest(arguments, "biosaur2 build-manifest")
