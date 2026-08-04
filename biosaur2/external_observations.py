"""Compact, immutable direct-observation cache for Project alignment."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .external_alignment import ExternalObservation, exact_ion_key
from .output import _temporary_neighbor, publish_staged_files
from .raw_ms1 import source_fingerprint


OBSERVATION_CACHE_VERSION = "2"


def observation_schema():
    return pa.schema(
        [
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("ion_key", pa.string(), nullable=False),
            pa.field("canonical_peptidoform", pa.string(), nullable=False),
            pa.field("charge", pa.int16(), nullable=False),
            pa.field("faims_cv", pa.float64()),
            pa.field("rt_apex_sec", pa.float64(), nullable=False),
            pa.field("q_value", pa.float64(), nullable=False),
            pa.field("assay_id", pa.int32(), nullable=False),
            pa.field("psm_id", pa.string(), nullable=False),
        ]
    )


def _signature(mzml_path, psm_path):
    # The single-run CLI represents an omitted --psm-path as an empty string;
    # Project manifests use None. Both forms mean there is no PSM input.
    psm_fingerprint = None if not psm_path else source_fingerprint(psm_path)
    return json.dumps(
        {
            "cache_version": OBSERVATION_CACHE_VERSION,
            "observation_policy": "unique_direct_id_positive_quant_v1",
            "mzml": source_fingerprint(mzml_path),
            "psm": psm_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _sidecar_path(paths):
    value = paths.get("external_observations")
    return None if not value else Path(value)


def read_observation_sidecar(run, paths):
    destination = _sidecar_path(paths)
    if destination is None or not destination.is_file():
        return None
    try:
        table = pq.read_table(destination)
        metadata = table.schema.metadata or {}
        expected = _signature(run.mzml_path, getattr(run, "psm_path", None)).encode()
        if metadata.get(b"biosaur2_external_observation_signature") != expected:
            return None
        return tuple(
            ExternalObservation(
                # Project manifests may intentionally use an ID different
                # from the mzML stem used by the single-run writer.
                run_id=run.run_id,
                ion_key=row["ion_key"],
                canonical_peptidoform=row["canonical_peptidoform"],
                charge=int(row["charge"]),
                faims_cv=row["faims_cv"],
                rt_apex_sec=float(row["rt_apex_sec"]),
                q_value=float(row["q_value"]),
                assay_id=int(row["assay_id"]),
                psm_id=row["psm_id"],
            )
            for row in table.to_pylist()
        )
    except (OSError, ValueError, pa.ArrowException):
        return None


def write_observation_sidecar(mzml_path, psm_path, destination, observations):
    """Atomically write source-provenanced direct observations."""

    if not destination:
        return
    final_path = Path(destination)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    schema = observation_schema().with_metadata(
        {
            b"biosaur2_external_observation_signature": _signature(
                mzml_path, psm_path
            ).encode()
        }
    )
    table = pa.Table.from_pylist(
        [observation.__dict__ for observation in observations], schema=schema
    )
    temporary = _temporary_neighbor(final_path)
    try:
        pq.write_table(table, temporary, compression="zstd")
        publish_staged_files([(temporary, final_path)])
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def observations_from_hybrid_rows(run_id, quant_rows, audit_rows, assay_rows):
    """Build the direct-only donor population before public outputs are read."""

    feature_rt = {
        int(row["feature_id"]): float(row["rt_apex_sec"])
        for row in quant_rows
        if row.get("feature_id") is not None
        and row.get("rt_apex_sec") is not None
        and row.get("quant_value") is not None
        and float(row["quant_value"]) > 0
    }
    feature_by_assay = {
        int(row["assay_id"]): int(row["feature_id"])
        for row in audit_rows
        if row.get("assay_id") is not None
        and row.get("feature_id") is not None
        and row.get("association_tier") == "direct_id"
    }
    selected = {}
    for assay in assay_rows:
        assay_id = int(assay["assay_id"])
        feature_id = feature_by_assay.get(assay_id)
        if (
            assay.get("conflict_status") != "unique"
            or feature_id not in feature_rt
        ):
            continue
        faims = assay.get("faims_cv")
        faims = None if faims is None else float(faims)
        ion_key = exact_ion_key(
            assay["canonical_peptidoform"], assay["charge"], faims
        )
        observation = ExternalObservation(
            run_id=run_id,
            ion_key=ion_key,
            canonical_peptidoform=assay["canonical_peptidoform"],
            charge=int(assay["charge"]),
            faims_cv=faims,
            rt_apex_sec=feature_rt[feature_id],
            q_value=float(assay["q_value"]),
            assay_id=assay_id,
            psm_id=assay["psm_id"],
        )
        rank = (observation.q_value, observation.assay_id, observation.psm_id)
        previous = selected.get(ion_key)
        if previous is None or rank < previous[0]:
            selected[ion_key] = (rank, observation)
    return tuple(selected[key][1] for key in sorted(selected))
