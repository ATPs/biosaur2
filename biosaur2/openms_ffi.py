"""OpenMS FeatureFinderIdentification rescue for Hybrid Biosaur2 outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from bisect import bisect_left, bisect_right
import json
import logging
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Iterable, Mapping
from uuid import uuid4

from lxml import etree
import pyarrow as pa
import pyarrow.parquet as pq

from .legacy_output import compact_feature
from .output import _temporary_neighbor, output_prefix, publish_staged_files
from .raw_ms1 import source_fingerprint
from .schema import HYBRID_SCHEMA_VERSION, hybrid_quant_output_columns


logger = logging.getLogger(__name__)

ORIGIN = "openms_ffi_rescue"
_UNIMOD = re.compile(r"\[UNIMOD:(\d+)\]")
_VERSION = re.compile(r"^Version:\s*(.+)$", re.MULTILINE)
_PPM = 8.0


class OpenMSFFIError(RuntimeError):
    """FeatureFinderIdentification could not produce a usable result."""


def resolve_executable(value: str) -> str | None:
    """Resolve an explicit executable path or a name from PATH."""

    candidate = Path(value)
    if "/" in value or "\\" in value or candidate.is_absolute():
        return str(candidate) if candidate.is_file() and candidate.stat().st_mode & 0o111 else None
    return shutil.which(value)


def _local_name(element):
    return etree.QName(element).localname


def _faims_value(value):
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _faims_equal(left, right):
    left = _faims_value(left)
    right = _faims_value(right)
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, abs_tol=1e-6, rel_tol=0.0)


def _faims_key(value):
    value = _faims_value(value)
    return None if value is None else int(round(value * 1e6))


def _faims_keys(value):
    key = _faims_key(value)
    return (None,) if key is None else (key - 1, key, key + 1)


def _openms_sequence(value: str) -> str:
    result = _UNIMOD.sub(r"(UniMod:\1)", str(value))
    if result.startswith("(UniMod:1)-"):
        result = "." + result.replace("-", "", 1)
    return result.replace("-", "")


def _write_idxml(path: Path, candidates: Iterable[Mapping]):
    root = etree.Element(
        "IdXML", version="1.5",
        nsmap={"xsi": "http://www.w3.org/2001/XMLSchema-instance"},
    )
    root.set(
        "{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation",
        "https://www.openms.de/xml-schema/IdXML_1_5.xsd",
    )
    etree.SubElement(
        root, "SearchParameters", id="SP_0", db="", db_version="",
        taxonomy="", mass_type="monoisotopic", charges="",
        enzyme="unknown_enzyme", missed_cleavages="0",
        precursor_peak_tolerance="0", precursor_peak_tolerance_ppm="false",
        peak_mass_tolerance="0", peak_mass_tolerance_ppm="false",
    )
    run = etree.SubElement(
        root, "IdentificationRun", date="0000-00-00T00:00:00",
        search_engine="Biosaur2 OpenMS FFId rescue", search_engine_version="",
        search_parameters_ref="SP_0",
    )
    etree.SubElement(
        run, "ProteinIdentification", score_type="q-value",
        higher_score_better="false", significance_threshold="0.0",
    )
    for candidate in candidates:
        identification = etree.SubElement(
            run, "PeptideIdentification", score_type="q-value",
            higher_score_better="false", significance_threshold="0.0",
            MZ=repr(float(candidate["selected_ion_mz"])),
            RT=repr(float(candidate["rt_sec"])),
            spectrum_reference=(
                "controllerType=0 controllerNumber=1 scan=%d"
                % int(candidate["native_scan_number"])
            ),
        )
        etree.SubElement(
            identification, "PeptideHit", score=repr(float(candidate["q_value"])),
            sequence=_openms_sequence(candidate["canonical_peptidoform"]),
            charge=str(int(candidate["charge"])),
        )
        etree.SubElement(
            identification, "UserParam", type="string", name="psm_id",
            value=str(candidate["psm_id"]),
        )
        etree.SubElement(
            identification, "UserParam", type="int", name="ms2_event_id",
            value=str(int(candidate["ms2_event_id"])),
        )
    etree.ElementTree(root).write(
        str(path), encoding="UTF-8", xml_declaration=True, pretty_print=True,
    )


def _direct_children(element, name):
    return [child for child in element if _local_name(child) == name]


def _value(element, name, default=None):
    for child in _direct_children(element, "UserParam"):
        if child.get("name") == name:
            return child.get("value", default)
    return default


def _feature_records(path: Path):
    """Yield final target featureXML records without constructing an XML DOM."""

    context = etree.iterparse(str(path), events=("end",), huge_tree=True)
    for _event, feature in context:
        if _local_name(feature) != "feature":
            continue
        parent = feature.getparent()
        if parent is None or _local_name(parent) != "featureList":
            feature.clear()
            continue
        if str(_value(feature, "OffsetPeptide", "false")).lower() == "true":
            feature.clear()
            continue
        positions = {
            child.get("dim"): child.text
            for child in _direct_children(feature, "position")
        }
        psm_events = []
        for peptide in _direct_children(feature, "PeptideIdentification"):
            psm_id = _value(peptide, "psm_id")
            event_id = _value(peptide, "ms2_event_id")
            if psm_id is None or event_id is None:
                continue
            try:
                psm_events.append((str(psm_id), int(event_id)))
            except (TypeError, ValueError):
                continue
        if psm_events:
            try:
                subordinate = next(
                    (child for child in _direct_children(feature, "subordinate")),
                    None,
                )
                yield {
                    "psm_events": tuple(psm_events),
                    "rt_sec": float(positions["0"]),
                    "mz": float(positions["1"]),
                    "intensity": float(next(child.text for child in _direct_children(feature, "intensity"))),
                    "quality": float(next(child.text for child in _direct_children(feature, "overallquality"))),
                    "charge": int(next(child.text for child in _direct_children(feature, "charge"))),
                    "rt_start_sec": float(_value(feature, "leftWidth", positions["0"])),
                    "rt_end_sec": float(_value(feature, "rightWidth", positions["0"])),
                    "peak_apices_sum": float(_value(feature, "peak_apices_sum", "0")),
                    "raw_intensity": float(_value(feature, "raw_intensity", "0")),
                    "isotope_count": 0 if subordinate is None else len(_direct_children(subordinate, "feature")),
                }
            except (KeyError, StopIteration, TypeError, ValueError) as exc:
                raise OpenMSFFIError("invalid final featureXML feature") from exc
        feature.clear()


def _openms_version(executable: str) -> str:
    """Return the OpenMS version without making version discovery fatal."""

    try:
        result = subprocess.run(
            [executable, "--help"], text=True, capture_output=True, check=False,
        )
    except OSError:
        return "unknown"
    match = _VERSION.search((result.stdout or "") + "\n" + (result.stderr or ""))
    return match.group(1).strip() if match else "unknown"


def _run_openms(executable: str, source: str, idxml: Path, featurexml: Path, workers: int):
    command = [
        executable, "-in", str(source), "-id", str(idxml), "-out", str(featurexml),
        "-extract:mz_window", "8", "-extract:isotope_pmin", "0.01",
        "-extract:rt_window", "60", "-detect:peak_width", "20",
        "-add_mass_offset_peptides", "11.0", "-threads", str(max(1, workers)),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic").strip()
        raise OpenMSFFIError(
            "FeatureFinderIdentification exited %d: %s"
            % (result.returncode, detail[-2000:])
        )
    if not featurexml.is_file() or featurexml.stat().st_size == 0:
        raise OpenMSFFIError("FeatureFinderIdentification did not write featureXML")


class _ScanGrid:
    """FAIMS-aware RT lookup for feature boundary scan assignment."""

    def __init__(self, ms1_rows):
        groups = defaultdict(list)
        for row in ms1_rows:
            rt = row.get("rt_sec")
            scan = row.get("scan_number", row.get("scan_id"))
            if rt is None or scan is None:
                continue
            faims_cv = _faims_value(row.get("faims_cv", row.get("FAIMS")))
            groups[_faims_key(faims_cv)].append((float(rt), int(scan), faims_cv))
        self._groups = {key: tuple(sorted(values)) for key, values in groups.items()}
        self._cache = {}

    def _entries(self, faims_cv):
        cache_key = _faims_value(faims_cv)
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = []
        for key in _faims_keys(faims_cv):
            result.extend(
                value for value in self._groups.get(key, ())
                if _faims_equal(value[2], faims_cv)
            )
        result.sort(key=lambda value: (value[0], value[1]))
        self._cache[cache_key] = (
            tuple(value[0] for value in result), tuple(result)
        )
        return self._cache[cache_key]

    def nearest(self, faims_cv, rt_sec):
        rts, entries = self._entries(faims_cv)
        if not entries:
            return None
        index = bisect_left(rts, float(rt_sec))
        candidates = entries[max(0, index - 1):index + 1]
        return min(candidates, key=lambda value: (abs(value[0] - rt_sec), value[1]))[1]

    def count(self, faims_cv, start_sec, end_sec):
        rts, _entries = self._entries(faims_cv)
        return bisect_right(rts, float(end_sec)) - bisect_left(rts, float(start_sec))


class _FeatureIndex:
    """Dynamic charge/FAIMS/mz index for OpenMS-to-Biosaur2 reuse."""

    def __init__(self, feature_rows, rt_scale=1.0):
        self._bucket_width = math.log1p(_PPM * 1e-6)
        self._entries = []
        self._buckets = defaultdict(list)
        for row in feature_rows:
            self.add(row, rt_scale=rt_scale)

    def _bucket(self, mz):
        mz = float(mz)
        return math.floor(math.log(mz) / self._bucket_width) if mz > 0 else None

    def add(self, row, rt_scale=1.0):
        try:
            feature_id = int(row["feature_idx"])
            charge = int(row["charge"])
            mz = float(row["mz"])
            start = float(row["rtStart"]) * float(rt_scale)
            end = float(row["rtEnd"]) * float(rt_scale)
        except (KeyError, TypeError, ValueError):
            return
        if charge < 1 or mz <= 0 or not math.isfinite(mz) or start > end:
            return
        entry = {
            "feature_idx": feature_id,
            "charge": charge,
            "mz": mz,
            "rt_start_sec": start,
            "rt_end_sec": end,
            "faims_cv": _faims_value(row.get("FAIMS", row.get("faims_cv"))),
        }
        position = len(self._entries)
        self._entries.append(entry)
        bucket = self._bucket(mz)
        if bucket is not None:
            self._buckets[(charge, _faims_key(entry["faims_cv"]), bucket)].append(position)

    def matches(self, feature):
        bucket = self._bucket(feature["mz"])
        if bucket is None:
            return []
        positions = set()
        for faims_key in _faims_keys(feature.get("faims_cv")):
            for value in (bucket - 2, bucket - 1, bucket, bucket + 1, bucket + 2):
                positions.update(self._buckets.get((int(feature["charge"]), faims_key, value), ()))
        matches = []
        for position in positions:
            existing = self._entries[position]
            if not _faims_equal(existing["faims_cv"], feature.get("faims_cv")):
                continue
            if abs(existing["mz"] - feature["mz"]) * 1e6 / feature["mz"] > _PPM:
                continue
            if max(existing["rt_start_sec"], feature["rt_start_sec"]) > min(existing["rt_end_sec"], feature["rt_end_sec"]):
                continue
            matches.append(existing)
        return matches


def _raw_feature(feature, feature_id, scan_grid):
    return {
        "massCalib": None,
        "rtApex": feature["rt_sec"], "intensityApex": feature["peak_apices_sum"],
        "intensitySum": feature["raw_intensity"], "charge": feature["charge"],
        "nIsotopes": feature["isotope_count"],
        "nScans": scan_grid.count(feature.get("faims_cv"), feature["rt_start_sec"], feature["rt_end_sec"]),
        "mz": feature["mz"], "rtStart": feature["rt_start_sec"],
        "rtEnd": feature["rt_end_sec"], "FAIMS": feature.get("faims_cv"),
        "im": None,
        "scanStart": scan_grid.nearest(feature.get("faims_cv"), feature["rt_start_sec"]),
        "scanApex": scan_grid.nearest(feature.get("faims_cv"), feature["rt_sec"]),
        "scanEnd": scan_grid.nearest(feature.get("faims_cv"), feature["rt_end_sec"]),
        "isoerror": None, "isoerror2": None, "feature_idx": feature_id,
        "area_sum": feature["intensity"],
    }


def _quant_row(run_id, feature, feature_id):
    intensity = feature["intensity"]
    return {
        "run_id": run_id, "feature_id": feature_id, "feature_origin": ORIGIN,
        "confidence_tier": "direct_id", "quant_value": intensity,
        "quant_method": "openms_feature_intensity", "quant_status": "quantified",
        "area_envelope_raw": intensity, "area_envelope_corrected": intensity,
        "area_mono_raw": None, "area_mono_corrected": None,
        "envelope_apex": feature["peak_apices_sum"], "quant_envelope_area": intensity,
        "quant_mono_area": None, "quant_envelope_apex": feature["peak_apices_sum"],
        "feature_quality_score": feature["quality"], "quality_flags": 0,
        "extraction_q_value": None, "supporting_psm_count": 0,
        "supporting_ms2_count": 0, "external_support_count": 0,
        "points_across_peak": 0, "isotope_cosine": None,
        "mass_error_ppm_median": None,
    }


def _shared_faims(rows):
    if not rows:
        return False, None
    value = _faims_value(rows[0].get("faims_cv"))
    return all(_faims_equal(value, row.get("faims_cv")) for row in rows), value


def _no_candidate_summary():
    return {
        "status": "no_candidates", "input_psm_count": 0,
        "mapped_psm_count": 0, "new_feature_count": 0,
        "attached_feature_count": 0, "ambiguous_feature_count": 0,
        "ambiguous_psm_count": 0, "unassigned_psm_count": 0,
        "generic_q_value_max": "not_used",
    }


def execute_rescue(*, source, run_id, candidates, existing_features, ms1_rows,
                   next_feature_id, workers, executable, existing_rt_scale=1.0):
    """Run FFId and return new rows, links, existing support deltas and summary."""

    candidates = list(candidates)
    if not candidates:
        return [], [], {}, {}, _no_candidate_summary()
    candidate_by_psm = {str(row["psm_id"]): row for row in candidates}
    if len(candidate_by_psm) != len(candidates):
        raise OpenMSFFIError("rescue candidates contain duplicate PSM IDs")
    version = _openms_version(executable)
    with tempfile.TemporaryDirectory(prefix="biosaur2-openms-ffi-") as temporary:
        temporary = Path(temporary)
        idxml = temporary / "input.idXML"
        featurexml = temporary / "result.featureXML"
        _write_idxml(idxml, candidates)
        _run_openms(executable, source, idxml, featurexml, workers)
        try:
            records = list(_feature_records(featurexml))
        except etree.XMLSyntaxError as exc:
            raise OpenMSFFIError("invalid featureXML output") from exc
    occurrences = Counter(
        psm_id for record in records for psm_id, event_id in record["psm_events"]
        if psm_id in candidate_by_psm
        and int(candidate_by_psm[psm_id]["ms2_event_id"]) == event_id
    )
    index = _FeatureIndex(existing_features, rt_scale=existing_rt_scale)
    scan_grid = _ScanGrid(ms1_rows)
    new_features = []
    quant_rows = []
    links = {}
    supports = defaultdict(lambda: {"psm_ids": set(), "event_ids": set()})
    current_id = int(next_feature_id)
    ambiguous_features = 0
    ambiguous_faims = 0
    attached = 0
    for record in records:
        event_rows = []
        for psm_id, event_id in record["psm_events"]:
            candidate = candidate_by_psm.get(psm_id)
            if candidate is None or int(candidate["ms2_event_id"]) != event_id:
                continue
            if occurrences[psm_id] == 1:
                event_rows.append(candidate)
        if not event_rows:
            continue
        same_faims, faims_cv = _shared_faims(event_rows)
        if not same_faims:
            ambiguous_faims += 1
            continue
        record = {**record, "faims_cv": faims_cv}
        matches = index.matches(record)
        if len(matches) > 1:
            ambiguous_features += 1
            continue
        if matches:
            feature_id = int(matches[0]["feature_idx"])
            attached += 1
        else:
            feature_id = current_id
            current_id += 1
            raw = _raw_feature(record, feature_id, scan_grid)
            new_features.append(raw)
            quant_rows.append(_quant_row(run_id, record, feature_id))
            index.add(raw)
        for candidate in event_rows:
            event_id = int(candidate["ms2_event_id"])
            links[event_id] = feature_id
            supports[feature_id]["psm_ids"].add(str(candidate["psm_id"]))
            supports[feature_id]["event_ids"].add(event_id)
    new_ids = {int(row["feature_id"]) for row in quant_rows}
    for row in quant_rows:
        support = supports[int(row["feature_id"])]
        row["supporting_psm_count"] = len(support["psm_ids"])
        row["supporting_ms2_count"] = len(support["event_ids"])
    support_deltas = {
        feature_id: (len(value["psm_ids"]), len(value["event_ids"]))
        for feature_id, value in supports.items() if feature_id not in new_ids
    }
    summary = {
        "status": "completed", "input_psm_count": len(candidates),
        "mapped_psm_count": len(links),
        "unassigned_psm_count": len(candidates) - len(links),
        "ambiguous_psm_count": sum(value > 1 for value in occurrences.values()),
        "final_target_feature_count": len(records),
        "new_feature_count": len(new_features),
        "attached_feature_count": attached,
        "ambiguous_feature_count": ambiguous_features,
        "ambiguous_faims_feature_count": ambiguous_faims,
        "executable": executable, "openms_version": version,
        "generic_q_value_max": "not_used",
    }
    return new_features, quant_rows, links, support_deltas, summary


def _apply_quant_support_deltas(quant_rows, deltas):
    by_id = {int(row["feature_id"]): row for row in quant_rows if row.get("feature_id") is not None}
    for feature_id, (psm_count, event_count) in deltas.items():
        row = by_id.get(int(feature_id))
        if row is None:
            continue
        row["supporting_psm_count"] = int(row.get("supporting_psm_count") or 0) + psm_count
        row["supporting_ms2_count"] = int(row.get("supporting_ms2_count") or 0) + event_count


def rescue_hybrid_results(*, source, run_id, assays, audit_by_event, feature_rows,
                          quant_rows, ms1_rows, ms2_rows, next_feature_id, args):
    """Mutate final Hybrid collections with OpenMS rescue rows when enabled."""

    if not args.get("feature_finder_identification", True):
        return {"status": "disabled"}, next_feature_id
    event_by_id = {int(row["ms2_event_id"]): row for row in ms2_rows}
    assay_by_event = {int(assay.ms2_event_id): assay for assay in assays}
    candidates = [
        {
            "psm_id": assay.psm_id, "ms2_event_id": assay.ms2_event_id,
            "canonical_peptidoform": assay.canonical_peptidoform,
            "charge": assay.charge, "rt_sec": assay.rt_sec,
            "selected_ion_mz": assay.selected_ion_mz, "q_value": assay.q_value,
            "faims_cv": assay.faims_cv,
            "native_scan_number": event_by_id.get(int(assay.ms2_event_id), {}).get("native_scan_number"),
        }
        for assay in assays
        if (
            assay.conflict_status == "unique"
            and float(assay.q_value) <= float(args.get("psm_q_value_max", 0.01))
            and audit_by_event.get(assay.ms2_event_id, {}).get("feature_id") is None
        )
    ]
    candidates = [
        row for row in candidates
        if row["native_scan_number"] is not None and row["selected_ion_mz"] is not None
    ]
    if not candidates:
        summary = _no_candidate_summary()
        summary["psm_q_value_max"] = float(args.get("psm_q_value_max", 0.01))
        return summary, next_feature_id
    executable_path = str(args.get("feature_finder_identification_path", "FeatureFinderIdentification"))
    executable = resolve_executable(executable_path)
    if executable is None:
        logger.warning("FeatureFinderIdentification not found (%s); skipping rescue", executable_path)
        return {
            "status": "executable_not_found", "executable": executable_path,
            "psm_q_value_max": float(args.get("psm_q_value_max", 0.01)),
            "generic_q_value_max": "not_used",
        }, next_feature_id
    try:
        new_features, new_quant, links, deltas, summary = execute_rescue(
            source=source, run_id=run_id, candidates=candidates,
            existing_features=feature_rows, ms1_rows=ms1_rows,
            next_feature_id=next_feature_id,
            workers=int(args.get("nprocs", args.get("workers", 1))),
            executable=executable,
        )
    except (OpenMSFFIError, etree.XMLSyntaxError, OSError) as exc:
        logger.warning("FeatureFinderIdentification rescue skipped: %s", exc)
        return {
            "status": "failed", "reason": str(exc),
            "psm_q_value_max": float(args.get("psm_q_value_max", 0.01)),
            "generic_q_value_max": "not_used",
        }, next_feature_id
    _apply_quant_support_deltas(quant_rows, deltas)
    feature_rows.extend(new_features)
    quant_rows.extend(new_quant)
    for event_id, feature_id in links.items():
        assay = assay_by_event[event_id]
        audit_by_event[event_id].update({
            "feature_id": feature_id, "association_tier": "openms_ffi_rescue",
            "status": "rescued_openms_ffi", "charge_used": assay.charge,
            "charge_source": "psm",
        })
    summary["psm_q_value_max"] = float(args.get("psm_q_value_max", 0.01))
    return summary, next_feature_id + len(new_features)


def standalone_candidates(identifications, linked_event_ids, mzml_events, q_value_max):
    """Build exact-formula candidates from final IDs and source MS2 metadata."""

    event_by_id = {int(row["ms2_event_id"]): row for row in mzml_events}
    result = []
    for row in identifications:
        event_id = row.get("ms2_event_id")
        charge = row.get("assay_charge")
        if (
            event_id is None or int(event_id) in linked_event_ids
            or row.get("formula_status") != "exact"
            or row.get("assay_status") != "accepted_direct_assay"
            or charge is None or not row.get("canonical_peptidoform")
            or row.get("q_value") is None or float(row["q_value"]) > float(q_value_max)
        ):
            continue
        event = event_by_id.get(int(event_id))
        if event is None:
            continue
        if (
            event.get("native_scan_number") is None
            or event.get("selected_ion_mz") is None
            or event.get("rt_sec") is None
            or not _faims_equal(row.get("assay_faims_cv"), event.get("faims_cv"))
        ):
            continue
        result.append({
            "psm_id": str(row["psm_id"]), "ms2_event_id": int(event_id),
            "canonical_peptidoform": str(row["canonical_peptidoform"]),
            "charge": int(charge), "rt_sec": float(event["rt_sec"]),
            "selected_ion_mz": float(event["selected_ion_mz"]),
            "q_value": float(row["q_value"]),
            "native_scan_number": int(event["native_scan_number"]),
            "native_id": event.get("native_id"), "faims_cv": event.get("faims_cv"),
        })
    return result


def public_feature_rows(raw_features, quant_rows, args, public_columns):
    quant_by_id = {int(row["feature_id"]): row for row in quant_rows}
    result = []
    for raw in raw_features:
        row = compact_feature(raw, args)
        quant = quant_by_id[int(raw["feature_idx"])]
        row.update({name: quant.get(name) for name in public_columns if name in quant})
        result.append(row)
    return result


def _parquet_targets(output: Path):
    if output.suffix.lower() != ".parquet" or not output.name.endswith(".features.parquet"):
        raise ValueError("--output must name a Hybrid .features.parquet file")
    prefix = output_prefix(output)
    return {
        "features": output,
        "ms2_events": Path("%s.ms2_events.parquet" % prefix),
        "identifications": Path("%s.identifications.parquet" % prefix),
    }


def _append_quality_column(table):
    if "feature_quality_score" in table.column_names:
        return table
    position = (
        table.column_names.index("quality_flags")
        if "quality_flags" in table.column_names
        else len(table.column_names)
    )
    return table.add_column(
        position, "feature_quality_score",
        pa.nulls(table.num_rows, type=pa.float32()),
    )


def _append_rows(table, rows):
    if not rows:
        return table
    appended = pa.Table.from_pylist(rows, schema=table.schema)
    return pa.concat_tables([table, appended])


def _standalone_args(features):
    names = set(features.column_names)
    return {
        "feature_mode": "hybrid",
        "write_mono_hills": "mono_hills_scan_lists" in names,
        "write_extra_details": "isotopes" in names,
        "write_quant_details": "area_envelope_raw" in names,
        "use64": str(features.schema.field("feature_idx").type) == "int64",
    }


def _apply_feature_support_deltas(table, deltas):
    if not deltas:
        return table
    rows = table.to_pylist()
    for row in rows:
        delta = deltas.get(int(row["feature_idx"]))
        if delta is None:
            continue
        row["supporting_psm_count"] = int(row.get("supporting_psm_count") or 0) + delta[0]
        row["supporting_ms2_count"] = int(row.get("supporting_ms2_count") or 0) + delta[1]
    return pa.Table.from_pylist(rows, schema=table.schema)


def _new_ms2_rows(links, candidates):
    by_event = {int(row["ms2_event_id"]): row for row in candidates}
    return [
        {
            "feature_idx": feature_id, "ms2_event_id": event_id,
            "native_id": by_event[event_id].get("native_id"),
            "native_scan_number": int(by_event[event_id]["native_scan_number"]),
            "rt_sec": float(by_event[event_id]["rt_sec"]),
            "precursor_mz": float(by_event[event_id]["selected_ion_mz"]),
            "charge": int(by_event[event_id]["charge"]),
        }
        for event_id, feature_id in sorted(links.items())
    ]


def _standalone_merge(source, feature_table, event_table, identification_table,
                      q_value_max, executable_path, workers):
    from .preprocessing import collect_mzml_metadata

    feature_table = _append_quality_column(feature_table)
    linked_ids = {
        int(value) for value in event_table.column("ms2_event_id").to_pylist()
        if value is not None
    }
    identifications = identification_table.to_pylist()
    if not any(
        row.get("formula_status") == "exact"
        and row.get("assay_status") == "accepted_direct_assay"
        and row.get("ms2_event_id") is not None
        and int(row["ms2_event_id"]) not in linked_ids
        and row.get("q_value") is not None
        and float(row["q_value"]) <= float(q_value_max)
        for row in identifications
    ):
        return feature_table, event_table, {}, _no_candidate_summary()
    ms1_rows, mzml_events = collect_mzml_metadata(
        {"file": str(source), "input_rt_unit": "seconds"}
    )
    candidates = standalone_candidates(
        identifications, linked_ids, mzml_events, q_value_max
    )
    if not candidates:
        return feature_table, event_table, {}, _no_candidate_summary()
    executable = resolve_executable(executable_path)
    if executable is None:
        raise OpenMSFFIError(
            "FeatureFinderIdentification not found: %s" % executable_path
        )
    run_ids = {
        str(value) for value in identification_table.column("run_id").to_pylist()
        if value is not None
    }
    if len(run_ids) != 1:
        raise OpenMSFFIError("Hybrid identifications must contain exactly one run_id")
    next_id = max(
        (int(value) for value in feature_table.column("feature_idx").to_pylist() if value is not None),
        default=0,
    ) + 1
    new_features, quant_rows, links, deltas, summary = execute_rescue(
        source=str(source), run_id=next(iter(run_ids)), candidates=candidates,
        existing_features=feature_table.to_pylist(), ms1_rows=ms1_rows,
        next_feature_id=next_id, workers=workers, executable=executable,
        existing_rt_scale=60.0,
    )
    summary["psm_q_value_max"] = float(q_value_max)
    args = _standalone_args(feature_table)
    public_rows = public_feature_rows(
        new_features, quant_rows, args,
        hybrid_quant_output_columns(bool(args["write_quant_details"])),
    )
    features = _append_rows(
        _apply_feature_support_deltas(feature_table, deltas), public_rows
    )
    return features, _append_rows(event_table, _new_ms2_rows(links, candidates)), deltas, summary


def _table_provenance(table):
    encoded = (table.schema.metadata or {}).get(b"biosaur2_provenance_json")
    if not encoded:
        raise OpenMSFFIError("Hybrid output lacks biosaur2 provenance metadata")
    try:
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise OpenMSFFIError("Hybrid output has invalid provenance metadata") from exc


def _validate_source_provenance(provenance, source):
    recorded_path = provenance.get("input_path")
    recorded_size = provenance.get("input_size")
    if not recorded_path or recorded_size is None:
        raise OpenMSFFIError("Hybrid output provenance lacks input identity")
    source = Path(source).resolve()
    recorded_fingerprint = provenance.get("input_fingerprint")
    fingerprint_path = (
        recorded_fingerprint.get("resolved_path")
        if isinstance(recorded_fingerprint, Mapping)
        else None
    )
    try:
        expected = Path(fingerprint_path or recorded_path).expanduser().resolve()
    except (TypeError, ValueError) as exc:
        raise OpenMSFFIError("Hybrid output provenance has invalid input_path") from exc
    if source != expected or source.stat().st_size != int(recorded_size):
        raise OpenMSFFIError("mzML source does not match existing Hybrid output")
    if recorded_fingerprint:
        if not isinstance(recorded_fingerprint, Mapping):
            raise OpenMSFFIError("Hybrid output has invalid input_fingerprint")
        actual = source_fingerprint(source)
        if (
            actual.get("size") != recorded_fingerprint.get("size")
            or actual.get("edge_sha256") != recorded_fingerprint.get("edge_sha256")
        ):
            raise OpenMSFFIError("mzML source fingerprint does not match existing Hybrid output")


def _validate_hybrid_tables(tables):
    required = {
        "features": {"feature_idx", "mz", "charge", "rtStart", "rtEnd", "supporting_psm_count", "supporting_ms2_count"},
        "ms2_events": {"feature_idx", "ms2_event_id"},
        "identifications": {"run_id", "psm_id", "ms2_event_id", "q_value", "formula_status", "assay_status", "assay_charge", "assay_faims_cv", "canonical_peptidoform"},
    }
    for name, columns in required.items():
        missing = columns - set(tables[name].column_names)
        if missing:
            raise OpenMSFFIError(
                "%s is not a supported Hybrid output: missing %s"
                % (name, ", ".join(sorted(missing)))
            )
    feature_ids = {
        int(value) for value in tables["features"].column("feature_idx").to_pylist()
        if value is not None
    }
    event_ids = set()
    for row in tables["ms2_events"].to_pylist():
        event_id = row.get("ms2_event_id")
        feature_id = row.get("feature_idx")
        if event_id is None or feature_id is None or int(feature_id) not in feature_ids:
            raise OpenMSFFIError("ms2_events contains an invalid feature link")
        if int(event_id) in event_ids:
            raise OpenMSFFIError("ms2_events contains duplicate ms2_event_id")
        event_ids.add(int(event_id))


def _updated_provenance(provenance, summary, source):
    updated = dict(provenance)
    try:
        hybrid_summary = json.loads(updated.get("hybrid_summary_json", "{}"))
    except (TypeError, ValueError) as exc:
        raise OpenMSFFIError("Hybrid output has invalid hybrid summary metadata") from exc
    hybrid_summary["openms_ffi_rescue"] = summary
    updated["hybrid_schema_version"] = HYBRID_SCHEMA_VERSION
    updated["hybrid_summary_json"] = json.dumps(
        hybrid_summary, sort_keys=True, default=str
    )
    updated["input_fingerprint"] = source_fingerprint(source)
    return updated


def _with_provenance(table, provenance):
    metadata = dict(table.schema.metadata or {})
    metadata[b"biosaur2_provenance_json"] = json.dumps(
        provenance, sort_keys=True, default=str
    ).encode("utf8")
    metadata[b"biosaur2_hybrid_schema_version"] = str(
        provenance["hybrid_schema_version"]
    ).encode("utf8")
    metadata[b"biosaur2_hybrid_summary_json"] = provenance[
        "hybrid_summary_json"
    ].encode("utf8")
    return table.replace_schema_metadata(metadata)


def _read_parquet_output(source, output):
    targets = _parquet_targets(output)
    missing = [path for path in targets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Hybrid output: %s" % ", ".join(map(str, missing)))
    tables = {name: pq.read_table(path) for name, path in targets.items()}
    _validate_hybrid_tables(tables)
    provenances = {name: _table_provenance(table) for name, table in tables.items()}
    identities = {
        json.dumps(
            {
                "input_path": value.get("input_path"),
                "input_size": value.get("input_size"),
                "input_fingerprint": value.get("input_fingerprint"),
            }, sort_keys=True, default=str,
        )
        for value in provenances.values()
    }
    if len(identities) != 1:
        raise OpenMSFFIError("Hybrid output tables disagree on source provenance")
    provenance = next(iter(provenances.values()))
    _validate_source_provenance(provenance, source)
    return targets, tables, provenance


def _publish_parquet(targets, tables, provenance):
    staged = []
    try:
        for name, table in tables.items():
            temporary = _temporary_neighbor(targets[name])
            staged.append((temporary, targets[name]))
            pq.write_table(_with_provenance(table, provenance), temporary, compression="zstd")
            verified = pq.read_table(temporary)
            if verified.num_rows != table.num_rows or verified.column_names != table.column_names:
                raise OpenMSFFIError("staged %s Parquet validation failed" % name)
            if _table_provenance(verified) != provenance:
                raise OpenMSFFIError("staged %s provenance validation failed" % name)
        publish_staged_files(staged)
    except BaseException:
        for temporary, _target in staged:
            temporary.unlink(missing_ok=True)
        raise


def rescue_completed_output(*, source, output, q_value_max, executable_path, workers):
    """Atomically merge FFId rescue results into an existing Hybrid output."""

    source = Path(source)
    output = Path(output)
    if not source.is_file():
        raise FileNotFoundError("mzML input does not exist: %s" % source)
    if output.suffix.lower() == ".duckdb":
        return _rescue_duckdb(source, output, q_value_max, executable_path, workers)
    targets, tables, provenance = _read_parquet_output(source, output)
    features, events, _deltas, summary = _standalone_merge(
        source, tables["features"], tables["ms2_events"], tables["identifications"],
        q_value_max, executable_path, workers,
    )
    if summary["status"] == "no_candidates":
        return summary
    updated = _updated_provenance(provenance, summary, source)
    _publish_parquet(
        targets,
        {"features": features, "ms2_events": events, "identifications": tables["identifications"]},
        updated,
    )
    return summary


def _rescue_duckdb(source, output, q_value_max, executable_path, workers):
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("DuckDB rescue requires the optional duckdb dependency") from exc
    if not output.is_file():
        raise FileNotFoundError("Hybrid DuckDB output does not exist: %s" % output)
    with duckdb.connect(str(output), read_only=True) as connection:
        names = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
        required = {"features", "ms2_events", "identifications", "runs"}
        if not required <= names:
            raise ValueError("--output is not a Hybrid Biosaur2 DuckDB result")
        tables = {
            name: connection.execute('SELECT * FROM "%s"' % name).to_arrow_table()
            for name in ("features", "ms2_events", "identifications")
        }
        rows = connection.execute("SELECT provenance_json FROM runs").fetchall()
    if len(rows) != 1 or not rows[0][0]:
        raise OpenMSFFIError("Hybrid DuckDB result must contain one provenance row")
    try:
        provenance = json.loads(rows[0][0])
    except (TypeError, ValueError) as exc:
        raise OpenMSFFIError("Hybrid DuckDB has invalid provenance metadata") from exc
    _validate_hybrid_tables(tables)
    _validate_source_provenance(provenance, source)
    original_feature_rows = tables["features"].num_rows
    original_event_rows = tables["ms2_events"].num_rows
    features, events, deltas, summary = _standalone_merge(
        source, tables["features"], tables["ms2_events"], tables["identifications"],
        q_value_max, executable_path, workers,
    )
    if summary["status"] == "no_candidates":
        return summary
    updated = _updated_provenance(provenance, summary, source)
    temporary = _temporary_neighbor(output)
    try:
        shutil.copy2(output, temporary)
        with duckdb.connect(str(temporary)) as connection:
            connection.execute("BEGIN")
            try:
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info('features')").fetchall()
                }
                if "feature_quality_score" not in columns:
                    connection.execute("ALTER TABLE features ADD COLUMN feature_quality_score FLOAT")
                    ordered_columns = features.column_names
                    projection = ", ".join(
                        '"%s"' % name.replace('"', '""')
                        for name in ordered_columns
                    )
                    temporary_table = (
                        "biosaur2_openms_ffi_features_" + uuid4().hex
                    )
                    connection.execute(
                        'CREATE TABLE "%s" AS SELECT %s FROM features'
                        % (temporary_table, projection)
                    )
                    connection.execute("DROP TABLE features")
                    connection.execute(
                        'ALTER TABLE "%s" RENAME TO features' % temporary_table
                    )
                for feature_id, (psm_count, event_count) in deltas.items():
                    connection.execute(
                        "UPDATE features SET supporting_psm_count = COALESCE(supporting_psm_count, 0) + ?, "
                        "supporting_ms2_count = COALESCE(supporting_ms2_count, 0) + ? WHERE feature_idx = ?",
                        [psm_count, event_count, feature_id],
                    )
                for table_name, rows_to_append in (
                    ("features", features.slice(original_feature_rows)),
                    ("ms2_events", events.slice(original_event_rows)),
                ):
                    if rows_to_append.num_rows:
                        connection.register("biosaur2_openms_ffi_append", rows_to_append)
                        try:
                            connection.execute(
                                'INSERT INTO "%s" BY NAME SELECT * FROM biosaur2_openms_ffi_append'
                                % table_name
                            )
                        finally:
                            connection.unregister("biosaur2_openms_ffi_append")
                connection.execute(
                    "UPDATE runs SET provenance_json = ?", [json.dumps(updated, sort_keys=True)]
                )
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        with duckdb.connect(str(temporary), read_only=True) as connection:
            if connection.execute("SELECT COUNT(*) FROM identifications").fetchone()[0] != tables["identifications"].num_rows:
                raise OpenMSFFIError("staged DuckDB modified identifications")
            if connection.execute("SELECT COUNT(*) FROM features").fetchone()[0] != features.num_rows:
                raise OpenMSFFIError("staged DuckDB feature validation failed")
            if connection.execute("SELECT COUNT(*) FROM ms2_events").fetchone()[0] != events.num_rows:
                raise OpenMSFFIError("staged DuckDB ms2 validation failed")
        publish_staged_files([(temporary, output)])
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return summary
