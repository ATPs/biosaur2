"""Spectrum ingestion and preprocessing for LC-MS and DIA workflows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
import math

import numpy as np

from .cutils import centroid_pasef_scan
from .output import input_stem
from .schema import (
    MS2_MISSING_CHARGE,
    MS2_MISSING_PRECURSOR_MZ,
    MS2_MISSING_PRECURSOR_MS1,
    MS2_UNRESOLVED_SPECTRUM_REF,
)
from .spectra import extract_scan_number, faims_value, retention_time_seconds
from .raw_ms1 import (
    RawMS1StoreBuilder,
    load_raw_ms1_cache,
    save_raw_ms1_cache,
)


logger = logging.getLogger(__name__)


@dataclass
class MzMLIngestion:
    spectra: list
    ms1_rows: list
    ms2_rows: list
    ms1_metadata: dict
    raw_ms1_store: object = None


def centroid_pasef_data(data, args, mz_step):
    """Collapse nearby PASEF points within each spectrum."""
    ion_mobility_accuracy = args["paseftol"]
    hill_mz_accuracy = args["htol"]
    for spectrum_index, spectrum in enumerate(data):
        logger.debug("PASEF scans analysis: %d/%d", spectrum_index + 1, len(data))
        if "ignore_ion_mobility" in spectrum:
            continue
        mz, intensity, mobility = centroid_pasef_scan(
            spectrum,
            mz_step,
            hill_mz_accuracy,
            ion_mobility_accuracy,
            args["pasefmini"],
            args["pasefminlh"],
        )
        spectrum["m/z array"] = np.asarray(mz)
        spectrum["intensity array"] = np.asarray(intensity)
        spectrum["mean inverse reduced ion mobility array"] = np.asarray(mobility)
    data[:] = [spectrum for spectrum in data if len(spectrum["m/z array"])]
    logger.info(
        "Number of MS1 scans after combining ion mobility peaks: %d", len(data)
    )
    return data


def process_profile(data):
    """Centroid a simple profile trace using the historical peak rule."""
    result = []
    for spectrum in data:
        output_mz = []
        output_intensity = []
        output_mobility = []
        best_mz = best_intensity = best_mobility = 0
        previous_mz = previous_intensity = None
        for mz, intensity, mobility in zip(
            spectrum["m/z array"],
            spectrum["intensity array"],
            spectrum["mean inverse reduced ion mobility array"],
        ):
            if previous_mz is None:
                best_mz, best_intensity, best_mobility = mz, intensity, mobility
            elif mz - previous_mz > 0.05 or (
                best_intensity > previous_intensity
                and intensity > previous_intensity
            ):
                output_mz.append(best_mz)
                output_intensity.append(best_intensity)
                output_mobility.append(best_mobility)
                best_mz, best_intensity, best_mobility = mz, intensity, mobility
            elif intensity > best_intensity:
                best_mz, best_intensity, best_mobility = mz, intensity, mobility
            previous_mz, previous_intensity = mz, intensity
        if previous_mz is not None:
            output_mz.append(best_mz)
            output_intensity.append(best_intensity)
            output_mobility.append(best_mobility)
        spectrum["m/z array"] = np.asarray(output_mz)
        spectrum["intensity array"] = np.asarray(output_intensity)
        spectrum["mean inverse reduced ion mobility array"] = np.asarray(
            output_mobility
        )
        result.append(spectrum)
    return result


def process_tof(data):
    """Apply the historical experimental TOF intensity filtering."""
    from .utils import calibrate_mass

    thresholds = {}
    samples = defaultdict(list)
    for spectrum_index, spectrum in enumerate(data[:25]):
        bins = spectrum["m/z array"] // 50
        for mz_bin in set(bins):
            if mz_bin in thresholds:
                continue
            values = np.log10(spectrum["intensity array"][bins == mz_bin])
            samples[mz_bin].extend(values)
            if len(samples[mz_bin]) > 150:
                values = np.asarray(samples[mz_bin])
                shift, sigma, covariance = calibrate_mass(
                    0.05, values.min(), values.max(), values
                )
                logger.debug(
                    "TOF calibration bin %s: shift=%s sigma=%s covariance=%s",
                    mz_bin,
                    shift,
                    sigma,
                    covariance,
                )
                thresholds[mz_bin] = 10 ** (shift + 2 * sigma)

    for spectrum in data:
        bins = spectrum["m/z array"] // 50
        keep = spectrum["intensity array"] > np.asarray(
            [thresholds.get(value, 150) for value in bins]
        )
        for key in (
            "m/z array",
            "intensity array",
            "mean inverse reduced ion mobility array",
        ):
            spectrum[key] = spectrum[key][keep]
    return [spectrum for spectrum in data if len(spectrum["m/z array"])]


def _filter_and_sort_spectrum(spectrum, min_intensity, min_mz, max_mz):
    keep = (
        (spectrum["intensity array"] >= min_intensity)
        & (spectrum["m/z array"] >= min_mz)
        & (spectrum["m/z array"] <= max_mz)
    )
    for key in (
        "m/z array",
        "intensity array",
        "mean inverse reduced ion mobility array",
    ):
        spectrum[key] = spectrum[key][keep]
    order = np.argsort(spectrum["m/z array"])
    for key in (
        "m/z array",
        "intensity array",
        "mean inverse reduced ion mobility array",
    ):
        spectrum[key] = spectrum[key][order]


def _merge_spectra(buffer):
    merged = {
        key: np.concatenate([spectrum[key] for spectrum in buffer])
        for key in (
            "m/z array",
            "intensity array",
            "mean inverse reduced ion mobility array",
        )
    }
    merged.update({key: value for key, value in buffer[0].items() if key not in merged})
    return merged


def _scan_info(spectrum):
    scans = spectrum.get("scanList", {}).get("scan", [])
    return scans[0] if scans else {}


def _finite_float(value, positive=False):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or (positive and numeric <= 0):
        return None
    return numeric


def _nullable_int(value, minimum=-32768, maximum=32767):
    numeric = _finite_float(value)
    if numeric is None or not numeric.is_integer():
        return None
    numeric = int(numeric)
    if numeric < minimum or numeric > maximum:
        return None
    return numeric


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _ms2_faims_value(spectrum):
    value = faims_value(spectrum)
    if value is not None:
        return value
    return _finite_float(_scan_info(spectrum).get("FAIMS compensation voltage"))


def _ms2_ion_mobility(spectrum, selected_ion):
    for source in (selected_ion or {}, _scan_info(spectrum), spectrum):
        value = _finite_float(source.get("inverse reduced ion mobility"))
        if value is not None:
            return value
    return None


def _ms2_rt_seconds(spectrum, args):
    scan_info = _scan_info(spectrum)
    value = scan_info.get("scan start time")
    if value is None:
        return None
    return retention_time_seconds(value, args.get("input_rt_unit", "seconds"))


def _append_ms2_rows(rows, spectrum, ms2_index, spectrum_index, preceding_ms1, args):
    precursor_list = spectrum.get("precursorList", {}).get("precursor")
    precursors = _as_list(precursor_list) or [None]
    event_id = len(rows)
    native_id = spectrum.get("id")
    for precursor in precursors:
        precursor = precursor or {}
        selected_list = precursor.get("selectedIonList", {}).get("selectedIon")
        selected_ions = _as_list(selected_list) or [None]
        isolation = precursor.get("isolationWindow", {})
        isolation_target = _finite_float(
            isolation.get("isolation window target m/z"), positive=True
        )
        for selected_ion in selected_ions:
            selected_ion = selected_ion or {}
            selected_mz = _finite_float(
                selected_ion.get("selected ion m/z"), positive=True
            )
            precursor_mz = selected_mz
            precursor_mz_source = "selected_ion" if selected_mz is not None else None
            if precursor_mz is None and isolation_target is not None:
                precursor_mz = isolation_target
                precursor_mz_source = "isolation_target"
            charge = _nullable_int(selected_ion.get("charge state"))
            if charge is None:
                charge = _nullable_int(selected_ion.get("possible charge state"))
            flags = 0
            if precursor_mz is None:
                flags |= MS2_MISSING_PRECURSOR_MZ
            if charge is None:
                flags |= MS2_MISSING_CHARGE
            rows.append(
                {
                    "run_id": input_stem(args["file"]),
                    "ms2_event_id": event_id,
                    "ms2_index": ms2_index,
                    "spectrum_index": spectrum_index,
                    "native_id": native_id,
                    "native_scan_number": extract_scan_number(spectrum),
                    "rt_sec": _ms2_rt_seconds(spectrum, args),
                    "precursor_ms1_index": None,
                    "precursor_resolution": None,
                    "selected_ion_mz": selected_mz,
                    "isolation_target_mz": isolation_target,
                    "isolation_lower_offset": _finite_float(
                        isolation.get("isolation window lower offset")
                    ),
                    "isolation_upper_offset": _finite_float(
                        isolation.get("isolation window upper offset")
                    ),
                    "precursor_mz": precursor_mz,
                    "precursor_mz_source": precursor_mz_source,
                    "charge": charge,
                    "selected_ion_intensity": _finite_float(
                        selected_ion.get("peak intensity")
                    ),
                    "faims_cv": _ms2_faims_value(spectrum),
                    "ion_mobility": _ms2_ion_mobility(spectrum, selected_ion),
                    "metadata_flags": flags,
                    "_spectrum_ref": precursor.get("spectrumRef"),
                    "_preceding_ms1_index": preceding_ms1,
                }
            )
            event_id += 1


def _resolve_ms2_precursors(rows, ms1_by_native_id):
    for row in rows:
        spectrum_ref = row.pop("_spectrum_ref")
        preceding_ms1 = row.pop("_preceding_ms1_index")
        exact_index = ms1_by_native_id.get(spectrum_ref)
        if spectrum_ref is not None and exact_index is not None:
            row["precursor_ms1_index"] = exact_index
            row["precursor_resolution"] = "spectrum_ref"
            continue
        if spectrum_ref is not None:
            row["metadata_flags"] |= MS2_UNRESOLVED_SPECTRUM_REF
        if preceding_ms1 is None:
            row["metadata_flags"] |= MS2_MISSING_PRECURSOR_MS1
            continue
        row["precursor_ms1_index"] = preceding_ms1
        row["precursor_resolution"] = "preceding_ms1"


def ingest_mzml(args):
    """Read source spectra once and return MS1 data plus optional sidecars."""
    from .utils import (
        _extract_ms1_scan_id,
        iter_ms1_and_ms2_metadata,
        iter_ms1_spectra,
    )

    combine_every = args["combine_every"]
    if not isinstance(combine_every, int) or combine_every <= 0:
        raise ValueError("combine_every must be a positive integer")
    if combine_every > 1:
        logger.info("Combining every %s MS1 scans.", combine_every)

    collect_ms1 = bool(args.get("write_ms1"))
    collect_ms2 = bool(args.get("write_ms2"))
    iterator = (
        iter_ms1_and_ms2_metadata(args["file"])
        if collect_ms2
        else iter_ms1_spectra(args["file"])
    )
    data = []
    ms1_rows = []
    ms2_rows = []
    buffer = []
    skipped = 0
    source_count = 0
    ms2_count = 0
    preceding_ms1 = None
    ms1_by_native_id = {}
    ms1_metadata = {}
    raw_builder = (
        RawMS1StoreBuilder()
        if args.get("feature_mode") == "hybrid" or args.get("_collect_raw_ms1")
        else None
    )

    for spectrum_index, spectrum in enumerate(iterator):
        ms_level = spectrum.get("ms level")
        if ms_level == 2 and collect_ms2:
            _append_ms2_rows(
                ms2_rows,
                spectrum,
                ms2_count,
                spectrum_index,
                preceding_ms1,
                args,
            )
            ms2_count += 1
            continue
        if ms_level != 1:
            continue

        fallback_index = source_count
        source_count += 1
        preceding_ms1 = fallback_index
        native_id = spectrum.get("id")
        if native_id is not None:
            ms1_by_native_id[str(native_id)] = fallback_index
        scan_info = _scan_info(spectrum)
        rt_sec = retention_time_seconds(
            scan_info["scan start time"], args.get("input_rt_unit", "seconds")
        )
        scan_number = extract_scan_number(spectrum)
        spectrum_faims = faims_value(spectrum)
        if raw_builder is not None:
            raw_builder.append(
                spectrum["m/z array"],
                spectrum["intensity array"],
                source_scan_index=fallback_index,
                scan_number=scan_number,
                rt_sec=rt_sec,
                faims_cv=spectrum_faims,
            )
        if collect_ms2:
            ms1_metadata[fallback_index] = {
                "rt_sec": rt_sec,
                "faims_cv": spectrum_faims,
            }
        if collect_ms1:
            ms1_rows.append(
                {
                    "scan_index": fallback_index,
                    "scan_number": scan_number,
                    "rt_sec": rt_sec,
                    "total_intensity": float(
                        spectrum.get(
                            "total ion current",
                            np.sum(spectrum.get("intensity array", [])),
                        )
                    ),
                    "faims_cv": spectrum_faims,
                    "ion_mobility_1_over_k0": None,
                }
            )
        spectrum["scan_index"] = fallback_index
        spectrum["scan_number"] = scan_number
        spectrum["scan_id"] = _extract_ms1_scan_id(spectrum, fallback_index)
        spectrum["rt_sec"] = rt_sec
        if "mean inverse reduced ion mobility array" not in spectrum:
            spectrum["ignore_ion_mobility"] = True
            spectrum["mean inverse reduced ion mobility array"] = np.zeros(
                len(spectrum["m/z array"])
            )
        _filter_and_sort_spectrum(
            spectrum, args["mini"], args["minmz"], args["maxmz"]
        )
        if combine_every == 1:
            if len(spectrum["m/z array"]):
                data.append(spectrum)
            else:
                skipped += 1
            continue
        buffer.append(spectrum)
        if len(buffer) == combine_every:
            merged = _merge_spectra(buffer)
            if len(merged["m/z array"]):
                data.append(merged)
            else:
                skipped += 1
            buffer = []
    if buffer:
        logger.info("Combining %s leftover MS1 scans.", len(buffer))
        merged = _merge_spectra(buffer)
        if len(merged["m/z array"]):
            data.append(merged)
        else:
            skipped += 1

    if collect_ms2:
        _resolve_ms2_precursors(ms2_rows, ms1_by_native_id)
    logger.info("Number of MS1 scans: %d", len(data))
    logger.info("Number of skipped MS1 scans: %d", skipped)
    if source_count == 0 or not data:
        raise ValueError("No usable MS1 scans remain after input filtering.")
    raw_store = None if raw_builder is None else raw_builder.finalize()
    raw_cache = args.get("raw_ms1_cache_dir")
    if raw_store is not None and raw_cache:
        from pathlib import Path

        cache_path = Path(raw_cache)
        if cache_path.exists():
            cached = load_raw_ms1_cache(cache_path, args["file"], mmap=True)
            if (
                cached.scan_count != raw_store.scan_count
                or cached.point_count != raw_store.point_count
            ):
                raise ValueError("existing raw MS1 cache does not match current ingestion")
            raw_store = cached
            logger.info("Validated and mapped existing raw MS1 cache: %s", cache_path)
        else:
            save_raw_ms1_cache(raw_store, cache_path, args["file"])
            raw_store = load_raw_ms1_cache(cache_path, args["file"], mmap=True)
            logger.info(
                "Published raw MS1 cache: %s (%d scans, %d points, %d bytes in arrays)",
                cache_path,
                raw_store.scan_count,
                raw_store.point_count,
                raw_store.memory_bytes,
            )
    return MzMLIngestion(
        data,
        ms1_rows,
        ms2_rows,
        ms1_metadata,
        raw_store,
    )


def process_mzml(args):
    """Compatibility wrapper returning the historical MS1 spectrum list."""
    return ingest_mzml(args).spectra


def collect_ms1_rows(args):
    """Compatibility helper retaining the historical standalone reader."""
    from .utils import iter_ms1_spectra

    rows = []
    for scan_index, spectrum in enumerate(iter_ms1_spectra(args["file"])):
        scan_info = _scan_info(spectrum)
        rows.append(
            {
                "scan_index": scan_index,
                "scan_number": extract_scan_number(spectrum),
                "rt_sec": retention_time_seconds(
                    scan_info["scan start time"], args.get("input_rt_unit", "seconds")
                ),
                "total_intensity": float(
                    spectrum.get(
                        "total ion current", np.sum(spectrum.get("intensity array", []))
                    )
                ),
                "faims_cv": faims_value(spectrum),
                "ion_mobility_1_over_k0": None,
            }
        )
    return rows


def process_mzml_dia(args):
    """Read and minimally filter MS2 spectra for experimental DIA modes."""
    from .utils import iter_all_spectra

    data = []
    skipped = 0
    ms1_scans = 0
    ms2_scans = 0
    for spectrum in iter_all_spectra(args["file"]):
        if spectrum["ms level"] == 1:
            ms1_scans += 1
            continue
        if spectrum["ms level"] != 2:
            continue
        ms2_scans += 1
        if "mean inverse reduced ion mobility array" not in spectrum:
            spectrum["ignore_ion_mobility"] = True
            spectrum["mean inverse reduced ion mobility array"] = np.zeros(
                len(spectrum["m/z array"])
            )
        _filter_and_sort_spectrum(spectrum, 0, 1, 1e6)
        if len(spectrum["m/z array"]):
            data.append(spectrum)
        else:
            skipped += 1
    logger.info("Number of MS2 scans: %d", len(data))
    logger.info("Number of skipped MS2 scans: %d", skipped)
    return data, ms1_scans, ms2_scans
