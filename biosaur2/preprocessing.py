"""Spectrum ingestion and preprocessing for LC-MS and DIA workflows."""

from __future__ import annotations

from collections import defaultdict
import logging

import numpy as np

from .cutils import centroid_pasef_scan
from .spectra import extract_scan_number, retention_time_seconds


logger = logging.getLogger(__name__)


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


def process_mzml(args):
    """Read, identify, filter and optionally combine source MS1 spectra."""
    from .utils import _extract_ms1_scan_id, iter_ms1_spectra

    combine_every = args["combine_every"]
    if not isinstance(combine_every, int) or combine_every <= 0:
        raise ValueError("combine_every must be a positive integer")
    if combine_every > 1:
        logger.info("Combining every %s MS1 scans.", combine_every)

    data = []
    buffer = []
    skipped = 0
    source_count = 0
    for fallback_index, spectrum in enumerate(iter_ms1_spectra(args["file"])):
        if spectrum["ms level"] != 1:
            continue
        source_count += 1
        spectrum["scan_index"] = fallback_index
        spectrum["scan_number"] = extract_scan_number(spectrum)
        spectrum["scan_id"] = _extract_ms1_scan_id(spectrum, fallback_index)
        spectrum["rt_sec"] = retention_time_seconds(
            spectrum["scanList"]["scan"][0]["scan start time"],
            args.get("input_rt_unit", "seconds"),
        )
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

    logger.info("Number of MS1 scans: %d", len(data))
    logger.info("Number of skipped MS1 scans: %d", skipped)
    if source_count == 0 or not data:
        raise ValueError("No usable MS1 scans remain after input filtering.")
    return data


def collect_ms1_rows(args):
    """Collect one modern summary row per source MS1 scan."""
    from .utils import iter_ms1_spectra

    rows = []
    for scan_index, spectrum in enumerate(iter_ms1_spectra(args["file"])):
        scan_info = spectrum["scanList"]["scan"][0]
        rows.append(
            {
                "scan_index": scan_index,
                "scan_number": extract_scan_number(spectrum),
                "rt_sec": retention_time_seconds(
                    scan_info["scan start time"],
                    args.get("input_rt_unit", "seconds"),
                ),
                "total_intensity": float(
                    spectrum.get(
                        "total ion current",
                        np.sum(spectrum.get("intensity array", [])),
                    )
                ),
                "faims_cv": (
                    float(spectrum["FAIMS compensation voltage"])
                    if spectrum.get("FAIMS compensation voltage") is not None
                    else None
                ),
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
