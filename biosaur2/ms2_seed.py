"""Bounded DDA MS2 precursor evidence for existing MS1 candidates."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
import math

import numpy as np


C13_C12_MASS_DIFF = 1.003354835
SEED_BONUS_CAP = 0.20

FLAG_ELIGIBLE = 0x0001
FLAG_MZ_MATCH = 0x0002
FLAG_RT_INSIDE = 0x0004
FLAG_RT_TOLERANCE = 0x0008
FLAG_SCAN_EXACT = 0x0010
FLAG_SCAN_ADJACENT = 0x0020
FLAG_SCAN_UNAVAILABLE = 0x0040
FLAG_ISOLATION_INTERSECTS = 0x0080
FLAG_ISOLATION_UNAVAILABLE = 0x0100
FLAG_ISOLATION_EXCLUDES = 0x0200
FLAG_INVALID_SELECTED_MZ = 0x0400
FLAG_INVALID_CHARGE = 0x0800
FLAG_INVALID_RT = 0x1000
FLAG_UNRESOLVED_FAIMS = 0x2000
FLAG_STANDARD_CANDIDATE = 0x4000
FLAG_SELECTED_CANDIDATE = 0x8000


def _finite(value, positive=False):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or (positive and value <= 0):
        return None
    return value


def _faims_matches(left, right):
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), abs_tol=1e-6)


def _event_faims(row, ms1_metadata):
    if row.get("faims_cv") is not None:
        return row["faims_cv"]
    precursor = row.get("precursor_ms1_index")
    if precursor is None:
        return None
    metadata = ms1_metadata.get(int(precursor), {})
    return metadata.get("faims_cv")


def _hill_rt_interval(hills_dict, hill_index, rt_by_local):
    scans = hills_dict["hills_scan_lists"][hill_index]
    return rt_by_local[scans[0]], rt_by_local[scans[-1]]


def _rt_distance(rt, interval):
    if interval[0] <= rt <= interval[1]:
        return 0.0
    return min(abs(rt - interval[0]), abs(rt - interval[1]))


def _source_scan_match(hills_dict, hill_indices, precursor_index, source_to_local):
    if precursor_index is None:
        return 0.6, None, FLAG_SCAN_UNAVAILABLE
    for distance in (0, 1):
        for source_scan in ({int(precursor_index)} if distance == 0 else {
            int(precursor_index) - 1, int(precursor_index) + 1
        }):
            local = source_to_local.get(source_scan)
            if local is None:
                continue
            if any(local in hills_dict["hills_scan_sets"][index] for index in hill_indices):
                return (1.0 if distance == 0 else 0.8), distance, (
                    FLAG_SCAN_EXACT if distance == 0 else FLAG_SCAN_ADJACENT
                )
    return 0.0, None, 0


def _window(row):
    target = _finite(row.get("isolation_target_mz"), positive=True)
    lower = _finite(row.get("isolation_lower_offset"))
    upper = _finite(row.get("isolation_upper_offset"))
    if target is None or lower is None or upper is None or lower < 0 or upper < 0:
        return None
    return target - lower, target + upper


def _point_in_window(hills_dict, hill_index, local_scan, window):
    scans = hills_dict["hills_scan_lists"][hill_index]
    position = bisect_left(scans, local_scan)
    if position == len(scans) or scans[position] != local_scan:
        return None
    mz = float(hills_dict["tmp_mz_array"][hill_index][position])
    if window[0] <= mz <= window[1]:
        return float(hills_dict["hills_intensity_array"][hill_index][position])
    return None


def _build_scan_point_index(hills_dict, scan_count):
    point_mz = [[] for _ in range(scan_count)]
    point_hill = [[] for _ in range(scan_count)]
    for hill_index, (scans, mz_values) in enumerate(zip(
        hills_dict["hills_scan_lists"], hills_dict["tmp_mz_array"]
    )):
        for local_scan, mz in zip(scans, mz_values):
            point_mz[local_scan].append(float(mz))
            point_hill[local_scan].append(hill_index)

    result = []
    for mz_values, hill_indices in zip(point_mz, point_hill):
        if not mz_values:
            result.append((np.empty(0, dtype=float), np.empty(0, dtype=np.int32)))
            continue
        mz_array = np.asarray(mz_values, dtype=float)
        hill_array = np.asarray(hill_indices, dtype=np.int32)
        order = np.argsort(mz_array, kind="stable")
        result.append((mz_array[order], hill_array[order]))
    return result


def _observed_isolation_hills(row, scan_point_index, source_to_local, ppm):
    """Return scan-local processed hills that could carry the isolated ion."""

    window = _window(row)
    selected_mz = _finite(row.get("selected_ion_mz"), positive=True)
    precursor = row.get("precursor_ms1_index")
    if precursor is None:
        return None
    if window is None:
        if selected_mz is None:
            return None
        delta = selected_mz * ppm * 1e-6
        lookup = (selected_mz - delta, selected_mz + delta)
    else:
        # A hill median can sit just outside a narrow isolation boundary while
        # its scan-local centroid lies inside it.  Expand only by the seed ppm
        # before validating the actual point below.
        delta = max(abs(window[0]), abs(window[1])) * ppm * 1e-6
        lookup = (window[0] - delta, window[1] + delta)

    observed = set()
    for source_scan in (
        int(precursor), int(precursor) - 1, int(precursor) + 1
    ):
        local = source_to_local.get(source_scan)
        if local is None:
            continue
        mz_values, hill_indices = scan_point_index[local]
        start = int(np.searchsorted(mz_values, lookup[0], side="left"))
        end = int(np.searchsorted(mz_values, lookup[1], side="right"))
        observed.update(int(value) for value in hill_indices[start:end])
    return observed


def prepare_seed_context(hills_dict, spectra, ms2_rows, ms1_metadata, faims_val,
                         rt_by_local, args, faims_group_count):
    """Build a local mono-hill seed index for one already processed FAIMS group."""

    source_to_local = {int(spectrum["scan_index"]): index
                       for index, spectrum in enumerate(spectra)}
    ordered = sorted(
        range(len(hills_dict["hills_mz_median"])),
        key=lambda index: float(hills_dict["hills_mz_median"][index]),
    )
    ordered_mz = [float(hills_dict["hills_mz_median"][index]) for index in ordered]
    scan_point_index = _build_scan_point_index(hills_dict, len(spectra))
    context = {
        "events": {}, "events_by_mono": {},
        "events_by_observed_hill": {}, "event_edges": {},
        "ordered": ordered, "seed_local": set(), "next_candidate_id": 0,
        "faims_val": faims_val, "source_to_local": source_to_local,
        "rt_by_local": rt_by_local, "summary": {"eligible_seed_count": 0,
        "seed_local_hill_count": 0, "local_candidate_counts": []},
    }
    ppm = float(args["ms2_seed_ppm"])
    tolerance = float(args["ms2_seed_rt_tolerance_sec"])
    for row in ms2_rows:
        flags = 0
        mz = _finite(row.get("selected_ion_mz"), positive=True)
        charge = row.get("charge")
        rt = _finite(row.get("rt_sec"))
        if mz is None:
            flags |= FLAG_INVALID_SELECTED_MZ
        if charge is None or not (args["cmin"] <= int(charge) <= args["cmax"]):
            flags |= FLAG_INVALID_CHARGE
        if rt is None:
            flags |= FLAG_INVALID_RT
        event_faims = _event_faims(row, ms1_metadata)
        if faims_group_count > 1 and (event_faims is None or not _faims_matches(event_faims, faims_val)):
            flags |= FLAG_UNRESOLVED_FAIMS
        elif event_faims is not None and faims_val is not None and not _faims_matches(event_faims, faims_val):
            flags |= FLAG_UNRESOLVED_FAIMS
        if flags:
            context["events"][row["ms2_event_id"]] = {"row": row, "eligible": False, "flags": flags}
            continue
        event = {"row": row, "eligible": True, "flags": FLAG_ELIGIBLE,
                 "mz": mz, "charge": int(charge), "rt": rt,
                 "offsets": tuple(args["ms2_seed_isotope_errors"])}
        context["events"][row["ms2_event_id"]] = event
        context["summary"]["eligible_seed_count"] += 1
        observed_isolation_hills = _observed_isolation_hills(
            row, scan_point_index, source_to_local, ppm
        )
        event["observed_isolation_hills"] = (
            observed_isolation_hills if _window(row) is not None else None
        )
        if event["observed_isolation_hills"] is not None:
            for hill_index in event["observed_isolation_hills"]:
                context["events_by_observed_hill"].setdefault(
                    hill_index, []
                ).append(event)
        seed_local_matches = set()
        support_lookup_matches = set()
        for offset in event["offsets"]:
            mono_mz = mz - offset * C13_C12_MASS_DIFF / event["charge"]
            delta = mono_mz * ppm * 1e-6
            start = bisect_left(ordered_mz, mono_mz - delta)
            end = bisect_right(ordered_mz, mono_mz + delta)
            for hill_index in ordered[start:end]:
                interval = _hill_rt_interval(hills_dict, hill_index, rt_by_local)
                if _rt_distance(rt, interval) > tolerance:
                    continue
                scan_support, _, _ = _source_scan_match(
                    hills_dict, [hill_index], row.get("precursor_ms1_index"), source_to_local
                )
                if row.get("precursor_ms1_index") is None or scan_support:
                    seed_local_matches.add(hill_index)
                if event["observed_isolation_hills"] is None:
                    # Missing window/precursor metadata cannot use the precise
                    # observed-hill reverse index, so retain the mono fallback.
                    support_lookup_matches.add(hill_index)
        context["summary"]["local_candidate_counts"].append(
            len(seed_local_matches)
        )
        for hill_index in support_lookup_matches:
            context["events_by_mono"].setdefault(hill_index, []).append(event)
        for hill_index in seed_local_matches:
            context["seed_local"].add(hill_index)
    context["summary"]["seed_local_hill_count"] = len(context["seed_local"])
    return context


def partition_mono_indices(sorted_indices, context):
    seeded = [index for index in sorted_indices if index in context["seed_local"]]
    remaining = [index for index in sorted_indices if index not in context["seed_local"]]
    return seeded, remaining


def annotate_candidate_support(candidate, hills_dict, context, args):
    """Attach the best bounded MS2 evidence to one standard-valid candidate."""

    for edge in candidate.get("_ms2_seed_edges", ()):
        event_edges = context["event_edges"].get(edge["event_id"])
        if event_edges is not None:
            context["event_edges"][edge["event_id"]] = [
                value for value in event_edges if value is not edge
            ]
    candidate_id = candidate.get("_ms2_seed_id")
    if candidate_id is None:
        candidate_id = context["next_candidate_id"]
        context["next_candidate_id"] += 1
        candidate["_ms2_seed_id"] = candidate_id
    mono_index = int(candidate["monoisotope idx"])
    edges = []
    hill_indices = [mono_index] + [int(item["isotope_idx"]) for item in candidate["isotopes"]]
    events_by_id = {
        event["row"]["ms2_event_id"]: event
        for event in context["events_by_mono"].get(mono_index, ())
    }
    for hill_index in hill_indices:
        for event in context["events_by_observed_hill"].get(hill_index, ()):
            events_by_id[event["row"]["ms2_event_id"]] = event
    mono_rt = _hill_rt_interval(hills_dict, mono_index, context["rt_by_local"])
    for event_id in sorted(events_by_id):
        event = events_by_id[event_id]
        row = event["row"]
        if int(candidate["charge"]) != event["charge"]:
            continue
        ppm_errors = []
        for offset in event["offsets"]:
            expected = float(candidate["hill_mz_1"]) + offset * C13_C12_MASS_DIFF / event["charge"]
            ppm_errors.append((abs(event["mz"] - expected) * 1e6 / expected, offset))
        ppm_error, offset = min(ppm_errors, key=lambda value: (value[0], abs(value[1]), value[1]))
        if ppm_error > args["ms2_seed_ppm"]:
            continue
        diagnostic_flags = FLAG_MZ_MATCH | FLAG_STANDARD_CANDIDATE
        event["diagnostic_flags"] = (
            event.get("diagnostic_flags", 0) | diagnostic_flags
        )
        distance_rt = _rt_distance(event["rt"], mono_rt)
        if distance_rt > args["ms2_seed_rt_tolerance_sec"]:
            continue
        if distance_rt == 0:
            diagnostic_flags |= FLAG_RT_INSIDE
        else:
            diagnostic_flags |= FLAG_RT_TOLERANCE
        event["diagnostic_flags"] |= diagnostic_flags
        scan_support, scan_distance, flags = _source_scan_match(
            hills_dict, hill_indices, row.get("precursor_ms1_index"), context["source_to_local"]
        )
        if scan_support == 0:
            continue
        diagnostic_flags |= flags
        event["diagnostic_flags"] |= diagnostic_flags
        window = _window(row)
        isolated = None
        if window is None:
            isolation_support = 0.75
            flags |= FLAG_ISOLATION_UNAVAILABLE
            diagnostic_flags |= FLAG_ISOLATION_UNAVAILABLE
        else:
            best = None
            precursor = row.get("precursor_ms1_index")
            if precursor is not None:
                for source_scan in (int(precursor), int(precursor) - 1, int(precursor) + 1):
                    local = context["source_to_local"].get(source_scan)
                    if local is None:
                        continue
                    for isotope_index, hill_index in enumerate(hill_indices):
                        intensity = _point_in_window(hills_dict, hill_index, local, window)
                        if intensity is not None:
                            value = (intensity, -isotope_index, isotope_index)
                            if best is None or value > best:
                                best = value
            else:
                for isotope_index, hill_index in enumerate(hill_indices):
                    mz = float(hills_dict["hills_mz_median"][hill_index])
                    if window[0] <= mz <= window[1]:
                        best = (0.0, -isotope_index, isotope_index)
                        break
            if best is None:
                event["diagnostic_flags"] |= (
                    diagnostic_flags | FLAG_ISOLATION_EXCLUDES
                )
                continue
            isolation_support = 1.0
            isolated = best[2] if row.get("precursor_ms1_index") is not None else None
            flags |= FLAG_ISOLATION_INTERSECTS
            diagnostic_flags |= FLAG_ISOLATION_INTERSECTS
        event["diagnostic_flags"] |= diagnostic_flags
        mz_support = max(0.0, 1.0 - ppm_error / args["ms2_seed_ppm"])
        if distance_rt == 0:
            rt_support = 1.0
            flags |= FLAG_RT_INSIDE
        elif args["ms2_seed_rt_tolerance_sec"]:
            rt_support = max(0.0, 1.0 - distance_rt / args["ms2_seed_rt_tolerance_sec"])
            flags |= FLAG_RT_TOLERANCE
        else:
            rt_support = 0.0
        support = mz_support * rt_support * scan_support * isolation_support
        flags |= FLAG_MZ_MATCH | FLAG_STANDARD_CANDIDATE
        edge = {"candidate": candidate, "event_id": row["ms2_event_id"], "support": support,
                "offset": offset, "ppm_error": ppm_error, "rt_distance": distance_rt,
                "scan_distance": scan_distance, "isolated_index": isolated,
                "flags": flags}
        edges.append(edge)
        context["event_edges"].setdefault(row["ms2_event_id"], []).append(edge)
    candidate["_ms2_seed_edges"] = edges
    if edges:
        best = max(edges, key=lambda edge: (edge["support"], -abs(edge["offset"]), -edge["offset"], -edge["event_id"]))
        candidate["_ms2_seed_support"] = best["support"]
        candidate["_ms2_seed_contributor"] = best["event_id"]
    else:
        candidate["_ms2_seed_support"] = 0.0
        candidate["_ms2_seed_contributor"] = None


def candidate_bonus(candidate):
    return min(SEED_BONUS_CAP, SEED_BONUS_CAP * float(candidate.get("_ms2_seed_support", 0.0)))


def build_link_rows(ms2_rows, context, final_candidates):
    selected = {candidate["_ms2_seed_id"]: candidate for candidate in final_candidates
                if "_ms2_seed_id" in candidate}
    rows = []
    summary = context["summary"]
    statuses = {}
    for source in ms2_rows:
        event_id = source["ms2_event_id"]
        event = context["events"].get(event_id, {"eligible": False, "flags": 0})
        result = {"run_id": source["run_id"], "ms2_event_id": event_id,
                  "feature_id": None, "status": "ineligible_seed",
                  "seed_eligible": bool(event.get("eligible")),
                  "seed_used_in_selection": False,
                  "selected_ion_isotope_offset": None, "isolated_isotope_index": None,
                  "mz_error_ppm": None, "rt_distance_sec": None,
                  "precursor_scan_distance": None, "seed_support": None,
                  "reason_flags": int(
                      event.get("flags", 0) | event.get("diagnostic_flags", 0)
                  )}
        if event.get("eligible"):
            edges = context["event_edges"].get(event_id, [])
            final_edges = [edge for edge in edges if edge["candidate"]["_ms2_seed_id"] in selected]
            if final_edges:
                final_edges.sort(key=lambda edge: (-edge["support"], abs(edge["offset"]), edge["offset"], edge["candidate"]["feature_idx"]))
                best = final_edges[0]
                if len(final_edges) > 1 and abs(best["support"] - final_edges[1]["support"]) <= 1e-6:
                    result["status"] = "ambiguous"
                else:
                    candidate = best["candidate"]
                    result.update({
                        "feature_id": candidate["feature_idx"],
                        "status": "matched_seeded_feature" if candidate.get("_ms2_seed_contributor") == event_id else "matched_existing_feature",
                        "seed_used_in_selection": candidate.get("_ms2_seed_contributor") == event_id,
                        "selected_ion_isotope_offset": best["offset"],
                        "isolated_isotope_index": best["isolated_index"],
                        "mz_error_ppm": best["ppm_error"], "rt_distance_sec": best["rt_distance"],
                        "precursor_scan_distance": best["scan_distance"], "seed_support": best["support"],
                        "reason_flags": result["reason_flags"] | best["flags"] | FLAG_SELECTED_CANDIDATE,
                    })
            elif edges:
                best = max(edges, key=lambda edge: edge["support"])
                result.update({"status": "seed_candidate_lost_conflict", "selected_ion_isotope_offset": best["offset"],
                               "isolated_isotope_index": best["isolated_index"], "mz_error_ppm": best["ppm_error"],
                               "rt_distance_sec": best["rt_distance"], "precursor_scan_distance": best["scan_distance"],
                               "seed_support": best["support"], "reason_flags": result["reason_flags"] | best["flags"]})
            else:
                result["status"] = "no_standard_candidate"
        statuses[result["status"]] = statuses.get(result["status"], 0) + 1
        rows.append(result)
    summary["status_counts"] = statuses
    return rows
