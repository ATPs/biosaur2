# Biosaur2 MS1 Pipeline Deep Dive for Code Reshaping

## Intent, Scope, and Non-Goals

This document explains how Biosaur2 performs **MS1 feature detection** from CLI entrypoint to `.features.tsv` output, with code-level references and algorithm details intended for refactoring and architectural redesign.

### In scope

- `biosaur2/search.py`
- `biosaur2/main.py`
- `biosaur2/utils.py`
- `biosaur2/cutils.pyx`

### Out of scope

- DIA branches in `biosaur2/main_dia.py` and `biosaur2/main_dia2.py`
- Packaging/setup mechanics outside behavior-critical runtime flow

---

## End-to-End Call Graph (MS1 path)

1. CLI argument parsing and dispatch starts in `search.run()` at `biosaur2/search.py:7`.
2. Parsed args are normalized:
   - `--stop_after_hills` can force `write_hills=True` in `biosaur2/search.py:60`.
   - On Windows, `nprocs` is forced to 1 in `biosaur2/search.py:71`.
3. For each input file, normal MS1 flow goes through `main.process_file()` in `biosaur2/search.py:84`.
4. In `main.process_file()` (`biosaur2/main.py:404`):
   - reads mzML scans through `utils.process_mzml()` (`biosaur2/main.py:425`, `biosaur2/utils.py:472`) or
   - reads precomputed hills via `.hills.tsv` (`biosaur2/main.py:548`).
5. Per FAIMS group, it performs preprocessing and hill construction:
   - optional TOF filtering (`biosaur2/main.py:459`, `biosaur2/utils.py:389`)
   - optional profile processing (`biosaur2/main.py:463`, `biosaur2/utils.py:333`)
   - optional PASEF centroiding (`biosaur2/main.py:468`, `biosaur2/utils.py:231`, `biosaur2/cutils.pyx:334`)
   - hill linking across scans via `detect_hills` (`biosaur2/main.py:511`, `biosaur2/cutils.pyx:618`)
   - hill splitting via `split_peaks_multi` -> `split_peaks` (`biosaur2/main.py:517`, `biosaur2/main.py:286`, `biosaur2/cutils.pyx:439`)
   - hill post-processing via `process_hills` (`biosaur2/main.py:519`, `biosaur2/cutils.pyx:750`)
6. If hills-only mode is disabled, feature detection continues in `process_features_iteration()` (`biosaur2/main.py:540`, `biosaur2/main.py:14`):
   - initial isotope cluster generation via `get_initial_isotopes` (`biosaur2/main.py:395`, `biosaur2/cutils.pyx:71`)
   - isotope mass-error self-calibration (`biosaur2/main.py:88`)
   - cluster pruning and overlap resolution (`biosaur2/main.py:150`, `biosaur2/main.py:199`)
   - feature finalization in `utils.calc_peptide_features` (`biosaur2/main.py:271`, `biosaur2/utils.py:125`)
   - output write in `utils.write_output` (`biosaur2/main.py:273`, `biosaur2/utils.py:169`)

---

## Pipeline Data Model (Core Runtime Shapes)

### 1. `data_for_analyse` scan list (from mzML)

Produced by `utils.process_mzml()` (`biosaur2/utils.py:472`), each scan dict typically contains:

- `'m/z array'` -> `np.ndarray`
- `'intensity array'` -> `np.ndarray`
- `'mean inverse reduced ion mobility array'` -> `np.ndarray`
- mzML metadata fields like `'scanList'`
- optionally `'FAIMS compensation voltage'`
- optional `'ignore_ion_mobility'` marker

### 2. Raw hill linkage state (`hills_dict` from `detect_hills`)

From `biosaur2/cutils.pyx:635-642`:

- `hills_idx_array`: hill id per peak instance
- `orig_idx_array`: original peak index inside its scan
- `scan_idx_array`: scan index
- `mzs_array`: flattened observed m/z values
- `intensity_array`: flattened observed intensities
- optional `im_array`

Interpretation: this is a flattened point-level representation where hill IDs are progressively unified across adjacent scans.

### 3. Processed hill objects (`process_hills`)

`process_hills()` (`biosaur2/cutils.pyx:750`) transforms point-level structure into hill-level structure:

- `hills_idx_array_unique`: unique hill identifiers
- `hills_mz_median`: intensity-weighted m/z centroid per hill
- optional `hills_im_median`
- `hills_scan_lists`, `hills_scan_sets`, `hills_lengths`
- `hills_intensity_array` and `tmp_mz_array` per hill
- fast lookup indexes:
  - `hills_mz_median_fast_dict` (bucketed by `int(mz/mz_step)`)
  - optional `hills_im_median_fast_dict`
- lazy caches:
  - `hills_idict`, `hill_sqrt_of_i`
  - `hills_intensity_apex`, `hills_scan_apex`

### 4. Candidate/final feature dict shape

Built in `get_initial_isotopes()` (`biosaur2/cutils.pyx:222`) and finalized in `calc_peptide_features()` (`biosaur2/utils.py:125`):

- monoisotope identity:
  - `monoisotope hill idx`
  - `monoisotope idx`
  - `hill_mz_1`
- isotope chain:
  - `isotopes` list of dicts (`isotope_number`, `isotope_hill_idx`, `isotope_idx`, `mass_diff_ppm`, `cos_cor`)
  - `nIsotopes`
- scoring/metadata:
  - `cos_cor_isotopes`
  - `charge`
  - `nScans`
  - `FAIMS`, `im`
  - `intensity_array_for_cos_corr`
- final export fields added later:
  - `massCalib`, `rtApex`, `rtStart`, `rtEnd`, `intensityApex`, `intensitySum`, `isoerror`, `isoerror2`, etc.

---

## Stage A: Input and Preprocessing

### Entry points

- `main.process_file()` in `biosaur2/main.py:404`
- `utils.process_mzml()` in `biosaur2/utils.py:472`

### Algorithm

1. Read MS1 scans using `MS1OnlyMzML` (`biosaur2/utils.py:16`), which XPath-filters to ms level 1.
2. For each scan, apply:
   - intensity threshold `-mini` (`biosaur2/utils.py:501`)
   - lower m/z `-minmz` (`biosaur2/utils.py:507`)
   - upper m/z `-maxmz` (`biosaur2/utils.py:513`)
   - sorting by m/z (`biosaur2/utils.py:519`)
3. Optionally merge every N scans if `-combine_every > 1` (`biosaur2/utils.py:483`, `biosaur2/utils.py:533`).
4. In `main.process_file`, split processing by FAIMS CV values (`biosaur2/main.py:429`, `biosaur2/main.py:437`).
5. Optional preprocessors:
   - TOF denoising (`-tof`) in `utils.process_tof` (`biosaur2/main.py:459`, `biosaur2/utils.py:389`)
   - profile peak reduction (`-profile`) in `utils.process_profile` (`biosaur2/main.py:463`, `biosaur2/utils.py:333`)
   - PASEF centroiding if ion mobility exists (`biosaur2/main.py:468`, `biosaur2/utils.py:231`)

### PASEF centroiding specifics

`centroid_pasef_scan` (`biosaur2/cutils.pyx:334`) groups near points by:

- m/z within dynamic tolerance (`mz * 1e-6 * htol`)
- IM bucket proximity and absolute IM difference (`<= paseftol`)
- summed intensity threshold `-pasefmini`
- group length threshold `-pasefminlh`

Output points are intensity-weighted averages in m/z and IM.

### Design observations

- `process_file` currently couples IO, calibration, hill construction, and output writing in one large function (`biosaur2/main.py:404`), making it difficult to isolate behavior for tests.
- Preprocessing mutates shared scan dicts in place, so downstream expectations depend on mutation order.

---

## Stage B: Hill Detection (`detect_hills`)

### Entry

- `detect_hills` in `biosaur2/cutils.pyx:618`

### Core idea

Treat each peak in each scan as initially belonging to a unique hill, then greedily connect peaks between adjacent scans when they are close in m/z (and IM if enabled) and pass tolerance.

### Detailed algorithm

1. Initialize every new scan’s peaks with new hill IDs (`biosaur2/cutils.pyx:655`).
2. Sort current scan peaks by descending intensity (`biosaur2/cutils.pyx:659`).
3. Build fast integer bins for current scan:
   - m/z bins via `get_fast_dict` (`biosaur2/cutils.pyx:673`)
   - IM bins similarly when IM enabled (`biosaur2/cutils.pyx:671`)
4. For each current peak:
   - gather candidate previous-scan peaks from same/neighbor bins (`fm-1, fm, fm+1`) (`biosaur2/cutils.pyx:679`)
   - optionally filter by IM neighboring bins (`biosaur2/cutils.pyx:698`)
   - among candidates not already used in this scan (`banned_prev_idx_set`), choose the **highest-intensity** predecessor that passes mass tolerance (`biosaur2/cutils.pyx:703-731`)
5. If match found, assign current peak to predecessor hill ID (`biosaur2/cutils.pyx:731`).
6. Record signed mass diff for calibration diagnostics (`biosaur2/cutils.pyx:735`).

### Mass-difference formula and correction

Raw signed ppm error per candidate:

`raw_ppm = ((mz_cur - mz_prev) / mz_cur) * 1e6`

Corrected error uses `md_correction` coefficient:

- `Orbi`: `coef = sqrt(600 / mz_cur)` (`biosaur2/cutils.pyx:714-715`)
- `Tof`: `coef = 1` (`biosaur2/cutils.pyx:716-717`)
- `Icr`: `coef = 600 / mz_cur` (`biosaur2/cutils.pyx:718-719`)

Then:

`err_ppm = raw_ppm * coef`

A candidate is valid if `abs(err_ppm) <= htol`.

### Consequence of design

- Matching is local to adjacent scans only and greedy by predecessor intensity.
- `banned_prev_idx_set` enforces one-to-one predecessor usage per scan transition, reducing many-to-one merges.

---

## Stage C: Hill Splitting (`split_peaks`)

### Entry

- Python orchestrator `split_peaks_multi` in `biosaur2/main.py:286`
- Cython splitter `split_peaks` in `biosaur2/cutils.pyx:439`

### Problem addressed

Hill linking can merge multiple chromatographic peaks into one long hill. This stage splits hills when valley evidence indicates multiple components.

### Algorithm

For each hill of length at least `2 * max(2, minlh)`:

1. Pull intensity trace in scan order (`biosaur2/cutils.pyx:466-469`).
2. Smooth with moving-average kernel width 3 (`meanfilt`) (`biosaur2/cutils.pyx:470`, `biosaur2/cutils.pyx:322`).
3. Scan candidate valley positions:
   - compute left ratio `l_r = left_max / valley`
   - compute right ratio `r_r = right_max / valley`
4. Valley accepted when both ratios exceed `hvf` and split preserves minimum-length constraints (`biosaur2/cutils.pyx:484-503`).
5. Reassign suffix segment to a new hill ID (`biosaur2/cutils.pyx:504`).

### Behavior notes

- Splitting is deterministic for given trace and params.
- Smoothing and strict min-length constraints suppress over-splitting of noisy short hills.

---

## Stage D: Hill Post-Processing (`process_hills`)

### Entry

- `process_hills` in `biosaur2/cutils.pyx:750`

### Responsibilities

1. Filter out hills shorter than `minlh` (`biosaur2/cutils.pyx:772-776`).
2. Sort flattened points by `(hill_id, scan)` (`biosaur2/cutils.pyx:780-783`).
3. For each hill:
   - collect scan list and intensity list
   - compute intensity-weighted m/z centroid (`biosaur2/cutils.pyx:816-821`)
   - compute weighted IM centroid if enabled (`biosaur2/cutils.pyx:822-824`)
4. Build lookup indexes:
   - m/z fast dict with neighboring buckets inserted (`biosaur2/cutils.pyx:832-836`)
   - IM fast dict likewise (`biosaur2/cutils.pyx:841-844`)
5. Initialize lazy caches for RT cosine and apex computations (`biosaur2/cutils.pyx:854-857`).

### Result

This stage converts low-level point links into searchable hill objects used by isotope discovery.

---

## Stage E: Initial Isotope Clustering (`get_initial_isotopes`)

### Entry

- `get_initial_isotopes` in `biosaur2/cutils.pyx:71`
- called via wrapper in `biosaur2/main.py:395`

### Inputs of interest

- charge range: `cmin..cmax`
- isotope index list: `1..9` (`isotopes_list[1:]`)
- isotope mass tolerance `itol`
- IVF split parameter `ivf`
- precomputed averagine intensity templates from binomial distribution (prepared in `biosaur2/main.py:21-34`)

### Algorithm

For each candidate monoisotopic hill (sorted by hill m/z):

1. Loop charges high-to-low (`biosaur2/cutils.pyx:90`).
2. For isotope number `k`, compute expected isotope m/z:
   - `mz_expected = mono_mz + 1.00335 * k / charge` (`biosaur2/cutils.pyx:119`)
3. Query nearby candidate hills from `hills_mz_median_fast_dict` and optional IM dict.
4. Require:
   - RT interval overlap (`biosaur2/cutils.pyx:126`)
   - at least one shared scan (`biosaur2/cutils.pyx:132`)
   - RT-shape cosine correlation >= 0.6 (`biosaur2/cutils.pyx:138-140`)
   - mass error within `itol` (with md correction) (`biosaur2/cutils.pyx:130`, `biosaur2/cutils.pyx:147-155`)
5. Build combinatorial candidate chains with `itertools.product(*candidates)` (`biosaur2/cutils.pyx:193`).
6. Score envelope fit:
   - theoretical isotope intensities from averagine template (`biosaur2/cutils.pyx:188-191`)
   - experimental apex intensities from selected isotope hills
   - IVF-based truncation when a later isotope rebounds strongly after a local minimum (`biosaur2/cutils.pyx:205-211`)
   - cosine-based carbon-pattern acceptance with threshold 0.6 (`biosaur2/cutils.pyx:216`)
7. Emit provisional feature candidate (`ready`) with monoisotope, isotope list, charge, cosine score, and intensity arrays (`biosaur2/cutils.pyx:222-236`).

### Charge banning heuristic

`charge_ban_map` and `banned_charges` avoid emitting lower-divisor charges when a higher-confidence chain was found (`biosaur2/cutils.pyx:83-85`, `biosaur2/cutils.pyx:238-239`).

---

## Stage F: Isotope Mass-Error Self-Calibration

### Entry

- `process_features_iteration` in `biosaur2/main.py:14`

### Purpose

The initial isotope candidates use static tolerances. This stage estimates dataset-specific isotope mass error distributions and prunes inconsistent tails.

### Algorithm

1. Build `isotopes_mass_error_map` for isotope numbers 1..9 (`biosaur2/main.py:94-105`).
2. If enough points (`>=1000`) for isotope 1..3:
   - fit Gaussian-like model via `utils.calibrate_mass` (`biosaur2/main.py:120`, `biosaur2/utils.py:24`)
   - store fitted shift and sigma.
3. If sparse, propagate estimates from earlier isotope numbers using incremental extrapolation (`biosaur2/main.py:132-143`).
4. If `ignore_iso_calib`, skip fitting and use `[0, itol]` for all isotope numbers (`biosaur2/main.py:88-92`).
5. For each candidate feature, keep isotope chain prefix where:
   - `abs(mass_diff_ppm - shift) <= 5 * sigma` (`biosaur2/main.py:161`)
6. Recompute cosine match on truncated intensity arrays (`biosaur2/main.py:169-179`).
7. Drop candidates failing recalculated cosine acceptance.

### Net effect

Self-calibration adapts isotope acceptance to observed instrument/run-specific bias and spread.

---

## Stage G: Conflict Resolution and Feature Export

### Entry

- conflict resolution in `biosaur2/main.py:199-266`
- feature completion in `biosaur2/utils.py:125`
- writing in `biosaur2/utils.py:169`

### Conflict resolution algorithm

1. Sort candidates by `-(nIsotopes + cos_cor_isotopes)` (`biosaur2/main.py:199`).
2. Maintain `ready_set` of already-claimed hill IDs.
3. Greedily accept a feature if monoisotope hill and isotope hills are all unclaimed (`biosaur2/main.py:216-222`).
4. If partially conflicting, truncate isotope chain until first claimed isotope and rescore (`biosaur2/main.py:227-248`).
5. Remove failed/empty candidates.

### Final feature field computation

`utils.calc_peptide_features` does:

- neutral mass:
  - `massCalib = mz*charge - 1.0072765*charge*(sign)` (`biosaur2/utils.py:134`)
- apex/sum intensity from monoisotope plus optional isotope contribution controlled by `-iuse` (`biosaur2/utils.py:140-150`)
- RT fields from either runtime `RT_dict` (mzML path) or imported hills values (`biosaur2/utils.py:154-161`)

### Output

- Features table columns in `biosaur2/utils.py:196-214`.
- Hills table columns in `biosaur2/utils.py:182-194`.
- `--stop_after_hills` skips Stage E/F/G after writing hills (`biosaur2/main.py:526` and `biosaur2/search.py:60`).

---

## Parameter-to-Algorithm Map

| CLI parameter | Where used | Behavioral effect |
|---|---|---|
| `-htol` | `detect_hills` (`biosaur2/cutils.pyx:631`), optional hill calibration (`biosaur2/main.py:475`) | Hill-link mass tolerance (ppm-like domain with md correction). |
| `-itol` | `process_features_iteration` (`biosaur2/main.py:15`) + `get_initial_isotopes` | Isotope candidate mass tolerance baseline. |
| `-hvf` | `split_peaks` (`biosaur2/cutils.pyx:448`) | Valley strength threshold for splitting hills. |
| `-ivf` | `get_initial_isotopes` (`biosaur2/cutils.pyx:209`) | Isotopic envelope truncation on rebound intensity. |
| `-minlh` | filtering in `split_peaks_multi` + `process_hills` | Minimum hill length retained and split constraints. |
| `-cmin`, `-cmax` | `process_features_iteration` + `get_initial_isotopes` | Search range of charge states. |
| `-md_correction` | parsed in `main.py:38-47` and `main.py:410-419`; consumed in Cython | Chooses mass-error normalization mode (`Orbi`, `Tof`, `Icr`). |
| `-use_hill_calib` | `main.py:475-509` | Optional auto-optimization of `htol` from temporary run statistics. |
| `-ignore_iso_calib` | `main.py:88` | Disables isotope self-calibration, uses static tolerance. |
| `--stop_after_hills` | `search.py:60`, `main.py:526` | Forces hills output and skips feature detection. |
| `-paseftol` | `main.py:473`, `cutils.pyx` IM logic | Enables IM-aware matching/centroiding and hill/link constraints. |
| `-pasefmini`, `-pasefminlh` | `centroid_pasef_scan` | Intensity and length thresholds during IM centroiding. |
| `-combine_every` | `utils.process_mzml` (`biosaur2/utils.py:483`) | Merges N consecutive scans before detection. |

---

## Refactor Hotspots and Engineering Risks

### 1) Multiprocessing partitioning in isotope stage

- Location: `biosaur2/main.py:66-80`
- Issue:
  - `step = int(len_full / n_procs)` drops remainder indices.
  - if `len_full < n_procs`, `step` becomes 0 and child slices are empty.
- Risk: missed monoisotope candidates and unstable results under high `nprocs`.
- Refactor guidance:
  - replace manual slicing with chunking that covers all indices exactly once.
  - explicitly guard `len_full == 0`.

### 2) Potential empty-list crash at `ready[0]`

- Location: `biosaur2/main.py:204`
- Issue: code assumes `ready` non-empty before conflict-resolution loop.
- Risk: runtime `IndexError` when no candidate isotopic clusters survive calibration/filtering.
- Refactor guidance:
  - fast-return empty output when `ready` is empty.

### 3) Sentinel bug risk in predecessor assignment

- Location: `biosaur2/cutils.pyx:703-734`
- Issue: `best_idx_prev` initialized to 0 and acceptance check is `if best_idx_prev != 0`.
- Risk: valid predecessor at index 0 can never be accepted as a match.
- Refactor guidance:
  - use `-1` sentinel or explicit boolean `found_match`.

### 4) Duplicated `md_correction` parsing

- Locations: `biosaur2/main.py:38-47` and `biosaur2/main.py:410-419`
- Issue: duplicate branching and fallback behavior in multiple functions.
- Risk: divergence during future edits.
- Refactor guidance:
  - centralize conversion in one utility function returning enum/int code.

### 5) Algorithm/IO coupling in `process_file`

- Location: `biosaur2/main.py:404`
- Issue: one method handles reading, preprocessing, calibration, hill processing, feature generation, and output writing.
- Risk: hard-to-test branch combinations and complex control flow.
- Refactor guidance:
  - split into pure compute stages plus orchestration shell.

### 6) Monolithic dict-of-arrays contracts

- Locations: broad (`hills_dict`, feature dicts)
- Issue: implicit schema, mutable cross-stage contracts, no typed validation.
- Risk: fragile refactors, hidden key dependencies.
- Refactor guidance:
  - introduce typed structures (`dataclass`/`TypedDict`) and conversion boundaries.

---

## Safe Reshape Plan (Module Boundaries and Mapping)

Target boundary design for incremental refactor:

1. `io_preprocess`
   - current mapping:
     - `utils.process_mzml`
     - `utils.process_profile`
     - `utils.process_tof`
     - `utils.centroid_pasef_data`
   - output contract: normalized scan stream object.
2. `hill_builder`
   - current mapping:
     - `cutils.detect_hills`
     - optional `use_hill_calib` helper path in `main.process_file`
   - output contract: raw hill linkage state.
3. `hill_splitter`
   - current mapping:
     - `main.split_peaks_multi`
     - `cutils.split_peaks`
   - output contract: relabeled hill linkage with split-applied hill IDs.
4. `isotope_linker`
   - current mapping:
     - `cutils.process_hills`
     - `cutils.get_initial_isotopes`
     - mass error calibration part of `main.process_features_iteration`
   - output contract: calibrated non-final isotope cluster candidates.
5. `feature_resolver`
   - current mapping:
     - overlap resolution in `main.process_features_iteration`
     - `utils.calc_peptide_features`
     - `utils.write_output`
   - output contract: stable feature rows + explicit writer interface.

Principle: isolate pure algorithm components from side effects (file IO/logging/process spawning), then wrap with thin orchestrators.

---

## Verification Checklist for Future Edits

Use this checklist whenever reshaping pipeline internals.

1. **Baseline traceability**
   - Confirm each stage (A-G) still exists conceptually and can be mapped to code path.
2. **Hills-only mode**
   - With `--stop_after_hills`, verify no feature rows are produced and hills rows are produced.
3. **IM-enabled behavior**
   - Confirm IM branch uses centroiding and IM-aware matching, and non-IM data cleanly disables IM logic.
4. **Sparse isotope calibration**
   - Validate fallback propagation path when isotope-level sample count is below threshold.
5. **Overlap invariant**
   - In final features, no hill ID should appear in more than one feature’s mono/isotope set.
6. **Stability envelope**
   - On a fixed dataset, track:
     - feature count
     - charge distribution
     - `nIsotopes` distribution
     - first-isotope `isoerror` mean/std
   - Expect only controlled shifts after intentional algorithm changes.
7. **Runtime scaling**
   - Compare `nprocs=1` vs `nprocs>1` for result parity and throughput.
8. **Regression watchpoints**
   - specifically test:
     - no-candidate path (avoid `ready[0]` crash),
     - predecessor index 0 matching,
     - full index coverage under multiprocessing partitioning.

---

## Public API / Interface Impact (Current Step)

1. No runtime API changes are introduced by this documentation task.
2. No CLI behavior changes are introduced by this documentation task.
3. This document defines a stable conceptual contract for internal structures (`hills_dict`, feature dict) to support future typed refactors.

---

## Assumptions Used in This Study Note

1. Language: English.
2. Scope is MS1 pipeline only.
3. Style target: algorithm explanation plus refactor hotspots.
4. References are based on current repository line positions at time of writing.
5. `dev/study/understand.biosaur2.md` was authored from empty state.
