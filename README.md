# biosaur2

Biosaur2 detects and quantifies isotope features in centroided LC-MS1 mzML
data. Version 0.5.0 adds an opt-in identification-aware hybrid DDA workflow:
it combines strict MS1 feature detection with q-value-filtered Percolator PSMs,
generic MS2 precursor evidence, intensity-conserving residual allocation and
explicit MS2-to-feature audit output.

For the algorithm and data-flow design, read [design.md](design.md). For the
implemented changes and scientific limits, read the
[2026-07-30 update](updates/2026-07-30.md#summary), its
[validation results](updates/2026-07-30.md#validation-results), and the
[expected-versus-actual conclusions](updates/2026-07-30.md#expected-versus-actual-outcome).

## Install and run

```bash
pip install biosaur2
biosaur2 sample.mzML.gz
```

The plain command writes `sample.features.tsv`. Existing output is not replaced
unless `--overwrite` is supplied. DuckDB is optional but is the preferred
Parquet writer:

```bash
pip install 'biosaur2[duckdb]'
biosaur2 sample.mzML.gz --feature-format parquet
```

Each requested Parquet product uses DuckDB Parquet V2, ZSTD level 6, and a
default row group of 122880. Feature Parquet is one `sample.features.parquet`,
not a directory or a set of normalized sidecars. DuckDB is also selected when
only hills or MS1 uses Parquet while features remain TSV. If DuckDB is
unavailable, Biosaur2 emits a visible warning and falls back to an optimized
PyArrow writer. Selecting `--parquet-engine pyarrow` explicitly avoids that
fallback check.

## Before you run

### Supported input files

The normal feature workflow accepts one or more centroided mzML files:

```text
sample.mzML
sample.mzML.gz
```

Use centroided MS1 spectra. Biosaur2 can read DDA MS2 precursor metadata from
the same mzML in `hybrid` mode. Profile processing and the
experimental DIA/DIA2 paths have different assumptions and should not be used
as a substitute for centroiding a conventional DDA experiment first.

The legacy detector can also consume reusable hill files:

```text
sample.hills.tsv
sample.hills.parquet
sample.hills.npz
```

Hills input does not support hybrid processing, MS2 sidecars, MS1 sidecars or
writing hills again. A hills file written with `--no-hill-list` is intentionally
small but cannot be reused for feature detection because its point-level trace
information was removed.

For direct identification evidence, provide a same-run Percolator target PSM
table, usually named like:

```text
sample.percolator.target.psms.tsv
sample.percolator.target.psms.tsv.gz
```

The reader accepts common tabular compression/encoding variants. It requires
semantic PSM ID, q-value and peptide columns, maps PSMs to MS2 by native
spectrum identity when possible, and otherwise validates a safe
`<run>_<scan>_<charge>_<rank>` PSM ID form. Fixed modifications must be passed
explicitly, for example `--fixed-mod C=UNIMOD:4`; Biosaur2 never silently
assumes carbamidomethylation from a mass shift.

A minimal tab-separated input looks like this (the common Percolator header
spellings are recognized semantically):

```text
PSMId	score	q-value	posterior_error_prob	peptide	proteinIds
sample_12345_2_1	3.17	0.0042	0.0018	PEPT[UNIMOD:35]IDE	sp|P01234|
```

`PSMId`, `q-value`, and `peptide` (or recognized aliases) are required. `score`,
PEP, proteins, scan, charge, rank, native ID and target/decoy columns are
optional. If scan/charge are not separate columns, the PSM ID must contain
enough identity to map safely. Passing a Percolator target table does not skip
q-value filtering, and rank greater than one is not rejected merely because of
rank.

### Choose the workflow

| Situation | Recommended command mode | Why |
| --- | --- | --- |
| Ordinary untargeted LC-MS1 feature detection | default `legacy` | Fast, established strict MS1 population; use when MS2 association is not required. |
| DDA data, no trustworthy PSM table | `--feature-mode hybrid` | Keeps strict MS1 detection and uses generic MS2 precursor/charge/C13 evidence with target/decoy control. |
| DDA data with Percolator PSMs | `--feature-mode hybrid --psm-path ...` | Adds q-filtered exact peptide/charge/isotope assays to improve association and local recovery. |
| Several comparable runs, with optional PSMs | `biosaur2 project run --mode hybrid` | Adds bounded project execution and optional RT-aligned external assays; recipient-run MS1 is always re-extracted. |
| Need compact downstream tables | add `--feature-format parquet` or `--duckdb-output ...` | Preserves feature information in compact structured output. |
| Repeated tuning on one large run | hybrid plus raw/strict/candidate caches | Avoids repeating mzML ingestion and hill work while rejecting stale cache state. |
| Standard hybrid is reliable but leaves difficult MS2 unresolved | audit first, then optionally add `--relaxed-ms2-feature` | Adds one guarded MS2-only retry; non-MS2 thresholds stay strict. |

Hybrid mode is opt-in and does not lower the standard for features without MS2.
It is designed to associate each MS2 event whenever the observed MS1 evidence
is scientifically defensible, not to force a feature for every precursor.

## Feature modes and hybrid DDA workflow

The single-file CLI has two explicit modes:

```text
--feature-mode legacy
--feature-mode hybrid
```

`legacy` preserves ordinary Biosaur2 detection. `hybrid` is an optional
identification-aware residual workflow and is off by default. It keeps the
strict untargeted feature population, then uses evidence in this order:

1. q-value-filtered direct PSM assays from the same run;
2. calibrated exact assays aligned from compatible project runs;
3. generic unidentified-MS2 precursor and C13-offset hypotheses;
4. a final strict untargeted detector on unallocated residual MS1 signal.

Direct PSMs are strong priors, not unconditional feature truth. A single
survey-scan precursor signal is audited as `precursor_signal_only` and is not
given a quantitative feature ID. Multiple PSMs or MS2 events may support one
feature, but its signal and abundance are allocated once.

### Suggested single-run commands

Start with the default legacy mode when the goal is ordinary strict MS1 feature
detection and quantification:

```bash
biosaur2 sample.mzML.gz
```

Use Parquet for most programmatic downstream workflows. Keep the point arrays
unless you know they are not needed for later hill/trace inspection:

```bash
biosaur2 sample.mzML.gz \
  --feature-format parquet \
  --parquet-sort mz_rt
```

Use hybrid mode without PSMs when the run is DDA and you want each MS2 precursor
evaluated against strict MS1 features plus generic, target/decoy-controlled
local evidence:

```bash
biosaur2 sample.mzML.gz \
  --feature-mode hybrid \
  --generic-ms2-refine \
  --generic-q-value-max 0.01 \
  --generic-ms2-ppm 10 \
  --ms2-rt-tolerance-sec 120 \
  --max-charge 7 \
  --feature-format parquet
```

Use a same-run Percolator PSM table when it is available. This is the preferred
hybrid command for DDA proteomics because exact peptide chemistry and selected
isotope information can guide local feature recovery:

```bash
biosaur2 sample.mzML.gz \
  --feature-mode hybrid \
  --psm-path sample.percolator.target.psms.tsv \
  --psm-q-value-max 0.01 \
  --fixed-mod C=UNIMOD:4 \
  --generic-q-value-max 0.01 \
  --generic-ms2-ppm 10 \
  --ms2-rt-tolerance-sec 120 \
  --max-charge 7 \
  --quant-method envelope_area \
  --feature-baseline edge_linear \
  --raw-ms1-cache-dir cache/sample-raw \
  --hybrid-stage-cache-dir cache/sample-strict \
  --hybrid-candidate-cache-dir cache/sample-local \
  --feature-format parquet
```

Use `--relaxed-ms2-feature` only after checking standard hybrid output and only
when additional MS2 coverage is worth evaluating. It is still conservative:
it is limited to unresolved MS2-supported candidates, keeps target/decoy or
direct q-value control, rejects single-point quantitative features, and remains
off by default.

```bash
biosaur2 sample.mzML.gz \
  --feature-mode hybrid \
  --psm-path sample.percolator.target.psms.tsv \
  --relaxed-ms2-feature \
  --feature-format parquet
```

For iterative tuning of hybrid settings, keep caches on fast storage. The raw
cache is required when a strict-stage cache is requested; stale scientific or
source fingerprints are refused automatically:

```bash
biosaur2 sample.mzML.gz \
  --feature-mode hybrid \
  --raw-ms1-cache-dir cache/sample.raw \
  --hybrid-stage-cache-dir cache/sample.strict \
  --hybrid-candidate-cache-dir cache/sample.candidates \
  --feature-format parquet
```

Percolator q-value filtering defaults to `0.01`; rank greater than one is not
discarded solely because of rank. Fixed modifications must be explicit and
are never inferred from precursor mass. The targeted RT tolerance defaults to
120 seconds. `--relaxed-ms2-feature` enables one conservative retry for
unresolved MS2-supported local evidence; it never lowers thresholds for
features without MS2. Generic and external extraction q-values are separate
from the Percolator q-value.

Hybrid mode writes one de-duplicated feature population plus compact sidecars:

```text
<stem>.features.parquet
<stem>.feature_quant.parquet
<stem>.ms2.parquet
<stem>.ms2_feature_links.parquet
<stem>.identifications.parquet
<stem>.id_assays.parquet
```

There is exactly one MS2 audit row per event, including honest null outcomes.
The hybrid summary reports audit coverage, observed local MS1-signal coverage,
quantitative-feature coverage, direct/generic coverage, and mutually exclusive
unresolved categories. Observed local signal is not itself a quantitative
feature.

## Hybrid quantification

Hybrid mode exposes exactly three feature-level methods:

| Method | Definition |
| --- | --- |
| `envelope_area` | trapezoidal area of the summed final assigned isotope traces on actual RT seconds (default) |
| `mono_area` | trapezoidal area of the final monoisotopic contribution |
| `envelope_apex` | maximum summed assigned isotope intensity at one common MS1 scan |

`--feature-baseline` is either `none` or `edge_linear`; it is preprocessing,
not a fourth quantification method. The compact quantification sidecar records
raw/corrected areas, the selected value, status, origin, evidence counts and
quality flags. Legacy `intensityApex`, `intensitySum`, and `area_sum` semantics
are unchanged.

## Project manifests and alignment

Build a deterministic manifest from exact normalized stems:

```bash
biosaur2 project make-manifest \
  --mzml-dir mzml \
  --psm-dir percolator \
  --psm-suffix .percolator.target.psms.tsv \
  --output runs.tsv
```

Run and validate a bounded project:

```bash
biosaur2 project run \
  --manifest runs.tsv \
  --output-dir results \
  --project-db results/project.duckdb \
  --mode hybrid \
  --run-workers 4 \
  --nprocs 20 \
  --allow-nested-parallelism \
  --psm-q-value-max 0.01 \
  --external-id \
  --external-q-value-max 0.01 \
  --generic-q-value-max 0.01 \
  --hybrid-stage-cache

biosaur2 project validate --project-db results/project.duckdb
```

Project execution also defaults to `legacy`; pass `--mode hybrid` explicitly
to enable identification/MS2-guided residual feature detection. This keeps the
new workflow opt-in for both single-run and manifest-driven commands.

Only `run_id` and `mzml_path` are required. Optional columns include
`psm_path`, `psm_format`, `identification_config`, `fixed_mods`, `q_value_max`,
sample/condition/replicate/fraction/batch metadata, and `alignment_group`.
External assays stay inside compatible groups, use robust monotonic RT
alignment, and always quantify recipient-run MS1 signal rather than copying
donor abundance. See `examples/hybrid_project_manifest.tsv` and
`examples/identification_config.json`.

Raw MS1, strict-stage, and local-candidate caches are fingerprinted and
atomically published. Reuse is refused when source, scientific parameters,
implementation signature, or residual ownership state differs. Project
parallelism is explicitly bounded; nested file/internal workers require
`--allow-nested-parallelism` so the total process budget is visible.

A minimal manifest needs only `run_id` and `mzml_path`:

```tsv
run_id	mzml_path
run_a	mzml/run_a.mzML.gz
run_b	mzml/run_b.mzML.gz
```

For direct PSM and alignment-aware hybrid processing, use the fuller shape in
[`examples/hybrid_project_manifest.tsv`](examples/hybrid_project_manifest.tsv):

```tsv
run_id	mzml_path	psm_path	fixed_mods	q_value_max	alignment_group
run_a	mzml/run_a.mzML.gz	percolator/run_a.percolator.target.psms.tsv	C=UNIMOD:4	0.01	batch_1
run_b	mzml/run_b.mzML.gz	percolator/run_b.percolator.target.psms.tsv	C=UNIMOD:4	0.01	batch_1
```

Use `alignment_group` only for runs that are scientifically comparable. External
assays are weaker than direct PSM assays, require enough good alignment anchors,
and always extract recipient-run MS1 intensity; they never copy donor-run
abundance. Use `--no-external-id` when runs should remain independent.

For high-throughput work, choose `run_workers` and `nprocs` from the available
CPU and memory budget. Their product is the visible maximum process budget only
when `--allow-nested-parallelism` is supplied. A conservative starting point is
one run worker with several internal workers; use multiple run workers when
files are independent and memory permits.

## Important command options

`biosaur2 --help` is the authoritative complete option list and prints the
parser default beside every option. The tables below explain how the options
affect practical use.

### Core detection and input

| Option | Default | Meaning and when to change it |
| --- | ---: | --- |
| `-mini` | 1 | Minimum centroid intensity considered during hill detection. Raise only when low-intensity noise dominates. |
| `-minmz`, `-maxmz` | 350, 1500 | Inclusive analysis m/z range. Narrow it only when acquisition or downstream scope is known. |
| `-htol`, `-itol` | 8 ppm, 8 ppm | Hill-linking and isotope-envelope mass tolerances. Keep instrument-appropriate; increasing them can merge unrelated evidence. |
| `-minlh`, `-pasefminlh` | 2, 1 | Minimum ordinary/PASEF hill point count. Do not use a lower ordinary hill length to manufacture quantitative features. |
| `-cmin`, `--max-charge` | 1, 7 | Charge search bounds. `--max-charge 7` is the current recommended ceiling for typical proteomics; increase only for credible higher-charge data. |
| `-iuse` | -1 | Isotopes used for legacy `intensityApex`, `intensitySum` and `area_sum`: -1 all assigned, 0 mono only, N mono plus N isotopes. |
| `-hvf`, `-ivf` | 1.3, 5.0 | Hill and isotope-pattern split sensitivity. Change only with a controlled validation set because both alter feature construction. |
| `-paseftol`, `-pasefmini` | 0.05, 100 | Ion-mobility/PASEF linking controls; leave defaults for ordinary non-PASEF data. |
| `-nm` | 0 | Ionization polarity: 0 positive, 1 negative. |
| `-tof`, `-profile`, `-use_hill_calib` | false | Experimental acquisition-specific paths. Validate separately before production use. |
| `--input-rt-unit` | seconds | Fallback only for metadata-free mzML or historical hills; mzML scan metadata takes precedence. |

### Hybrid evidence and quantification

| Option | Default | Meaning and when to change it |
| --- | ---: | --- |
| `--feature-mode` | `legacy` | Select strict untargeted `legacy` or opt-in `hybrid`. |
| `--psm-path` | empty | Same-run Percolator target PSM table for exact direct assays. Empty is valid for generic-only hybrid processing. |
| `--psm-q-value-max` | 0.01 | PSM quality filter applied before direct-assay construction. Do not raise it merely to increase links. |
| `--psm-pep-max` | none | Optional additional PEP filter. |
| `--fixed-mod` | none | Repeatable explicit fixed modification `SITE=MOD`, for example `C=UNIMOD:4`. |
| `--direct-id` | true | Enable direct PSM association/local recovery in hybrid mode. |
| `--generic-ms2-refine` | true | Enable generic unidentified-MS2 hypotheses, target/decoy association and local recovery. |
| `--generic-q-value-max` | 0.01 | Generic target/decoy extraction q-value threshold; separate from the PSM q-value. |
| `--generic-ms2-ppm` | 10 ppm | Selected-ion tolerance for generic precursor hypotheses. |
| `--ms2-rt-tolerance-sec` | 120 s | Initial local RT window around an MS2 event. Calibration may tighten a direct retry. |
| `--relaxed-ms2-feature` | false | One bounded, MS2-only retry for unresolved evidence. Keep false for the broad standard configuration; enable only for a measured A/B. |
| `--quant-method` | `envelope_area` | `envelope_area`, `mono_area`, or `envelope_apex` from final assigned signal. |
| `--feature-baseline` | hybrid: `edge_linear` | `none` or `edge_linear` baseline preprocessing. It is not a separate quantitative method. |

### Output, cache and execution

| Option | Default | Meaning and when to change it |
| --- | ---: | --- |
| `-o` | beside input | Output feature path for one file, or output directory for multiple files. Existing targets require `--overwrite`. |
| `--feature-format` | tsv | `tsv` for compatibility; `parquet` for typed downstream access. |
| `--write-hills`, `--write-ms1`, `--write-ms2` | false | Write optional hill, MS1 summary and DDA precursor sidecars. Hybrid automatically writes its MS2 audit sidecars. |
| `--hills-format`, `--ms1-format` | tsv, tsv | Format of requested hill/MS1 sidecars. |
| `--no-hill-list`, `--no-mono-hills` | false | Reduce output size. The former prevents reuse of hills input; the latter removes feature trace arrays. |
| `--write-extra-details` | false | Add large diagnostic nested feature columns for investigation. |
| `--duckdb-output` | empty | Write requested ordinary products into one DuckDB database; requires DuckDB. |
| `--parquet-engine` | duckdb | Preferred V2 writer. `pyarrow` is an explicit alternative; automatic fallback is logged. |
| `--64` | false | Use 64-bit structured output values instead of compact default storage. Detection remains float64 internally either way. |
| `--intensity-decimals` | 0 | Output-only intensity rounding; use `none` to retain fractions. |
| `--raw-ms1-cache-dir` | empty | Persist a compact memory-mappable raw MS1 cache for hybrid reuse. |
| `--hybrid-stage-cache-dir` | empty | Persist/reuse strict-stage hybrid context; requires a raw MS1 cache. |
| `--hybrid-candidate-cache-dir` | empty | Persist expensive target/decoy local candidates for the exact residual state. |
| `-nprocs` | 4 | Internal worker count per file. |
| `--run-workers` | 1 | File-level concurrency for ordinary mzML feature processing. |
| `--continue-on-error` | false | Continue independent multi-file work after a failure, while preserving a nonzero final status. |
| `--overwrite` | false | Atomically replace existing outputs. |

### Experimental DIA controls

`-dia`, `-dia2`, `-diahtol`, `-diaminlh`, `-diadynrange`, `-min_ms2_peaks`
and `-mgf` are retained for experimental DIA/DIA2 workflows. They are not part
of hybrid residual feature detection, do not support DuckDB output, and should
be validated separately for a dataset before routine use.

## Practical tips and important notes

- Use a PSM table from the same mzML run for direct assays. A PSM file with a
  similar name but a different run will not provide trustworthy direct RT/scan
  context.
- PSM q-value, generic extraction q-value and external-assay q-value are
  separate controls. Passing a PSM filter does not automatically accept a
  feature; observed MS1 isotope and chromatographic evidence is still required.
- A single MS1 survey point at the precursor is not a quantitative feature.
  Null `ms2_feature_links` outcomes are expected for sparse traces and are more
  trustworthy than forced associations.
- Several MS2 events may legitimately link to one feature. Do not sum linked
  event rows to obtain abundance; use `feature_quant.parquet` once per feature.
- Keep the default strict configuration for feature-only/no-MS2 populations.
  `--relaxed-ms2-feature` must never be treated as a global sensitivity switch.
- Run `biosaur2 project validate --project-db ...` after a project job. It
  verifies output contracts, IDs, audit coverage and published paths.
- Use `--overwrite` deliberately. Publication is atomic, but the option still
  replaces a complete existing target.
- Read [design.md](design.md) before changing hill splitting, residual,
  confidence or quantification settings in a method-development study.

## Compact feature output

Default feature columns, in order, are:

```text
massCalib rtApex intensityApex intensitySum charge nIsotopes nScans mz
rtStart rtEnd FAIMS im mono_hills_scan_lists mono_hills_intensity_list
scanApex isoerror isoerror2 feature_idx area_sum
```

`--no-mono-hills` removes the two large `mono_hills_*` arrays. This is useful
for downstream preprocessing that does not reuse the embedded monoisotopic
trace. `--write-extra-details` keeps its original role and appends `isotopes`,
`intensity_array_for_cos_corr`, `monoisotope hill idx`, and `monoisotope idx`
to the same feature file. It is intended for diagnostics and increases size.

The default Parquet/DuckDB physical types are:

| Columns | Type |
| --- | --- |
| floating scalars and float-list elements | FLOAT32 |
| `charge`, `nIsotopes` | INT8 |
| `nScans` | INT16 |
| `scanApex`, `feature_idx`, scan-list elements | INT32 |

`--64` widens structured numeric output. Internal detection, calibration,
scoring, conflict resolution, and area calculation continue in float64.
Narrow-integer overflow is an output error rather than a wrap.

Feature and hill `rtStart`, `rtApex`, and `rtEnd` are in minutes, matching the
0.3.2 column contract. MS1 `RT` is in seconds. Missing FAIMS, compatible ion
mobility (`im`), native `scanApex`, `isoerror2`, and `area_sum` values are null;
missing FAIMS is not encoded as zero.

`feature_idx` is deterministic, one-based, and always present. Hills use the
same value to identify their assigned feature; an unassigned hill has
`feature_idx = -1`.

## Intensities and area

`-iuse -1`, the default, uses all assigned isotopes for `intensityApex`,
`intensitySum`, and `area_sum`; `-iuse 0` selects mono only, and `-iuse N`
selects mono plus up to N assigned isotopes.

`area_sum` is the sum of raw trapezoidal trace areas for that isotope subset in
instrument-intensity × seconds. It is null if any selected trace cannot be
integrated. Exact per-point RT is used when available. For older hills without
point RT, Biosaur2 interpolates from the hill start/apex/end anchors and records
that fact in structured-output provenance.

Output intensities are rounded half away from zero to zero decimal places by
default and remain stored as floating point. This happens only during output;
it cannot alter feature detection or area calculation. Use
`--intensity-decimals none` to preserve fractional output or provide a
nonnegative decimal count.

Derived QC, shape, score, status, flag, isotope-row, and trace-point columns are
not written by default. The compact file retains baseline-compatible values
and only the additional `feature_idx` and `area_sum` needed for assignment and
area quantification.

## Hills, MS1, MS2, and DuckDB database output

```bash
biosaur2 sample.mzML.gz \
  --write-hills --hills-format parquet \
  --write-ms1 --ms1-format parquet \
  --write-ms2

# Small preprocessing-oriented outputs
biosaur2 sample.mzML.gz \
  --write-hills --hills-format parquet --no-hill-list \
  --no-mono-hills \
  --write-ms1 --ms1-format parquet \
  --feature-format parquet
```

Hills retain their compact nested shape. When point lists are enabled,
`hills_rt_list` stores exact per-point seconds so a hills file can be reused
without reconstructing RT. `--no-hill-list` removes all large hill point arrays
and makes that output unsuitable as feature-detection input. MS1 contains only
`scan_id`, `RT`, and `total_intensity`. `--write-ms2` writes one separate
`<input-stem>.ms2.parquet` precursor sidecar. It contains no fragment arrays or
chromatogram payloads. Its indexes are zero-based; RT is seconds; selected-ion
m/z and isolation target m/z remain distinct; and missing precursor m/z,
charge, unresolved `spectrumRef`, and missing preceding MS1 are recorded in
the documented `metadata_flags` bitmask. `ion_mobility` is populated only for
inverse reduced mobility (1/K0), so drift-time values are not mixed into that
column. The flags are `0x0001` missing precursor m/z, `0x0002` missing charge,
`0x0004` unresolved `spectrumRef`, and `0x0008` missing precursor MS1.

Parquet features, hills, and MS1 are separate requested products; the feature
product itself is always one file. To keep requested products in one database:

```bash
biosaur2 sample.mzML.gz --duckdb-output sample.biosaur2.duckdb
biosaur2 sample.mzML.gz --duckdb-output sample.biosaur2.duckdb \
  --write-hills --write-ms1
```

A `.duckdb` file contains a small `runs` provenance table, compact `features`,
and only explicitly requested `hills` and `ms1` tables. With `--write-ms2`,
the MS2 product remains the separate Parquet sidecar beside the database and
is published atomically with it. Unlike ordinary Parquet fallback, explicit
`--duckdb-output` requires DuckDB.

Structured outputs record schema and package versions, input path and size,
parameters, units, numeric and rounding policies, writer settings, and
calibration/area provenance. They do not reread the input to calculate a
content hash. Publication is atomic. TSV has a header but no metadata preamble
or sidecar.

## Input RT and multiple files

mzML retention-time metadata takes precedence. `--input-rt-unit` is the
fallback for metadata-free mzML or hills input and defaults to seconds. Use
`--input-rt-unit minutes` for pre-0.4 hills files whose scalar RT values are in
minutes.

Multiple inputs are accepted. With ordinary output, `-o` must be an output
directory; with multiple `--duckdb-output` inputs, that option must also be a
directory. All target collisions are checked before processing begins.
Hills inputs are feature-detection inputs only; `--stop-after-hills`,
`--write-hills`, and `--write-ms1` are rejected for them.

Use `--run-workers N` to bound normal mzML file-level parallelism. The default
is `1`, preserving the existing per-file `-nprocs` behavior. When `N > 1`, each
file runs in a fresh spawned process with effective `nprocs=1`; Biosaur2 logs
the requested and effective configuration and never nests file workers with
internal detection workers. At most `N` files are active, so completed file
memory is released before the worker handles another input. A final batch
report is printed in input order. By default the first failed file stops new
submissions; `--continue-on-error` completes independent files but still exits
nonzero if any fail. `--write-ms2` and `--run-workers > 1` are supported only
by the normal mzML feature workflow, not experimental DIA/DIA2 or hills input.

DIA and DIA2 remain experimental and receive compatibility smoke coverage.
`.duckdb` output is not supported in DIA modes. `--no-mono-hills` is not
accepted with `-dia`, because that workflow consumes the monoisotopic arrays.

## 0.5 migration notes

Version 0.5.0 retains the 0.4 output migration: the experimental normalized
three-file layout and the public options `--parquet-layout`, `--legacy-columns`,
`--scalar-float`, `--intensity-float`, `--quantification`,
`--output-rt-unit`, and DuckDB Parquet V1 selection. They are rejected instead
of silently acting as compatibility aliases. Use `--64` for wide structured
storage and `--input-rt-unit` only to describe metadata-free input.

The hybrid workflow is opt-in through `--feature-mode hybrid`; legacy remains
the default strict workflow. Hybrid output adds sidecars
for quantitative feature values, normalized MS2 events, MS2 audit links, PSM
mapping and direct assays. Read [design.md](design.md) before comparing hybrid
and legacy abundance semantics, and [the v0.5.0 update](updates/2026-07-30.md)
for validation evidence.

Run `biosaur2 --help` for the complete detection/output controls, current
defaults, units, types, file naming and command examples. Run
`biosaur2 project run --help` for multi-run defaults and external-assay controls.

## Citation

Abdrakhimov et al., “Biosaur: An open-source Python software for liquid
chromatography-mass spectrometry peptide feature detection with ion mobility
support.” https://doi.org/10.1002/rcm.9045
