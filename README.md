# biosaur2

Biosaur2 detects isotope features in centroided LC-MS1 mzML data. Version
0.4.0 improves deterministic processing, retention-time and native-scan
correctness, multiprocessing reliability, calibration fallbacks, and compact
output.

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

## Feature modes and hybrid DDA workflow

The single-file CLI has three explicit modes:

```text
--feature-mode legacy
--feature-mode weak-ms2
--feature-mode hybrid
```

`legacy` preserves ordinary Biosaur2 detection. `weak-ms2` preserves the
bounded compatibility seed (the existing `--ms2-seed` flag is its alias).
`hybrid` is an optional identification-aware residual workflow and is off by
default. It keeps the strict untargeted feature population, then uses evidence
in this order:

1. q-value-filtered direct PSM assays from the same run;
2. calibrated exact assays aligned from compatible project runs;
3. generic unidentified-MS2 precursor and C13-offset hypotheses;
4. a final strict untargeted detector on unallocated residual MS1 signal.

Direct PSMs are strong priors, not unconditional feature truth. A single
survey-scan precursor signal is audited as `precursor_signal_only` and is not
given a quantitative feature ID. Multiple PSMs or MS2 events may support one
feature, but its signal and abundance are allocated once.

A direct/generic single-run example is:

```bash
biosaur2 sample.mzML.gz \
  --feature-mode hybrid \
  --psm-path sample.percolator.target.psms.tsv \
  --psm-q-value-max 0.01 \
  --fixed-mod C=UNIMOD:4 \
  --generic-q-value-max 0.01 \
  --ms2-seed-rt-tolerance-sec 120 \
  --max-charge 7 \
  --quant-method envelope_area \
  --feature-baseline edge_linear \
  --raw-ms1-cache-dir cache/sample-raw \
  --hybrid-stage-cache-dir cache/sample-strict \
  --hybrid-candidate-cache-dir cache/sample-local \
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

## 0.4 migration notes

Version 0.4.0 intentionally removes the experimental normalized three-file
layout and the public options `--parquet-layout`, `--legacy-columns`,
`--scalar-float`, `--intensity-float`, `--quantification`,
`--output-rt-unit`, and DuckDB Parquet V1 selection. They are rejected instead
of silently acting as compatibility aliases. Use `--64` for wide structured
storage and `--input-rt-unit` only to describe metadata-free input.

Run `biosaur2 --help` for all detection and output controls, defaults, units,
types, file naming, and examples.

## Citation

Abdrakhimov et al., “Biosaur: An open-source Python software for liquid
chromatography-mass spectrometry peptide feature detection with ion mobility
support.” https://doi.org/10.1002/rcm.9045
