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
