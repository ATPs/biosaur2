# Outputs and quantification

This guide uses an input named `sample.mzML.gz`, whose input stem is
`sample`. Unless an output location is supplied, files are written beside the
input. For one input, `-o results/name.features.parquet` selects the feature
file path. In that case, MS2 and hybrid sidecars use the input stem in the same
directory (for example, `results/sample.ms2.parquet`), while ordinary hill and
MS1 sidecars use the feature prefix (for example,
`results/name.features.hills.tsv`). Use `-o results/name` when the ordinary
files should instead share the simpler `name.*` prefix. For several inputs,
`-o` is an output directory.

## Output file map

Every normal output that Biosaur2 can write is shown below. An option only
creates its corresponding file when it is requested, except that hybrid mode
always creates its evidence sidecars.

| Workflow or option | Files written | What each row represents |
| --- | --- | --- |
| Default | `sample.features.tsv` | One accepted MS1 isotope feature. |
| `--feature-format parquet` | `sample.features.parquet` | The same feature table in typed, columnar Parquet. |
| `--write-hills` | `sample.hills.tsv` | One chromatographic hill, a group of nearby signal points used to build features. |
| `--write-hills --hills-format parquet` | `sample.hills.parquet` | The hill table in Parquet. |
| `--stop-after-hills` | `sample.hills.tsv` | Hills only; feature detection is not run. Use `--hills-format parquet` for `sample.hills.parquet`. |
| `--write-ms1` | `sample.ms1.tsv` | One MS1 scan summary: scan ID, retention time in seconds, and total intensity. |
| `--write-ms1 --ms1-format parquet` | `sample.ms1.parquet` | The MS1 summary in Parquet. |
| `--write-ms2` | `sample.ms2.parquet` | One DDA precursor event; it contains precursor metadata, not fragment arrays. |
| `-dia` or `-dia2` | a requested `-mgf` path | Experimental DIA/DIA2 MGF spectra. Without `-mgf`, Biosaur2 derives a path from the input name. |

`--write-hills`, `--write-ms1`, and `--write-ms2` can be combined. The feature
format does not change the selected hill or MS1 format. `-dia` and `-dia2` are
experimental, separate workflows and do not produce the normal feature files.

### Ordinary feature detection

```bash
biosaur2 sample.mzML.gz
```

```text
sample.features.tsv
```

Use Parquet when the next tool can read it:

```bash
biosaur2 sample.mzML.gz --feature-format parquet
```

```text
sample.features.parquet
```

### Hills and optional scan sidecars

This command requests every ordinary sidecar in Parquet format:

```bash
biosaur2 sample.mzML.gz \
  --feature-format parquet \
  --write-hills --hills-format parquet \
  --write-ms1 --ms1-format parquet \
  --write-ms2
```

```text
sample.features.parquet   accepted MS1 isotope features
sample.hills.parquet      chromatographic hills used to construct features
sample.ms1.parquet        MS1 scan summaries
sample.ms2.parquet        normalized DDA precursor events
```

To inspect or save hills before feature assembly, stop after hill detection:

```bash
biosaur2 sample.mzML.gz --stop-after-hills --hills-format parquet
```

```text
sample.hills.parquet
```

Hills written with their point lists can be supplied as experimental input to a
later feature-detection command. Do not combine `--no-hill-list` with that
plan: it deliberately omits the point lists needed for reuse.

### Experimental DIA MGF output

Give `-mgf` an explicit filename when using the experimental DIA modes:

```bash
biosaur2 sample.mzML.gz -dia -mgf results/sample.dia.mgf
```

```text
results/sample.dia.mgf
```

## Hybrid output

Hybrid mode automatically writes four evidence sidecars. Add `--write-ms2` to
also keep the normalized MS2 event table. Empty identification or assay files
are still created with their column definitions, making the absence of accepted
evidence explicit.

```bash
biosaur2 sample.mzML.gz \
  --feature-mode hybrid \
  --psm-path sample.percolator.target.psms.tsv \
  --feature-format parquet \
  --write-ms2 \
  -o results/sample.features.parquet
```

```text
results/sample.features.parquet             one accepted MS1 isotope feature
results/sample.feature_quant.parquet        one abundance record per feature
results/sample.ms2.parquet                  one normalized MS2 precursor event
results/sample.ms2_feature_links.parquet    one association or unresolved result per MS2 event
results/sample.identifications.parquet      one parsed and mapped PSM record
results/sample.id_assays.parquet            one accepted peptide-specific assay
```

Without `--write-ms2`, the other five files in this example are still written,
but `sample.ms2.parquet` is not. The feature and quantification tables contain
the same unique `feature_id` values. Several MS2 events may link to one
feature, so do not use the link table to sum feature abundance. Hybrid commands
can also request the optional hill and MS1 sidecars listed above.

## DuckDB output

`--duckdb-output` stores the normal feature, hill, and MS1 tables in one
database. MS2 and all hybrid evidence products remain Parquet sidecars so they
are easy to inspect and join directly.

```bash
biosaur2 sample.mzML.gz \
  --duckdb-output results/sample.biosaur2.duckdb \
  --write-hills --write-ms1 --write-ms2
```

```text
results/sample.biosaur2.duckdb
  tables: features, hills, ms1
results/sample.ms2.parquet
```

With several input files, pass a directory to `--duckdb-output`. Biosaur2 then
creates one `<input-stem>.biosaur2.duckdb` database per input in that directory.

## Project output

Project processing keeps each run's data in a directory and uses the project
database as an index of the work, rather than copying every feature into one
large table. For a manifest run `run_a` whose input stem is `sample`:

```bash
biosaur2 project run \
  --manifest runs.tsv \
  --output-dir results \
  --project-db results/project.duckdb \
  --mode hybrid
```

```text
results/project.duckdb
results/run_a/run_a.features.parquet
results/run_a/sample.ms2.parquet
results/run_a/sample.feature_quant.parquet
results/run_a/sample.ms2_feature_links.parquet
results/run_a/sample.identifications.parquet
results/run_a/sample.id_assays.parquet
results/run_a/sample.external_id_evidence.parquet   only after external-ID extraction
```

`project.duckdb` records run status, resolved options, published paths, cache
stage status, retention-time alignment models, external-extraction summaries,
and validation metadata. `external_id_evidence` is written only in the
optional cross-run stage. It records both accepted and rejected recipient-run
measurements; it does not copy a donor run's abundance. See
[Project workflow](project-workflow.md) for the manifest and alignment steps.

## What the files look like

TSV files are plain text and can be opened in a spreadsheet. Parquet and
DuckDB files are binary, so inspect them with a table reader such as DuckDB.
For example:

```bash
duckdb -c "SELECT * FROM read_parquet('sample.ms2.parquet') LIMIT 2"
```

The examples below show selected columns and illustrative values; `...` means
the file has additional columns. Column names and values are the same whether a
table is TSV or Parquet.

### Feature table: `sample.features.tsv` or `sample.features.parquet`

`rtApex`, `rtStart`, and `rtEnd` are in minutes. `feature_idx` is the feature
identifier used to join this table to hybrid output.

```text
massCalib  rtApex  intensityApex  intensitySum  charge  nIsotopes  nScans  mz        rtStart  rtEnd  feature_idx  area_sum
997.3856   12.34   98231          641203        2       3          12      499.7001  12.18    12.51  42           641203
1201.5942  18.76   45612          338905        3       2          9       401.2052  18.64    18.87  43           338905
```

The full default table also contains FAIMS and ion-mobility values, the apex
scan, isotope error values, and monoisotopic-hill point lists.

### Hill table: `sample.hills.tsv` or `sample.hills.parquet`

Each row is one chromatographic trace. `feature_idx` identifies the feature it
was assigned to; `-1` means the hill was not assigned to an accepted feature.

```text
rtApex  intensityApex  intensitySum  nScans  mz        rtStart  rtEnd  scanApex  hill_idx  feature_idx
12.34   45213          293018        12      499.7001  12.18    12.51  1284      901       42
12.33   31784          198185        11      500.2018  12.19    12.49  1283      902       42
18.76   22106          150240        9       401.2052  18.64    18.87  1957      903       -1
```

Unless `--no-hill-list` is used, the table also includes lists of the scans,
intensities, m/z values, and retention times that make up each hill.

### MS1 scan summary: `sample.ms1.tsv` or `sample.ms1.parquet`

`RT` is in seconds, unlike the feature and hill retention-time columns.

```text
scan_id  RT      total_intensity
1283     738.1   18274630
1284     740.3   19650421
1285     742.6   19011782
```

### Normalized MS2 event table: `sample.ms2.parquet`

Each row represents one precursor entry in an MS2 spectrum. An MS2 spectrum
with multiple precursor entries therefore produces multiple rows. It contains
precursor metadata only, not fragment-peak arrays.

"Normalized" here does not change the spectrum or manufacture measurements.
It converts differently structured mzML precursor metadata into one fixed
table, so every event has the same column names. Biosaur2 chooses
`selected_ion_mz` as `precursor_mz` when available, otherwise uses
`isolation_target_mz`; `precursor_mz_source` records that choice. It resolves
the parent MS1 scan through the mzML `spectrumRef` when possible, otherwise
uses the preceding MS1 scan, with `precursor_resolution` recording which rule
worked. `metadata_flags` is a bit field that records missing precursor m/z,
missing charge, an unresolved `spectrumRef`, or no preceding MS1 scan.

```text
run_id  ms2_event_id  native_scan_number  rt_sec  precursor_ms1_index  precursor_resolution  selected_ion_mz  isolation_target_mz  precursor_mz  precursor_mz_source  charge  metadata_flags
sample  317           1542                741.7   1284                spectrum_ref          499.7001         500.0000             499.7001      selected_ion         2       0
sample  318           1543                743.2   1285                preceding_ms1         null             401.2052             401.2052      isolation_target     null    2
```

In the second row, `metadata_flags` value `2` means the charge was absent. The
other values are `1` for missing precursor m/z, `4` for an unresolved
`spectrumRef`, and `8` for a missing preceding MS1 scan; values add when more
than one condition applies.

### Experimental MGF: `sample.dia.mgf`

MGF is a text spectrum format produced only by the experimental `-dia` or
`-dia2` modes. It contains fragment m/z-intensity pairs, unlike the MS2 event
table above.

```text
BEGIN IONS
TITLE=sample.mzML.1542.1542.2
RTINSECONDS=741.700000
PEPMASS=499.700100 98231.000000
CHARGE=2+
126.0550 1842.0
175.1198 6321.0
END IONS
```

### Hybrid quantification: `sample.feature_quant.parquet`

There is one row for every accepted feature. `feature_id` is the same value as
`feature_idx` in the feature table.

```text
run_id  feature_id  feature_origin      confidence_tier  quant_value  quant_status        quant_envelope_area  quant_mono_area  quant_envelope_apex
sample  42          direct_identified   direct_id        428519.7     baseline_corrected  428519.7            241006.2        9831.0
sample  43          strict_untargeted   strict           231480.4     baseline_corrected  231480.4            152704.9        6412.0
```

### MS2-to-feature links: `sample.ms2_feature_links.parquet`

This is an audit table. It keeps unresolved MS2 events rather than dropping
them, so a null `feature_id` is meaningful.

```text
run_id  ms2_event_id  feature_id  association_tier  status                              mz_error_ppm  rt_error_sec  extraction_q_value
sample  317           42          direct_id         matched_strict_feature              1.4           0.0           null
sample  318           null        none              unresolved_no_direct_identification  null          null          null
```

### Parsed PSMs: `sample.identifications.parquet`

This table reports every parsed PSM and its mapping/assay outcome, including
PSMs that could not become direct assays.

```text
run_id  psm_id       ms2_event_id  mapping_status  q_value  peptide_raw  canonical_peptidoform  formula_status  assay_status
sample  scan=1542_1  317           mapped          0.0021   PEPTIDEK     PEPTIDEK               exact           accepted_direct_assay
sample  scan=1543_1  318           mapped          0.0068   ACDEK        ACDEK                  exact           accepted_direct_assay
```

### Accepted direct assays: `sample.id_assays.parquet`

Only exact, non-conflicting direct assays appear here. The table is empty when
no PSM passes those checks.

```text
run_id  assay_id  ms2_event_id  psm_id       canonical_peptidoform  charge  rt_sec  monoisotopic_mz  q_value  conflict_status
sample  1         317           scan=1542_1  PEPTIDEK               2       741.7   464.7348          0.0021   unique
sample  2         356           scan=1601_1  ACDEK                  2       812.4   283.1176          0.0040   unique
```

### Cross-run evidence: `sample.external_id_evidence.parquet`

This project-only table documents each aligned donor assay tested in a recipient
run. It records accepted and rejected extraction attempts and does not copy the
donor abundance.

```text
target_run  source_run  canonical_peptidoform  charge  predicted_rt_sec  competition_winner  extraction_q_value  status                             feature_id
run_b       run_a       PEPTIDEK               2       755.8             target              0.0031              accepted_matched_existing_feature  87
run_b       run_a       ACDEK                  2       826.4             decoy               1.0000              decoy_winner                       null
```

### DuckDB files

`sample.biosaur2.duckdb` contains the `features`, `hills`, and optional `ms1`
tables shown above. For example, this query returns ordinary features:

```sql
SELECT feature_idx, mz, rtApex, intensitySum
FROM features
LIMIT 2;
```

`project.duckdb` is a separate project index. Its `runs` and `qc_metrics`
tables look like this when queried:

```text
run_id  status   features_path                               feature_quant_path
run_a   success  results/run_a/run_a.features.parquet        results/run_a/sample.feature_quant.parquet
run_b   success  results/run_b/run_b.features.parquet        results/run_b/sample.feature_quant.parquet

run_id  feature_count  ms2_count  linked_ms2_count  quant_feature_count
run_a   18432          2311       968               18432
run_b   17905          2198       901               17905
```

## The three abundance measurements

`--quant-method all` is the default and writes all three final values:

| Column | Definition | Typical use |
| --- | --- | --- |
| `quant_envelope_area` | Trapezoidal area of all assigned isotope traces over actual RT seconds. | Default robust feature abundance. |
| `quant_mono_area` | Area of the monoisotopic trace only. | Comparisons that deliberately avoid heavier isotopes. |
| `quant_envelope_apex` | Maximum summed isotope intensity at one common MS1 scan. | Apex-based workflows. |
| `quant_value` | Envelope area when the method is `all`. | Compatibility and project donor ranking. |

The quantification table also keeps raw and baseline-corrected diagnostic area
columns, quantification status, points across the peak, evidence counts,
feature origin, quality flags, isotope similarity, and mass error.

Feature and hill intensities are written with fixed zero-decimal,
half-away-from-zero rounding for the compact output contract. Quantification
areas and hybrid quantitative values remain floating-point calculations.

## Example row

```text
run_id  feature_id  quant_method  quant_envelope_area  quant_mono_area  quant_envelope_apex
run_a   1842        all           428519.7             241006.2         9831.0
```

This says feature `1842` has three alternative summaries of the same assigned
MS1 traces. It does not represent three separate features.

## Safe joins

To inspect the feature abundance linked to each MS2 event:

```sql
SELECT
  links.ms2_event_id,
  links.status,
  quant.feature_id,
  quant.quant_envelope_area
FROM read_parquet('sample.ms2_feature_links.parquet') AS links
LEFT JOIN read_parquet('sample.feature_quant.parquet') AS quant
  USING (feature_id);
```

To obtain one abundance row per feature, query the quantification table
directly rather than summing the joined rows:

```sql
SELECT feature_id, quant_envelope_area
FROM read_parquet('sample.feature_quant.parquet');
```

## Downstream quantification

Biosaur2 provides de-duplicated feature-level abundance and evidence needed to
connect features to MS2 events and accepted assays. A downstream workflow can
then perform sample normalization, missing-value handling, peptide roll-up,
and protein inference. Those operations depend on the experimental design and
are intentionally not performed automatically by Biosaur2.

Units and lower-level schemas are specified in [the design](../design.md).
