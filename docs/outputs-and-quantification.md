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
