# Outputs and quantification

This guide uses `sample.mzML.gz`, so the input stem is `sample`. Selected
columns are shown to keep examples readable; `...` means more columns exist.
TSV displays nested values as JSON text. Parquet and DuckDB retain typed lists
and structs.

## File map

One `--format` controls all requested tables. The automatic default is TSV in
legacy mode and Parquet in Hybrid mode.

| Workflow | Files or tables | Row grain |
| --- | --- | --- |
| Legacy | `sample.features.tsv` | One accepted strict MS1 feature. |
| Legacy `--format parquet` | `sample.features.parquet` | The same feature table in typed Parquet. |
| Hybrid | `sample.features.parquet`, `sample.identifications.parquet` | One feature; one accepted parsed PSM. |
| `--format duckdb` | `sample.biosaur2.duckdb` | One database per input with the same tables. |
| `--write-hills` | `sample.hills.<format>` | One chromatographic hill. |
| `--write-ms1` | `sample.ms1.<format>` | One MS1 scan summary. |
| Legacy `--write-ms2` | `sample.ms2.<format>` | One normalized precursor entry from an MS2 spectrum. |
| Project Hybrid external stage only | `sample.external_id_evidence.<format>` or DuckDB table | One source-run support row for a weak candidate, plus compact rejected-candidate audit rows. |
| Project index | `project.duckdb` | Run status, QC, alignment and stage summaries. |
| Experimental `-dia`/`-dia2` | requested `.mgf` | One text block per exported spectrum. |

For one input, `-o results/sample.features.parquet` fixes the feature path and
places `results/sample.identifications.parquet` beside it. With several
inputs, `-o results` is a directory and each input gets its own stem. DuckDB
also creates one database per input, never one shared run database.

## Main Hybrid feature table

`sample.features.parquet` is the main result. Every row combines:

- strict/recovered feature coordinates;
- feature origin, confidence and quality;
- all requested abundance values;
- a typed `ms2_events` list containing zero, one or multiple linked events.

Selected scalar columns look like this:

```text
feature_idx  mz        charge  rtStart  rtApex  rtEnd  feature_origin     quant_value  quant_envelope_area  quant_mono_area  quant_envelope_apex  supporting_ms2_count
42           499.7001  2       12.18    12.34   12.51  direct_identified  428519.7     428519.7            241006.2        9831.0               2
43           401.2052  3       18.64    18.76   18.87  strict_untargeted  231480.4     231480.4            152704.9        6412.0               0
44           612.3210  2       21.02    21.14   21.27  generic_local      118042.8     118042.8             72011.5        4208.0               1
```

`rtStart`, `rtApex` and `rtEnd` are minutes for compatibility with legacy
features. Hybrid also includes `rt_start_sec`, `rt_apex_sec` and `rt_end_sec`
in seconds. `feature_idx` is the stable positive feature key.

The same three rows have conceptually these nested events:

```text
feature_idx  ms2_events
42           [{ms2_event_id:317, association_tier:"direct_id", status:"matched_strict_feature", assay_id:1, rt_sec:741.7},
              {ms2_event_id:325, association_tier:"direct_id", status:"matched_strict_feature", assay_id:3, rt_sec:746.2}]
43           []
44           [{ms2_event_id:411, association_tier:"generic_ms2", status:"generic_recovered_local_feature", generic_isotope_error:1, extraction_q_value:0.006}]
```

Several MS2 events may point to one feature. Do not explode the list and sum
the repeated feature abundance. MS2 events with no feature are not placed in
this list.

## Main Hybrid identifications table

`sample.identifications.parquet` merges PSM parsing/mapping fields with the
direct assay created from that PSM, when one could be constructed.

```text
psm_id       ms2_event_id  mapping_status  q_value  peptide_raw            canonical_peptidoform  formula_status  assay_status             assay_id  assay_charge  monoisotopic_mz  assay_conflict_status
scan1542_1   317           mapped          0.0021   K.PEPTIDEK.R           PEPTIDEK               exact           accepted_direct_assay    1         2             464.7348         unique
scan1543_1   318           mapped          0.0068   K.AC[UNIMOD:4]DEK.R   AC[UNIMOD:4]DEK        exact           accepted_direct_assay    2         2             321.1335         unique
scan1602_1   356           mapped          0.0082   X.BADMODPEP.Y          BADMODPEP              unsupported     unsupported_formula       null      null          null             null
```

An accepted PSM can remain here even when its MS2 event did not obtain a
feature. This makes assay construction and mapping failures inspectable. An
MS2 event with neither a feature nor a PSM is omitted row-by-row; aggregate
outcomes remain in Hybrid summary metadata and logs.

## Legacy feature table

`sample.features.tsv` or `sample.features.parquet` contains the established
strict feature columns. Hybrid uses these same base coordinates before adding
its quantitative/evidence columns.

```text
massCalib  rtApex  intensityApex  intensitySum  charge  nIsotopes  nScans  mz        rtStart  rtEnd  FAIMS  feature_idx  area_sum
997.3856   12.34   98231          641203        2       3          12      499.7001  12.18    12.51  null   42           641203.4
1201.5942  18.76   45612          338905        3       2           9      401.2052  18.64    18.87  null   43           338905.1
```

Feature RT columns are minutes. `area_sum` is trapezoidal intensity x seconds
over the legacy `-iuse` isotope subset. Compact feature/hill intensities are
rounded to zero decimals at output; Hybrid area values remain floating point.

## Hill diagnostic

`--write-hills` creates `sample.hills.<format>`. A hill is one m/z trace across
nearby MS1 scans. It is an algorithm diagnostic, not the final quantitative
feature table.

```text
hill_idx  mz        rtStart  rtApex  rtEnd  nScans  intensityApex  feature_idx
901       499.7001  12.18    12.34   12.51  12      45213          42
902       500.2018  12.19    12.33   12.49  11      31784          42
903       401.2052  18.64    18.76   18.87   9      22106          -1
```

`feature_idx = -1` means the hill was not assigned to a final feature. Point
lists are included unless `--no-hill-list` is supplied.

## MS1 scan diagnostic

`--write-ms1` creates `sample.ms1.<format>`:

```text
scan_id  RT      total_intensity
1283     738.1   18274630
1284     740.3   19650421
1285     742.6   19011782
```

Here `RT` is seconds. This is a scan-level summary, not a feature abundance
table.

## Normalized MS2 event diagnostic

Legacy-only `--write-ms2` creates `sample.ms2.<format>`. One row represents
one precursor entry in an MS2 spectrum, not the fragment peaks. A spectrum
with multiple precursor entries produces multiple rows.

```text
ms2_event_id  native_scan_number  rt_sec  precursor_ms1_index  precursor_resolution  selected_ion_mz  isolation_target_mz  precursor_mz  precursor_mz_source  charge  metadata_flags
317           1542                741.7   1284                 spectrum_ref          499.7001         500.0000             499.7001      selected_ion         2       0
318           1543                743.2   1285                 preceding_ms1         null             401.2052             401.2052      isolation_target     null    2
319           1544                744.0   null                 unresolved            null             null                 null          missing              3       13
```

“Normalized” means Biosaur2 converts mzML files that express precursor
metadata differently into one fixed row schema. It does not normalize
intensity, alter spectra, or manufacture measurements. Biosaur2 prefers
`selected_ion_mz` for `precursor_mz`, otherwise uses `isolation_target_mz`;
`precursor_mz_source` records the choice. It resolves the parent MS1 through
`spectrumRef` when possible, otherwise uses the preceding MS1 scan;
`precursor_resolution` records the rule.

`metadata_flags` is additive: 1 missing precursor m/z, 2 missing charge, 4
unresolved spectrum reference, and 8 missing preceding MS1. Thus 13 means
1 + 4 + 8. Hybrid does not write this standalone table: linked fields are
inside `features.ms2_events`, while PSM-bearing events remain in
`identifications`.

## Cross-run external evidence

This output is produced only by a Hybrid Project with external-ID enabled.
Single-file Hybrid processing may create private strong/weak sidecars, but it
does not write public external evidence because no cross-run competition has
occurred.

For Parquet and TSV, each run receives
`<run>.external_id_evidence.<format>`; DuckDB output receives an
`external_id_evidence` table. One accepted weak candidate may have several
rows, one for each distinct source-run support retained up to
`--external-max-support-runs`. A rejected candidate normally has one compact
audit row, or a row with nullable source fields when it had no support.

```text
target_run  weak_candidate_id  source_run  source_feature_id  support_rank  support_score  target_score  decoy_score  acceptance_q_value  status
run_b       1842               run_a       9173               1             0.96           3.71          0.81         0.024               accepted_matched_weak_feature
run_b       1842               run_c       8821               2             0.94           3.71          0.81         0.024               accepted_matched_weak_feature
run_b       1901               null        null               null          null           null          0.72         1.000               no_external_support
run_b       1917               run_d       1044               null          0.66           0.66          0.83         1.000               decoy_winner
```

`support_score` is one source run's best strong-feature match.
`target_score` and `decoy_score` are sums from up to the configured maximum
number of distinct runs. `acceptance_q_value` is the Project feature-transfer
q-value. Alignment method, anchor count, held-out MAD, predicted RT, ppm error
and RT error make each reported support auditable.

Accepted weak rows are appended to the target run's ordinary feature table with
origin `aligned_external_weak`, confidence tier `external_id_weak`, and
`external_support_count` equal to the number of distinct source runs used.
Their feature boundaries and abundance come from the pre-existing target-run
weak candidate; source abundance is never copied and Project does not re-read
raw mzML for targeted extraction.

Rejected candidates remain absent from the feature table. Common statuses are
`no_accepted_alignment`, `no_external_support`,
`insufficient_target_support_runs`, `decoy_winner` and
`target_q_value_above_limit`. An empty evidence table is valid when no weak
candidate can be evaluated; Project funnel summaries distinguish that state
from an external stage that was disabled.

## Per-input DuckDB

`--format duckdb` writes one `sample.biosaur2.duckdb` for each input. A Hybrid
database normally contains:

```text
table_name             example rows
features               42, 43, 44 ...
identifications         scan1542_1, scan1543_1 ...
runs                    one provenance row
external_id_evidence    project Hybrid only
```

Explicit legacy diagnostics add `hills`, `ms1`, or `ms2`. Example queries:

```sql
SELECT feature_idx, mz, rtApex, quant_envelope_area
FROM features
ORDER BY quant_envelope_area DESC
LIMIT 3;

SELECT feature_idx, event.ms2_event_id, event.status
FROM features, UNNEST(ms2_events) AS t(event)
LIMIT 3;
```

## Project index DuckDB

`project.duckdb` records paths and status rather than combining every run's
feature rows. Its `runs` table resembles:

```text
run_order  run_id  status   output_format  features_path                         identification_path
0          run_a   success  parquet        results/run_a/run_a.features.parquet  results/run_a/run_a.identifications.parquet
1          run_b   success  parquet        results/run_b/run_b.features.parquet  results/run_b/run_b.identifications.parquet
```

Selected `qc_metrics` and `hybrid_summary` rows look like:

```text
run_id  feature_count  linked_ms2_count  quant_feature_count
run_a   18432          968               18432
run_b   17905          901               17905

run_id  strict_feature_count  recovered_feature_count  generic_decoy_only_count
run_a   17680                 752                      83
run_b   17144                 761                      79
```

Other tables store stage/cache status, identification summaries, RT alignment
models, external summaries and resolved project options.

## Experimental MGF

`-dia` or `-dia2` can write an experimental MGF. Unlike the normalized MS2
event table, MGF includes m/z-intensity fragment pairs:

```text
BEGIN IONS
TITLE=sample.mzML.1542.1542.2
RTINSECONDS=741.700000
PEPMASS=499.700100 98231.000000
CHARGE=2+
126.0550 1842.0
175.1198 6321.0
END IONS

BEGIN IONS
TITLE=sample.mzML.1543.1543.3
RTINSECONDS=743.200000
PEPMASS=401.205200 45612.000000
CHARGE=3+
147.1128 2510.0
204.1343 4322.0
END IONS
```

## Three abundance measurements

Hybrid defaults to `--quant-method all` and reports:

| Column | Definition | Typical use |
| --- | --- | --- |
| `quant_envelope_area` | Trapezoidal area across assigned isotope traces and actual RT seconds. | Default robust feature abundance. |
| `quant_mono_area` | Area of only the monoisotopic trace. | Workflows that deliberately exclude heavier isotopes. |
| `quant_envelope_apex` | Maximum summed isotope intensity at one common MS1 scan. | Apex-based comparisons. |
| `quant_value` | Envelope area when `--quant-method all` is used. | Stable primary value for generic consumers. |

These are alternative summaries of the same feature, not separate features.
The table also reports raw/corrected areas, baseline status, points across the
peak, evidence counts, isotope cosine, mass error, origin and quality flags.

Biosaur2 stops at feature-level measurements. Sample normalization,
missing-value policy, peptide roll-up and protein inference depend on the
experimental design and belong in downstream analysis.

For algorithm invariants and complete schema intent, read
[the design](../design.md).
