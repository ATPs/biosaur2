# Outputs and quantification

## Hybrid files

For `sample.mzML.gz`, hybrid Parquet processing writes:

| File | One row represents |
| --- | --- |
| `sample.features.parquet` | One accepted MS1 isotope feature. |
| `sample.feature_quant.parquet` | One abundance record for that feature. |
| `sample.ms2.parquet` | One normalized MS2 precursor event. |
| `sample.ms2_feature_links.parquet` | One association or unresolved outcome per MS2 event. |
| `sample.identifications.parquet` | One parsed and mapped PSM record. |
| `sample.id_assays.parquet` | One accepted peptide-specific assay. |

The feature and quantification tables have the same unique feature-ID set.
Several MS2 rows may reference one feature ID.

## The three abundance measurements

`--quant-method all` is the default and writes all three final values:

| Column | Definition | Typical use |
| --- | --- | --- |
| `quant_envelope_area` | Trapezoidal area of all assigned isotope traces over actual RT seconds. | Default robust feature abundance. |
| `quant_mono_area` | Area of the monoisotopic trace only. | Comparisons that deliberately avoid heavier isotopes. |
| `quant_envelope_apex` | Maximum summed isotope intensity at one common MS1 scan. | Apex-based workflows. |
| `quant_value` | Envelope area when the method is `all`. | Compatibility and project donor ranking. |

The sidecar also retains raw and baseline-corrected diagnostic area columns,
quantification status, points across the peak, evidence counts, feature origin,
quality flags, isotope similarity, and mass error.

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
