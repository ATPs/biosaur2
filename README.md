# Biosaur2

Biosaur2 finds and quantifies peptide-like isotope features in centroided
LC-MS data. It reads the MS1 signal that contains precursor abundance and can
use DDA MS2 events or Percolator identifications to associate difficult local
signals without counting the same feature more than once.

This repository builds on [markmipt/biosaur2](https://github.com/markmipt/biosaur2).
The upstream project remains a straightforward choice for strict untargeted
LC-MS1 feature detection. This version adds an optional hybrid DDA workflow,
target/decoy-controlled recovery, auditable MS2 links, feature-level
quantification sidecars, multi-run RT alignment, unified caches, and bounded
project execution. See [what changed](updates/2026-07-30.md) and the
[validation results](updates/2026-07-30.md#validation-results); performance and
coverage depend on the data and are not guaranteed for every experiment.

## Install

```bash
pip install biosaur2
```

DuckDB is recommended for compact Parquet output:

```bash
pip install 'biosaur2[duckdb]'
```

Input mzML files should normally contain centroided MS1 spectra. Hybrid mode
also reads DDA MS2 precursor metadata from the same file.

## Choose a workflow

Strict LC-MS1 feature detection:

```bash
biosaur2 sample.mzML.gz --feature-format parquet
```

Hybrid processing without peptide identifications:

```bash
biosaur2 sample.mzML.gz \
  --feature-mode hybrid \
  --feature-format parquet
```

Hybrid processing with a same-run Percolator PSM table:

```bash
biosaur2 sample.mzML.gz \
  --feature-mode hybrid \
  --psm-path sample.percolator.target.psms.tsv \
  --fixed-mod C=UNIMOD:4 \
  --feature-format parquet
```

Several comparable runs:

```bash
biosaur2 project run \
  --manifest runs.tsv \
  --output-dir results \
  --project-db results/project.duckdb \
  --mode hybrid \
  --workers 16

biosaur2 project validate --project-db results/project.duckdb
```

Start with [Getting started](docs/getting-started.md) if MS1, MS2, isotope
features, PSMs, or q-values are unfamiliar.

## Reuse a cache

By default, all hybrid cache layers are written below
`.biosaur2_cache` in the current directory and removed when the command ends.
Add `--keep-cache` to retain them. A later command reuses every compatible
layer and recomputes any layer whose fingerprint no longer matches.

```bash
biosaur2 sample.mzML.gz \
  --feature-mode hybrid \
  --cache-dir .biosaur2_cache --keep-cache \
  --feature-format parquet \
  -o results/first.features.parquet

biosaur2 sample.mzML.gz \
  --feature-mode hybrid \
  --cache-dir .biosaur2_cache --keep-cache \
  --feature-format parquet \
  -o results/second.features.parquet
```

The second output path is different because Biosaur2 does not overwrite an
existing result unless `--overwrite` is supplied. Reuse is reported in the
log. Project cache reuse is shown in the
[project workflow guide](docs/project-workflow.md#reuse-project-caches).

## CPU use

`--workers` is the total CPU worker-process budget for the command and defaults
to `4`. One input receives the whole budget. With several inputs, Biosaur2
automatically runs an appropriate number of files concurrently, targeting
about four workers per active run without exceeding the total. The detected
available CPU count can reduce the effective budget; the final allocation is
logged.

## Quantification output

Hybrid mode defaults to `--quant-method all`. It writes one quantification row
per accepted feature, including:

| Column | Meaning |
| --- | --- |
| `quant_envelope_area` | Area under all assigned isotope traces. |
| `quant_mono_area` | Area under the monoisotopic trace. |
| `quant_envelope_apex` | Largest summed isotope intensity at one MS1 scan. |
| `quant_value` | Primary value; envelope area when the method is `all`. |
| `feature_id` | Stable key used to join the feature and evidence tables. |

Do not sum MS2-link rows: several MS2 events may point to one feature. Use the
single row in `<stem>.feature_quant.parquet` for abundance. Biosaur2 provides
feature-level measurements and evidence; downstream normalization, peptide
roll-up, and protein inference remain separate analysis steps.

Read [Outputs and quantification](docs/outputs-and-quantification.md) for file
names, example rows, joins, units, and DuckDB queries.

## Key concepts

- `--ms2-rt-tolerance-sec` is a same-run local search window around one MS2
  event. It does not match features between runs.
- The generic extraction q-value estimates false associations by comparing
  real precursor hypotheses with deliberately shifted decoy hypotheses. It is
  separate from the Percolator PSM q-value.
- Local recovery searches nearby raw MS1 scans for a defensible isotope trace;
  it may return no feature rather than force an association.
- Cross-run matching exists only in hybrid project processing. It aligns
  retention times with shared identification anchors and always measures
  intensity from the recipient run.

Detailed explanations:

- [Getting started and glossary](docs/getting-started.md)
- [Hybrid evidence, q-values, and local recovery](docs/hybrid-workflow.md)
- [Projects and cross-run alignment](docs/project-workflow.md)
- [Outputs and quantification](docs/outputs-and-quantification.md)
- [Parameter guide](docs/parameters.md)
- [Algorithm design](design.md)

Run `biosaur2 --help` and `biosaur2 project run --help` for the complete current
option list.

## Citation

Abdrakhimov et al., “Biosaur: An open-source Python software for liquid
chromatography-mass spectrometry peptide feature detection with ion mobility
support.” https://doi.org/10.1002/rcm.9045
