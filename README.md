# Biosaur2

Biosaur2 detects peptide-like isotope features in centroided LC-MS data and
reports one quantitative record per accepted feature. It uses MS1 signal for
abundance. In Hybrid mode, DDA MS2 precursor metadata and optional same-run
Percolator PSMs help connect or recover difficult features without copying MS2
fragment intensity into the measurement.

This repository extends [markmipt/biosaur2](https://github.com/markmipt/biosaur2)
with a complete Hybrid workflow: PSM-guided and unidentified-MS2 evidence,
target/decoy-controlled local recovery, three named abundance measures,
multi-run RT alignment, strong-to-weak feature matching, unified reusable caches, and
bounded multi-file execution. The strict upstream-style detector remains
available as `legacy` mode. See [the update and validation record](updates/2026-07-30.md)
for scope and limitations.

## Install

```bash
pip install biosaur2
```

DuckDB is recommended for Parquet and database output:

```bash
pip install 'biosaur2[duckdb]'
```

Input mzML files should contain centroided MS1 spectra. Hybrid mode also reads
DDA precursor metadata from MS2 spectra in the same mzML.

## Run

Strict untargeted detection defaults to TSV:

```bash
biosaur2 sample.mzML.gz
```

Hybrid mode defaults to Parquet and needs no PSM file:

```bash
biosaur2 sample.mzML.gz --feature-mode hybrid
```

Add a same-run Percolator target PSM table when available:

```bash
biosaur2 sample.mzML.gz \
  --feature-mode hybrid \
  --psm-path sample.percolator.target.psms.tsv \
  --fixed-mod C=UNIMOD:4 \
  --quant-method all \
  -o results/sample.features.parquet
```

Hybrid writes two primary files for this input:

- `results/sample.features.parquet`: feature coordinates, quality,
  quantification, and zero or more linked MS2 events in `ms2_events`.
- `results/sample.identifications.parquet`: parsed PSM and direct-assay fields.

An MS2 event with neither a feature nor a PSM is counted in run summaries but
is not stored as a row. See [Outputs and quantification](docs/outputs-and-quantification.md)
for several example rows from every output type.

Use one output format for all requested tables:

```bash
biosaur2 sample.mzML.gz --format tsv
biosaur2 sample.mzML.gz --format parquet
biosaur2 sample.mzML.gz --feature-mode hybrid --format duckdb
```

The automatic default is TSV for `legacy` and Parquet for `hybrid`. DuckDB
creates one `<stem>.biosaur2.duckdb` per input.

## Several runs

Hybrid project mode lets comparable files assist one another. Shared
high-confidence peptide/charge observations fit a retention-time mapping. A
peptide seen in a donor run can guide a search in a recipient run, but the
reported abundance is always measured from that recipient's own MS1 scans.

```bash
biosaur2 project run \
  --manifest runs.tsv \
  --output-dir results \
  --project-db results/project.duckdb \
  --mode hybrid \
  --workers 16

biosaur2 project validate --project-db results/project.duckdb
```

Each input still receives its own features and identifications outputs. The
project database records run status, paths, alignment and weak-feature rescue
summaries. Read [Project workflow](docs/project-workflow.md) for the manifest
and the difference between same-run search and cross-run matching.

## Cache reuse

`--cache-dir` stores raw-MS1, strict-stage, candidate and project caches under
one root. It defaults to `.biosaur2_cache` in the current directory and is
cleaned after the job. Add `--keep-cache` to retain compatible layers:

```bash
biosaur2 sample.mzML.gz --feature-mode hybrid \
  --cache-dir .biosaur2_cache --keep-cache \
  -o results/first.features.parquet

biosaur2 sample.mzML.gz --feature-mode hybrid \
  --cache-dir .biosaur2_cache --keep-cache \
  -o results/recheck.features.parquet
```

The second command reuses fingerprint-compatible layers and logs each cache
hit. A changed scientific option invalidates only dependent layers. A new
output path avoids overwriting the first result.

## CPU use

`--workers` is the total CPU budget and defaults to 4. With several input
files, Biosaur2 distributes that budget dynamically, targeting about four
workers per active file without exceeding the total.

Biosaur2 CLI commands set OpenMP, BLAS, NumExpr, vecLib and Arrow CPU/I/O
thread pools to one before loading numerical libraries. This prevents hidden
native pools from exceeding the explicit `--workers` process budget.

## Learn the terms

- [Getting started](docs/getting-started.md): MS1/MS2, feature, PSM, required
  PSM columns, peptide notation, and fixed modifications.
- [Parameter guide](docs/parameters.md): everyday and `--help-all` options,
  including both q-values and local width `auto`.
- [Hybrid workflow](docs/hybrid-workflow.md): association, target/decoy and
  local recovery.
- [Project workflow](docs/project-workflow.md): cross-run assistance and cache
  reuse.
- [Outputs and quantification](docs/outputs-and-quantification.md): every file,
  sample rows, units and queries.
- [Algorithm design](design.md): authoritative implementation contract.

Run `biosaur2 --help` for everyday controls and `biosaur2 --help-all` for all
advanced and diagnostic controls.

## Citation

Abdrakhimov et al., “Biosaur: An open-source Python software for liquid
chromatography-mass spectrometry peptide feature detection with ion mobility
support.” https://doi.org/10.1002/rcm.9045
