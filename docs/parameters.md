# Parameter guide

Run `biosaur2 --help` for everyday controls and `biosaur2 --help-all` for all
advanced and diagnostic controls. Defaults are starting points; validate
scientific tolerance changes on representative data.

## Everyday options

| Option | Default | Meaning |
| --- | --- | --- |
| `--feature-mode` | `legacy` | `legacy` is strict untargeted detection; `hybrid` adds DDA evidence, local recovery and named quantification. |
| `--format` | legacy: `tsv`; hybrid: `parquet` | One format for features and every explicitly requested diagnostic table. `duckdb` stores all tables for one input in one database. |
| `-o` | beside input | Output file/prefix for one input; output directory for several inputs. |
| `--workers` | 4 | Total CPU budget shared dynamically across input files and their per-run work. |
| `--log-level` | `info` | Console verbosity: `quiet` keeps errors only; `warning`, `info`, and `debug` add progressively more detail. Every log line includes a time. Debug records stage start/completion and `runtime_sec`. Project runs stream child-process logs live; those lines identify the run and process. |
| `--cache-dir` | `.biosaur2_cache` | Root for raw, strict-stage, candidate and project caches. |
| `--keep-cache` | false | Retain compatible caches; otherwise the job's cache namespace is cleaned at completion. |
| `--overwrite` | false | Atomically replace existing complete outputs. |

## PSM and Hybrid evidence

| Option | Default | Meaning |
| --- | --- | --- |
| `--psm-path` | empty | Same-run Percolator target PSM TSV. See [the input example](getting-started.md#psm-input). |
| `--psm-q-value-max` | 0.01 | Maximum Percolator PSM q-value used to build direct peptide assays. |
| `--fixed-mod` | none | Repeatable `SITE=MOD`, for example `C=UNIMOD:4` or `peptide_n_term=UNIMOD:1`. |
| `--generic-ms2-refine` | true | For MS2 without a usable direct assay, test precursor hypotheses with target/decoy control and local recovery. |
| `--generic-q-value-max` | 0.01 | Maximum q-value for Biosaur2's unidentified-MS2 target/decoy associations, not a peptide-identification q-value. |
| `--generic-ms2-ppm` | 10 | Selected-ion precursor tolerance for generic hypotheses. |
| `--ms2-rt-tolerance-sec` | 120 | Initial same-run raw-MS1 search distance on each side of an MS2 event. It does not match runs. |
| `--quant-method` | `all` | Report envelope area, mono area and envelope apex; `quant_value` uses envelope area. |

The two q-value thresholds are independent. `--psm-q-value-max` asks whether a
peptide assignment is reliable. `--generic-q-value-max` asks whether an
unidentified MS2 precursor was associated with MS1 signal more convincingly
than shifted decoys. See [Hybrid workflow](hybrid-workflow.md).

## Generic local recovery (`--help-all`)

| Option | Default | Meaning |
| --- | ---: | --- |
| `--generic-ms2-isotope-errors` | `0,1,2,3` | Candidate selected-isotope indices. For `N`, `mono_mz = selected_ion_mz - N * 1.003354835 / charge`. Negative values from -8 to -1 are accepted only when explicitly supplied and should be instrument-validated. |
| `--generic-local-isotope-count` | 5 | Isotope channels examined for a candidate envelope. |
| `--generic-local-min-mono-points` | 3 | Nonzero MS1 points required in the monoisotopic channel. |
| `--generic-local-min-channel-points` | 3 | Points required for one isotope channel to count as supported. |
| `--generic-local-min-supported-channels` | 2 | Supported channels required for standard recovery. |
| `--generic-local-min-isotope-cosine` | 0.90 | Minimum similarity between integrated observed and averagine envelopes. |
| `--generic-local-max-width-sec` | `auto` | Reject a recovered component wider than the adaptive or explicit limit. |
| `--generic-relaxed-min-mono-points` | 2 | Monoisotopic points for the optional relaxed retry. |
| `--generic-relaxed-min-channel-points` | 2 | Points per supported channel in the relaxed retry. |
| `--generic-relaxed-min-supported-channels` | 2 | Supported channels in the relaxed retry. |
| `--generic-relaxed-min-isotope-cosine` | 0.95 | Higher similarity required to offset the relaxed point count. |

`--generic-local-max-width-sec auto` calculates the 99th percentile (`q99`) of
`rt_end_sec - rt_start_sec` among strict features in that run, then clamps the
result to 15-60 seconds. If no strict widths exist, it uses 30 seconds. This is
a candidate-width rejection rule, not the MS2 search window. Supplying an
explicit positive number disables adaptation.

`--relaxed-ms2-feature` is false by default. It permits one guarded retry for
otherwise unresolved MS2 evidence; it does not relax features with no MS2
support.

## Other advanced controls

`--help-all` also exposes the established hill/detection tolerances,
experimental DIA/profile paths, optional `--write-hills`, `--write-ms1`, and
legacy-only `--write-ms2` diagnostics, TSV precision, and Parquet
engine/compression/row-group/sort controls. These options are hidden from the
ordinary help screen to keep the common workflow readable.

`--write-ms2` exports normalized precursor-event metadata only in legacy mode.
Hybrid already embeds linked events in `features.ms2_events` and keeps PSM-only
events in `identifications`, so using `--write-ms2` with Hybrid is an error.

For lower-level algorithm effects, read [the design](../design.md).
