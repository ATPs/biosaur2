# Parameter guide

The defaults are intended as a starting point. Change scientific tolerances
only with instrument knowledge and a validation dataset.

## Everyday options

| Option | Default | What it means |
| --- | ---: | --- |
| `--feature-mode` | `legacy` | Use `hybrid` for DDA-aware evidence, audit, and quantification sidecars. |
| `--feature-format` | `tsv` | Use `parquet` for typed, compact downstream analysis. |
| `-o` | beside input | Output file for one input or output directory for several inputs. |
| `--workers` | 4 | Total CPU worker-process budget for the complete command. |
| `--cache-dir` | `.biosaur2_cache` | Root used for every hybrid and project cache layer. |
| `--keep-cache` | false | Retain valid caches for a later command instead of cleaning this job's namespace. |
| `--overwrite` | false | Atomically replace complete existing output targets. |

## Hybrid evidence

| Option | Default | What it means and when to change it |
| --- | ---: | --- |
| `--psm-path` | empty | Same-run Percolator target PSM table. Leave empty for generic-only hybrid processing. |
| `--psm-q-value-max` | 0.01 | Identification confidence filter. Raising it accepts less reliable peptide assignments. |
| `--psm-pep-max` | none | Optional additional posterior-error filter. |
| `--fixed-mod` | none | Explicit repeatable modification such as `C=UNIMOD:4`; Biosaur2 does not guess it. |
| `--generic-ms2-refine` | true | Test unidentified-MS2 precursor hypotheses with target/decoy control. |
| `--generic-q-value-max` | 0.01 | Estimated false-discovery limit for generic extraction, not for PSMs. |
| `--generic-ms2-ppm` | 10 | Allowed precursor m/z error for generic hypotheses. Change only for justified instrument accuracy. |
| `--ms2-rt-tolerance-sec` | 120 | Initial same-run search distance before and after one MS2 event. It is not cross-run matching. |
| `--relaxed-ms2-feature` | false | One guarded retry for unresolved MS2 evidence; evaluate before enabling routinely. |

## Quantification

| Option | Default | What it means |
| --- | ---: | --- |
| `--quant-method` | `all` | Write envelope area, mono area, and envelope apex; envelope area remains `quant_value`. |
| `--feature-baseline` | hybrid: `edge_linear` | Estimate a line between trace edges before quantification; `none` uses raw traces. |
| `-iuse` | -1 | Isotope subset for legacy compact intensity and `area_sum`; it does not remove the hybrid named quant columns. |

## Detection tolerances

| Option | Default | What it controls |
| --- | ---: | --- |
| `-mini` | 1 | Minimum centroid intensity considered during hill construction. |
| `-minmz`, `-maxmz` | 350, 1500 | m/z interval searched for features. |
| `-htol` | 8 ppm | Point-to-hill mass tolerance. |
| `-itol` | 8 ppm | Isotope-envelope mass tolerance. |
| `-minlh` | 2 | Minimum MS1 points in an ordinary hill. |
| `-cmin`, `--max-charge` | 1, 7 | Charge hypotheses considered. |

Broadening tolerances can merge unrelated evidence or increase false
candidates. Lowering minimum signal or hill length can turn isolated signal
into poor quantitative candidates. Read [Hybrid workflow](hybrid-workflow.md)
and [the design](../design.md) before method-development changes.
