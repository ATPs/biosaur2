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
| `--psm-path` | empty | Percolator target PSM TSV (compressed text supported) or Parquet. A multi-run table is filtered to the current mzML stem when it has a run/idn field. See [the input example](getting-started.md#psm-input). |
| `--psm-q-value-max` | 0.01 | Maximum Percolator PSM q-value used to build direct peptide assays. |
| `--fixed-mod` | none | Repeatable `SITE=MOD`, for example `C=UNIMOD:4` or `peptide_n_term=UNIMOD:1`. |
| `--generic-ms2-refine` | true | For MS2 without a usable direct assay, test precursor hypotheses with target/decoy control and local recovery. |
| `--generic-q-value-max` | 0.05 | Maximum estimated false-discovery rate for generic, unidentified-MS2-to-feature associations. It does not filter Percolator peptide IDs. |
| `--generic-ms2-ppm` | 10 | Selected-ion precursor tolerance for generic hypotheses. |
| `--ms2-rt-tolerance-sec` | 120 | Initial same-run raw-MS1 search distance on each side of an MS2 event. It does not match runs. |
| `--quant-method` | `all` | Report envelope area, mono area and envelope apex; `quant_value` uses envelope area. |
| `--write-mono-hills` | false | Add monoisotopic scan/intensity point arrays to Hybrid features. |
| `--write-quant-details` | false | Add raw and baseline-corrected envelope/mono area columns. |

These two same-run q-value thresholds are independent. `--psm-q-value-max`
asks whether a peptide assignment is reliable. `--generic-q-value-max` asks
whether an unidentified MS2 precursor was associated with MS1 signal more
convincingly than shifted decoys. Project external rescue has a third,
independent `--external-q-value-max` described below. See
[Hybrid workflow](hybrid-workflow.md).

### Choosing `--generic-q-value-max`

Use this option only to control **generic MS2 associations**: cases where an
MS2 scan has no usable direct peptide/PSM assay, but its precursor m/z, charge,
retention time, and isotope pattern may still identify a real MS1 feature.
It answers this practical question: "Can this otherwise unidentified MS2 scan
be linked to this MS1 feature with enough evidence?"

For each such scan, Biosaur2 evaluates two alternatives using the same rules:

1. The **target** is the measured precursor hypothesis.
2. The **decoy** is an intentionally shifted, false precursor hypothesis.
3. If target-like matches win much more often than decoy-like matches, the
   program estimates how many of the accepted target links could be false.

That estimate is the generic q-value. It is a property of the accepted set,
not a guarantee that any one MS2-to-feature link has exactly that probability
of being wrong. With the default `0.05`, the accepted generic links are allowed
an estimated false-discovery rate of at most 5%. As a simple interpretation,
if 100 generic links pass at this limit, roughly five false links are permitted
on average by this control method.

`--generic-q-value-max` is independent of peptide identification quality. It
does **not** mean that Percolator PSMs with q-values up to 5% are used: direct
PSM assays remain controlled by `--psm-q-value-max` (default `0.01`). It also
does not change strict untargeted feature detection, m/z tolerance, or create a
feature for an MS2 scan that lacks adequate MS1 evidence.

| Value | Practical effect |
| --- | --- |
| `0.05` (default and maintained baseline) | Permits about 5% estimated false discoveries among accepted generic links. |
| Higher than `0.05` | More permissive: may increase coverage, but permits a larger estimated false-discovery rate. Validate on a representative panel before using it. |

Future development, regression testing, and scientific acceptance use `0.05`.
Older `0.01` results are historical comparisons, not a compatibility target.
If this value is explicitly overridden, check the target/decoy,
quantitative-link, and unresolved-event summaries after changing it.

The advanced `--psm-*-column` options, visible only in `biosaur2 --help-all`,
override automatic semantic-column detection for composite PSM IDs, split
run/scan/charge/rank identity, peptide, q-value and optional annotations.
Extra source columns are accepted but are not copied to Biosaur2 outputs.

## External weak-feature rescue (`--help-all`)

External rescue is feature match-between-runs, not peptide-ID transfer and not
targeted extraction from raw mzML. It has three distinct layers:

1. Each Hybrid run performs normal strict detection. Only when `--external-id`
   is enabled, detector rejects are also screened and written to private weak
   sidecars. They are not yet public features.
2. Project mode aligns runs using their final strong features. It then asks
   whether strong features in other runs support each weak candidate already
   measured in the target run.
3. Target matches compete with deterministically shifted decoy matches. Only a
   target winner below the external q-value threshold is appended to the
   target run's feature output.

A single-file Hybrid run can create the sidecars but has no other runs from
which to obtain support. `biosaur2 project run --mode hybrid` performs all
three layers. `--no-external-id` skips weak-candidate collection and the
Project external stage entirely.

### Run alignment

| Option | Default | Valid values | Detailed effect |
| --- | ---: | --- | --- |
| `--external-ppm` | 8 | finite, >0 | m/z tolerance used both to construct mutual-nearest strong-feature RT anchors and to match a target-run weak candidate to source-run strong features. Charge and FAIMS must also match exactly. This does not alter hill detection or feature boundaries. |
| `--external-rt-tolerance-sec` | 120 | finite, >=0 | Maximum absolute difference between the target weak apex and the source strong apex after RT alignment. The same value is the maximum held-out alignment q90 absolute error. It does not change local weak detection and is unrelated to `--ms2-rt-tolerance-sec`. |
| `--external-alignment-min-anchors` | 20 | integer, >=1 | Minimum number of **fit** anchors required for one directed RT-alignment edge after validation anchors have been reserved. Anchors are mutual-nearest final strong features with the same charge and FAIMS and within `--external-ppm`; a longest strictly increasing RT chain removes crossing/isobaric matches. Every fifth chain anchor is held out, so the default normally needs at least 20 fit plus 5 validation anchors. An edge below either minimum is rejected. |
| `--external-alignment-max-mad-sec` | 30 | finite, >=0 | Acceptance limit applied separately to the absolute median signed error (bias) and median absolute deviation (MAD) on held-out anchors. This measures whether an alignment generalizes; it is not the 120-second feature matching window. `0` is allowed but accepts only exactly zero held-out bias and MAD. |
| `--external-alignment-max-anchors` | 256 | integer, >=1 | Maximum number of non-validation anchors fitted for one directed edge. If more survive, deterministic RT-stratified sampling retains at most this many, preferring higher-quality anchors within each stratum. Held-out validation anchors are not consumed by this cap. Raising it increases fit work and memory; lowering it does not lower `--external-alignment-min-anchors` and can make the configuration impossible when set below that minimum. |

For each declared `alignment_group`, runs are ranked by strong-feature count.
Each run tries up to four high-coverage reference candidates. Accepted
bidirectional edges form a deterministic reference-rooted forest; disconnected
runs remain in separate components and cannot support one another. A support
may traverse multiple accepted edges. Besides the two limits above, held-out
validation requires q90 absolute RT error no greater than
`--external-rt-tolerance-sec`. Alignment counts, bias, MAD, q90 and rejection
status are recorded in `project.duckdb.rt_alignment_models`.

Lowering `--external-alignment-min-anchors` may connect sparse runs, but also
increases the chance that an accidental m/z ordering produces an unstable RT
map. Raising it is more conservative but can split the forest and produce
`no_accepted_alignment` outcomes. Increasing the RT window cannot repair a
rejected edge; it only relaxes the q90 validation limit and subsequent support
matching.

### Local weak-candidate gates

| Option | Default | Valid values | Detailed effect |
| --- | ---: | --- | --- |
| `--external-weak-min-mono-points` | 2 | integer, >=1 | Minimum raw points in the monoisotopic hill of a detector reject. |
| `--external-weak-min-secondary-points` | 2 | integer, >=1 | Minimum points required in at least one non-mono isotope hill. With both point defaults at 2, the local structural minimum is often called the `2+2` gate. |
| `--external-weak-min-isotope-cosine` | 0.60 | finite [0,1] | Minimum cosine similarity between integrated observed isotope intensities and the expected averagine envelope. Lower values admit less isotope-like candidates; higher values improve shape specificity but reduce the pool. |
| `--external-weak-max-strong-overlap` | 0.30 | finite [0,1] | Maximum fraction of the weak candidate's original raw hill intensity already claimed by final same-run strong features. The comparison is inclusive: exactly 0.30 passes at the default. |

The overlap fraction is

```text
intensity at the candidate's raw points already owned by the final strong ledger
-------------------------------------------------------------------------------
              all original raw intensity in the candidate footprint
```

It is an ownership guard against publishing the same signal twice. It is not
the chromatographic RT-overlap percentage and not the fraction of isotope
channels shared. A separate strong-equivalence gate rejects any candidate
already represented by a final same-run strong feature with matching charge,
FAIMS, 8 ppm m/z and RT interval, even when its ownership fraction is below
the configured limit.

Lowering `--external-weak-max-strong-overlap` is conservative and removes
more candidates sharing strong-feature signal. Raising it enlarges the weak
pool, but increases the risk of shared-signal or double-counted quantitative
features. `1.0` disables only this fractional gate; the strong-equivalence,
point, cosine and positive-quantification gates still apply. Candidate
quantification must be finite and positive. Candidates are de-duplicated by
FAIMS, charge and monoisotopic hill before sidecar persistence.

These four options exist on both the single-run Hybrid advanced CLI and the
Project CLI because they control local sidecar production. They have no cost
in ordinary mode: weak candidates are generated only for Hybrid runs with
external-ID enabled. Changing any one invalidates the local weak sidecar and
replays local weak postprocessing; compatible raw MS1 and strong-stage caches
remain independently reusable.

### Cross-run support and transfer FDR

| Option | Default | Valid values | Detailed effect |
| --- | ---: | --- | --- |
| `--external-min-support-runs` | 1 | integer 1-16 | Minimum number of distinct source runs needed for a target score or a decoy score to be valid. The rule is symmetric. At the default, one run makes a candidate eligible for competition but does not guarantee rescue. |
| `--external-max-support-runs` | 4 | integer 1-16, >= min | Maximum distinct-run supports retained, combined as empirical log-likelihood evidence and reported on each target or decoy side. |
| `--external-q-value-max` | 0.10 | finite [0,1] | Maximum project-level feature-transfer q-value for publishing a weak candidate. This q-value is independent of Percolator PSM q-values and generic-MS2 extraction q-values. |

Within an accepted alignment component, each source run contributes only its
single best matching strong feature. Supports are ranked by match score, and
the raw score decreases with normalized m/z and aligned-RT error. This raw
geometric score is not treated as a probability. Biosaur2 bins the component's
target and shifted-decoy support scores in 32 equal-width bins over `[0,1]`,
adds a pseudocount of `1.0` to each side of every bin, applies a monotonic
pooled adjacent violators (PAVA) fit, and estimates an empirical per-support
log-likelihood ratio.
Evidence from at most `--external-max-support-runs` distinct runs is then added
in log space. Deterministic two-fold cross-fitting ensures that a weak
candidate is scored by a calibration that did not use that candidate's target
or decoy supports. A very strong single support can therefore outrank several
weak supports, while repeated high-quality supports still accumulate strong
evidence.

The exact same minimum, maximum, m/z/RT rules and empirical LLR transform are
applied to a deterministically m/z-shifted decoy candidate. Target/decoy competition is
calibrated within each alignment component. A weak candidate is published only
when the target wins, its target support count reaches the minimum, and its
external q-value is at most the configured threshold. Typical funnel outcomes
are `no_accepted_alignment`, `no_external_support`,
`insufficient_target_support_runs`, `decoy_winner`,
`target_q_value_above_limit`, and `accepted_matched_weak_feature`.

Changing q-value or min/max support settings reruns only the in-memory Project
competition when local sidecars remain compatible. Accepted evidence contains
up to the configured maximum support rows; `external_support_count` records
the actual number of distinct source runs used. Rescued weak features are not
promoted into the strong donor index during the same Project run.

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
legacy-only `--write-ms2` diagnostics, Hybrid `--write-mono-hills` and
`--write-quant-details`, TSV precision, and Parquet
engine/compression/row-group/sort controls. These options are hidden from the
ordinary help screen to keep the common workflow readable.

`--write-ms1` writes `scan_id`, RT seconds and total intensity. It is enabled
by default in Hybrid and disabled by default in Legacy; `--no-write-ms1`
suppresses it explicitly.

`--write-ms2` exports normalized precursor-event metadata only in legacy mode.
Hybrid writes compact linked events in its `ms2_events` table and keeps
PSM-only events in `identifications`, so using `--write-ms2` with Hybrid is an
error.

For lower-level algorithm effects, read [the design](../design.md).
