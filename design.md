# Biosaur2 algorithm design

## Document status

This document describes the algorithm and workflow implemented in the repository
as of 2026-07-30. It covers the legacy detector and the opt-in hybrid residual
workflow. It is an implementation design
document, not a promise that every MS2 event can always be assigned a feature.

The central scientific objective is to build an accurate, nearly complete,
de-duplicated MS1 feature population with one reliable quantitative value per
accepted feature. MS2 and q-value-filtered PSM evidence are used to improve and
associate that population, but they never justify fabricating chromatographic
evidence or duplicating shared MS1 intensity.

## Design principles

1. **MS1 features are the primary population.** A feature does not require an
   MS2 event. Features without MS2 always retain the ordinary strict acceptance
   thresholds.
2. **Every MS2 receives an audit outcome.** The workflow tries to associate
   every event with the best defensible MS1 evidence, while allowing explicit
   null outcomes such as insufficient evidence, ambiguity, precursor-only
   signal, or no signal.
3. **PSMs are strong priors, not unconditional truth.** Percolator input is
   filtered before assay construction. A valid PSM contributes exact peptide
   chemistry, charge, monoisotopic mass, isotope distribution, MS2 location and
   C13 selection information, but an accepted feature still requires coherent
   MS1 evidence.
4. **Signal ownership is non-negative and conserved.** Accepted features own
   only their assigned contribution to raw centroid points. The implementation
   does not delete entire scans, windows or hills.
5. **One feature has one abundance.** Multiple PSMs and MS2 events may link to
   one feature; they update support counts but do not create duplicate feature
   rows or duplicate quantitative values.
6. **Relaxation is local and MS2-only.** Any moderately relaxed recovery is
   opt-in, bounded, and restricted to unresolved MS2-supported candidates.
   Untargeted/non-MS2 standards are never lowered.
7. **Ambiguity is preferable to false precision.** When different envelopes
   share signal and the decomposition is not identifiable, the event remains a
   conflict/ambiguity instead of assigning the same intensity several times.
8. **Reproducibility is part of the algorithm.** Candidate order, conflict
   decisions, feature IDs, cache fingerprints, project execution and output
   publication are deterministic and bounded.

## User-visible modes

| Mode | Activation | Behavior |
|---|---|---|
| `legacy` | default | Ordinary Biosaur2 hill construction, isotope-envelope detection and legacy feature output. |
| `hybrid` | `--feature-mode hybrid` or project `--mode hybrid` | Enables q-filtered direct assays, generic MS2 hypotheses, residual allocation, local recovery, target/decoy confidence, explicit quantification and exhaustive MS2 audit. |

Hybrid mode is deliberately opt-in. `--relaxed-ms2-feature` is a second,
independent opt-in switch and defaults to false. Disabling hybrid keeps the
legacy strict workflow unchanged.

## Core data model

### Raw MS1 store

`RawMS1Store` is a compact, scan-indexed representation of the original MS1
centroids. It retains scan identifiers, real RT seconds, m/z, intensity, FAIMS
and ion-mobility metadata required by local extraction. The original signal is
immutable and can be memory-mapped from a fingerprinted cache.

### Hills

A hill is a mass-consistent trace through consecutive MS1 scans. Hills are
useful candidate-building structures, but they are not indivisible scientific
ownership units. The strict detector may split hills at chromatographic
valleys; hybrid local refinement may merge, extend, relink or segment evidence
on a common scan/RT grid.

### Strict candidates and features

The ordinary Biosaur2 detector links compatible hills into isotope envelopes,
checks charge spacing, isotope mass error, isotope-pattern agreement,
coelution, scan support and configured thresholds, then resolves greedy hill
ownership. Accepted candidates become the initial strict feature population.

### Direct assays

A direct assay is constructed from a same-run, q-value-filtered PSM. It records
the canonical modified peptide, explicit fixed modifications, charge, elemental
formula state, theoretical monoisotopic/isotope m/z values, sequence-specific
isotope probabilities, MS2 RT, precursor scan context and selected C13 isotope.

### Generic MS2 hypotheses

An unidentified MS2 event is represented by its RT, precursor MS1 relationship,
selected/isolation m/z, isolation offsets, available charge or bounded charge
hypotheses, FAIMS/IM, and plausible isotope offsets such as M, M+1 and M+2.
Expected isotope intensity is approximated with an averagine model.

### Residual ledger

For every raw point, the ledger maintains the invariant:

```text
observed intensity = sum(accepted feature contributions) + residual intensity
```

Allocations are non-negative, cannot exceed the observed intensity, and retain
provenance to the original scan/m/z positions. Shared or uncertain intensity
stays residual unless a bounded decomposition is identifiable.

## End-to-end single-run workflow

```text
mzML/mzML.gz + optional Percolator PSM table
                    |
                    v
       one-pass MS1/MS2 ingestion and raw cache
                    |
                    v
      strict hill and isotope-envelope detection
                    |
                    +--> capture bounded direct-relevant losing competitors
                    |
                    v
       initial strict feature population and ledger ownership
                    |
                    v
      direct PSM association and bounded local recovery
                    |
                    v
     generic MS2 -> strict association -> local target/decoy recovery
                    |
                    v
       optional guarded MS2-only relaxed retry
                    |
                    v
       unchanged strict detector on remaining residual MS1
                    |
                    v
       recheck unresolved direct and generic MS2 events
                    |
                    v
 de-duplicated features + one quant row/feature + one audit row/MS2
```

### Stage 0: ingestion and metadata normalization

The mzML reader processes the file once and collects:

- usable MS1 scans and their original centroid points;
- every DDA MS2 event, RT and spectrum/native scan identity;
- precursor MS1 resolution, selected-ion/isolation-window metadata and charge;
- FAIMS and ion-mobility values when available;
- compact MS1/MS2 rows and the raw store needed by residual extraction.

Retention times are normalized to seconds. Scan identity is preserved rather
than inferred from row position when native scan metadata exists. Optional raw
cache publication is atomic; stale caches are rejected using source and
scientific-parameter fingerprints.

### Stage 1: Percolator parsing and exact chemistry

The identification adapter handles common compression, BOM/encoding,
delimiter/header variations and maps PSMs to MS2 using native identity when
possible, otherwise safe run/scan parsing with charge validation.

Default PSM filtering is q-value <= 0.01 and is configurable. Rank alone is not
used to discard a PSM. Fixed modifications must be explicit, for example
`C=UNIMOD:4`; they are never silently inferred. Peptidoforms are normalized
against the pinned local Unimod subset and classified as exact-formula,
mass-only or unavailable. Only usable mappings become direct assays.

When the instrument selected a heavy isotope, the assay jointly evaluates the
selected isotope index and precursor mass error rather than assuming that every
precursor m/z is monoisotopic.

### Stage 2: strict MS1 detection

The production strict detector performs the established Biosaur2 operations:

1. link centroid points into m/z-consistent hills;
2. split chromatographically multi-modal hills using configured valley rules;
3. calculate hill apex, intensity, mass and scan properties;
4. enumerate charge and isotope-envelope candidates;
5. require mass accuracy, scan support, isotope consistency and coelution;
6. resolve competing candidates deterministically;
7. produce the initial strict untargeted population.

Direct-relevant processed-hill candidates that pass ordinary mass/cosine gates
but would lose destructive greedy ownership can be captured immediately before
the conflict pass. Capture is bounded to the top three deterministic
representations per direct PSM and persisted in strict-stage cache version 2.
These candidates only guide a later retry; they cannot bypass raw extraction,
conflict, quantification or conservation gates.

### Stage 3: initialize strict ownership

All accepted initial strict features are converted to raw-point contributions
and allocated once in the residual ledger. If strict ownership cannot be
reconstructed safely, later final-residual detection is suppressed rather than
operating on an invalid residual state.

This ordering deliberately protects the complete strict MS1 population,
including features without MS2. Direct and generic candidates must either link
to that population or demonstrate defensible residual evidence without taking
already-owned strict signal.

### Stage 4: direct identified association and recovery

Each exact direct assay first competes against the strict population using
charge, calibrated ppm, RT interval, FAIMS and isotope-selection compatibility.
If a strict feature explains the event, the MS2/PSM is linked to that existing
feature and only its support metadata changes.

For an unresolved assay, the workflow extracts isotope XICs from raw MS1 around
the MS2 event. It jointly selects the chromatographic component nearest the
event, checks multi-scan support, isotope channels, isotope cosine, mass error,
coelution, apex alignment, isolation context and boundary quality.

Run-specific calibration is learned from reliable direct-to-strict matches:

- mass-error center and retry ppm;
- MS2-to-feature apex RT offset and robust RT tolerance;
- typical feature width.

At most one retry is compared monotonically with the original candidate. A
captured processed-hill competitor may supply a bounded RT center, width and
mass shift to this same retry. Selection of a retry is not acceptance: all
strict-hill conflict, recovered-feature equivalence, raw-point conflict,
residual allocation and quantitative gates still apply.

Equivalent repeated PSMs/MS2 events reuse the same recovered feature. Conflicting
identifications or non-identifiable overlapping local candidates remain null
or ambiguous.

### Stage 5: quantify the initial and direct-recovered population

Initial strict features and accepted direct-recovered features are quantified
from their final assigned traces. Direct PSM/MS2 support counts are accumulated
after de-duplication. A PSM does not copy intensity into the feature and does
not generate a second abundance row.

### Stage 6: generic MS2 association with strict features

For every still-relevant MS2 event, the workflow creates bounded target
hypotheses and deterministic paired decoy hypotheses. Candidate scoring combines
calibrated evidence such as:

- precursor and isotope m/z agreement;
- charge and isotope-offset consistency;
- RT and precursor-scan localization;
- isolation-window compatibility;
- event-apex and selected-intensity support;
- isotope count, mass accuracy and averagine cosine;
- coelution and chromatographic point support.

Weights can be calibrated from paired direct anchors. Target and decoy candidates
compete within the same family, and extraction q-values are calculated
independently of the Percolator PSM q-value. A generic event that already has a
defensible strict feature is associated without changing that feature's
abundance.

### Stage 7: generic residual local recovery

Unresolved generic events are evaluated against raw and residual isotope traces.
With `--generic-local-max-width-sec auto`, the run-derived limit is q99 of
strict-feature `(rt_end_sec - rt_start_sec)`, clamped to 15-60 seconds, with a
30-second fallback when no strict widths exist. An explicit positive value
disables this adaptation. The limit rejects broad candidate components; it is
not the before/after search window controlled by `--ms2-rt-tolerance-sec`.
Local refinement may propose:

- `split`: common RT segmentation across isotope traces;
- `merge`: combine compatible adjacent trace fragments;
- `extend`: recover a short supported boundary/gap;
- `relink`: replace an incorrect local hill relationship.

Candidate envelopes are clustered so repeated compatible MS2 events may share
one feature. Small identifiable overlaps may use non-negative local
deconvolution. A candidate is rejected when there are too few mono points,
insufficient isotope channels, incoherent isotope apexes, weak averagine fit,
excessive width, event-scan absence, isolation inconsistency, a decoy win, a
q-value failure, strict-hill ownership, or an unresolved raw-point conflict.

Accepted generic features receive new IDs and residual allocations only once.

### Stage 8: optional guarded relaxed retry

`--relaxed-ms2-feature` retries only unresolved, MS2-supported candidates. It
does not modify the strict detector or non-MS2 feature thresholds.

The relaxation is deliberately limited:

- direct relaxation requires a same-run PSM with q-value < 0.01;
- generic relaxation retains paired target/decoy competition;
- traces remain multi-scan; single-point features are forbidden;
- partial envelopes still require multiple supported channels;
- generic relaxed candidates retain strong isotope cosine and localization;
- a clearly superior equivalent strict candidate wins;
- cross-envelope sharing is rejected when joint allocation is not identifiable.

This guard makes “prefer the MS2-supported model when similarly plausible” a
bounded tie-breaker, not permission to replace a clearly better strict feature.

### Stage 9: final strict detector on residual MS1

After targeted stages, the ordinary strict detector runs again on materialized
residual MS1 using unchanged untargeted thresholds. It discovers features that
were hidden by earlier overlap or were not selected in the initial population.

Residual strict candidates are checked against every accepted strict, direct
and generic feature to prevent rediscovery. Accepted candidates must allocate
successfully in the same ledger and are quantified as strict untargeted
features. This stage never receives an MS2-only threshold relaxation.

### Stage 10: final MS2 recheck and audit finalization

Unresolved direct assays are rechecked against new final-residual strict
features. Unidentified events are re-evaluated through a separate final-strict
target/decoy family. Failed prior audit reasons are preserved unless a valid
feature association is obtained.

Finally, every MS2 event has exactly one audit row. Quantitative feature links,
precursor-only signal, statistical rejection, metadata failure, insufficient
chromatographic evidence and ambiguity are mutually exclusive final outcomes.

## Hill repair and overlap policy

All isotope traces for a candidate are evaluated on a common real scan/RT grid.
The algorithm avoids independently splitting isotope hills at unrelated
boundaries. Edits must improve a bounded objective and retain reversible
provenance.

For overlapping envelopes:

1. equivalent candidates are de-duplicated;
2. repeated MS2 events may share one feature;
3. clearly superior strict evidence protects the strict representation;
4. identifiable same-envelope intensity sharing may use non-negative least
   squares/local decomposition;
5. different envelopes with non-identifiable shared raw points remain conflict
   or ambiguity;
6. no candidate may subtract more intensity than exists.

The design intentionally does not claim universal decomposition. Some overlap
regions are mathematically underdetermined from the observed MS1 data.

## Feature quantification

Hybrid mode calculates three feature-level measurements:

| Method | Definition |
|---|---|
| `envelope_area` | Trapezoidal integration of the sum of final assigned isotope traces over actual RT seconds. |
| `mono_area` | Trapezoidal integration of the final monoisotopic contribution. |
| `envelope_apex` | Maximum summed assigned isotope intensity at one common MS1 scan. |

The default `--quant-method all` writes these values as
`quant_envelope_area`, `quant_mono_area`, and `quant_envelope_apex`, while
keeping envelope area as the primary `quant_value`. Selecting one named method
changes the primary scalar but does not remove the three explicit columns.
`edge_linear` or `none` is optional baseline preprocessing. Raw and corrected areas are retained. If baseline
correction would be unreliable, the raw value is retained with an explicit
status/quality flag.

Every accepted feature has exactly one positive quantification row containing
its method, status, origin, confidence tier, quality flags, isotope cosine,
mass error, RT boundaries, point count and supporting PSM/MS2 counts. Legacy
`intensitySum`, `intensityApex` and `area_sum` semantics remain available in
the ordinary feature output.

## Confidence and quality control

Confidence is evidence-family specific:

- Percolator q-values control direct PSM entry, default <= 0.01;
- direct MS1 extraction must still pass chromatographic/isotope gates;
- generic candidates use paired target/decoy extraction q-values;
- external assays have a separate aligned-transfer target/decoy q-value;
- relaxed direct and generic candidates carry explicit relaxed origin/flags;
- strict features retain strict confidence independent of MS2 association.

Audit reason codes remain visible so coverage cannot be increased by silently
dropping difficult events from the denominator.

## Project and external-assay workflow

Project mode reads a deterministic manifest containing at least `run_id` and
`mzml_path`, plus optional PSM/configuration and experimental grouping fields.
Each run first completes its own single-run workflow and publishes atomically.

For compatible alignment groups, the project stage then:

1. selects reference runs deterministically;
2. builds shared high-confidence peptide/charge anchors;
3. fits robust monotonic RT mappings with minimum-anchor and MAD checks;
4. plans missing exact assays in recipient runs;
5. extracts and quantifies recipient-run MS1 signal; predicted RT is a local
   centre rather than a required exact event scan;
6. applies separate target/decoy and isotope-quality controls; a separate
   donor-guided weak family can accept a 2-point mono plus 2-point secondary
   isotope component at q <= 0.05 after residual-overlap control computed
   against strict external claims already accepted for that recipient;
7. writes external evidence and project summaries.

Donor intensity is never copied into a recipient. An external assay may add no
feature when the recipient lacks defensible MS1 evidence. Weak recovery rejects
candidates whose intensity is more than 20% already explained by accepted
recipient features, subtracts the remaining assigned intensity, and quantifies
only the residual recipient component. Raw workers return compact sparse
raw-point footprints; the recipient process compares them only after current
strict ownership has been allocated. On resume, it deterministically rebuilds
claims for published external features from the pre-external ownership cache
before new recovery. Weak transferred features never become donors themselves.

## Caching and performance design

Four cache layers accelerate repeated development without changing scientific
results:
results:

1. **Raw MS1 cache**: compact memory-mappable original centroids and scan
   metadata.
2. **Strict-stage cache**: ingestion products, strict contexts/features and
   bounded direct processed-hill competitors. Current format is cache v2.
3. **Candidate cache**: expensive generic target/decoy local candidates keyed
   by the residual ownership state.
4. **Residual-ownership cache**: final sparse raw-point intensity claims used
   by project external recovery to prevent double quantification.

Cache keys include source fingerprints, scientific parameters and relevant
implementation signatures. Scheduling-only/downstream options do not
unnecessarily invalidate upstream caches. A stale or partially published cache
is rejected rather than reused. Outputs and cache directories use staging plus
atomic publication.

`--cache-dir` defines one root for all three layers in single-run and project
processing. The default root is `.biosaur2_cache` in the current directory.
Single-run commands use an isolated invocation namespace without
`--keep-cache`. Project mode instead uses a deterministic project workspace and
an atomic checkpoint: interruption retains compatible cache layers and the next
invocation resumes automatically. On a successful Project, strict/candidate
layers are deleted after their local consumer, raw/ownership layers after the
external recipient, and the remaining workspace is deleted. With
`--keep-cache`, stable source-keyed run directories allow later compatible
commands to reuse every layer.

For Project mode, `--workers` is a busy-core target. The manager starts a
four-worker cohort and samples the owned process tree's CPU/PSS plus system
available memory from Linux `/proc`. After three low-CPU samples it adds
one-worker runs; declared allocations are capped at 1.5 times the target and
new submission stops when CPU or `--max-memory` (integer GiB, no swap) would be
exceeded. Local and external-recipient work share this manager. Atomic per-run
and per-recipient checkpoint records make default resume skip published work.
Each recipient record fingerprints its exact donor plans, alignment model,
external science options, implementation and published outputs. A changed plan
recomputes only affected recipients; a missing or changed output is never
treated as complete. Checkpoint records are independently atomically published
per run, with a heartbeat lease for cross-host recovery, so large Projects do
not rewrite a growing global checkpoint after every completion.
CLI startup fixes implicit OpenMP, BLAS, NumExpr, vecLib and Arrow CPU/I/O pools
at one thread before numerical modules load.

## Output contract

Hybrid mode publishes a single de-duplicated population in two primary tables:

| Output | Contract |
|---|---|
| `<stem>.features.parquet` | One row per accepted MS1 feature, including its quantification fields and a typed list of zero or more linked MS2 event/audit structs. |
| `<stem>.identifications.parquet` | One accepted parsed PSM row with nullable direct-assay fields merged into the same row. PSM-bearing events may remain even when no feature was obtained. |
| `<stem>.external_id_evidence.parquet` | Project-only donor-assay attempts in a recipient run, including accepted and rejected target/decoy outcomes. |
| `<stem>.biosaur2.duckdb` | Per-input alternative containing the same run tables; project processing adds external evidence to that run's database. |
| `project.duckdb` | Run status, paths, resolved options, stage/cache summaries, alignment and validation metadata. |

Feature IDs are positive and unique. Every feature has exactly one merged
quantitative record. Every persisted `ms2_events` member references its parent
positive-quant feature. Internal audit finalization still classifies every MS2
event, but an event with neither a feature nor a PSM is retained only in
summary counts, not as a public row. Project validation checks the published
contracts before considering a run successful.

One public `--format {tsv,parquet,duckdb}` controls all requested outputs.
Legacy defaults to TSV and Hybrid defaults to Parquet. `--write-ms2` is a
legacy-only normalized-precursor diagnostic because Hybrid stores linked
events with features and PSM-bearing events with identifications.

## Important defaults

| Parameter | Current default | Meaning |
|---|---:|---|
| feature/project mode | `legacy` | Hybrid residual processing is opt-in. |
| PSM q-value maximum | 0.01 | Direct-assay input control. |
| targeted MS2 RT tolerance | 120 s | Initial bounded local search window; run calibration may tighten retries. |
| maximum charge | 7 | Charge hypotheses/features up to z=7. |
| output format | legacy: `tsv`; hybrid: `parquet` | One format control for all run tables. |
| quantification | `all` | Report envelope area, mono area and envelope apex; envelope area is `quant_value`. |
| project baseline | `edge_linear` | Optional baseline preprocessing. |
| generic extraction q-value maximum | 0.01 | Separate target/decoy family; configurable. |
| generic selected-isotope errors | `0,1,2,3` | Test a selected peak interpreted as M through M+3. |
| generic local isotope channels | 5 | Channels evaluated for a generic envelope. |
| generic local point minima | mono 3; channel 3; supported channels 2 | Standard local-recovery support. |
| generic local isotope cosine | 0.90 | Standard observed/averagine envelope agreement. |
| generic local maximum width | `auto` | Strict-feature width q99 clamped to 15-60 s; fallback 30 s. |
| relaxed local minima/cosine | mono 2; channel 2; channels 2; cosine 0.95 | Guarded retry defaults. |
| relaxed MS2 feature | false | Conservative MS2-only retry is disabled by default. |
| project workers | 4 | Busy-core target for the adaptive Project manager; declared allocation is bounded at 1.5x. |
| project max memory | physical RAM | Integer-GiB Project admission limit, excluding swap. |

## Module responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | Strict detection orchestration, strict-stage cache and final residual strict detector. |
| `hills.py`, `cutils.pyx` | Deterministic hill normalization and performance-critical detection routines. |
| `preprocessing.py`, `raw_ms1.py` | mzML ingestion, MS1/MS2 metadata and compact raw store/cache. |
| `identifications.py`, `chemistry.py` | Percolator parsing/mapping, modification normalization, exact formulas and isotope libraries. |
| `hybrid.py` | Direct/generic association, residual recovery, strict protection, quantification assembly and audit finalization. |
| `generic_association.py`, `generic_local.py` | Generic precursor association, scoring, local extraction and target/decoy candidates. |
| `local_refinement.py`, `optimization.py`, `cutils.pyx` | Bounded trace edits, local components and non-negative decomposition. Generic-local component gating (channel support, integration, cosine, apex and score inputs) runs in the Cython numeric kernel; Python retains the evidence policy and output objects. |
| `residual.py` | Reversible intensity ownership and conservation ledger. |
| `direct_competitors.py` | Pre-conflict capture of bounded direct-relevant losing hill candidates. |
| `confidence.py` | Deterministic decoys, competitions and extraction q-values. |
| `quantification.py` | Area/apex calculations and baseline handling. |
| `external.py`, `alignment.py` | Multi-run RT alignment and recipient-run external assay extraction. |
| `stage_cache.py`, `postprocess_cache.py` | Fingerprinted strict and local candidate caches. |
| `project.py`, `project_manifest.py` | Bounded multi-run execution, resume/validation and project metadata. |
| `output.py`, `legacy_output.py`, `duckdb_output.py` | Atomic output lifecycle, schemas and compact formats. |

## Validated behavior as of 2026-07-30

- Standard residual mode on the frozen 12-run panel linked
  363,117/514,529 MS2 events to positive-quant features: 70.5727%.
- Guarded relaxed mode on four runs linked 115,693/165,776 events: 69.7888%,
  adding 1,119 links over its same-input standard baseline with zero lost
  standard links and zero lost baseline final-strict scientific rows.
- The guarded four-run output contained 794,341 features and exactly 794,341
  positive quant rows; all four residual ledgers conserved intensity.
- Repeated-MS2 behavior reuses one feature abundance rather than duplicating it.
- Real cache-v2 validation captured 465 losing-hill matches for 443 direct
  assays. Seventeen retries were attempted and 14 locally selected, but no new
  feature/link passed all final gates. This is retained as a safe bounded path,
  not claimed as a measured coverage improvement.
- The most recent full test baseline is 215 passed; build, compile, CLI and
  output validation also passed.

## Known boundaries

- Reliable coverage is not forced to 100%. Sparse chromatographic traces,
  missing event-scan isotope signal and non-identifiable overlaps remain null.
- Universal decomposition of distinct overlapping envelopes is not always
  mathematically identifiable. Such cases remain explicit conflicts.
- External aligned assays are implemented, but the current validation panel
  produced zero accepted q<=0.01 gain. Confidence thresholds were not weakened.
- The real processed-hill cache-v2 experiment demonstrated execution and safety
  but zero final coverage gain on run 1555082.

These boundaries must not be “fixed” by lowering non-MS2 thresholds, accepting
single-survey-scan features, duplicating shared intensity, copying donor
abundance, or weakening q-value control.

## Design maintenance

Update this document whenever a change materially alters feature population
construction, hill ownership/repair, residual allocation, evidence hierarchy,
confidence control, quantification semantics, output contracts, cache validity,
mode defaults, or project execution. Small refactors that do not change design
or observable behavior do not require a design rewrite, but their update notes
should still state that the design is unchanged.
