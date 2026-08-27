# Biosaur2 algorithm design

## Document status

This document describes the algorithm and workflow implemented in the
repository. It covers the legacy detector and the opt-in hybrid residual
workflow. It is an implementation design document, not a promise that every
MS2 event can always be assigned a feature.

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

The identification adapter accepts compressed text or Parquet, handles common
BOM/encoding, delimiter/header variations and semantic column aliases, and
maps PSMs to MS2 using native identity when possible, otherwise safe
PSMId or split run/scan parsing with charge validation. Extra source columns
are ignored. Aggregate PSM inputs with a run/idn field are filtered to the
current mzML filename stem before mapping; advanced column-name overrides are
available only through `--help-all`.

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

For an initially conflict-free strict population, workers may map immutable
raw-point footprints concurrently. They write compact numeric footprint
artifacts, then the ledger owner consumes those artifacts in original feature
ID order. The owner retains all mutable claims, exact overallocation fallback
and failure accounting; parallel transport cannot change ownership order.

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

Strict trace quantification and final legacy-compatible strict-row preparation
are independent per feature after conflict decisions complete. They may run in
ordered worker ranges and are merged by the parent in the pre-existing feature
order. Quantification and final feature row values, IDs and schemas remain the
same.

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

When a run has more than one worker, the read-only target and paired-decoy link
passes may run as a two-process forked pair.  Each child receives the immutable
strict context through copy-on-write state and returns only its link rows and
summary.  Score calibration, q-value competition and audit mutation remain in
the parent, so worker completion order cannot alter scientific output.

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
The parent applies accepted candidates in input order.  Its private incremental
index caches each accepted candidate's raw-point footprint and narrows possible
m/z/charge/FAIMS equivalents and raw-point conflicts before rerunning the same
exact predicates in prior acceptance order.  The index is an execution detail:
feature IDs, ledger allocation order, audit outcomes and quantification order
are unchanged.

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

Residual `detect_hills` remains scan-sequential. After it, calibrated candidate
filtering can be range-parallel, and the greedy selector can process independent
connected components of shared mono/isotope hill IDs in parallel. Each component
runs the unchanged greedy rule, then the parent applies the established final
feature sort before assigning IDs. This optimization is restricted to the final
residual detector and is disabled for weak-candidate audit collection.

### Stage 10: final MS2 recheck and audit finalization

Unresolved direct assays are rechecked against new final-residual strict
features. Unidentified events are re-evaluated through a separate final-strict
target/decoy family, using the same independent read-only worker pair when
available. Failed prior audit reasons are preserved unless a valid feature
association is obtained.

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
`edge_linear` or `none` is optional baseline preprocessing. The compact output
retains the selected values and an explicit status/quality flag. Raw and
corrected areas are added only with `--write-quant-details`.

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
- external weak-feature transfers have a separate aligned-transfer target/decoy q-value;
- relaxed direct and generic candidates carry explicit relaxed origin/flags;
- strict features retain strict confidence independent of MS2 association.

Audit reason codes remain visible so coverage cannot be increased by silently
dropping difficult events from the denominator.

## Project feature match-between-runs workflow

Project mode reads a deterministic manifest containing at least `run_id` and
`mzml_path`, plus optional PSM/configuration and experimental grouping fields.
Each run first completes its own single-run workflow and publishes atomically.

With `--external-id`, each local hybrid run additionally persists calibrated
isotope envelopes that lose the ordinary conflict selection.  These private
weak candidates require at least two mono points, one secondary isotope raw
hill with at least two points, isotope cosine >= 0.6, positive quantification,
and at most 0.30 final-strong ownership overlap. They are not public
features unless Project transfer accepts them.  Normal output remains the
strong population; every default feature, including direct/generic recovery,
is a valid strong reference and PSMs are not required for external acceptance.

For each compatible alignment group, Project then:

1. builds charge/FAIMS/mz mutual-nearest anchors from strong features only and
   fits a bounded reference-star RT forest; default minimum anchors is 20 and
   fitting is capped at 256 anchors per edge;
2. matches each recipient weak candidate to source-run strong features using
   exact charge/FAIMS, 8 ppm and the unchanged 120 second aligned RT window;
3. keeps one best support per source run, then at most four supports from
   distinct source runs. Supports are ranked by normalized joint m/z/RT
   distance with deterministic error, quality and feature-ID tie breaks;
4. repeats the match after a deterministic neutral-mass decoy shift and fits a
   monotone empirical target/decoy log-likelihood ratio over support-score bins
   with deterministic two-fold cross-fitting;
5. sums the calibrated contributions separately for each weak candidate's
   retained target and decoy supports, competes those aggregate scores, and
   estimates q-values independently inside each alignment group using the
   conservative +1 decoy correction;
6. promotes target winners at q <= 0.10 into the normal feature output with
   `feature_origin=aligned_external_weak`, `external_support_count` and a
   complete evidence record.

The Project stage does not open mzML or raw MS1 caches and never creates a new
feature by donor-guided extraction.  It only validates measured recipient weak
candidates against measured strong features.  Source strong features may
support multiple recipient weak candidates, as matching is candidate-centric.
Evidence has one row per accepted support and one compact audit row for each
rejected candidate.

## Caching and performance design

Four cache layers accelerate repeated development without changing scientific
results:

1. **Raw MS1 cache**: compact memory-mappable original centroids and scan
   metadata.
2. **Strict-stage cache**: ingestion products, strict contexts/features and
   bounded direct processed-hill competitors. Current format is cache v3.
3. **Candidate cache**: expensive generic target/decoy local candidates keyed
   by the residual ownership state.
4. **External feature-MBR sidecars**: source-provenanced compact strong-feature
   and full weak-candidate Parquet sidecars written during local hybrid
   postprocessing. They are invalidated by source or weak-gate changes and let
   Project matching run without raw MS1 access.

For a cold Hybrid run, strict-stage payload construction and atomic cache
publication may execute in one background process after strict detection while
the parent begins independent Hybrid postprocessing. The cache remains absent
until its atomic rename completes. The parent joins the writer before reporting
successful completion and propagates any writer error, so cache validity and
subsequent-cache behavior are unchanged.

Split-hill workers write numeric labels to invocation-local `.npy` artifacts.
The parent memory-maps them in worker-range order, maps labels by first
encounter exactly as before, and removes the artifacts after successful merge
or failure. This avoids sending million-element Python lists through process
queues while preserving deterministic hill IDs.

Cache keys include source fingerprints, scientific parameters and relevant
implementation signatures. Scheduling-only/downstream options do not
unnecessarily invalidate upstream caches. A stale or partially published cache
is rejected rather than reused. Outputs and cache directories use staging plus
atomic publication.

`--cache-dir` defines one root for all cache layers in single-run and project
processing. The default root is `.biosaur2_cache` in the current directory.
Single-run commands use an isolated invocation namespace without
`--keep-cache`. In that temporary single-run mode only the raw MS1 mmap store
is created: reusable strict-stage and generic-candidate caches are disabled so
their serialization cannot compete with the current analysis. Project mode instead uses a deterministic project workspace and
an atomic checkpoint: interruption retains compatible cache layers and the next
invocation resumes automatically. Completed local raw, strict and candidate
layers are deleted after their compact external sidecars and outputs are
published, and the remaining workspace is deleted after Project success. With
`--keep-cache`, stable source-keyed run directories allow later compatible
commands to reuse every layer.

For Project mode, `--workers` is a busy-core target. The manager normally uses
four workers per run. The default `--scheduler-resource-mode auto` reads only
host available memory every five seconds. Every 60 seconds it
walks active wrapper trees through Linux `children` files and reads one `stat`
line per owned process to aggregate RSS, CPU use and thread counts; it does not
enumerate unrelated host processes or read `smaps_rollup`. Cold admission
reserves 16 GiB per run, then uses compatible resumed or completed per-run
peaks times 1.2, constrained to 1-30 GiB. Each newly started run retains that
estimate until both its wall time and its aggregate owned-process CPU time
reach three minutes. The default heartbeat aggregates those CPU seconds from
the already-read owned process `stat` lines; it does not add scans or increase
sampling frequency. After both conditions are met, host available memory
already reflects actual RSS, so auto mode does not reserve possible future
growth. In auto mode, `--max-memory` is an integer-GiB host-use ceiling, so the
manager retains at least the greater of 8 GiB, 5% physical memory, and
`physical_memory - max_memory`. CPU pressure pauses submission; a safety-floor
breach terminates newest work, requeues it, and reports `MemoryLimitExceeded`
after the third preemption. The explicit
`detailed` resource mode retains complete Project PSS accounting.
Declared allocations remain capped at 1.5 times the target and at three times the
host's logical CPU count, including the existing preemptible eight-worker and
one-worker overcommit tiers. Local work uses this manager while feature-MBR
matching is a bounded in-memory Project stage. Sidecar reads and independent
per-run feature/evidence publication use one bounded spawned pool, constrained
by the effective worker budget, host CPU/memory limits and 32 processes.
Alignment, target/decoy calibration and q-value assignment remain deterministic
parent work. Atomic per-run checkpoint records make default resume
skip compatible local work. Project alignment and competition are then rerun
deterministically from source-fingerprinted strong/weak sidecars; there is no
raw-recipient or external-recipient checkpoint. Run checkpoint records are
independently atomically published with a heartbeat lease for cross-host
recovery, so large Projects do not rewrite a growing global checkpoint after
every completion. The adjacent append-only `scheduler-events.jsonl` records
start, completion, memory-preemption and summary history. It is diagnostic and
seeds memory estimates after resume; only atomic per-run records determine
which outputs are complete.
CLI startup fixes implicit OpenMP, BLAS, NumExpr, vecLib and Arrow CPU/I/O pools
at one thread before numerical modules load. DuckDB output staging closes its
unused host-sized default scheduler and opens a connection limited to the run's
effective worker allocation.

After completed local work, Project summary construction uses bounded spawned
readers. Parquet feature/MS2 row counts come from file metadata, while direct
assay counts stream only the nullable `assay_id` column in bounded batches. The
parent remains the only Project DuckDB writer: it restores manifest order,
inserts summary rows in one transaction, closes the neighboring temporary DB
and then atomically publishes it. Summary readers are capped by the resolved
Project budget, host CPU/memory limits and 32 concurrent processes.

Explicit `project validate` uses the same read-only per-run process model.
`project validate --workers N` overrides the recorded Project worker budget;
the parent restores `run_order` before reporting problems, so parallel
completion cannot change diagnostics. Validation readers are likewise capped
by host CPU/memory limits and 32 processes. Both feature-MBR and validation
stay serial below an 8 MiB input/output threshold, avoiding spawn overhead for
small Projects.

## Output contract

Hybrid mode publishes a single de-duplicated population in four primary tables:

| Output | Contract |
|---|---|
| `<stem>.features.parquet` | One row per accepted MS1 feature, including compact quantification and evidence fields. |
| `<stem>.ms1.parquet` | One row per MS1 survey scan with canonical `scan_id`, RT seconds and total intensity. |
| `<stem>.ms2_events.parquet` | One row per feature-linked MS2 event, containing `feature_idx`, stable/raw spectrum references, RT, precursor m/z and charge. |
| `<stem>.identifications.parquet` | One accepted parsed PSM row with nullable direct-assay fields merged into the same row. PSM-bearing events may remain even when no feature was obtained. |
| `<stem>.external_id_evidence.parquet` | Project-only source-run support rows and rejected weak-candidate target/decoy outcomes. |
| `<stem>.biosaur2.duckdb` | Per-input alternative containing the same run tables; project processing adds external evidence to that run's database. |
| `project.duckdb` | Run status, paths, resolved options, stage/cache summaries, alignment and validation metadata. |

Feature IDs are positive and unique. Every feature has exactly one merged
quantitative record. Every persisted `ms2_events.feature_idx` references its
parent positive-quant feature and every `ms2_event_id` is unique within a run.
Every Hybrid feature has `scanStart`, `scanApex` and `scanEnd` values that
resolve to `ms1.scan_id`. The legacy-compatible `rtStart`, `rtApex` and
`rtEnd` minute columns remain in features; duplicate Hybrid RT-second columns
are not public.
Internal audit finalization still classifies every MS2
event, but an event with neither a feature nor a PSM is retained only in
summary counts, not as a public row. Project validation checks the published
contracts before considering a run successful.

One public `--format {tsv,parquet,duckdb}` controls all requested outputs.
Legacy defaults to TSV and Hybrid defaults to Parquet. `--write-ms2` is a
legacy-only normalized-precursor diagnostic because Hybrid stores compact
linked-event references separately and PSM-bearing events in identifications.
MS1 output defaults on in Hybrid and off in Legacy; `--write-ms1` and
`--no-write-ms1` override the mode default.

## Important defaults

| Parameter | Current default | Meaning |
|---|---:|---|
| feature/project mode | `legacy` | Hybrid residual processing is opt-in. |
| PSM q-value maximum | 0.01 | Direct-assay input control. |
| targeted MS2 RT tolerance | 120 s | Initial bounded local search window; run calibration may tighten retries. |
| maximum charge | 7 | Charge hypotheses/features up to z=7. |
| output format | legacy: `tsv`; hybrid: `parquet` | One format control for all run tables. |
| MS1 basic table | legacy: omitted; hybrid: written | `--write-ms1`/`--no-write-ms1` override the mode default. |
| quantification | `all` | Report envelope area, mono area and envelope apex; envelope area is `quant_value`. |
| Hybrid mono traces | omitted | `--write-mono-hills` adds the two monoisotopic point arrays. |
| Hybrid raw/corrected areas | omitted | `--write-quant-details` adds four baseline diagnostic columns. |
| project baseline | `edge_linear` | Optional baseline preprocessing. |
| generic extraction q-value maximum | 0.05 | Separate target/decoy family; the maintained development and validation baseline. |
| generic selected-isotope errors | `0,1,2,3` | Test a selected peak interpreted as M through M+3. |
| generic local isotope channels | 5 | Channels evaluated for a generic envelope. |
| generic local point minima | mono 3; channel 3; supported channels 2 | Standard local-recovery support. |
| generic local isotope cosine | 0.90 | Standard observed/averagine envelope agreement. |
| generic local maximum width | `auto` | Strict-feature width q99 clamped to 15-60 s; fallback 30 s. |
| relaxed local minima/cosine | mono 2; channel 2; channels 2; cosine 0.95 | Guarded retry defaults. |
| relaxed MS2 feature | false | Conservative MS2-only retry is disabled by default. |
| standalone external ID | false | Use `--external-id` to collect weak candidates; Project keeps this enabled by default. |
| project workers | 4 | Busy-core target for the adaptive Project manager; normal runs use four workers and declared allocation is bounded at 1.5x. |
| scheduler heartbeat | 60 sec | Owned-process RSS/CPU/thread scan interval; auto mode reads only host memory every 5 sec. |
| scheduler resource mode | `auto` | Fast host-memory admission; `detailed` retains complete Project PSS accounting. |
| project max memory | physical RAM | Integer-GiB host-use ceiling in auto mode; Project PSS admission limit in detailed mode. |

## Module responsibilities

| Module | Responsibility |
|---|---|
| `main.py` | Strict detection orchestration, strict-stage cache and final residual strict detector. |
| `candidate_selection.py`, `peak_splitting.py`, `strict_cache_writer.py` | Deterministic candidate conflict selection, disk-backed split result transport and asynchronous retained-cache publication. |
| `hills.py`, `cutils.pyx` | Deterministic hill normalization and performance-critical detection routines. |
| `preprocessing.py`, `raw_ms1.py` | mzML ingestion, MS1/MS2 metadata and compact raw store/cache. |
| `identifications.py`, `chemistry.py` | Percolator parsing/mapping, modification normalization, exact formulas and isotope libraries. |
| `hybrid.py` | Stable compatibility imports for Hybrid data types, entry points and selected helpers. |
| `hybrid_assays.py`, `hybrid_local.py`, `hybrid_strict.py` | Direct assay construction, bounded local extraction, strict-feature association and ownership protection. |
| `hybrid_direct_stage.py`, `hybrid_generic_stage.py`, `hybrid_residual_stage.py`, `hybrid_postprocess.py` | Hybrid stage orchestration, final residual rechecks, quantification assembly and audit finalization. |
| `hybrid_generic_association.py`, `hybrid_generic_local.py` | Hybrid-specific generic association, score calibration, local recovery and audit summaries. |
| `generic_association.py`, `generic_local.py` | Generic precursor association, scoring, local extraction and target/decoy candidates. |
| `local_refinement.py`, `optimization.py`, `cutils.pyx` | Bounded trace edits, local components and non-negative decomposition. Generic-local component gating (channel support, integration, cosine, apex and score inputs) runs in the Cython numeric kernel; Python retains the evidence policy and output objects. |
| `residual.py` | Reversible intensity ownership and conservation ledger. |
| `direct_competitors.py` | Pre-conflict capture of bounded direct-relevant losing hill candidates. |
| `confidence.py` | Deterministic decoys, competitions and extraction q-values. |
| `quantification.py` | Area/apex calculations and baseline handling. |
| `external_weak.py`, `external_mbr.py`, `external_alignment.py`, `alignment.py` | Private weak-candidate gates, feature-only RT alignment, strong support indexing and Project target/decoy transfer competition. |
| `stage_cache.py`, `postprocess_cache.py` | Fingerprinted strict and local candidate caches. |
| `project.py`, `project_manifest.py` | Bounded multi-run execution, resume/validation and project metadata. |
| `output.py`, `legacy_output.py`, `duckdb_output.py` | Atomic output lifecycle, schemas and compact formats. |

## Known boundaries

- Reliable coverage is not forced to 100%. Sparse chromatographic traces,
  missing event-scan isotope signal and non-identifiable overlaps remain null.
- Universal decomposition of distinct overlapping envelopes is not always
  mathematically identifiable. Such cases remain explicit conflicts.

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

Date-specific validation results and test counts belong in `updates/`, where
their dataset and implementation context can be retained without becoming part
of the current design contract.
