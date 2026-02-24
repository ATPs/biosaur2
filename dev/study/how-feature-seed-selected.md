# How Feature Seeds Are Selected (from `main.py`)

This note explains how Biosaur2 chooses feature "seeds" (monoisotopic hills) and how it keeps the final seed set non-overlapping.

## What is a seed here?

In this code path, a **seed** is the hill used as `monoisotope idx` / `monoisotope hill idx` when building an isotope cluster candidate.

- Candidate creation happens in `get_initial_isotopes(...)` ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):71).
- Orchestration happens in `process_features_iteration(...)` ([biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):14).

## 1. Seed iteration order

`process_features_iteration` builds the list of hill indices sorted by hill median m/z:

- `sorted_idx_full = ... sorted(..., key=lambda x: x[-1])`
  ([biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):54, [biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):66)

That sorted list is passed to `get_initial_isotopes(...)`, and each index `idx_1` is treated as a potential monoisotopic seed:

- loop: `for idx_1 in sorted_idx_child_process:`
  ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):92)

So the raw seed traversal order is **ascending hill m/z**.

## 2. How a seed is accepted as a candidate feature

For each seed hill `idx_1`, the algorithm tries charges from high to low:

- `charges = ... [::-1]` and `for charge in charges:`
  ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):90, [biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):111)

For each isotope number `k = 1..9`, expected isotope m/z is:

- `m_to_check = hill_mz_1 + (1.00335 * isotope_number / charge)`
  ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):119)

Candidate isotope hills must pass all of these filters:

1. m/z fast-bin lookup hit (`hills_mz_median_fast_dict`) ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):122)
2. IM compatibility when `paseftol > 0` ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):124)
3. RT interval overlap ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):126)
4. absolute mass difference within `itol` tolerance ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):130)
5. at least one shared scan ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):132)
6. RT-shape cosine correlation >= 0.6 ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):138, [biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):140)

If isotope `k` has no candidates, search for that charge stops early:

- `if len(candidates) < isotope_number: break`
  ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):171)

Then all isotope combinations are evaluated (`itertools.product`), compared to averagine theoretical envelope, IVF tail-truncation is applied, and only combos with isotope-envelope cosine >= 0.6 are emitted:

- combo generation: ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):193)
- IVF rule: ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):205)
- isotope cosine gate: ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):216, [biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):218)

When accepted, a record is added to `ready` with this seed as monoisotope:

- `monoisotope idx`, `monoisotope hill idx`
  ([biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):223, [biosaur2/cutils.pyx](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/cutils.pyx):224)

## 3. Post-generation pruning before final seed set

Back in `main.py`, provisional candidates (`ready`) are pruned by isotope mass-error calibration and re-checked for envelope cosine:

- build/calibrate `isotopes_mass_error_map`: ([biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):88)
- keep isotope prefix while `|mass_diff_ppm - shift| <= 5*sigma`: ([biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):161)
- recompute cosine and drop failures: ([biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):172)

## 4. Final seed selection (non-overlapping greedy selection)

Final accepted features are chosen from `ready` using a greedy no-overlap rule:

1. Sort by `-(nIsotopes + cos_cor_isotopes)`
   ([biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):199, [biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):203)
2. Keep `ready_set` of already claimed hill IDs
   ([biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):202)
3. Accept candidate if monoisotope and all isotope hills are unclaimed
   ([biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):216)
4. If partially conflicting, truncate isotopes at first claimed hill and rescore cosine; keep only if still valid
   ([biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):227, [biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):242)

This step determines the **final selected seeds** because once a hill is claimed in `ready_set`, it cannot seed another accepted feature.

## Minimal pseudocode

```python
# seed enumeration
for seed in hills_sorted_by_mz:
    for charge in range(cmax, cmin - 1, -1):
        isotope_lists = find_isotope_candidates(seed, charge)
        for combo in product(*isotope_lists):
            if envelope_cosine(seed, combo) >= 0.6:
                ready.append(feature(seed, combo, charge))

# pruning
ready = filter_by_isotope_mass_error_and_rescore(ready)

# final seed selection
ready.sort(key=lambda f: -(f.nIsotopes + f.cos_cor_isotopes))
for f in ready:
    if no_hill_overlap(f, ready_set):
        accept(f)
        mark_claimed_hills(f, ready_set)
    else:
        f2 = truncate_before_first_conflict_and_rescore(f)
        if f2:
            reconsider(f2)
```

## Important implementation note

In multiprocessing mode, seed list splitting uses:

- `step = int(len_full / n_procs)` and slices `sorted_idx_full[i*step:i*step+step]`
  ([biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):68, [biosaur2/main.py](/data/p/xiaolong/biosaur2/biosaur2/biosaur2/main.py):70)

This can leave remainder indices unassigned, so some potential seeds may never be evaluated when `len_full % n_procs != 0`.
