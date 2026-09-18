# Session State — 2026-09-18 (Session 2: dataset re-verification + community intel)

## What happened this session

Two parts: (1) re-verified project state (see `state-2026-09-18.md` from
earlier the same day — still zero code/data/git-history as of the prior
check), (2) reviewed community intelligence the user found — a Kaggle EDA
notebook and a public-LB-rank-1 discussion post — and wrote it up as a new
doc, `docs/05-community-intel.md`.

### Dataset re-verification

User pasted the official Kaggle Data-tab description in full. Cross-checked
against our existing `docs/02-dataset.md` — **matches almost exactly**,
confirming that doc is accurate and doesn't need correction. No changes made
to `02-dataset.md` this session.

### Community intel reviewed

1. **"CASMI 2026: A Complete Statistical Tour"** (EDA notebook by
   josefreitasalvesneto) — could NOT be fetched; Kaggle notebook pages are
   client-side rendered, `web_fetch` returned empty content on two attempts
   (truncated mode and selective mode). Flagged as a next-session TODO to
   review manually in a browser.
2. **Discussion post: "[0.339 Top 1 Solution] 4-Channel Mass-Shifted Analog
   Propagation & Neural Bayes Reranking"** — user pasted full text (Kaggle
   discussion pages also not fetchable by our tools — confirmed empty
   output attempting to fetch the competition discussion URL directly).
   Public-LB rank 1 at time of posting, score progression 0.149 → 0.193 →
   0.205 → 0.239 → 0.339. Full writeup now in `docs/05-community-intel.md`.

### Key takeaways written into `docs/05-community-intel.md`

- Their hypothesized class-distribution estimate: Tier1(~10-15%)/
  Tier2(~45-55%)/Tier3(~30-40%), roughly mapping to our Class 1/2/3 — this
  is convergent validation of our existing novelty-class framing from an
  independent competitor, not a new idea.
- **Most useful new idea**: "mass-shifted analog propagation" — search
  training spectra within a mass-shift window (±CH2, ±OH, ±hexose etc.),
  score candidates by spectral-similarity-to-analog × fingerprint-Tanimoto-
  to-analog. This sits between our existing Strategy A (exact library
  search) and Strategy B (formula/fingerprint DB retrieval) and is NOT
  currently in our `04-method-landscape.md` — proposed as a new "Strategy
  A2" but NOT YET MERGED into the canonical plan, pending user confirmation
  (flagged explicitly since it adds a new pipeline stage beyond original
  session-1 scope).
- Their other channels (exact spectral match, chemical/substructure
  heuristic filtering, spectrum->fingerprint transformer + GBT reranking)
  are functionally equivalent to strategies already planned in our
  `04-method-landscape.md` — validates the plan direction, no changes
  needed there.
- **What did NOT work for them** (actionable avoidance guidance): blind
  full-PubChem isomer expansion hurt ranking (too many decoys) — prefer
  curated NP databases (COCONUT, ChEBI, LipidMaps) over raw PubChem
  breadth. Overly wide precursor mass tolerance also hurt — their empirical
  sweet spot was ~±8.5 ppm (empirical for their pipeline, not universal,
  but a reasonable starting point for us to sweep from).
- **Efficiency trick worth reproducing**: under independent-Bernoulli-bits
  assumption, fingerprint-based candidate scoring reduces to a linear term
  computable via one vectorized matmul across the whole candidate pool —
  applicable when we build our own Strategy B fingerprint reranking step.
- Explicit caveat documented: public LB score (0.339) is computed on a
  subset of test data per standard Kaggle code-competition practice; final
  scoring re-runs against a hidden set, so this number is informative, not
  a guarantee of final standing.

## Files changed this session

- Created `docs/05-community-intel.md` (new doc, community/competitor
  intelligence — distinct from `03-literature-review.md` which is academic
  papers only)
- Updated `.kiro/steering/codebase-map.md` — added the new doc to the file
  tree listing
- Updated `.kiro/steering/project-context.md` — added a pointer to the new
  doc in the "What this project is" section
- Created this file, `.kiro/memory/state-2026-09-18-session2.md`

## LATE SESSION 2 ADDENDUM — notebook source reviewed, leakage found, plan written

User supplied the rank-1 notebook's **full source code and output**. Key
outcome: **the 0.297/0.339 public score is leakage-driven.** The notebook's
own diagnostic prints `best_library_sim` = exactly 1.0, std 0.0, for 100% of
400 visible test molecules, and "Molecules powered by Analog Propagation &
Neural Model: 0.0%". The visible test.parquet is a train sample (organizer-
documented), so Channel 1 exact-lookup explains essentially the whole score
and Channels 2-4 are untested by their own numbers. Their cited
hyperparameter sweeps were all measured under this leakage -> treat as
starting points, not optima. Notebook UI also shows Public Score **0.297**,
not the 0.339 claimed in the title.

Also verified from source: 712,199-structure candidate pool (COCONUT 2.0
436,389 + 275,810 train structures deduped by inchikey14); ChEBI/LipidMaps
attached but DISABLED (`USE_BIO_DB=False`, "prevent decoy dilution");
Channel 1 uses Numba spectral *entropy* similarity (Li et al. 2021 entropy
weighting), not plain cosine; Channel 2 takes max(direct, mass-shifted) sim
over ±200 Da with `sim^4` Tanimoto weighting; Channel 3 is a real 1-2 bond
in-silico fragmenter scoring sqrt-intensity-weighted peak explanation;
Channel 4 FPNet is a 6-layer d=512 transformer embedding both m/z AND
neutral loss per peak, ensembling a single-spectrum and a merged-spectrum
checkpoint, scored as `cand_fp @ logits` (one matmul); reranker is 8 HistGBMs
(2 priors x 4 seeds) over 142,762 x 31 precomputed features. Runtime 53 min
on 2x T4.

**Wrote `docs/06-implementation-plan.md`** — the concrete build plan, now the
"start here" doc. Contents: validation-first design (inchikey14 split,
NP-weighted hold-out, synthesized Class-1/2/3 proxies, MRR@25 scorer with
unit tests before anything else); baseline defined as Channel1+Channel2 with
rank fusion scored on the honest split (NOT a dummy submission, NOT the
public LB); 5-channel pipeline; 10-step build order; Azure 4x T4 vs
no-internet-Kaggle work split; their hyperparameters recorded as starting
points with explicit caveat.

**Our deliberate differences from the rank-1 approach**: keep de novo
generation (they are retrieval-only, structurally incapable of Class 3 which
they estimate at 30-40% of test — this is our edge); default candidate pool
to COCONUT+train with ChEBI/LipidMaps behind a flag we validate ourselves;
mandatory stagewise per-channel attribution; simpler fingerprint model
before justifying a transformer.

Also updated `05-community-intel.md` with a prominent leakage warning at the
top plus a full verified-architecture section, and put a "CRITICAL: the
visible test set leaks" section in `project-context.md`.

## IMPLEMENTATION ADDENDUM — code written, 9 atomic commits

### Corrections to earlier state files

**`state-2026-09-17.md` and `state-2026-09-18.md` are STALE on two points:**
1. They say "nothing committed or pushed". Wrong — commit `9f19493`
   ("Initial project setup: docs, literature review, folder structure") **is
   already on `origin/main`**. Verified with `git log origin/main`.
2. They say no code / no pyproject.toml exists. No longer true.

### Environment

`.venv` via `uv`, Python 3.12.10 (found at
`C:\Users\aagne\AppData\Local\Programs\Python\Python312\python.exe` — plain
`python` is NOT on PATH, the WindowsApps shim fails). `uv` is at
`C:\Users\aagne\.local\bin\uv.exe`.

Run tests: `.venv\Scripts\python.exe -m pytest tests/ -q`
Lint: `.venv\Scripts\python.exe -m ruff check src/ tests/ scripts/`

Installed: numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1, numba 0.67.0,
sklearn 1.9.1, rdkit 2026.03.3 (pinned to scoring env), torch 2.14.0+cpu
(optional `train` extra, installed locally only to verify the training code
actually runs).

### What was built

317 tests, all passing, ruff clean. Nine commits this session:

| Commit | Contents |
|---|---|
| `71351eb` | pyproject.toml (uv/hatchling, RDKit pinned 2026.3.3, torch optional) |
| `946c8a5` | chem.py, adducts.py, spectra.py, config.py + tests |
| `47f8975` | eval/metrics.py (MRR@25) + tests |
| `b15613e` | data/loaders.py, data/split.py + tests + conftest fixtures |
| `afbcc0f` | candidates/pool.py + tests |
| `4757e77` | channels/{library,analog,fusion}.py + tests |
| `37741f4` | pipeline.py + tests |
| `283af96` | models/{fpnet,train}.py + tests |
| `d253aea` | scripts/{build_split,build_pool,run_baseline,train_fpnet}.py + CLI tests |

Plus a final docs commit (see below).

**NOTHING FROM SESSION 2 IS PUSHED.** All nine commits are local. Pushing
needs user confirmation.

### Verified behaviour (real run, synthetic train.parquet)

Ran the full script chain end-to-end. Output:

```
[OK] leakage check passed
  train_rows: 16, val_rows: 20
  class1_structures: 2, class2_structures: 4, class3_structures: 2
hold-out composition: riken 3, massbank 2, gnps 2, enveda-np-examples 1
                      (ZERO enveda-180 -> NP weighting works)

cohort            n   MRR@25   recall     top1
overall           8   0.7500   0.7500   0.7500
class1            2   1.0000   1.0000   1.0000
class2            4   1.0000   1.0000   1.0000
class3            2   0.0000   0.0000   0.0000   <- pool exclusion works

Channel contribution: library 0.0%, analog_or_other 100.0%
```

Class-3 at exactly 0.0 is the important line: it proves the synthetic class-3
cohort is genuinely unreachable by retrieval, so that metric is real rather
than decorative.

### Bugs the tests caught (all fixed)

1. `_clean_peaks` assumed ascending m/z input. Real parquet data is sorted, but
   the similarity kernel merge-aligns two peak lists, so unsorted input would
   have produced a wrong-but-plausible similarity score — invisible in
   aggregate. Now enforced with an O(n) check plus conditional sort.
2. `merge_spectra` normalisation semantics (per-spectrum, not global).
3. `compute_bit_weights` floor semantics — bits at >=50% frequency all collapse
   to weight 1.0 by design (boost rare bits, never suppress common ones).
4. A wrong NH4 mass constant in my own test (18.0338 vs correct 18.0344).

### Key design decisions (rationale in docs/07-codebase-guide.md)

- Split by `inchikey14`, never `spectrum_id`, never full `inchikey`.
- Hold-out NP-weighted via `SplitConfig.library_weights`; a structure in both
  an NP library and enveda-180 takes the MAX weight (counts as in-domain).
- Class 1 keeps sibling spectra in train; class 2 removes all spectra; class 3
  removes the structure from the candidate pool too.
- Leak-proofing enforced at load time (`keep_rows=train_mask`), not by
  downstream filtering. `train_fpnet.py --split` is mandatory for the same
  reason.
- Fingerprints bit-packed in the pool (7GB -> 900MB for ~700k structures).
- FPNet scoring is `cand_fp @ logits`, one matmul; a test asserts this is
  rank-equivalent to the exact Bernoulli log-likelihood.
- Peaks embedded by m/z AND neutral loss (neutral loss transfers across
  molecule sizes, which matters given the train/test domain gap).
- Fusion is transparent weighted scoring, NOT a learned reranker yet, so
  per-channel contribution stays measurable.

### Deliberate divergences from the rank-1 reference solution

- De novo generation stays in the plan (they are retrieval-only, structurally
  incapable of class 3, which they estimate at 30-40% of test). This is our
  intended edge.
- Candidate pool defaults to COCONUT + train structures; ChEBI/LipidMaps
  behind a flag we validate ourselves (they shipped theirs disabled).
- Mandatory per-channel attribution before accepting any score change.
- Simpler fingerprint model justified before a 6-layer transformer.

### Not implemented (steps 6-10 of docs/06-implementation-plan.md)

- FPNet not wired into `pipeline.py` (model + `score_candidates()` exist and
  are tested; the pipeline does not call them)
- Channel 5 in-silico fragmentation (MetFrag-lite)
- Learned GBM reranker replacing weighted fusion
- De novo generation for the class-3 tail
- **No competition data downloaded.** `data/raw/` still empty. All verification
  is against synthetic fixtures. The scripts have never run against the real
  3 GB parquet files.

## Not yet done / next-session TODOs
- **Manually review** (in an actual browser, not automated fetch) the
  statistical-tour notebook and the two notebooks linked from the 0.339
  discussion post (`0.339 Top 1: 4-Channel Transformer Analog Ensemble` and
  `SOTA 0.339 Review & Methodology`) — our fetch tooling cannot render
  Kaggle's JS-heavy notebook/discussion pages, confirmed by two independent
  empty-result attempts this session.
- **Decide whether to formally merge "Strategy A2 — analog propagation"**
  into `04-method-landscape.md` — written up as a recommendation in
  `05-community-intel.md` but deliberately NOT auto-merged into the
  canonical plan since it's a scope addition, needs user go-ahead first.
- All prior next-session TODOs from `state-2026-09-17.md` /
  `state-2026-09-18.md` still apply and are unchanged: download data,
  decide Python scaffolding, implement local MRR@25 scorer first, first
  EDA notebook, first git commit+push (needs user confirmation).

## Orientation for whoever picks this up next

Read order unchanged: `README.md` -> `docs/00-brief.md` -> `.kiro/steering/
project-context.md` -> `docs/02-dataset.md` -> `docs/03-literature-review.md`
-> `docs/04-method-landscape.md` -> `docs/05-community-intel.md` (new, read
last since it's competitor intel layered on top of our own plan, not a
replacement for it).
