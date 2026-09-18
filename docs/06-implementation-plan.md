# Implementation Plan — Baseline, Pipeline, Validation, Compute

Written 2026-09-18 (session 2), after reviewing the public-LB-leading
notebook's actual source code (see `05-community-intel.md`). This supersedes
the ordering in `04-method-landscape.md` where they disagree; `04` remains
the reference for *why* each strategy exists, `06` is the concrete build plan.

## The single most important context for this plan

The visible `test.parquet` is a sample drawn from `train.parquet` (organizer-
documented). The rank-1 public notebook's own diagnostic output shows
`best_library_sim` = exactly 1.0 for 100% of 400 visible test molecules, and
"Molecules powered by Analog Propagation & Neural Model: 0.0%". Its public
score is therefore almost entirely earned by exact library lookup against a
leaked test set, and says nothing about performance on the real hidden test
set that gets swapped in at re-run.

**Consequence for us: the public leaderboard is not a usable optimization
target.** We need our own honest offline validation before we can trust any
number, including our own. That's why validation is step 1 below, not step 5.

## Validation design (build FIRST, before any modeling)

### Split rule

Split `train.parquet` by **`inchikey14`** — never by `spectrum_id`, never by
full `inchikey`. Same compound recurs across the 11 source libraries and
appears at multiple adducts/collision energies; splitting by spectrum leaks
near-identical spectra of the same structure across the train/val boundary,
and splitting by full `inchikey` still leaks stereoisomers of the same 2D
skeleton (which is what actually gets scored).

### Hold out ~1,500–2,500 structures, NP-weighted

Do **not** sample held-out structures uniformly at random across train.
`enveda-180` contributes 1.15M of ~2.5M spectra but is synthetic drug-like
screening chemistry — a uniform random split produces a validation set that
is mostly off-domain and will report an MRR@25 that does not transfer to the
natural-product test distribution. Weight the held-out sample toward the
NP-relevant libraries: `riken`, `gnps`, `enveda-np-examples`, with
`massbank`/`mona`/`spectraverse`/`msdial` as secondary.

Use `enveda-np-examples` (250 compounds, same instrument + same processing
pipeline as the real test set) as a separate small high-fidelity sanity
check — it is the closest available proxy to the true test distribution.

### Synthesize the three novelty classes

Aggregate MRR@25 alone will hide which strategy is doing the work. Construct
per-class proxies inside the held-out set so we can measure class-conditional
MRR:

| Proxy | How to construct | Simulates |
|---|---|---|
| Class-1-like | Held-out structure still has *other* spectra of the same `inchikey14` remaining in the train side | Public reference spectra exist -> library match should fire |
| Class-2-like | All spectra of that `inchikey14` removed from train side, but structure IS present in the candidate database (COCONUT/train-structures) | Known structure, no spectra -> retrieval must carry it |
| Class-3-like | All spectra removed from train side AND structure removed from the candidate database for that query | Novel structure -> only de novo can reach it |

Class-3-like is the only honest way to measure de novo value, and it is
exactly the regime the rank-1 notebook cannot address at all (retrieval-only
pipeline, no generative component).

### The scorer itself

`src/casmi/eval/` — RDKit tautomer canonicalization -> InChIKey14 ->
compare against ranked candidate list -> MRR@25 (reciprocal rank of first
correct hit, 0 if absent from top 25). Pin RDKit 2026.03.3 to match the
competition's scoring environment. This module needs unit tests before it is
trusted for anything: every downstream decision depends on it being correct,
and a subtly wrong scorer invalidates all experiments run against it.

Never iterate against the real Kaggle `test.parquet` — schema/format
reference only.

## Baseline definition

Our baseline is **Channel 1 + Channel 2, simple rank fusion, no neural
model**, scored on the honest held-out split:

- Channel 1: exact/near-exact spectral match against the train side
  (spectral entropy similarity, which is a published method — Li et al.
  2021 — plus cosine as a cross-check)
- Channel 2: mass-shifted analog propagation
- Fusion: reciprocal-rank or weighted fusion, no learned reranker yet

Rationale: this is buildable quickly, exercises the full data ->
candidates -> ranked list -> score path, and produces a trustworthy floor
that every later addition must beat. A dummy/constant submission is not a
useful baseline; neither is anything measured on the public LB.

Report baseline as three numbers (Class-1-like / Class-2-like / Class-3-like
MRR@25) plus the aggregate, always. Aggregate alone is what let the rank-1
notebook not notice that 3 of its 4 channels contributed nothing.

## Pipeline

```
Query molecule (1-16 spectra — aggregate ALL of them, never score one in isolation)
   |
   |-- Neutral mass derivation (adduct arithmetic) + multi-spectrum peak fusion
   |
   |-- Channel 1: Exact library match (entropy + cosine sim vs train spectra)
   |-- Channel 2: Mass-shifted analog propagation (spectral sim x fingerprint Tanimoto^p)
   |-- Channel 3: Formula/mass-constrained candidate retrieval (COCONUT + train structures)
   |-- Channel 4: Spectrum -> fingerprint model (scored as linear dot product vs candidate FPs)
   |-- Channel 5: In-silico fragmentation plausibility (MetFrag-lite) — filter/prior
   |
   |-- Per-candidate feature assembly (all channel scores, mass deltas, rank-agreement features)
   |-- Reranker: rank fusion first, learned GBM once features are stable
   |-- Dedup by inchikey14 -> Top-25 SMILES
```

### Deliberate differences from the rank-1 notebook

1. **De novo generation stays in the plan.** That notebook is retrieval-only
   and therefore structurally incapable of Class 3. Their own estimate puts
   Class 3 at ~30–40% of test molecules. If that is even roughly right, a
   retrieval-only pipeline has a hard ceiling, and de novo is where our edge
   is — not in out-engineering their retrieval.
2. **Candidate pool default = COCONUT + train structures.** ChEBI/LipidMaps
   behind an experiment flag, enabled only if it beats baseline on *our*
   split. They shipped it disabled (`USE_BIO_DB=False`, commented "prevent
   decoy dilution") — we should reach our own conclusion rather than inherit
   one tuned against leaked data.
3. **Stagewise attribution is mandatory.** Add one channel, measure the
   delta, keep or drop. No multi-channel jumps.
4. **Simpler fingerprint model first.** Start with an MLP/CNN over binned
   spectra before a 6-layer transformer. On-domain (NP) training data is
   thin; justify the transformer by beating the simpler model on held-out
   MRR, don't assume it.

### Hyperparameter starting points (from their code — starting points, NOT confirmed optima)

Their sweeps were run against the leaked visible test set, so treat these as
reasonable priors to re-tune against our own held-out split, not as settled:

- Precursor mass window: ±8.5 ppm (they report ±10 -> 0.521, ±8.5 -> 0.524
  Class-2 MRR), with a ±30 ppm fallback when the tight window returns nothing
- Analog search window: ±200 Da, keep top ~100 analogs
- Analog similarity power weighting: `sim^4` (they report p=1 -> 0.498,
  p=3 -> 0.521, p=4 -> 0.525 — real gain from power-weighting over linear,
  diminishing past 4)
- Spectral cleaning: 0.2% relative intensity floor, top-256 peaks, entropy
  weighting for low-entropy spectra
- Candidate pruning before expensive scoring: coarse-rank by
  `lib_sim*100 - mass_diff`, keep top ~80
- Fingerprint: Morgan r2 (4096) + Morgan r3 (4096) + RDKit FP (2048) +
  MACCS (167), bit-subset to the model's trained width
- Reranker: HistGradientBoosting, `max_depth=6, max_iter=500, lr=0.03,
  min_samples_leaf=80, l2=1.0`; bag over seeds and 2 class-prior weightings

## Build order

1. **Project scaffolding + data pipeline + scorer.** `pyproject.toml`/deps,
   parquet loaders, matchms-based cleaning, `inchikey14` NP-weighted split
   with class proxies, MRR@25 scorer + unit tests. Gate: scorer passes tests.
2. **Channel 1** (exact library search). Gate: first honest non-zero
   class-conditional MRR@25.
3. **Channel 3** (candidate pool + fingerprint infrastructure) — built
   before Channel 2/4 because both depend on it.
4. **Channel 2** (analog propagation). Expected largest single lift for
   Class-2-like.
5. **Rank fusion -> record BASELINE.**
6. **Channel 4** (spectrum->fingerprint model, simple architecture first).
7. **Channel 5** (fragmentation plausibility).
8. **Learned GBM reranker** replacing rank fusion.
9. **De novo generation** for the Class-3 tail (fingerprint->SMILES decoder,
   formula-constrained). Our differentiator.
10. **Iterate**: DreaMS backbone, ensembling, class-conditional weighting,
    uncertainty-based channel gating.

## Compute plan (Azure 4x T4)

Split work by internet access, because the scored Kaggle notebook has none:

**On Azure (has internet, 4x T4 — ~2x the rank-1 notebook's 2x T4):**
- Download `train.parquet`/`test.parquet` once; do all EDA there
- Train FPNet / de novo models
- Build + fingerprint the candidate pool (COCONUT etc.)
- Run all held-out validation sweeps and hyperparameter tuning

**Kaggle submission notebook (no internet, 9h cap):**
- Inference only. Load pre-attached artifacts (model weights, candidate pool
  + precomputed fingerprints, RDKit wheel) as Kaggle Datasets/Models.
- Reference point for budget: their pipeline ran 400 molecules in ~53 min on
  2x T4 (~370s startup + ~6–7s/molecule). Hidden test is similar molecule
  count but ~2,500 spectra vs ~1,500 visible, so expect somewhat more —
  still comfortably inside 9h with room for added channels.

**Artifact flow**: heavy compute and 3GB raw data stay on Azure; only small
derived artifacts (checkpoints, fingerprint arrays, candidate pool, feature
matrices) get pulled down and attached to Kaggle. Do not re-download the raw
parquet repeatedly, and do not attempt model training inside the scored
notebook.

## Open decisions needing a call

- **Resolved**: packaging is `uv` + pip-installed RDKit wheel (2026.3.3,
  matches the competition's own scoring environment).
- **Resolved**: code is pushed to `origin/main` (session 2's implementation
  plus session 3's Azure/Kaggle tooling).
- Whether to pursue DreaMS pretrained embeddings — requires verifying the
  license permits bundling weights as a Kaggle artifact (flagged in
  `03-literature-review.md` §9, still unresolved)
- Whether/when to add COCONUT to the candidate pool. `build_pool.py` supports
  `--coconut`, but session 3's real-data pool build used train structures
  only (275,810). Session 2's synthetic-fixture tests exercised the COCONUT
  path; it just hasn't been given a real source file yet.
- Azure box `rogii-gpu` root disk is at 99% full (3GB free) as of session 3 —
  needs resizing or cleanup before more work happens there.
