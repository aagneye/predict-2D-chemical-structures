# Community Intelligence — Kaggle Notebooks & Discussion

Tracks findings from other competitors' public notebooks and discussion
posts on the competition's Kaggle page, as opposed to `03-literature-review.md`
(academic papers) or `04-method-landscape.md` (our own plan). Update this
file as new public notebooks/discussions surface — it's a living log, not
a snapshot.

**Caveat on public leaderboard scores below**: The competition uses the
usual Kaggle split — the *public* leaderboard score is computed against a
subset of the test set only, and Kaggle code competitions re-run submitted
notebooks against a hidden test set for final scoring. A public LB score of
0.339 does not guarantee that score on the private/final leaderboard, and
overfitting to public-LB-visible patterns is a known risk in code
competitions generally. Read the numbers below as informative, not as
ground truth about final standings.

> **UPDATE (2026-09-18, after reading the notebook source): the caveat above
> turned out to be far more serious than "standard Kaggle variance." The
> rank-1 public score is driven almost entirely by test-set leakage in the
> visible `test.parquet`, not by the pipeline's modelling. See
> "§ VERIFIED FINDING: the score is leakage-driven" below — read that section
> before drawing any conclusion from this document. The architectural ideas
> here are still worth adopting; the *scores* are not evidence that they
> work.**

## Source 1 — "CASMI 2026: A Complete Statistical Tour" (EDA notebook)

Author: josefreitasalvesneto. URL:
https://www.kaggle.com/code/josefreitasalvesneto/casmi-2026-a-complete-statistical-tour

Not yet independently reviewed in detail — Kaggle notebook pages are
client-side rendered and not fetchable by our current tooling (both
`web_fetch` attempts against the URL returned empty content). Flagged as a
**next-session TODO**: open manually in a browser and pull out anything
that adds to or corrects `docs/02-dataset.md`, e.g. concrete distributions
for `precursor_error_ppm`, per-adduct peak-count stats, formula-space
overlap between `enveda-180` and the test set, or a first real look at
class-distribution proxies. This doc's dataset write-up already matches the
Kaggle Data-tab description closely (verified directly against the pasted
dataset description in session), so the statistical-tour notebook is likely
additive detail/visualization rather than a correction.

## Source 2 — Discussion post: "[0.339 Top 1 Solution] 4-Channel Mass-Shifted Analog Propagation & Neural Bayes Reranking"

Public-LB rank 1 at time of posting. Author's own framing: a post-mortem /
open experiment log, not a final answer — explicitly inviting critique and
flags several things as "estimates, not ground truth." Two linked notebooks
referenced by the author but not yet independently fetched/reviewed:
"0.339 Top 1: 4-Channel Transformer Analog Ensemble" and "SOTA 0.339 Review
& Methodology" — **next-session TODO** to open manually and review.

Score progression reported: 0.149 → 0.193 → 0.205 → 0.239 → 0.339.

### Their working hypothesis on class distribution (their estimate, not organizer-confirmed)

| Tier (their terms) | Maps to our Class | Rough share (their estimate) | Strategy that works |
|---|---|---|---|
| 1 — Library Match | Class 1 | ~10–15% | Exact spectral matching |
| 2 — Database Known | Class 2 | ~45–55% | Analog propagation + neural reranking |
| 3 — Novel / De Novo | Class 3 | ~30–40% | Substructure reasoning + in-silico fragmentation |

This implies **Class 2 (known structure, database retrievable) is the
single largest bucket**, bigger than either Class 1 or Class 3 alone — if
accurate, this reinforces `04-method-landscape.md`'s existing call that
Strategy B (formula-first fingerprint/database retrieval) should be the
backbone with the biggest expected marginal MRR gain, not a secondary
check behind Strategy A. Worth explicitly noting: their tiers roughly
mirror our existing Class 1/2/3 framing already documented in
`02-dataset.md` and `04-method-landscape.md` — this is convergent
validation of the novelty-class framing from an independent competitor,
not a new idea to adopt.

### Their pipeline: 4 evidence channels + learned reranking

```
Query Spectrum
  -> Precursor Mass / Adduct Filtering -> Candidate Pool
  -> Channel 1: Exact Spectral Matching
  -> Channel 2: Mass-Shifted Analog Propagation
  -> Channel 3: Chemical / Substructure Heuristics
  -> Channel 4: Spectrum -> Fingerprint Transformer -> Neural Reranking
  -> Multi-feature Ranking (31 features)
  -> HistGBM Ensemble (4 seeds x 2 calibrated priors)
  -> Dynamic Confidence Gating
  -> Top-25 Candidates
```

**Channel 1 — Exact library matching.** Conventional: precursor mass
filter, spectral entropy similarity, cosine similarity, reference peak
matching. This is our existing Strategy A, same idea.

**Channel 2 — Mass-shifted analog propagation.** *This is the most useful
new idea here relative to our current plan.* Rather than only asking "is
this exact spectrum in my library," they also search for **structurally
related reference spectra within a mass-shift window** — differing by
interpretable transformations like ±CH₂, ±OH, ±hexose — and propagate
structural evidence through molecular similarity even when no analog's
*exact* structure is known. Their scoring intuition combines spectral
similarity between query and analog with **fingerprint Tanimoto similarity
between the candidate structure and the analog's structure** — i.e., "if a
close spectral analog exists, structures similar to that analog's
structure get a boosted prior." This is conceptually a generalization of
Strategy A that doesn't require an exact spectral match, and sits between
our Strategy A (exact library search) and Strategy B (formula/fingerprint
retrieval) — **not currently an explicit strategy in our `04-method-
landscape.md`, worth adding** (see recommendation below).

**Channel 3 — Chemical heuristics ("MetFrag-lite").** Explainable filters,
not a primary scorer: approximate bond-dissociation reasoning, neutral-loss
matching, substructure consistency, simple in-silico fragmentation rules.
Framed explicitly as a candidate *filter/prior* ahead of final ranking, not
a structure-elucidation method on its own. Conceptually close to our
existing precursor-mass/formula hard-filter idea in `04-method-
landscape.md`'s re-ranking section, generalized to substructure-level
consistency checks.

**Channel 4 — Spectrum-to-fingerprint transformer ("FPNet").** A 6-layer
transformer predicting logits over 6,930 fingerprint bits directly from the
spectrum. This is functionally identical to the "spectrum → fingerprint"
step already planned in our Strategy B (CSI:FingerID-style). Their
contribution is architectural (small transformer, not a GNN/deeper model)
and in how the output is used for scoring (see below) — not a new idea vs.
our existing plan, more a validated existence proof that a compact
transformer works for this task at competitive scores.

**Why the neural fingerprint score is cheap.** Under an independent-
Bernoulli-bits assumption, candidate log-likelihood reduces to a term that
is constant across candidates (for a fixed query spectrum) plus a term
linear in the predicted fingerprint logits and the candidate's fingerprint
vector — meaning **the entire candidate pool can be scored via one
vectorized matrix multiply** against predicted logits. This is a genuinely
useful, concrete implementation detail: it means fingerprint-similarity
reranking against tens of thousands of candidates is cheap enough to not
need approximate/pruned search, which matters for our 9h runtime budget
once we scale up the PubChem+COCONUT candidate pool size in Strategy B.

### Final ranking / ensembling detail

31-feature ranking matrix combining exact spectral similarity, analog
relationships, precursor mass differences, molecular similarity, chemical
heuristics, transformer fingerprint scores, and other retrieval stats, fed
into a **HistGradientBoosting ensemble** (4 seeds × 2 calibrated prior
settings) plus a confidence-gating stage. This matches our own `04-method-
landscape.md` re-ranking section's "learned re-ranker" option — validates
that a GBT reranker over multi-strategy features is a reasonable choice,
not just the "simple weighted rank fusion" fallback we listed as v1.

### What did NOT work for them (worth avoiding ourselves)

- **Blind PubChem isomer expansion.** Aggressively expanding candidates
  without a strong prior introduced too many synthetic/irrelevant decoys
  and *hurt* ranking on harder classes. Curated natural-product-specific
  DBs (COCONUT 2.0, ChEBI, LipidMaps) worked better for them than raw
  PubChem breadth. **Actionable for us**: when we build Strategy B's
  PubChem+COCONUT retrieval, don't just widen the candidate pool for its
  own sake — prioritize NP-relevant curated subsets (COCONUT, ChEBI,
  LipidMaps) over blind full-PubChem isomer enumeration, and treat pool
  size as a tunable hyperparameter to ablate, not "more is better."
- **Overly wide precursor mass windows.** Wider tolerance adds decoys
  faster than it adds true candidates. Their empirical sweet spot was
  **~±8.5 ppm** for their pipeline/data. **Actionable for us**: don't
  default to a generous mass tolerance "to be safe" in Strategy B's
  formula-filtering step — treat ppm tolerance as a knob to sweep
  empirically against our own local MRR@25 validation set, with ~8.5 ppm
  as a reasonable starting point rather than an arbitrary wider default.

### Open problem they (and we) still don't have a good answer for

Tier 3 / our Class 3 (fully novel structures, not in any database) remains
the weak point even for the public LB leader. Their stated open directions:
better in-silico fragmentation, substructure assembly, graph-based
candidate generation, stronger spectrum-to-structure models, contrastive
spectra/fingerprint learning, better uncertainty estimation for dynamic
channel weighting, and analog-specific retrieval training. This matches our
own `04-method-landscape.md` Strategy C framing (modest expectations, MRR@25
partial credit is the realistic goal, not top-1 exact match) — no need to
revise our Class-3 expectations downward or upward based on this, it's
consistent with the literature-review SOTA numbers we already have (MIST+
MolForge ~28% top-1 best-in-class).

## VERIFIED FINDING: the score is leakage-driven

Added 2026-09-18 after the user supplied the notebook's full source and
output (the notebook page itself is not machine-fetchable — Kaggle renders
client-side; `web_fetch` returned empty on every attempt, including the
`/notebook` suffix and the `kernels/scriptcontent` download endpoint, which
404s).

The notebook's own printed diagnostic summary over all 400 visible test
molecules:

```
       n_candidates  best_library_sim  best_analog_sim  top_prob
mean         63.570               1.0            0.930     0.752
std          23.708               0.0            0.130     0.103
min           1.000               1.0            0.444     0.256
max          80.000               1.0            1.000     0.964

Molecules with confident library hit (>0.85): 100.0%
Molecules powered by Analog Propagation & Neural Model: 0.0%
```

`best_library_sim` is **exactly 1.0 with zero standard deviation for 100% of
400 molecules**. A genuine spectral-similarity search against held-out
molecules does not produce a perfect score for every single query. This is
direct evidence that the exact spectrum of every visible test molecule is
present in `train.parquet` — consistent with the organizers' own statement
(quoted in `02-dataset.md`) that the visible `test.parquet` "is comprised of
examples from the training dataset; it will be replaced by the hidden test
set during re-run."

Implications, in order of importance:

1. **The public leaderboard is not a usable optimization target for this
   competition.** Public scores right now largely measure "did you implement
   a library lookup," not "can you identify unseen molecules."
2. **Channels 2, 3 and 4 of this pipeline are untested by its own numbers.**
   The author's diagnostic explicitly reports 0.0% of molecules being carried
   by analog propagation or the neural model. Their entire reported score
   progression (0.149 -> 0.339) was measured under leakage, so the per-channel
   and per-hyperparameter gains they cite (e.g. `sim^4` power weighting,
   ±8.5 ppm window) are tuned against a contaminated signal. Treat those
   settings as reasonable starting points, not validated optima.
3. **On re-run against the real hidden test set, Channel 1 will not fire like
   this**, and the final score will depend entirely on the channels that are
   currently unexercised.
4. This is *not* an accusation of cheating — the leak is in the competition's
   own provided sample file, disclosed by the organizers, and the author's
   pipeline is legitimately built and well-engineered. The problem is purely
   one of *inference from the score*.

### Score discrepancy worth noting

The notebook's own UI reports **Public Score 0.297** (Best Score 0.297, V2),
while the discussion post title and notebook header both claim 0.339. Either
the 0.339 run is a version not published here, or the numbers drifted between
posts. The verified figure from the notebook itself is 0.297.

## Verified architecture details (from source code)

Confirmed by reading the actual implementation, superseding the inferences
made from the discussion post alone:

**Candidate pool — 711,705 / 712,199 structures** (log output says 712,199):
COCONUT 2.0 contributes 436,389 structures preloaded as `coco_fp.npy` /
`coco_mass.npy` / `coco_meta.pkl`; plus all 275,810 unique train structures,
deduped against COCONUT by `inchikey14`, with fingerprints computed at
runtime via a 4-worker multiprocessing pool (~253s). Pool is sorted by
neutral exact mass with `searchsorted` mass-window lookup.

**ChEBI/LipidMaps is attached but DISABLED**: `USE_BIO_DB = False`, commented
"disabled by default to prevent decoy dilution." Concrete confirmation of the
discussion post's "blind expansion hurt" claim — they built the integration,
tested it, and shipped it off.

**Fingerprint**: concatenation of Morgan radius-2 (4096 bits) + Morgan
radius-3 (4096 bits) + RDKit FP (2048 bits, maxPath=6) + MACCS keys (167),
then indexed down by a saved `fp_bits.npy` mask to the 6,930 bits FPNet was
trained against. Stored bit-packed (`np.packbits`), unpacked per query.

**Channel 1** uses spectral *entropy* similarity (Numba-JIT, `parallel=True`),
not plain cosine — an actual published method (entropy weighting per Li et
al. 2021: spectra with Shannon entropy < 3.0 get intensities raised to
power `0.25 + 0.25*S` and renormalized). Peak alignment tolerance 0.01 Da.

**Channel 2** implementation: takes one representative spectrum per unique
structure (the one with the most peaks), searches ±200 Da in neutral mass,
and evaluates `max(direct_similarity, mass_shifted_similarity)` where the
shift is applied to the reference spectrum's m/z values. Analog evidence then
enters the feature matrix as `max over analogs of (Tanimoto(cand, analog) *
sim(analog)^4)`, plus linear-weighted, best-Tanimoto, top-analog and
mean-weighted variants.

**Channel 3 (MetFrag-lite)** is a real simplified in-silico fragmenter:
enumerates connected components after breaking all 1-bond and all 2-bond
combinations (skipped if the molecule has >34 bonds — a scalability guard),
computes candidate fragment ion masses at hydrogen shifts of -2..+2 plus
proton/deprotonation, and scores the **sqrt-intensity-weighted fraction of
observed peaks explained** within 0.01 Da. Intensity-weighted, not
count-based.

**Channel 4 (FPNet)**: 6-layer transformer, d=512, 8 heads, GELU FFN at 4x
width, dropout 0.1. Each peak is embedded as sinusoidal(m/z) +
sinusoidal(neutral loss = precursor - m/z) + intensity — encoding neutral
loss alongside raw m/z is a meaningful choice, since neutral losses are the
chemically interpretable quantity. A global/CLS token carries sinusoidal
precursor mass, collision energy /100, ionization mode, log1p(precursor)/10,
plus learned adduct and instrument-family embeddings. Output head takes
`concat(CLS, masked-mean-of-peaks)` -> 2048 -> 6,930 logits. Input peaks
capped at 128 via a top-8-per-50-Da-window selection (preserving low-mass
fragments), intensities sqrt-scaled.

Two checkpoints are ensembled: `fp_single_s2.pt` (trained on individual
spectra) and `fp_merged_m1.pt` (trained on merged multi-spectrum peak lists),
logit-averaged — i.e. ensembling over *how a molecule's multiple spectra are
aggregated*, not just over seeds.

**The Bayes/dot-product trick, confirmed in code**: candidate scoring is
literally `cand_fp @ logits` — one matmul for the whole candidate set. Also
fed to the reranker in z-scored, rank-normalized, and
length-normalized (`/sqrt(bitcount)`) variants, plus "rank agreement"
features measuring whether the neural model and library/analog channels
agree on their top pick.

**Reranker**: 8 `HistGradientBoostingClassifier`s = 2 class priors
(W1 = 0.30, 0.60) x 4 seeds (0-3), each `max_depth=6, max_iter=500,
lr=0.03, min_samples_leaf=80, l2=1.0`, trained on a precomputed
`rank_train.npz` of 142,762 rows x 31 features, predictions averaged.
Sample weights encode the class prior. Candidate set is pruned to top-80 by
`lib_sim*100 - mass_diff` before the expensive per-candidate work.

**Engineering patterns worth copying**: heavy artifacts (model weights,
candidate pool + fingerprints, reranker training features, an offline RDKit
2026.3.3 wheel `pip install`ed from `/kaggle/input` at runtime) are all
pre-attached Kaggle Datasets, keeping the scored notebook inference-only.
Submission is strictly audited before writing — row count, `molecule_id`
alignment against `sample_submission.csv`, no duplicates, no nulls, and
exactly 25 semicolon-separated candidates per row (padded with `CCO` if
short). Runtime ~3,204s (53 min) for 400 molecules on 2x T4, of which ~370s
is startup.



Not yet applied — proposing here first since it changes the strategy list:

1. **Add "Strategy A2 — Mass-shifted analog propagation"** as a distinct
   step between Strategy A (exact library search) and Strategy B (formula/
   fingerprint database retrieval). Search training spectra within a mass-
   shift window (informed by common NP modifications: ±CH₂, ±OH, ±hexose,
   etc.), score by spectral-similarity-to-analog × fingerprint-Tanimoto-to-
   analog, and add resulting candidates to the pooled set. This is the
   single most concrete, previously-missing idea from this competitor's
   write-up and plausibly explains a meaningful chunk of their score jump
   (0.205 → 0.239 → 0.339 coincides with adding channels 2–4 per their own
   narrative, though they don't give a clean per-channel ablation).
2. When implementing Strategy B's candidate pool, **default to curated NP
   databases (COCONUT, ChEBI, LipidMaps) over raw full-PubChem isomer
   expansion**, and treat pool size / mass tolerance as hyperparameters to
   sweep against local MRR@25 rather than fixed generously-wide defaults.
   Starting point: ~±8.5 ppm precursor tolerance (empirical, not derived).
3. Confirm the "independent Bernoulli fingerprint bits -> linear scoring ->
   vectorized matmul" trick when we implement our own Strategy B fingerprint
   reranking step — it's a real efficiency win worth reproducing regardless
   of whose idea it originated from.

These are proposed additions, not yet written into `04-method-landscape.md`
— flag for user confirmation before merging into the canonical plan, since
it does add a new pipeline stage beyond what was scoped in session 1.

## Next-session TODOs from this session

- Manually review (browser, not automated fetch) the statistical-tour
  notebook and the two linked notebooks in the 0.339 discussion post —
  automated fetching returned empty content both times (Kaggle notebook/
  discussion pages are JS-rendered, not scrapable by our current tools).
- Decide whether to formally add "Strategy A2 — analog propagation" to
  `04-method-landscape.md` (see recommendations above) — pending user
  confirmation.
- Keep monitoring the discussion thread for follow-up comments/ablations
  the author promised to report back ("happy to try new ideas suggested by
  the community and report results").
