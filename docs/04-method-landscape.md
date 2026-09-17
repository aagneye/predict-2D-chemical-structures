# Method Landscape & Recommended Approach

Synthesizes `03-literature-review.md` into a concrete, phased plan mapped to
the competition's three hidden novelty classes and the MRR@25 metric.

## Design principle: cascade of strategies feeding one ranked-25 list

Since the class distribution is hidden per-molecule, don't try to classify-
then-dispatch. Instead, **run every applicable strategy for every molecule**,
collect a pooled candidate set of (SMILES, score, source-strategy), dedupe by
InChIKey14 (the actual matching key — dedupe early to avoid wasting slots on
the same skeleton twice), and produce one final ranking by a learned or
heuristic re-ranker. MRR@25 means: getting the right answer at rank 5 instead
of rank 1 only costs a little (0.2 vs 1.0) — prioritize **recall in the top
25** first, precision-at-1 second.

## Strategy A — Spectral library search (targets Class 1)

- Cosine / modified-cosine spectral similarity (matchms has this) against
  all of `train.parquet`, restricted to spectra with matching/compatible
  adduct + reasonable precursor-mass tolerance.
- Cheapest, most reliable signal when it fires. Should be the first
  candidate source computed for every molecule.
- Aggregate across a molecule's multiple test spectra (different CE/adduct)
  by taking best-per-candidate-structure score, not just first-spectrum
  score — the brief explicitly requires evidence fusion across spectra.

## Strategy B — Formula-first database retrieval (targets Class 2, backs up Class 1)

1. Predict/derive **molecular formula** per molecule:
   - Precursor m/z + adduct arithmetic gives candidate neutral masses
     directly (adduct mass tables are standard/well known) — a good, cheap
     first pass before reaching for a learned formula predictor.
   - MIST-CF or SIRIUS for cases where isotope-pattern/fragment info is
     needed to disambiguate formula candidates within mass-tolerance.
2. **Fingerprint prediction**: spectrum → molecular fingerprint (CSI:FingerID-
   style, or MIST transformer, or train our own on `train.parquet` — we have
   275k labeled structures, plenty to train a fingerprint predictor from
   scratch if pretrained checkpoints prove hard to attach/license).
3. **Retrieval**: search a merged **PubChem + COCONUT** candidate pool
   (filtered by predicted formula first, dramatically shrinking the search),
   ranked by fingerprint Tanimoto similarity to the predicted fingerprint.
4. This is the best-validated, most standard pipeline in the field (SIRIUS+
   CSI:FingerID lineage) — should be the backbone even for Class 1 as a
   sanity cross-check against Strategy A.

## Strategy C — De novo generation (targets Class 3, the frontier)

- Expectations should be modest per the literature: SOTA exact top-1 is
  ~28% (MIST+MolForge on MassSpecGym), DiffMS ~2.3%. Do not expect to "solve"
  Class 3 — aim for **some correct structures landing somewhere in top 25**,
  which is a much lower bar than top-1 exact match and is exactly what
  MRR@25 rewards partial credit for.
- Candidate architecture, in order of buildability given the 9h/no-internet
  Kaggle constraint:
  1. **Fingerprint → SMILES decoder** (MolForge-style transformer, or
     reproduce a simpler LSTM decoder à la MSNovelist) fed by our own
     Strategy-B fingerprint predictor. Reuses most of Strategy B's
     infrastructure — cheapest to add.
  2. Graph-based generation (DiffMS-style diffusion) — more implementation
     effort, marginal-but-real gains per the literature; consider only if
     time allows after A/B/C.1 are solid.
  3. Direct spectrum→SMILES end-to-end transformer (Spec2Mol-style) as a
     third independent candidate generator to diversify the pool (different
     failure modes than the fingerprint-mediated path can help top-25
     recall even if top-1 accuracy is similar or lower).
- Constrain generation with the predicted molecular formula (fixes atom
  composition) — consistently improves results across the literature and
  narrows an otherwise intractable search space.

## Strategy D — Spectrum embeddings as a shared backbone

- Consider using **DreaMS** (pretrained on 24M+ spectra, cross-instrument) as
  a frozen or fine-tuned encoder feeding *both* Strategy A's similarity
  search (embedding-space nearest neighbor instead of/alongside raw cosine
  similarity) and Strategy B/C's fingerprint predictors. This could give a
  meaningful edge over training a spectrum encoder from scratch on only
  our ~2.5M competition spectra, especially for out-of-domain (natural
  product) chemistry underrepresented in `enveda-180`.
- Must verify DreaMS weights can legally/practically be bundled as a Kaggle
  Dataset/Model attachment (no-internet constraint) — open item, see
  literature review §9.

## Re-ranking / final list assembly

- Once strategies A–C produce a pooled, deduped (by InChIKey14) candidate
  list with per-strategy scores, blend into a single ranking. Options,
  roughly in increasing sophistication:
  - Simple weighted rank fusion (e.g. Borda count or reciprocal-rank fusion
    across strategies) — good v1 baseline, no extra training needed.
  - Learned re-ranker (small gradient-boosted tree or logistic regression)
    using per-strategy scores + spectrum/precursor features as inputs,
    trained via cross-validation on held-out `train.parquet` structures
    (held out **by InChIKey14**, not by spectrum, to avoid leakage) — safer
    once we have a working pooled-candidate pipeline to generate features
    from.
- Precursor-mass/formula compatibility should hard-filter or heavily
  penalize any candidate whose exact mass doesn't match precursor_mz for the
  stated adduct within a tight ppm tolerance — a cheap, high-value filter
  applicable to every strategy's output.

## Validation strategy

- Build a local held-out split of `train.parquet` **by InChIKey14** (not by
  spectrum_id, and ideally not even by exact InChIKey to avoid diastereomer
  leakage) to estimate MRR@25 offline before touching the Kaggle test set.
- Use `enveda-np-examples` (the 250-compound, same-instrument, same-pipeline
  library) as a small, high-fidelity sanity-check set, given it's explicitly
  the closest available proxy for the real test distribution.
- Track class-conditional performance if possible by tagging local
  validation molecules as "found in a spectral library used for training
  (Class-1-like)" vs "not" vs "not in PubChem/COCONUT either (Class-3-like)"
  — lets us estimate per-class MRR contributions even though real test class
  labels are hidden.

## Compute / engineering constraints to keep in mind

- 9-hour inference cap for ~400 molecules / ~2,500 hidden test spectra is
  generous per molecule (~1.3 min/molecule average budget) but rules out
  brute-force combinatorial structure enumeration; keep any de novo search
  bounded (beam search with fixed width/depth, not exhaustive enumeration).
- No internet at submission time — every model checkpoint, database dump
  (PubChem/COCONUT subsets), and library file must be pre-fetched and
  attached as a Kaggle Dataset/Model well before the final submission.
- Train/experiment iteration should happen outside the submission notebook
  (local GPU or a separate Kaggle notebook with internet on) — the
  competition notebook is inference-only.

## Phased plan (suggested order of implementation)

1. **Data pipeline**: parquet loaders, spectral cleaning (matchms-based),
   train/val split by InChIKey14, EDA notebook.
2. **Strategy A** (library search) end-to-end, scored offline via local
   MRR@25 — cheapest path to a first real (non-dummy) submission.
3. **Strategy B** (formula + fingerprint + PubChem/COCONUT retrieval) —
   biggest expected marginal MRR gain, addresses Class 2 and backs up
   Class 1.
4. **Re-ranking / fusion** of A+B into one list — likely the first genuinely
   competitive submission.
5. **Strategy C** (de novo) added last, as a supplementary candidate source
   for whatever molecules A+B leave empty or low-confidence on.
6. Iterate: DreaMS backbone swap-in, learned re-ranker, ensembling multiple
   de novo generators, formula-prediction accuracy improvements.
