# Project Context — Enveda CASMI 2026

## What this project is

Kaggle code competition: predict 2D chemical structures (SMILES, up to 25
ranked candidates per molecule) from LC-MS/MS spectra. Domain: natural-
product / metabolomics small-molecule structure elucidation. NOT the BioHub
cell-tracking project — this is a separate, unrelated repo/competition.
Full brief: `docs/00-brief.md`. Rules: `docs/01-competition-rules.md`.
Dataset: `docs/02-dataset.md`. Literature: `docs/03-literature-review.md`.
Method plan: `docs/04-method-landscape.md`. Community/competitor intel
(Kaggle notebooks & discussion): `docs/05-community-intel.md`.
**Concrete build plan (baseline, pipeline, validation, compute):
`docs/06-implementation-plan.md` — start here for "what do I do next".**

## CRITICAL: the visible test set leaks

The visible `test.parquet` is a sample of `train.parquet` (organizer-
documented). Verified evidence: the rank-1 public notebook's own diagnostics
show `best_library_sim` == exactly 1.0 for 100% of 400 visible test
molecules, with its analog + neural channels contributing 0.0%. **The public
leaderboard is therefore not a usable optimization target** — it mostly
measures whether you implemented a library lookup. All iteration must happen
against our own held-out split (by `inchikey14`, NP-weighted, with
synthesized Class-1/2/3 proxies). See `06-implementation-plan.md`.

## Repo

- Remote: `https://github.com/aagneye/predict-2D-chemical-structures.git`
  (origin, branch `main`)
- Everything through session 2's implementation plus session 3's Azure/
  Kaggle-tooling commits **is pushed** to `origin/main`.

## Key facts to hold in working memory

- **Metric: MRR@25** — reward for the *first* correct guess in a ranked
  list of ≤25 SMILES per molecule. Getting many plausible candidates into
  the top 25 matters more than nailing rank 1 exactly.
- **Matching**: RDKit tautomer canonicalization → InChIKey14 (2D skeleton
  only). Stereochemistry and tautomer form are NOT scored — don't spend
  modeling effort on stereo.
- **3 hidden novelty classes** per test molecule: (1) in public spectral
  libraries, (2) known structure (PubChem/COCONUT) but no public spectra,
  (3) fully novel, not in any database. Distribution is secret — must handle
  all 3 simultaneously per molecule, not classify-then-dispatch.
- **No internet at submission time.** Any pretrained weights / DB dumps
  must be pre-attached as Kaggle Dataset/Model artifacts.
- **9-hour run-time cap** (CPU or GPU) for the scored notebook.
- Predictions are **per molecule_id**, aggregating evidence across that
  molecule's 1–16 (median 3) spectra at different adducts/collision
  energies — never score against a single spectrum in isolation.
- Data license: **CC BY-NC 4.0** — non-commercial, attribution required.
- `test.parquet` visible today is a stand-in sample from train, NOT the
  real hidden test — don't overfit EDA conclusions to its exact contents,
  treat it as schema/format reference only.

## Recommended architecture (see docs/04 for full detail)

Cascade, not classifier-dispatch: (A) spectral library search, (B) formula-
first fingerprint-based database retrieval (PubChem+COCONUT), (C) de novo
generation (fingerprint→SMILES decoder, formula-constrained) as a lower-
expectation supplementary source, then (D) fuse/re-rank all candidate pools
into one deduped (by InChIKey14) top-25 list per molecule.

## Status as of 2026-09-18 (session 2)

**Code exists and is tested.** 317 tests pass, ruff clean. Implemented:
validation split (leak-proof, NP-weighted, synthetic novelty classes), MRR@25
scorer, Channel 1 (library search), Channel 2 (mass-shifted analog
propagation), weighted fusion, baseline pipeline with leakage diagnostics,
FPNet spectrum→fingerprint transformer + training loop, and four CLI scripts.
See `docs/07-codebase-guide.md`.

Environment: `.venv` via `uv`, Python 3.12, RDKit pinned 2026.3.3. Run tests
with `.venv/Scripts/python -m pytest tests/ -q`.

**Still no competition data downloaded** — `data/raw/` is empty; everything is
verified against synthetic fixtures only.


## Status as of 2026-09-18 (session 3): Azure training + Kaggle submission

Real competition data downloaded to the Azure GPU box (`rogii-gpu`, 4x T4,
`azureuser@20.51.160.183`) — NOT locally; local `data/raw/` is still empty.
FPNet trained on the real 2.5M-row train set (30k steps, val cosine
similarity 0.71), checkpoint saved persistently at
`~/casmi_checkpoints/fpnet_final.pt` on that box. Candidate pool (275,810
structures) built from real train data.

Ran the Channels-1+2 baseline against the real `test.parquet` and
independently reproduced the documented leak: 100% of 400 visible test
molecules get a perfect library match. Packaged an RDKit wheel + the pool +
our source as a Kaggle Dataset, submitted a notebook, and pushed it to the
leaderboard via `kaggle competitions submit -k ... -v ...`. **Public score
was still PENDING as of session end** — check
`.kiro/memory/state-2026-09-18-session3.md` or the Kaggle site directly for
the resolved score before reporting it to anyone.

**Azure box root disk is at 99% full** — fix before doing more work there.

Not yet done: FPNet is trained but not wired into `pipeline.py`; fragmentation
channel, learned GBM reranker, de novo generation all still pending. See
`docs/06-implementation-plan.md` steps 6-10 and session 3's memory file for
the full next-session TODO list.

Not yet done: FPNet wired into `pipeline.py`, fragmentation channel, learned
GBM reranker, de novo generation (the intended differentiator). See
`docs/06-implementation-plan.md` steps 6-10.

Earlier: session 1 (2026-09-17) produced repo init, docs, literature review,
dataset breakdown. See `.kiro/memory/state-*.md` for per-session logs.

## Status as of 2026-09-19 (session 4): 4-channel pipeline + GBM reranker

Confirmed the Kaggle public score from session 3: **0.152** (Channels 1+2
only, weighted fusion). Target/reference: a community notebook (rank 1, gold
zone), public score **0.339**, using 4 channels (library, analog, MetFrag-lite
fragmentation, an FPNet-style transformer) fused by a bagged
`HistGradientBoostingClassifier` reranker.

**Implemented and pushed to `origin/main` this session** (386 tests pass,
ruff clean):
- Channel 5, MetFrag-lite in-silico fragmentation (`src/casmi/channels/
  fragmentation.py`) — 1-2 bond-break fragment enumeration, peak-explain-ratio
  scoring.
- Channel 4 wired into `pipeline.py` — FPNet (already trained, session 3)
  was never called by the pipeline; now `predict_molecule`/`run_pipeline`
  accept optional `fpnet_model`/`fpnet_config`/`fpnet_device` params.
- A learned GBM reranker (`src/casmi/channels/ranker.py`) replacing weighted
  fusion as an option — bagged `HistGradientBoostingClassifier` ensemble over
  `RankerConfig.class1_priors x seeds`, matching the reference's own bagging
  design.
- Scripts to build the reranker's training data from the held-out split
  (`scripts/build_rank_train.py`) and fit it (`scripts/train_ranker.py`);
  `scripts/run_baseline.py` extended with `--fragmentation`/
  `--fpnet-checkpoint`/`--ranker` flags to measure honest per-channel deltas.
- **A real bug found and fixed**: `DEFAULT_WEIGHTS` in `fusion.py` never
  included `fragmentation_score`/`fpnet_score`, so those channels changed
  diagnostics but never the actual ranking under weighted fusion (features
  absent from the weights dict are implicitly zero-weighted). Fixed in
  commit `7dd3a79` with 3 regression tests.
- Kaggle notebook and dataset-build script rewritten for the 4-channel +
  reranker pipeline, with clean fallback to weighted fusion / no-FPNet if
  either artifact is missing.

**Honest MRR@25 on the real held-out split (2000 molecules), so far**:
library+analog baseline 0.337; +fragmentation 0.351; +fpnet 0.467 (FPNet is
the single biggest lever found so far, especially for class 2). The full
4-channel + GBM reranker number — the one to compare against the reference's
0.339 — was **still running on Azure as of session end**. See
`.kiro/memory/state-2026-09-19.md` for the exact stage reached, the ordered
next-steps list (rebuild Kaggle dataset, push notebook, submit, verify), and
Azure operational details.

**Not yet done**: de novo generation for class 3 (still the intended
differentiator vs. the retrieval-only reference solution — class-3 MRR
remains ~0.01-0.02, near the floor, across every stage tried so far).
