# Project Context — Enveda CASMI 2026

## What this project is

Kaggle code competition: predict 2D chemical structures (SMILES, up to 25
ranked candidates per molecule) from LC-MS/MS spectra. Domain: natural-
product / metabolomics small-molecule structure elucidation. NOT the BioHub
cell-tracking project — this is a separate, unrelated repo/competition.
Full brief: `docs/00-brief.md`. Rules: `docs/01-competition-rules.md`.
Dataset: `docs/02-dataset.md`. Literature: `docs/03-literature-review.md`.
Method plan: `docs/04-method-landscape.md`.

## Repo

- Remote: `https://github.com/aagneye/predict-2D-chemical-structures.git`
  (origin, branch `main`)
- No commits pushed yet as of 2026-09-17 — repo was freshly `git init`'d
  this session. Nothing has been pushed to the remote.

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

## Status as of 2026-09-17 (session 1)

Completed: repo init + remote, folder structure, .gitignore, full literature
review, full dataset column/library breakdown, method landscape doc, this
steering doc, codebase-map.md. No code written yet — no data downloaded
locally yet either. See `.kiro/memory/state-2026-09-17.md` for the fuller
session log and next-session TODOs.
