# CASMI 2026 — Project Brief

## What this is

**Enveda CASMI 2026 — Molecule ID From Mass Spectra** is a Kaggle competition
(hosted by Enveda Biosciences) reviving the academic **CASMI** (Critical
Assessment of Small Molecule Identification) contest series, founded in 2012
by Emma Schymanski and Steffen Neumann, run through 2022, now revived with
their permission and aimed at the broader ML community.

- Kaggle URL: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra
- Repo (this project): https://github.com/aagneye/predict-2D-chemical-structures
- Prize pool: $50,000 (1st $16k / 2nd $12k / 3rd $9k / 4th $7k / 5th $6k)
- Format: **Code competition** — submit via Kaggle Notebooks only
  - CPU or GPU notebook, ≤ 9h run-time
  - **No internet access** during submission run
  - Freely & publicly available external data/pretrained models ARE allowed
    (just must be attached to the notebook, not fetched live)
  - Output file must be named `submission.csv`

## Timeline

| Date | Event |
|---|---|
| 2026-09-14 | Competition start |
| 2026-12-07 | Entry deadline (must accept rules) |
| 2026-12-07 | Team merger deadline |
| 2026-12-14 | Final submission deadline |

All deadlines 11:59 PM UTC. ~3 months of runway from today (2026-09-17).

## The task, in one paragraph

Given one-or-more LC-MS/MS spectra (Bruker timsTOF) for a `molecule_id`,
output up to 25 candidate 2D structures (SMILES), ranked best guess first.
Predictions are made **per molecule**, not per spectrum — you must fuse
evidence across all spectra (different adducts / collision energies) for
that molecule into one ranked list. Roughly 400 test molecules, ~1,500
spectra total (median 3 spectra/molecule), masses 157–1,159 Da (median 348 Da).
Test spectra are all from Enveda's own unpublished acquisitions — real
natural products, hypothesized natural products, NP analogs, or biologically
plausible synthetics. Domain: **natural product / metabolomics small-molecule
structure elucidation**, not proteomics.

## Why it's hard: three novelty classes (hidden at test time)

1. **Class 1 — in public spectral libraries.** Reachable by spectral library
   search / similarity to training spectra. Easiest.
2. **Class 2 — known structure, no public spectra.** Structure exists in
   PubChem or COCONUT (natural products DB) but no reference spectrum exists.
   Requires **database retrieval** conditioned on a predicted
   fingerprint/formula, not just library lookup.
3. **Class 3 — truly novel structure.** Not in PubChem at all. Must be
   generated **de novo** — no database contains the answer. This is the
   frontier ML problem (see `03-literature-review.md`); current SOTA de novo
   top-1 exact-match accuracy on comparable benchmarks is only ~28%.

The mix of classes in the actual test set is secret, so a robust submission
needs a strategy that **works across all three regimes simultaneously**
(library search → database retrieval → de novo generation), typically as an
ensemble/fallback cascade contributing candidates into one ranked-25 list
per molecule.

## Evaluation

**MRR@25** (Mean Reciprocal Rank, mean over molecules, considering only the
first correct guess in the top 25). Rank 1 → 1.0, rank 2 → 0.5, ... rank 25 →
0.04, no correct guess in top 25 → 0.

**Matching rule:** both prediction and answer SMILES are run through RDKit
tautomer canonicalization (pinned RDKit 2026.03.3), reduced to **InChIKey14**
(first block of InChIKey = 2D skeleton, ignores stereochemistry), and
compared as strings. So: stereochemistry doesn't matter, tautomers don't
matter, only 2D atom connectivity (skeleton) matters. This significantly
simplifies the generation problem — no need to get wedge/hash bonds right.

## Submission format

CSV, header `molecule_id,smiles`. One row per test molecule\_id (every
molecule\_id exactly once). `smiles` field = up to 25 SMILES strings
separated by `;`, best guess first. Rejected if: missing either column,
empty, nulls, duplicate molecule_id, or >25 guesses for any molecule.
Fewer than 25 is fine — no penalty beyond forfeiting those ranks.

## See also

- `01-competition-rules.md` — code-competition constraints, external data
  rules, prizes detail
- `02-dataset.md` — full column/library breakdown of train/test parquet
- `03-literature-review.md` — CASMI history + prior/related ML methods
- `04-method-landscape.md` — concrete architecture options mapped to the
  three novelty classes, with a recommended phased approach
