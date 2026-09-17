# Enveda CASMI 2026 — Predict 2D Chemical Structures from Mass Spectra

Kaggle code competition: predict up to 25 ranked candidate 2D chemical
structures (SMILES) for each test molecule from its LC-MS/MS spectra.

- Competition: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra
- This repo: https://github.com/aagneye/predict-2D-chemical-structures
- Prize pool: $50,000 · Entry/merger deadline: 2026-12-07 · Final submission: 2026-12-14

## Start here

1. `docs/00-brief.md` — what the competition is, task summary, timeline, scoring
2. `docs/01-competition-rules.md` — code-competition constraints (9h runtime,
   no internet at submission, external-data policy), license, prizes
3. `docs/02-dataset.md` — full column/library breakdown of train/test data,
   novelty classes, curation recipes
4. `docs/03-literature-review.md` — CASMI history + survey of MS/MS→structure
   ML methods (SIRIUS, MIST/MIST-CF, MSNovelist, DiffMS, DreaMS, etc.)
5. `docs/04-method-landscape.md` — our concrete phased implementation plan

For AI-assisted sessions: `.kiro/steering/project-context.md` and
`.kiro/steering/codebase-map.md` hold condensed orientation; `.kiro/memory/`
holds dated session logs.

## The task in brief

Given 1–16 (median 3) MS/MS spectra for a `molecule_id` (all same molecule,
different adducts/collision energies), output up to 25 ranked SMILES
candidates. Scored by **MRR@25** — mean reciprocal rank of the first correct
guess, where "correct" means the same RDKit-tautomer-canonicalized
**InChIKey14** (2D skeleton, stereochemistry-agnostic) as the true structure.

Three hidden novelty classes make this hard: (1) structure has public
reference spectra — findable by library search; (2) structure is known
(PubChem/COCONUT) but has no public spectra — needs database retrieval;
(3) structure isn't in any public database — needs de novo generation. The
class mix per molecule is secret, so a robust solution must combine all
three strategies.

## Repo structure

```
docs/           Competition brief, rules, dataset reference, literature review, method plan
src/casmi/      Main Python package (data, formula, fingerprint, retrieval, denovo, rerank, eval)
notebooks/      EDA and experiment notebooks
scripts/        CLI entry points for pipeline stages
tests/          pytest suite
data/           Gitignored. raw/ (downloaded parquet/csv) and processed/ (derived artifacts)
```

## Setup

Data files are **not committed** (see `.gitignore`, CC BY-NC 4.0 license,
3.04 GB total). Download from the Kaggle competition's Data tab into
`data/raw/`:

```
data/raw/train.parquet
data/raw/test.parquet
data/raw/sample_submission.csv
```

Python environment: not yet finalized (no `pyproject.toml`/`requirements.txt`
committed as of this writing — see `.kiro/steering/codebase-map.md` for
current status). Anticipated core dependencies: `polars` or `pandas` +
`pyarrow` (parquet), `rdkit` (canonicalization/InChIKey/fingerprints),
`matchms` (spectral cleaning/similarity), plus whichever ML framework the
chosen models need (PyTorch, most likely, given the literature).

## Submission format

CSV with header `molecule_id,smiles`, one row per test molecule, `smiles`
field = up to 25 SMILES joined by `;`, best guess first. See
`docs/00-brief.md` for full validity rules.

## License note

Competition data is **CC BY-NC 4.0** (non-commercial, attribution required).
This governs use of the provided train/test data specifically — keep this in
mind for any reuse of derived artifacts outside the competition itself.
