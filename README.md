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
5. `docs/04-method-landscape.md` — phased architecture plan (the "why")
6. `docs/05-community-intel.md` — competitor intel, and **why the public
   leaderboard is not a usable target** (the visible test set leaks)
7. `docs/06-implementation-plan.md` — baseline definition, pipeline, validation
   design, Azure compute plan. **Start here for "what do I do next".**
8. `docs/07-codebase-guide.md` — what exists in `src/`, design rationale,
   end-to-end workflow

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

Python environment (`uv`, Python 3.12):

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"          # baseline pipeline + test suite
uv pip install -e ".[dev,train]"    # adds torch for FPNet training
```

RDKit is pinned to **2026.3.3** to match the competition's scoring environment,
since tautomer canonicalization output is version-sensitive. `torch` is an
optional `train` extra so the baseline installs without CUDA wheels.

Run the test suite (317 tests, synthetic fixtures — no data needed):

```bash
.venv/Scripts/python -m pytest tests/ -q
```

See `docs/07-codebase-guide.md` for the module map, design rationale, and the
end-to-end workflow.

## Submission format

CSV with header `molecule_id,smiles`, one row per test molecule, `smiles`
field = up to 25 SMILES joined by `;`, best guess first. See
`docs/00-brief.md` for full validity rules.

## License note

Competition data is **CC BY-NC 4.0** (non-commercial, attribution required).
This governs use of the provided train/test data specifically — keep this in
mind for any reuse of derived artifacts outside the competition itself.
