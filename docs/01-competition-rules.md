# Competition Rules & Constraints

## Code requirements (hard constraints on the final submission)

Submission is through a **Kaggle Notebook**, committed, with "Submit" button
enabled only when:

- CPU notebook run-time ≤ 9 hours, OR GPU notebook run-time ≤ 9 hours
- **Internet access disabled** during the scored run
- Freely & publicly available external data is allowed, **including
  pretrained model weights** — but they must be attached as a Kaggle
  Dataset/Model to the notebook beforehand, not downloaded live
- Output file must be exactly named `submission.csv`

Practical implications:
- Any pretrained model we want to use (DreaMS, MIST, SIRIUS/CSI:FingerID
  fingerprint predictors, RDKit, COCONUT/PubChem structure dumps) must be
  **pre-downloaded and attached as a Kaggle Dataset** before the run, since
  the scored notebook can't hit the internet.
- 9-hour budget covers ~1,500 test spectra / ~400 molecules — generous per
  molecule, but rules out expensive per-molecule combinatorial search unless
  it's tightly bounded (e.g. bounded-depth fragment-tree enumeration, not
  brute-force graph enumeration over all isomers).
- We should design the training/experimentation pipeline locally (or on our
  own compute) and only port the **inference-only** path into the submitted
  notebook.

## Timeline detail

- Start: 2026-09-14
- Entry deadline: 2026-12-07 23:59 UTC (must accept rules to compete)
- Team merger deadline: 2026-12-07 23:59 UTC
- Final submission deadline: 2026-12-14 23:59 UTC

Organizers reserve the right to change the timeline.

## Data & license

- Files: `train.parquet` (~2.5M spectra / ~275k unique structures, 18 cols),
  `test.parquet` (~1,500 spectra / ~400 molecules, 12 cols — no structure
  label), `sample_submission.csv`.
- Total size 3.04 GB.
- **License: CC BY-NC 4.0** (Attribution-NonCommercial). This constrains what
  we can do with the data/derived artifacts outside the competition —
  non-commercial use only, attribution required. Flag before any reuse
  beyond the competition itself.
- The public `test.parquet` visible now is a **stand-in** sampled from
  train — it will be swapped for the real hidden test set on rerun, of
  approximately the same size. **Do not overfit hyperparameters to the
  visible test.parquet's exact contents** — treat it only as a schema/EDA
  reference, not a leaderboard-representative sample.

## External data policy

"Freely & publicly available external data is allowed, including pre-trained
models." This is generous — we can bring in:
- PubChem, COCONUT (natural products DB, CC0 licensed) for retrieval
  candidates for Class 2.
- Pretrained spectral embedding models: DreaMS (from GeMS pretraining
  corpus), MIST / MIST-CF checkpoints, CSI:FingerID / SIRIUS.
- Other public MS/MS libraries not already in train.parquet, e.g. via
  FragHub, GNPS, MassIVE (raw unannotated spectra — good for further
  self-supervised pretraining if we have compute for it, though DreaMS
  already published such a model).

All of the above must be "freely and publicly available" — no paywalled or
personal-license-only data/tools. If in doubt about a specific dataset's
license before using it, check before integrating.

## Team rules

- Team merger deadline same as entry deadline (2026-12-07). After that,
  team composition is locked.

## Prizes

Total $50,000 across top 5 (`$16k/$12k/$9k/$7k/$6k`). Standard Kaggle
prize competition — check the full rules PDF on Kaggle for eligibility
(this brief doesn't capture eligibility/geographic restrictions, which
should be checked directly on the competition Rules tab before assuming
prize eligibility).
