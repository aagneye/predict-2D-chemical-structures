# Codebase Map — Enveda CASMI 2026

Updated 2026-09-18 (session 2). Baseline pipeline, validation and FPNet are
implemented and tested; see `docs/07-codebase-guide.md` for the detailed guide.

```
Enveda CASMI/
├── pyproject.toml          # uv/hatchling. RDKit pinned to 2026.3.3 (scoring env).
│                           # torch is an optional 'train' extra.
├── README.md
├── .gitignore              # data/, checkpoints/, submissions/, runs/ excluded
├── docs/
│   ├── 00-brief.md              # Competition summary: task, timeline, scoring
│   ├── 01-competition-rules.md  # Constraints, external data policy, prizes
│   ├── 02-dataset.md            # Column reference, 11 libraries, novelty classes
│   ├── 03-literature-review.md  # CASMI history + MS/MS->structure ML survey
│   ├── 04-method-landscape.md   # Phased architecture plan (the "why")
│   ├── 05-community-intel.md    # Competitor intel + THE LEAKAGE FINDING
│   ├── 06-implementation-plan.md # Baseline/pipeline/validation/compute plan
│   └── 07-codebase-guide.md     # What exists in src/, design rationale, workflow
├── src/casmi/
│   ├── config.py            # Frozen dataclasses. '(ref)' values need re-sweeping.
│   ├── chem.py              # inchikey14() scoring key, fingerprints, tanimoto
│   ├── adducts.py           # mass table, neutral_mass_array() — upstream of all
│   ├── spectra.py           # Numba entropy similarity (+mass shift), cleaning
│   ├── pipeline.py          # Orchestration, diagnostics, submission writer
│   ├── eval/metrics.py      # MRR@25, recall@25, per-novelty-class breakdown
│   ├── data/
│   │   ├── loaders.py       # Parquet -> flat CSR SpectralLibrary, QueryMolecule
│   │   └── split.py         # LEAK-PROOF SPLIT + verify_no_leakage() hard gate
│   ├── candidates/pool.py   # Mass-sorted pool, bit-packed fingerprints
│   ├── channels/
│   │   ├── library.py       # Channel 1: direct spectral match (class 1)
│   │   ├── analog.py        # Channel 2: mass-shifted propagation (class 2)
│   │   └── fusion.py        # Weighted fusion, dedup, top-25
│   └── models/              # OPTIONAL (torch). Not imported by the baseline.
│       ├── fpnet.py         # Spectrum->fingerprint transformer, Bayes scoring
│       └── train.py         # Training loop, DataParallel for 4x T4
├── scripts/
│   ├── build_split.py       # RUN FIRST. Leakage gate + hold-out composition.
│   ├── build_pool.py        # COCONUT + train structures -> pool.npz
│   ├── run_baseline.py      # THE baseline number, per novelty class
│   └── train_fpnet.py       # Azure box. --split is mandatory (anti-leak).
├── tests/                   # 317 tests, all passing. Synthetic fixtures only.
│   ├── conftest.py          # Fixtures from real small molecules
│   ├── test_adducts.py / test_chem.py / test_spectra.py
│   ├── test_eval_metrics.py / test_data_loaders.py / test_data_split.py
│   ├── test_candidates.py / test_channels.py / test_pipeline.py
│   ├── test_models.py       # Skips cleanly without torch
│   └── test_scripts_cli.py  # Runs the real scripts as subprocesses
├── notebooks/            # Empty
├── data/                 # GITIGNORED, EMPTY. No data downloaded yet.
└── .kiro/
    ├── steering/{project-context,codebase-map}.md
    └── memory/state-*.md
```

## Orientation for a new session

1. `docs/06-implementation-plan.md` — what to do next and why
2. `docs/07-codebase-guide.md` — what exists, design rationale, how to run it
3. `docs/05-community-intel.md` § "VERIFIED FINDING" — why the public
   leaderboard cannot be used as a target

## Facts that are easy to get wrong

- **No competition data is downloaded.** `data/raw/` is empty. Every test uses
  synthetic fixtures. The scripts have been verified end-to-end against a
  synthetic `train.parquet`, not against the real 3 GB files.
- **The visible `test.parquet` leaks** (it is a sample of train). Iterate
  against `scripts/build_split.py` output only.
- **Git**: commit `9f19493` (docs/scaffolding from session 1) **is pushed** to
  `origin/main`. Session 2's nine implementation commits are **local only** —
  not pushed, pending user confirmation.
- FPNet exists but is **not yet wired into `pipeline.py`**. The model and
  `score_candidates()` are tested; the pipeline does not call them.
- De novo generation for class 3 is **not implemented** and is the intended
  differentiator versus the retrieval-only reference solution.
