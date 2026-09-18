# Codebase Map — Enveda CASMI 2026

Updated 2026-09-18 (session 3). Baseline pipeline, validation and FPNet are
implemented and tested, trained on the real dataset on Azure, and submitted
to Kaggle via a notebook. See `docs/07-codebase-guide.md` for the detailed
guide and `.kiro/memory/state-2026-09-18-session3.md` for the Azure/Kaggle
operational details.

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
├── kaggle/
│   ├── probe/               # Environment probe: confirmed no rdkit, 4 CPUs on Kaggle
│   └── submission/          # Submitted notebook + kernel-metadata.json
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

- **Competition data IS downloaded — on the Azure box, not locally.**
  `/mnt/casmi/data/raw/` on `azureuser@20.51.160.183` has the real
  `train.parquet` (2,539,608 rows) and `test.parquet` (400 molecules, 1,213
  spectra). Local `data/raw/` is still empty and gitignored; don't assume
  "no data downloaded" without checking the Azure box.
- **FPNet has been trained on the real data.** Checkpoint at
  `~/casmi_checkpoints/fpnet_final.pt` on the Azure box (persistent home dir,
  survives VM deallocation — do NOT rely on `/mnt/casmi/*`, which is
  ephemeral and wiped on deallocation). Val cosine similarity 0.71. It is
  trained but **not wired into `pipeline.py`** yet — see below.
- **A baseline submission has been made to Kaggle.** This IS a notebook-only
  competition — plain CSV upload via the API returns 400. Submission requires
  a committed notebook (see `kaggle/submission/`) plus
  `kaggle competitions submit -k <kernel> -f submission.csv -v <version> -m
  <msg>`. Running/completing a notebook does NOT auto-submit it. As of
  session 3's end, submissions were still PENDING — public score not yet
  known.
- **The visible `test.parquet` leaks** (it is a sample of train). Our own
  pipeline, run against the real data, independently reproduced this exactly:
  100% of 400 visible test molecules get a perfect (1.0) library-similarity
  match. Iterate against `scripts/build_split.py` output only; don't trust
  the public LB score.
- **Git**: everything through commit `9c39083` (session 2's implementation)
  plus session 3's Azure/Kaggle-tooling commits are pushed to `origin/main`.
- FPNet exists, is trained, but is **not yet wired into `pipeline.py`**. The
  model, checkpoint and `score_candidates()` are all ready; the pipeline
  still only runs Channels 1+2.
- De novo generation for class 3 is **not implemented** and is the intended
  differentiator versus the retrieval-only reference solution.
- **Azure box `rogii-gpu` root disk was at 99% full (3GB free)** as of
  session 3's end — needs fixing before more work happens there.
