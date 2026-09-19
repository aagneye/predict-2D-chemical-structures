# Codebase Map — Enveda CASMI 2026

Updated 2026-09-19 (session 4). Baseline pipeline, validation, FPNet,
fragmentation (Channel 5), and a learned GBM reranker are all implemented
and tested. Honest held-out validation for the 4-channel + reranker
pipeline was **in progress on Azure as of session 4's end** — see
`.kiro/memory/state-2026-09-19.md` for exact status, real MRR@25 numbers so
far, and what to check first next session.

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
│   ├── chem.py               # inchikey14() scoring key, fingerprints, tanimoto
│   ├── adducts.py             # mass table, neutral_mass_array() — upstream of all
│   ├── spectra.py             # Numba entropy similarity (+mass shift), cleaning
│   ├── pipeline.py            # Orchestration incl. Channels 4/5 + reranker, diagnostics
│   ├── eval/metrics.py        # MRR@25, recall@25, per-novelty-class breakdown
│   ├── data/
│   │   ├── loaders.py         # Parquet -> flat CSR SpectralLibrary, QueryMolecule
│   │   └── split.py           # LEAK-PROOF SPLIT + verify_no_leakage() hard gate
│   ├── candidates/pool.py     # Mass-sorted pool, bit-packed fingerprints
│   ├── channels/
│   │   ├── library.py         # Channel 1: direct spectral match (class 1)
│   │   ├── analog.py          # Channel 2: mass-shifted propagation (class 2)
│   │   ├── fragmentation.py   # Channel 5: MetFrag-lite in-silico fragmentation
│   │   ├── ranker.py          # Learned GBM reranker (feature assembly, bagging)
│   │   └── fusion.py          # Weighted fusion (all 4/5 channels weighted), dedup, top-25
│   └── models/                # OPTIONAL (torch). Not imported unless FPNet is used.
│       ├── fpnet.py            # Channel 4: spectrum->fingerprint transformer, Bayes scoring
│       └── train.py            # Training loop, DataParallel for 4x T4
├── scripts/
│   ├── build_split.py        # RUN FIRST. Leakage gate + hold-out composition.
│   ├── build_pool.py         # COCONUT + train structures -> pool.npz
│   ├── run_baseline.py       # THE baseline/validation number(s), per novelty class.
│   │                         # Flags: --fragmentation, --fpnet-checkpoint, --ranker
│   ├── build_rank_train.py   # Generates rank_train.npz (candidate features + labels)
│   ├── train_ranker.py       # Fits the GBM reranker ensemble from rank_train.npz
│   ├── train_fpnet.py        # Azure box. --split is mandatory (anti-leak).
│   └── run_full_validation.sh # Detached 7-stage validation+reranker-training sequence
├── kaggle/
│   ├── probe/                # Environment probe: confirmed no rdkit, 4 CPUs on Kaggle
│   └── submission/           # 4-channel+reranker notebook + kernel-metadata.json
├── tests/                  # 386+ tests, all passing. Synthetic fixtures only.
│   ├── conftest.py          # Fixtures from real small molecules
│   ├── test_adducts.py / test_chem.py / test_spectra.py
│   ├── test_eval_metrics.py / test_data_loaders.py / test_data_split.py
│   ├── test_candidates.py / test_channels.py / test_pipeline.py
│   ├── test_fragmentation.py    # Channel 5 unit tests
│   ├── test_ranker.py           # Reranker unit tests
│   ├── test_pipeline_channels.py # Channel 4/5/reranker pipeline integration + regression
│   ├── test_models.py       # Skips cleanly without torch
│   └── test_scripts_cli.py  # Runs the real scripts as subprocesses
├── notebooks/            # Empty
├── data/                 # GITIGNORED, EMPTY. No data downloaded locally.
└── .kiro/
    ├── steering/{project-context,codebase-map}.md
    └── memory/state-*.md
```

## Orientation for a new session

1. **`.kiro/memory/state-2026-09-19.md` first** — has the exact Azure job
   status, real MRR@25 numbers so far, and the ordered next-steps list.
2. `docs/06-implementation-plan.md` — the overall plan (steps 6-8 are what
   session 4 implemented; step 9, de novo generation, is still untouched)
3. `docs/07-codebase-guide.md` — what exists, design rationale (written
   session 2/3; does not yet describe Channels 4/5/reranker wiring — the
   memory file and this map are the current source of truth for that)
4. `docs/05-community-intel.md` § "VERIFIED FINDING" — why the public
   leaderboard cannot be used as a target

## Facts that are easy to get wrong

- **Competition data IS downloaded — on the Azure box, not locally.**
  `/mnt/casmi/data/raw/` on `azureuser@20.51.160.183` has the real
  `train.parquet` (2,539,608 rows) and `test.parquet` (400 molecules, 1,213
  spectra). Local `data/raw/` is still empty and gitignored.
- **FPNet is trained AND wired into `pipeline.py`** (session 4). Checkpoint
  at `~/casmi_checkpoints/fpnet_final.pt` on the Azure box (persistent home
  dir). Val cosine similarity 0.71. `predict_molecule`/`run_pipeline` accept
  optional `fpnet_model`/`fpnet_config`/`fpnet_device` — omitting them
  reproduces the exact Channel-1+2-only baseline (torch stays unimported).
- **Channel 5 (fragmentation) and the GBM reranker both exist and are wired**
  (session 4) — see `src/casmi/channels/fragmentation.py` and
  `src/casmi/channels/ranker.py`. Both are optional/additive params on
  `predict_molecule`/`run_pipeline`.
- **A real bug was found and fixed this session**: `DEFAULT_WEIGHTS` in
  `fusion.py` originally omitted `fragmentation_score`/`fpnet_score`, so
  enabling those channels changed diagnostics but not the actual ranking
  under weighted fusion (any feature absent from the weights dict is
  implicitly multiplied by zero). Fixed in commit `7dd3a79`. If you see a
  channel's diagnostics change but MRR staying flat, check this exact class
  of bug first.
- **A validation run against the real held-out split is what matters**, not
  the public LB. As of session 4's end this was in progress on Azure —
  check `.kiro/memory/state-2026-09-19.md` for the exact stage reached and
  real numbers obtained so far (library+analog baseline: MRR@25=0.337;
  +fragmentation: 0.351; +fpnet: 0.467 — FPNet is the single biggest lever
  found so far). The final number (4-channel + reranker) was still pending.
- **A baseline submission has been made to Kaggle**, public score
  **0.152 CONFIRMED COMPLETE** (not pending — resolved during session 4).
  Notebook-only competition; plain CSV upload returns 400. The updated
  4-channel+reranker notebook was rebuilt (session 4) but **not yet
  resubmitted** — waiting on the Azure run to produce `ranker.pkl` first.
- **The visible `test.parquet` leaks** (it is a sample of train). Don't trust
  the public LB score; iterate against `scripts/build_split.py` output only.
- **Git**: everything through this session's commits (ending at `7dd3a79`)
  is pushed to `origin/main`.
- De novo generation for class 3 is **still not implemented** — the intended
  differentiator versus the reference solution. Class-3 MRR remains
  ~0.01-0.02 (near the floor) across every stage tried so far, exactly as
  expected since nothing implemented yet can reach it.
- **Azure box `rogii-gpu` root disk**: was 99% full (3GB free) at session 3's
  end; cleared uv/pip caches at session 4's start, now ~13GB free. Check
  `df -h /` before generating `rank_train.npz`/`ranker.pkl` if picking this
  up much later, since those are the largest new artifacts.
