# Codebase Map — Enveda CASMI 2026

Session-1 snapshot (2026-09-17). No source code exists yet — this documents
the intended structure so future sessions/teammates know where things go.

```
Enveda CASMI/
├── .gitignore              # Python/ML gitignore — data/, checkpoints/, submissions/ all excluded
├── README.md               # Entry point: competition summary, setup, repo map
├── docs/
│   ├── 00-brief.md              # Competition summary: task, timeline, scoring, submission format
│   ├── 01-competition-rules.md  # Code-competition constraints, external data policy, prizes
│   ├── 02-dataset.md            # Full column reference, library breakdown, novelty classes
│   ├── 03-literature-review.md  # CASMI history + MS/MS->structure ML methods survey
│   └── 04-method-landscape.md   # Concrete phased architecture plan mapped to novelty classes
├── src/
│   └── casmi/               # Main package (empty scaffold as of session 1)
│       # Intended submodules (not yet created):
│       #   data/         - parquet loaders, matchms-based cleaning, train/val split by InChIKey14
│       #   formula/      - molecular formula prediction (adduct arithmetic, MIST-CF/SIRIUS wrappers)
│       #   fingerprint/  - spectrum -> fingerprint models
│       #   retrieval/    - PubChem/COCONUT candidate search by fingerprint similarity
│       #   denovo/       - fingerprint -> SMILES decoders, graph/diffusion generators
│       #   library_search/ - spectral similarity search against train.parquet
│       #   rerank/       - candidate pool fusion / final top-25 ranking
│       #   eval/         - local MRR@25 scorer, InChIKey14 matching (RDKit tautomer canonicalization)
├── notebooks/            # EDA and experiment notebooks (empty as of session 1)
├── scripts/              # CLI entry points for pipeline stages (empty as of session 1)
├── tests/                # pytest suite (empty as of session 1)
├── data/                 # GITIGNORED. raw/ and processed/ subdirs created, empty.
│   ├── raw/                  # Downloaded train.parquet, test.parquet, sample_submission.csv go here
│   └── processed/             # Cleaned/derived artifacts
└── .kiro/
    ├── steering/
    │   ├── project-context.md   # High-level orientation, key facts, current status
    │   └── codebase-map.md      # This file
    └── memory/
        └── state-2026-09-17.md  # Session log
```

## Notes for next session

- No Python project files yet (`pyproject.toml`/`requirements.txt`/`setup.py`
  not created). Decide on packaging approach before writing code in
  `src/casmi/` — recommend `pyproject.toml` + `uv` or plain `pip` + venv,
  whichever the team prefers; not yet decided.
- No data has been downloaded locally yet. `data/raw/` and `data/processed/`
  exist as empty gitignored directories.
- Repo has been `git init`'d with `origin` pointed at
  `https://github.com/aagneye/predict-2D-chemical-structures.git`, branch
  renamed to `main`, but **nothing has been committed or pushed yet** as of
  end of session 1.
