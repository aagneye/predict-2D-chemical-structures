# Codebase Guide

What exists in `src/casmi/`, why each piece is shaped the way it is, and how to
run it. Implements `06-implementation-plan.md`.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"          # baseline pipeline + tests
uv pip install -e ".[dev,train]"    # adds torch for FPNet (use on Azure)
```

RDKit is pinned to **2026.3.3**, the competition's scoring version. Tautomer
canonicalisation output is version-sensitive, so an unpinned RDKit would make
our local InChIKey14 matching disagree with Kaggle's scoring.

Run tests: `.venv/Scripts/python -m pytest tests/ -q` (317 tests, ~2 min).
Lint: `.venv/Scripts/python -m ruff check src/ tests/ scripts/`.

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | All hyperparameters as frozen dataclasses. Values marked `(ref)` came from the public-LB-leading notebook and were tuned against a leaked test set — re-sweep them. |
| `chem.py` | Tautomer-canonical `inchikey14()` (the scoring key), `FingerprintCalculator`, matmul `tanimoto_matrix()`. |
| `adducts.py` | Monoisotopic mass table, `neutral_mass_array()`. Upstream of everything. |
| `spectra.py` | Numba entropy similarity (Li et al. 2021) with mass-shift support, peak cleaning, multi-spectrum fusion. |
| `eval/metrics.py` | MRR@25, recall@25, top-1, per-novelty-class breakdown. |
| `data/loaders.py` | Parquet to flat CSR-style `SpectralLibrary`; `QueryMolecule` grouping. |
| `data/split.py` | The leak-proof split. `verify_no_leakage()` is a hard gate. |
| `candidates/pool.py` | Mass-sorted `CandidatePool` with bit-packed fingerprints. |
| `channels/library.py` | Channel 1 — direct spectral match (class 1). |
| `channels/analog.py` | Channel 2 — mass-shifted analog propagation (class 2). |
| `channels/fusion.py` | Weighted score fusion, dedup by InChIKey14, top-25. |
| `pipeline.py` | Orchestration, diagnostics, submission writer. |
| `models/fpnet.py` | Spectrum to fingerprint transformer + Bayes dot-product scoring. Optional (torch). |
| `models/train.py` | FPNet training loop, multi-GPU aware. Optional (torch). |

## Design decisions worth knowing

**Split by `inchikey14`, never `spectrum_id`.** Compounds recur across all 11
source libraries and appear at multiple adducts and collision energies. A
spectrum-level split puts near-identical spectra of the same structure on both
sides. Splitting by full `inchikey` still leaks stereoisomers of the same 2D
skeleton, which is what actually gets scored.

**The hold-out is NP-weighted, not uniform.** `enveda-180` is ~46% of spectra
but synthetic drug-like screening chemistry. A uniform random hold-out measures
the wrong chemical space. `SplitConfig.library_weights` upweights
`riken`/`gnps`/`enveda-np-examples`; a structure appearing in both an NP library
and `enveda-180` takes the max weight, so it counts as in-domain.

**Three novelty classes are synthesised and always reported separately.**
Class 1 keeps sibling spectra in the library (that is what "reference spectra
exist" means); class 2 removes all its spectra but keeps the structure in the
pool; class 3 removes it from the pool too. Aggregate-only reporting is how the
public-LB leader failed to notice three of its four channels contributed
nothing.

**Held-out spectra are removed at load time, not filtered later.**
`load_spectral_library(..., keep_rows=train_mask)` means a forgotten downstream
filter cannot leak. Likewise `train_fpnet.py` requires `--split`.

**Fingerprints are bit-packed in the pool.** ~700k structures x 10,407 bits is
~7 GB unpacked, ~900 MB packed — the difference between fitting in a Kaggle
notebook and not.

**Candidate scoring against FPNet is one matmul.** Under independent Bernoulli
bits, a candidate's log-likelihood is `f·z` plus a candidate-independent term,
so `f·z` gives an identical ranking. `tests/test_models.py` asserts that
equivalence against the exact expression rather than assuming it.

**Peaks are embedded by m/z *and* neutral loss.** Neutral loss
(`precursor - m/z`) is the transferable quantity: 18.011 Da is water regardless
of precursor mass. This matters because the training libraries and the test set
occupy different chemical space.

**Fusion is transparent weighted scoring, not a learned reranker (yet).** With
a learned model it is hard to tell whether a change came from better evidence or
from the reranker memorising something. A GBM replaces it once per-channel
deltas are known. Fusion is score-based rather than rank-based because the
channels' scores are mutually calibrated: a 1.0 library similarity is near-proof
of identity, while a correct analog answer commonly sits near 0.5, and
reciprocal-rank fusion would discard that.

## Workflow

Order matters — the split gates everything.

```bash
# 1. Build the honest validation split. Fails loudly if it would leak.
python scripts/build_split.py \
    --train data/raw/train.parquet \
    --out data/processed/split.npz \
    --n-holdout 2000

# 2. Build the candidate pool (COCONUT + train structures).
python scripts/build_pool.py \
    --coconut data/raw/coconut.csv \
    --train data/raw/train.parquet \
    --out data/processed/pool.npz

# 3. Baseline: Channel 1 + Channel 2, scored per novelty class.
python scripts/run_baseline.py \
    --train data/raw/train.parquet \
    --split data/processed/split.npz \
    --pool data/processed/pool.npz \
    --out runs/baseline

# 4. FPNet — Azure 4x T4, not the Kaggle notebook.
python scripts/train_fpnet.py \
    --train data/raw/train.parquet \
    --split data/processed/split.npz \
    --out checkpoints/fpnet \
    --max-steps 20000 --batch-size 64
```

`run_baseline.py` prints per-class MRR@25 and channel contribution, and warns
if library search carries >95% of molecules — on a held-out split that means
leakage, not strength.

### Interpreting the output

```
cohort            n   MRR@25   recall     top1   cands
------------------------------------------------------
overall        2000   ?.????   ?.????   ?.????    ??.?
class1          500   ...
class2          900   ...
class3          600   ...
```

Read `recall` alongside `mrr`: recall@25 is the ceiling re-ranking can reach, so
a high recall with low MRR means invest in ranking, while low recall means
invest in retrieval (a structure absent from the pool can never be predicted).

## Current state and what is deliberately absent

Implemented: validation, Channels 1-2, fusion, baseline pipeline, FPNet
architecture and training.

Not yet implemented (steps 6-10 of `06-implementation-plan.md`):

- Wiring FPNet logits into the pipeline as a ranking feature (the model and
  `score_candidates()` exist; `pipeline.py` does not call them yet)
- Channel 5, in-silico fragmentation (MetFrag-lite)
- The learned GBM reranker replacing weighted fusion
- De novo generation for the class-3 tail — **our intended differentiator**,
  since the reference solution is retrieval-only and therefore structurally
  incapable of class 3, which it estimates at 30-40% of the test set

## Testing approach

No competition data is present locally (`data/raw/` is empty, files are 3 GB),
so everything is exercised against synthetic fixtures in `tests/conftest.py`
built from real small molecules — RDKit behaviour stays realistic while correct
answers are known by construction. Fixtures deliberately reproduce the awkward
properties of the real data: compounds recurring across libraries, variable
spectra per molecule, unknown adducts, empty peak lists.

`tests/test_scripts_cli.py` runs the actual scripts as subprocesses against a
synthetic `train.parquet`. Unit tests would not catch an argument mismatch or a
column missing from a script's read list.

Verified behaviour on that synthetic run: leakage gate passes; hold-out draws
from riken/massbank/gnps/enveda-np-examples with zero `enveda-180`; class-1 and
class-2 recover their structures while class-3 MRR is exactly 0.0, confirming
the class-3 cohort is genuinely unreachable rather than nominally so.

Four real bugs were caught by tests while writing this: the spectral cleaner
assumed m/z-sorted input (silently wrong similarity on unsorted data, now
enforced), `merge_spectra` normalisation semantics, `compute_bit_weights` floor
behaviour, and a wrong NH4 mass constant in a test.
