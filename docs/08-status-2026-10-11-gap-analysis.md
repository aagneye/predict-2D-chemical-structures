# Status 2026-10-11 — Why the public score is 0.148 and what to do about it

Written after re-reading the whole repo, the VM, the Kaggle leaderboard, the
top public notebooks and the discussion forum. **This document corrects a
wrong conclusion that runs through `docs/05`, `docs/06` and the `.kiro`
steering/memory files.**

## 1. The public leaderboard IS a real score (the docs were wrong)

The leaderboard page states: *"This leaderboard is calculated with
approximately 31% of the test data."* Kaggle re-runs every submitted notebook
against the **real hidden test set** (~400 Enveda natural-product molecules);
the visible `test.parquet` (a sample of train) is only what the notebook sees
during the editor/commit run. The earlier sessions saw the reference
notebook's editor-run diagnostics ("library sim = 1.0 for 100%") and concluded
the LB was leak-driven. It is not. Proof from our own numbers:

| Measurement | Value |
|---|---|
| Honest held-out MRR@25 (4-channel + GBM reranker, session 5) | 0.592 |
| Public LB for the same pipeline | **0.148** |
| Public LB for Channels 1+2 only | 0.152 |

If the LB were scored on leaked train data, our class-1 MRR of 0.93 would have
given a public score near 0.9. The hosts also confirmed (discussion 745715)
there is **no `molecule_id` overlap** between the dummy test and the hidden
rerun, so the id-hardcoding trick some notebooks use is noise.

Consequences:

- Our held-out split **massively overestimates** real performance. It is
  built from public-library spectra, and the class-2 proxy ("structure is in
  our pool") is artificial because our pool is only the 275,810 train
  structures.
- The public LB is noisy (~130 molecules, one molecule ≈ 0.008, seed noise
  ±0.007 per the forum) but it is the right direction signal. Compare it with
  an **identity-disjoint** validation on natural-product molecules, not with
  the current split.

## 2. Where we stand

| Item | State |
|---|---|
| Public LB | **0.148, rank 2405 / 3040**. Top is 0.484 (MarvinTMB); ranks 2-49 are 0.481-0.441, a dense block of public-notebook forks. |
| Submissions | 4 total: 0.152 (Sep 18, ch 1+2) and 0.148 (Sep 20/21, 4-channel). Latest kernel ran 3,314 s on CPU for 400 molecules. |
| Code | 85 tracked files, 6.7k lines in `src/casmi`, ~390 tests. Channels 1/2/4/5/6 + GBM reranker all implemented. |
| Azure VM `rogii-gpu` | Rebooted 2026-09-24. **`/mnt` (ephemeral) was wiped**: repo clone, `train.parquet`, the de novo MolForge checkpoints, `ranker_25feat.pkl`, `rank_train.npz`, all `runs/`. Only `~/casmi_checkpoints/` survived (`fpnet_final.pt`, `pool.npz`, `split.npz`). The VM is now **shared** with another project running on GPUs 0-2; GPU 3 is free. Today: re-created `/mnt/casmi`, re-cloned the repo, re-downloaded the data, started `scripts/azure_setup.sh`. |
| De novo (Channel 6) | Code exists, **trained model is lost**. Local `checkpoints/denovo_final.pt` is a 39 KB test fixture. |
| Kaggle dataset `aagneye/casmi26-baseline-assets` | pool.npz (train only), fpnet_final.pt, ranker.pkl (25-feature), RDKit wheel, source. `ranker.pkl` was pickled with sklearn 1.9.1 but Kaggle has 1.6.1 (InconsistentVersionWarning in the log). |
| Local dev env | `.venv` had to be rebuilt (the Python it pointed at was uninstalled). Now Python 3.12.13 + RDKit 2026.03.3 + kaggle CLI, via `uv`. |

## 3. Why 0.148 — the diagnosis

The hidden test is ~16% class 1, ~45% class 2, ~39% class 3 (community
estimate from LB probing, discussion 741597). Our pipeline, on that mix:

1. **Class 2 is unreachable for us.** Our candidate pool is the 275,810 train
   structures only. A class-2 molecule is by definition *not* in train, so its
   structure is never a candidate and the best possible score for 45% of the
   test is 0. The top notebooks use train ∪ **COCONUT** (436k) ∪ ChEBI/LipidMaps
   as the main pool (~711k) plus a **105.9M-structure PubChem tier** as a
   second channel. On a 250-molecule natural-product panel, 246/250 answers
   are inside the 10-ppm PubChem window.
2. **Class 3 is unreachable** (de novo model lost; even when trained, the
   forum consensus is that generated candidates have not yet paid off on the LB).
3. So our entire 0.148 comes from class 1 (library match). 0.16 × 0.93 ≈ 0.15.
   This is exactly the "library-only" score every team starts from
   (prvsiyan's pure library search also scores 0.151).
4. Adding FPNet/fragmentation/reranker changed nothing (0.152 → 0.148) because
   there was nothing new to rank: the true structure was not in the list.

**The single biggest lever is candidate-pool coverage (COCONUT + PubChem),
not better ranking of the current pool.** Ranking matters second: once the
answer is a candidate, "most database-route errors are isomer errors" —
the wrong winner has the same molecular formula ~98% of the time
(discussion 743254).

## 4. What the 0.43-0.48 public pipeline actually does

All top-50 public notebooks are forks of one lineage (prvsiyan → ahmedberatozer
v4 → bobthebot369 → obstacledeveloper → flexonafft → canqiang). Verified from
the source of `canqiang/casmi26-e006-window-40-ice-fallback-true` and its 16
attached datasets:

| Component | What it is | Reported LB effect |
|---|---|---|
| Main pool | train ∪ COCONUT 2.0 (710,701) with 6,930 selected fingerprint bits, precomputed fragment masses | baseline |
| Engine 1 (v4n) | library + adduct-shifted library + analog propagation + MetFrag-lite + FPNet (`fpnet_full1`, trained on *all* train) + 4-booster LightGBM ranker over feature families (derivation priors, fragmentation 2.0, analog-structure relations, DreaMS-FP views) | ~0.36-0.38 |
| Engine 2 | prvsiyan's 31-feature HistGBM ranker + a second 51-feature ranker on simulated rows; fused with Engine 1 by weighted reciprocal rank (1/(3+r) + 0.6/(3+r)) | +0.02 |
| PubChem-only channel | 105.9M PubChem structures, 10-ppm window, FPNet A+B dot-product, top-5000 shortlist; **popularity prior** 0.25·(log1p substances + log1p PubMed) | +0.004 (prior) |
| Popularity re-rank of the pool list | z(ranker) + 0.15·pop | +0.010 |
| ICEBERG + GLACIER | forward MS/MS simulators (Coley group `ms-pred`, MIT, MassSpecGym-trained weights); predict each candidate's spectrum, entropy-similarity to the measured one; re-rank **same-formula groups** by z(score)+z(ICE)+z(GL). GLACIER on [M+H]+ only | ICE +0.013 / GL +0.006 (disputed) |
| Gated PubChem merge | PubChem candidates interleaved into slots 2,4,6,8,10 (or 4,8,...) when no strong library match; promoted to rank 1 when z-score > 6 and popularity ≥ 5 | ~+0.02-0.04 |
| Metric-safe tail | dedupe by the *official* key (stereo stripped → tautomer canonical → InChIKey14), backfill vacant slots with ranked overflow and mass-compatible DreaMS candidates | small |

Key caveats from the forum (discussion 748068, "never validated by CV"):
every gain after ahmedberatozer's early versions was accepted on the 130-
molecule public LB alone; `fpnet_full1` saw all training data so the lineage
cannot be validated offline; several thresholds are "LB 0.403 rule" magic
numbers. Expect a private-LB drop for blind forks. MarvinTMB (rank 1, 0.484)
says NIST or extra data is **not** needed.

Licensing (host answers, discussions 741359/745072/744470): train.parquet
derivatives, COCONUT, PubChem, MassSpecGym-trained ICEBERG/GLACIER (MIT) are
fine for prize-eligible solutions; **NIST is forbidden** (disqualification).

## 5. Facts that change our validation design

- `inchikey14` stored in train differs from the scorer's tautomer-canonical
  key for ~1.6% of structures (10.4% of COCONUT). Build pools and dedupe with
  the scorer key, stripping stereo *before* tautomer canonicalisation.
- The popularity prior is enormous on known natural products: FPNet alone
  ranks the answer first for 40% of a 250-NP panel; adding 0.25·pop gives 93%
  (dmitriigluzdov's panel benchmark). The correct structure is better
  documented than 99.7% of its mass-window neighbours (median).
- An honest NP benchmark exists: `dmitriigluzdov/casmi26-natural-product-panel-benchmark`
  (250 timsTOF NPs, three scenarios, every PubChem candidate pre-scored) and
  `casmi26-identity-disjoint-validation-folds` (5 folds, identity- and
  scaffold-disjoint). Pretrained public FPNet checkpoints have seen all folds.
- FPNet generalisation gap: 89% top-1 on molecules it trained on vs 42% on
  unseen NPs. Our own FPNet (val cosine 0.71) is likely similar or worse.
- Mass window width is irrelevant on timsTOF (99% within 5 ppm); formula
  scoring and curated-DB membership flags did not help; in-silico
  fragmentation **parsimony** (fewer cleavages, down-weight 2-bond breaks,
  linear intensity, 0.005 Da, H shifts -2..+3) gave +0.023 held-out.
- "Novel molecule" simulations by deleting the structure from the pool
  overstate de novo reach ~3×; validate class-3 ideas on molecules genuinely
  absent from PubChem.

## 6. Recommended plan (ordered by expected LB gain per unit work)

1. **Candidate pool: train ∪ COCONUT ∪ PubChem tier.** Rebuild `pool.npz`
   with COCONUT (CC BY 4.0, `prvsiyan/coconut-casmi26-candidates` already has
   it fingerprinted with the same 4096/4096/2048/167 layout; or download
   COCONUT 2.0 and run `scripts/build_pool.py --coconut`). Add a PubChem
   channel using `ahmedberatozer/casmi26-pubchem-tier` (5.5 GB, mass-sorted
   SMILES) or PubChemLite CC0 (385k). Expected: this alone unlocks class 2,
   the largest bucket. Target ≥0.30.
2. **Popularity prior** from `dmitriigluzdov/casmi26-pubchem-popularity-prior`
   (pool_popularity.csv keyed by InChIKey14; arrays for the PubChem tier).
   Cheap, large, and chemically sensible for "known natural products".
3. **Honest validation**: adopt identity-disjoint folds + the 250-NP timsTOF
   panel; retrain FPNet on 4 folds so the ranker can be validated. Report
   candidate coverage before MRR. Stop using the current `split.npz` numbers
   as a proxy for the LB.
4. **Isomer separation**: fragmentation parsimony changes (cheap), then
   forward-spectrum re-scoring of same-formula groups with ICEBERG/GLACIER
   (MIT, offline packages exist) on GPU within the 9 h budget.
5. **Ship fixes**: dedupe with the official metric key (stereo-strip first),
   backfill to 25, re-pickle the reranker with Kaggle's sklearn 1.6.1 (or
   export to a version-stable format), enable GPU in the kernel.
6. **De novo (class 3) last**: retrain MolForge on the VM (GPU 3) with
   checkpoints under `~/casmi_checkpoints/`, but only insert generated
   candidates behind database candidates and validate on PubChem-absent
   molecules. Forum evidence says this has not yet paid off for anyone.

## 7. Operational notes

- VM: use `CUDA_VISIBLE_DEVICES=3`; GPUs 0-2 belong to another project. Keep
  anything valuable in `~/casmi_checkpoints/`, never only on `/mnt`.
- Local: `.venv` rebuilt with `uv venv --python 3.12` + `uv pip install -e ".[dev]" kaggle`.
  Kaggle CLI: `.venv/Scripts/kaggle`.
- Kaggle rerun: hidden test has ~2,500 spectra (vs 1,213 visible); our CPU
  run took 55 min for 1,213, so budget ~2 h CPU, far less on GPU.
- Hosts updated the rerun test data on 2026-10-08 (electron-mass fix) and
  re-scored all submissions.

## 8. Build log (same day)

Implemented and pushed (atomic commits `c0100b3..5fb8e5e`):

| Change | Where | Why |
|---|---|---|
| Scorer-key fix: strip stereo before tautomer canonicalisation | `src/casmi/chem.py` | matches the official metric; the old key mis-identified skeletons whose stereocentre pins a tautomer |
| PubChem popularity prior as feature + post-rank blend | `src/casmi/priors.py`, pipeline/fusion/ranker, `--popularity`, `--popularity-mu` | the single biggest isomer-separation signal measured by the community (40% → 93% top-1 on a 250-NP panel) |
| Parallel, scorer-keyed pool build (COCONUT + PubChemLite + train) | `scripts/build_pool.py`, `pool_from_records` | unlocks class 2; ~1.1M structures in minutes on 48 cores |
| Natural-product panel split | `--holdout-lib enveda-np-examples`, `--cohort-keys` | an honest timsTOF class-2 proxy instead of the old split |
| Fragmentation parsimony defaults | `src/casmi/channels/fragmentation.py` | +0.023 held-out in the community ablation |
| Kaggle notebook: GPU, prior, reranker feature guard; sklearn 1.6 pin | `kaggle/submission/`, `pyproject.toml` | no more silent fallbacks or pickle-version crashes |

Running on the VM (do not poll; read the logs when back):
`/mnt/casmi/logs/phase_a.log` (pool v2 → prior → split v2 → six baseline
measurements, artifacts to `~/casmi_checkpoints/`) and
`/mnt/casmi/logs/fpnet_v2.log` (FPNet retrain on the new split, GPU 3).

Next: rebuild `rank_train.npz` with `--popularity --fragmentation` and
`fpnet_v2`, retrain the ranker, measure, `build_kaggle_dataset.sh`, push the
kernel, submit. Then the PubChem 106M channel and ICEBERG/GLACIER isomer
re-scoring (phase B).
