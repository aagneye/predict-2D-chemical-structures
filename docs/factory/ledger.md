# Factory ledger — goal: top 10 on the public LB (10th = 0.484 on 2026-10-26; > 0.46 was rank 778)

Newest first. Every entry: what changed, offline emulator number, Kaggle public score.

## Kaggle submissions

| Version | Date | Public | What changed |
|---|---|---|---|
| base-0467 | 2026-10-26 | **0.467** | unchanged copy of alexchilton/casmi26-lb-0-467-one-knob (public lineage: corrected PubChem + neural 0.25, RRF offset 1); reproduces exactly; rank ≈ 66 |
| fork-0462 | 2026-10-26 | 0.462 | flexonafft v49 cells 0–6 (the 0.462 challenger recipe), unchanged; reproduces exactly |
| v9 | 2026-10-26 | 0.255 | v8 + Channel 8 analog derivation + 33-feature CV reranker. Emulator said 0.287: class 3 transferred ≈ +0.003 (recall 12%, wrong sites), class-2 "gain" was an apples-to-oranges split comparison |
| v8 | 2026-10-20 | 0.252 | evidence-based pruning, PubChem 106M tier channel, CV reranker (29 feat, OOF 0.66), popularity v2 |
| v7 | 2026-10-12 | 0.186 | pool = train ∪ COCONUT ∪ PubChemLite (1.07M), popularity prior, FPNet v2, parsimony fragmentation, weighted fusion |
| v4 | 2026-09-20 | 0.148 | train-only pool, FPNet, fragmentation, in-sample 22-feat reranker |
| v2 | 2026-09-18 | 0.152 | library + analog only |

Leaderboard reference (2026-10-29, ≈ 4,300 teams): 1st 0.517, 2nd 0.513, 10th 0.484,
gold line ≈ rank 18 at 0.481; **231 teams sit at exactly 0.477** — the new best public
notebook (busyaprime "LB 0.477 · Two 0.472 Fixes, Merged the Right Way", submission
57722897): the same 0.465 pipeline + neural weight 0.75 on ranks 2–25 + Roman Tamrazov's
fragment check moving rank 1 (taken from the 0.50 order). Its own ablations, all on the
hidden test: long-trained neural model −0.019, NX isomer reranker −0.058, provenance
reranker ±0, three leader-rule thresholds ±0, vote step .025 → .015 −0.004. Fork
`aagneye/casmi26-lb-0-477-base-unchanged-public-copy` built (unchanged notebook, same 7
datasets, image `9a6bcfda…`); `scripts/push_when_quota.sh` pushes and submits it as soon
as the GPU quota resets.
Leaderboard reference (2026-10-26, 4,059 teams): 1st 0.504, 10th 0.484, 66th 0.467,
778th 0.460; we are 2,886th (0.252). Best public notebook 0.462 (flexonafft V47).
(2026-10-19: 1st 0.495, 10th ≈ 0.464 — the top is rising ≈ 0.01/week.)

## Offline numbers (split v3, 2,238 molecules, grouped 5-fold OOF reranker)

| Config | Overall | Class 1 | Class 2 | Class 3 | Note |
|---|---|---|---|---|---|
| v8 stack | 0.6625 | 0.873 | 0.772 | 0.275 | class 3 inflated: PubChem channel could still find class-3 truths (fixed by `pubchem_blocklist`, 2026-10-21) |

CV → LB ratio so far ≈ 0.38 (0.66 → 0.252). Main suspects: real class 3 (~40–52% of
test) is in no database; validation spectra are mostly non-timsTOF; popularity prior
transfers ~7× weaker to the LB than to np-examples.

### LB emulator, round 1 (2026-10-22, `scripts/run_factory_round1.sh`)

Test-like split `split_emu` (600 molecules, 200 per class; a class-2 truth stays in the
pool only when COCONUT/PubChemLite has it — 126/200; class 3 blocked everywhere, PubChem
included), each configuration re-run with its *own* submitted code (git worktrees):

| Config | Class 1 | Class 2 | Class 3 | Emulator | Kaggle |
|---|---|---|---|---|---|
| v2 (33dd6ea) | 0.915 | 0.160 | 0.000 | 0.154 | 0.152 |
| v7 (2c35b8b) | 0.787 | 0.293 | 0.000 | 0.180 | 0.186 |
| v8 (HEAD, 29-feat ranker) | 0.904 | 0.493 | 0.000 | 0.255 | 0.252 |
| **v9** (33 feat + Channel 8; split v3 OOF, 2,244 mol) | 0.877 | 0.580 | **0.101** | **0.287** | pending |

Fit: `public ≈ 0.50 × (0.23·m1 + 0.61·m2 + 0.16·m3)`, RMSE 0.004. Only `α·s1 = 0.115`
and `α·s2 = 0.306` are identified: every calibration point has m3 = 0, so the class-3
share is a free parameter until v9 (m3 > 0) is scored. Reading: the public score is
≈ 2.7× more sensitive to class 2 than to class 1; our class-1 proxy (truth spectra in
train) is easier than the real "has public spectra" class; and perfect class 1 + class 2
caps our stack near 0.42 — class 3 and the domain factor α have to move to pass 0.46.

### Overlay simulation (2026-10-26, `scripts/simulate_overlay.py`, split v3 rows)

Base = non-derived candidates ranked by the v8 ranker (no class-3 channel): 0.933 /
0.611 / 0.002. (Correction 2026-10-26: the public recipes *do* have an analogue generator,
`EngineCfg(generate=True)` in engine 1, so this base is weaker on class 3 than theirs.) Inserting Channel 8 structures with
`casmi.overlay.merge_overlay`:

| Gate | Insert at | k | Gated | Class 1 | Class 2 | Class 3 | Δ public (fitted s3) | Δ (class 3 = ½) |
|---|---|---|---|---|---|---|---|---|
| lib_max < 0.8 | below rank 1 | 1 | 68% | 0.932 | 0.614 | **0.054** | **+0.005** | **+0.014** |
| always | below rank 1 | 1 | 100% | 0.929 | 0.612 | 0.061 | +0.005 | +0.015 |
| always | below rank 2 | 1 | 100% | 0.933 | 0.615 | 0.041 | +0.004 | +0.011 |
| lib_max < 0.8 | below rank 1 | 3 | 68% | 0.931 | 0.610 | 0.055 | +0.004 | +0.013 |

Replacing the base's rank 1 (insert at 0) never makes the top 15 under either weighting.

### Channel 8 miss diagnosis (2026-10-26, `scripts/diagnose_derivation.py`, 400 class-3 molecules)

| Category | Share | Meaning |
|---|---|---|
| generated | 11.8% | truth produced (= recall) |
| reachable_1 | 28.2% | truth formula = parent + one transform delta, but not produced |
| reachable_2 | 18.5% | two deltas away, not produced |
| reachable_3 | 11.8% | three deltas away (we allow 2) |
| zero_delta | 4.2% | truth is an isomer of a parent |
| no_parent | 25.5% | no spectral analog ≥ 0.3 similarity |
| unreachable | 0% | — |

The transform *vocabulary* covers every class-3 formula within 3 steps; the misses are
site chemistry (methyl on aliphatic chains / S instead of aromatic C; thioether
demethylation), rearranged analogs (same delta, different arrangement), caps and depth,
and 25% with no parent at all. Example pairs are in `/mnt/casmi/runs/diag_derivation.json`.

Recall after the 19 site-level edits (same 400 molecules; "generated" = truth among the
products, before ranking):

| Setting | Generated | No parent |
|---|---|---|
| old edits (31), 2 steps, floor 0.3 | 11.8% | 25.5% |
| + 19 edits, 2 steps, floor 0.3 | 21.8% | 24.5% |
| + 3 steps, 1,000 products/path | 25.0% | 24.5% |
| + parent floor 0.15, 30 parents (2 steps) | 25.5% | 15.2% |
| natural-product-flagged class 3 only (198 mol, 2 steps) | 18.7% | 21.7% |

Of the 113 one-step misses with the old edits, 77 were "rearranged" (the parent is not a
substructure of the truth): no single-site edit can reach them — the ceiling of the
edit-a-known-analog idea is roughly 50–60% of class 3.

### LB emulator, round 2 (2026-10-26) — class 3 does not transfer

v9's own code on `split_emu`: 0.910 / 0.494 / **0.078** → Kaggle 0.255 (v8: 0.904 / 0.493 /
0.000 → 0.252). Refit on four points: shares 0.27 / 0.72 / **0.01**, α = 0.425, RMSE 0.0034.
Fixed alternatives fit worse (0.30/0.50/0.20: RMSE 0.009; 0.16/0.45/0.39: 0.010;
0.28/0.20/0.52: 0.021). Reading: a class-3 MRR of 0.08 on our train-derived proxy bought
≈ +0.000–0.003 on the board. Either class 3 is a small part of the public test, or (more
likely, and what the forum reports) our class-3 proxy is easier than the hidden novel
natural products. Consequence: the overlay's expected gain is 0 … +0.005, not a route to
the top 10 by itself; class-2 discrimination on the strong base is.

### Testing centre for the public base (2026-10-27)

`scripts/run_public_base.sh` runs a public notebook on the VM in a no-network sandbox
(`unshare -rmn`, Kaggle-shaped `/kaggle` bind-mounted, `~/.kaggle` hidden) with a
Python 3.12 env matching Kaggle (numpy 2.0.2, pandas 2.3.3, pyarrow 24, sklearn 1.6.1,
torch 2.10 cu128) and all 7 inputs of the 0.467 notebook (39 GB in
`/mnt/casmi/base0467/input`). Smoke run (12 visible molecules): **RUN_OK**, 3,014 s, of
which ≈ 2,800 s is setup (library, pools, engines) and ≈ 90 s the simulators. ICEBERG
scored 11/12 molecules (47 pred/s on a T4), GLACIER 11/11 (62 pred/s, [M+H]+ only), N1
proxies 0 (no non-[M+H]+ molecules in the sample). The "CLAW STAGE HEALTH | ICE NOT RUN"
line is a stale label; "STAGE SCORED ice=11 gl=11" is the real count. Rerun budgets:
ICE 5,400 s, GL 4,000 s, N1 1,200 s. Full 400-molecule rerun-mode run started for
stage timings and per-molecule simulator coverage (accuracy on the visible test is
meaningless — it is a sample of train).

Full rerun-mode run, 400 visible molecules (1,213 spectra, all timsTOF): **RUN_OK in
5,153 s (86 min) of the 9 h limit** — ≈ 7 h of headroom. ≈ 69 candidates per molecule go
to the simulators (27,594). ICEBERG 1,226 s, 371/400 molecules; GLACIER [M+H]+ only,
366/400; N1 proxies (NH4p/WLp) 2 molecules. **34 molecules (8.5%) have no [M+H]+
spectrum: 25 negative-only [M-H]-, 5 [M+Na]+, 2 [M+NH4]+, 2 formate** — 32 of them get no
GLACIER score and ≈ 29 no ICEBERG score. 124/400 molecules have negative-mode spectra.
Leads for the next base experiments (each verified to run in the sandbox first):
(E1) the Channel 8 overlay, ready; (E2) an [M-H]- proxy arm for the N1 stage — the
forum's repeatable +0.008 steps were exactly this kind of simulator coverage for
non-[M+H]+ molecules; (E3) spend the 7 h headroom on more candidates per molecule in the
simulator window. Offline accuracy stays unmeasurable on the base (its models saw the
train molecules), so these are confirmed on the LB after the GPU quota resets.

CPU-only run (2026-10-29, `GPU=none CPUS=0-3 RERUN=1`, a Kaggle CPU notebook's 4 cores):
**RUN_FAILED — the 0.467 base cannot be submitted as a CPU notebook.** ICEBERG falls back
to CPU by itself but scores 2.6 predictions/s (≈ 1 candidate/s; a T4 does ≈ 20): its
5,400 s budget covered 86 of 371 molecules, and the final pass asserts
`ice status == 'ok'`. GLACIER on CPU is fine (16 pred/s, 366/366 in 3,235 s); the rest
of the pipeline takes ≈ 4,100 s on 4 cores. A full CPU ICEBERG would need ≈ 6.7 h on its
own, so CPU only works with fewer ICEBERG candidates — a different recipe, which would
confound every A/B against the GPU base. Where ICEBERG enters: cell 41 (z(score) +
z(ice) + z(gl) for engine and PubChem lists), cell 45 (`CELL19_ICE`), and cell 52 (ICE
only where a molecule has no GLACIER score); cell 49 passes `{}` for ICE.

### E2 probe: can [M-H]- spectra stand in for [M+H]+? (2026-10-29, `scripts/probe_negative_proxy.py`)

Measured spectra only, no model: 3,000 train molecules with both [M+H]+ and [M-H]-, each
ranked among its same-formula molecules (mean 25.7 candidates) by entropy similarity to
their measured [M+H]+ spectra.

| Query | MRR | Top-1 | Top-5 |
|---|---:|---:|---:|
| chance | 0.261 | | |
| [M-H]- unshifted (control) | 0.463 | 30% | 66% |
| **[M-H]- shifted +2 protons (E2 proxy)** | **0.522** | 37% | 70% |
| a second [M+H]+ spectrum (ceiling) | 0.885 | 82% | 96% |

The proxy adds +0.06 over the unshifted control even against *measured* [M+H]+ spectra;
GLACIER predictions are weaker than measurements, so the usable signal for the ≈ 7%
negative-only molecules is small and a z-score fusion at weight 1 could hurt.
**E2 deprioritised.**

### Kaggle GPU quota

2026-10-26 03:1x UTC: `Kernel push error: Maximum weekly GPU quota of 30.00 hours reached`
— the overlay kernel (needs the GPU base) cannot run until the weekly reset (weekend,
00:00 UTC; check kaggle.com/settings). Use the week for Channel 8 recall on the VM.
Reading: one structure directly below rank 1 is nearly free for classes 1–2 (−0.001 to
−0.004) and lifts class 3 to 0.05–0.06 (vs 0.10 when the v9 ranker may put it first).
On a 0.467 base that is ≈ +0.005…+0.015 → 0.47–0.48; the top 10 (0.484) needs better
Channel 8 recall (12% @25 now) on top of this.

## Built, not yet measured

- 2026-10-26 **Channel 8 site coverage**: 19 new edits (chain ±CH2, aliphatic / amide /
  aniline / S methylation, S-demethylation, methyl esters, N-acetyl, aromatic C-hexose,
  aliphatic sulfate, phosphate, aldehyde / imine reduction, epoxide, CH2 → C=O, Cl / Br),
  mixed-sign deltas, and lazy scorer keys in `derive_candidates` (pool by SMILES, key
  only the best-scoring products) so the caps can grow. Measuring: recall via
  `diagnose_derivation.py` (`diag_v2`, and `diag_v2_s3` = 3 steps, 1,000 per path).

- 2026-10-21 **Channel 8, analog derivation** (`src/casmi/channels/derivation.py`):
  applies the biosynthetic edit implied by each analog's mass shift (31 reaction
  SMARTS: hydroxyl, methyl, sugars, acyls, redox, (de)hydration, decarboxylation; up to
  2 edits) to the analog, keeps products on the measured mass, ranks by FPNet.
  Features `derived`, `derivation_parent_sim`, `derivation_steps`, `derivation_support`
  (33 ranker features). Tests: every edit fires with its declared mass delta; a
  database-absent isorhamnetin is recovered from kaempferol and ranked first.
  **Measured 2026-10-22**: class-3 OOF MRR 0.101 (recall@25 0.124, top-1 0.087) on
  split v3 with the realistic pool and PubChem blocklist — the first non-zero class-3
  number in this project; `ranker_v9.pkl` (33 features) trained on those rows.
- 2026-10-21 fix: FPNet logits were skipped when the pool window was empty — the
  exact novel-molecule case — which also silently disabled the PubChem channel there.
- 2026-10-21 `pubchem_blocklist`: evaluation keeps class-3 truths out of PubChem.

## Failed / do not repeat

See `docs/11-status-2026-10-18-competitor-comparison.md` §3. Short list: train-only pool;
raw FPNet dot product in fusion; mass-only pruning to 80; stored-key split; in-sample
ranker numbers; MolForge-from-bits for class 3; artifacts on `/mnt`.
2026-10-22: `dmitriigluzdov/casmi26-open-model-and-candidate-resources` is no longer
public (403); external keys now come from `pool_popularity.csv` (`np_coconut`) and
`pubchemlite_annotations.parquet` (`ik14`) — 649/1,149 split-v3 class-2 keys reachable.
The COCONUT/PubChemLite metadata elsewhere ships only as pickles: never unpickle them.
2026-10-26: kernel v9 pushed 3 s after `kaggle datasets status` said "ready" ran against
the *previous* dataset version (`No module named casmi.channels.derivation`). Wait a few
minutes after a dataset version before `kaggle kernels push`, or pin the version.
Same day, overlay kernel v1: `No module named casmi.overlay` — the dataset is a snapshot
of `src/casmi`; any new module needs a dataset version *before* the kernel push.

## Open bets toward the top 10 (2026-10-29, on the 0.477 base)

A. **Submit the 0.477 fork** (queued: `push_when_quota.sh` on the VM, log
   `~/casmi_checkpoints/logs/push_0477.log`). Expected 0.477.
B. **Neural seed ensemble (next).** Every public step 0.465 → 0.477 came from re-ranking
   the final 25 with sanpier's spectrum-candidate scorer (weight 0.25 → 0.50 → 0.75) plus
   Tamrazov's fragment check. The scorer is tiny (3-layer transformer, 16 epochs × 500
   steps = **≈ 5 GPU-minutes**, 17 s/epoch) and has public training code; a longer-trained
   copy lost 0.019 on the board (its random-holdout gain was memorisation). So: keep the
   exact recipe and length, train 4 more seeds on the VM, average scores
   (`kaggle/neural_ensemble/neural_scores.py`, a one-module patch with the notebook's own
   SHA-256 pin updated by `scripts/build_module_patch_kernel.py`; with no seed files it
   equals v1). Kernel `aagneye/casmi26-lb-0-477-neural-seed-ensemble` built; needs dataset
   `aagneye/casmi26-neural-seeds` (`seed*/final_model_seed*.pt`). Offline check:
   `scripts/neural_seed_ensemble.py eval` (same held-out molecules and candidates for each
   seed and their average; relative only).
C. ~~E3 wider simulator window~~: no effect by construction — simulator scores only act
   inside the final 40-candidate fusion shortlist (base top 60 + engine-2 top 40 are
   already in the ICE input). ~~E2~~: see the probe above. E1 (class-3 overlay): one
   submission at most; the forum reports the same negative result for rule-based
   biotransformation.

## Open bets toward > 0.46 (2026-10-26, kept for history)

1. ~~Fork the public recipes~~ done 2026-10-26: 0.467 and 0.462 reproduced under our account.
2. ~~Submit v9~~ done: 0.255. Channel 8 must find far more before it moves the board.
0. **Testing centre for the 0.467 base (next)**: run the public base on the VM against
   identity-disjoint labelled molecules (dmitriigluzdov's validation folds / NP panel
   datasets exist for this) so changes *on the base* are measured offline. Then the
   forum's levers on that base: forward-model (ICEBERG/GLACIER) coverage for molecules
   lacking scores (+0.008 per step historically) and retraining engine 1's ranker with
   source flags (10–40 CPU h per setting — fits the 64-vCPU VM). Needs the user's OK to
   run third-party code on the shared VM.
3. **Channel 8 recall 12% → 40%+** (deprioritised by emulator round 2; recall now 25%): add the missing
   site chemistry (aliphatic C-methylation / chain homologation ±CH2, S-(de)methylation,
   methyl esters, amide N-methylation, C=N/aromatic reductions), raise the per-path caps,
   allow 3 steps when the mass demands it, lower the parent floor / take parents from the
   base's own lists for the 25% without a parent. Measure with `diagnose_derivation.py`
   (generated share) then `run_baseline.py` on `split_emu` → emulator.
4. **Overlay kernel** (`aagneye/casmi26-lb0467-class3-overlay`, built and tested; v1
   failed only because the dataset predated `casmi.overlay`; dataset fixed) — push when
   the GPU quota resets, with the improved Channel 8. Offline: +0.005…+0.014 on 0.467.
   Prize-eligibility risk: some lineage weights lack public training code.
   Public notebook scores checked 2026-10-22 (public LB, ±0.01 noise):
   flexonafft/casmi26-closing-the-evidence-gap **0.462 best (V47)**, 0.457 latest (V49,
   Apache 2.0); ahmedberatozer/casmi26-v14-pubchemlite 0.455; guhongbin v461-e006-ccby-glob
   0.454; seyitkaangunes e006-n1a-chebi 0.452; mayankraaz "0.458+" 0.448 (title overstates);
   alexchilton v461-depth7 0.448. V49 = V47's challenger recipe (cells 0-6, writes
   `submission.csv`) + a cell-7 CE-ensemble treatment that overwrote it and cost 0.005, so
   V49 without cell 7 should reproduce 0.462. Fork built on the VM at
   `~/forks/ours_0462` (17 public inputs, pinned image); pushing it waits on the user's OK
   because it runs third-party code under our Kaggle account (approved 2026-10-26).
4. **Class-2 lift in our own stack** (the emulator's biggest lever, α·s2 = 0.31, m2 = 0.58):
   ICEBERG/GLACIER same-formula re-scoring; CC-BY ChEBI/LMSD and the full PubChemLite
   candidate set in the pool; engine-2-style rank-2 insertion.
5. FPNet ensemble (merged-spectra variant + seeds) on a free GPU.
6. Hard validation split S2 (enveda-180 structures with no public spectra).
