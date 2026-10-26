# Factory ledger — goal: top 10 on the public LB (10th = 0.484 on 2026-10-26; > 0.46 was rank 778)

Newest first. Every entry: what changed, offline emulator number, Kaggle public score.

## Kaggle submissions

| Version | Date | Public | What changed |
|---|---|---|---|
| v8 | 2026-10-20 | **0.252** | evidence-based pruning, PubChem 106M tier channel, CV reranker (29 feat, OOF 0.66), popularity v2 |
| v7 | 2026-10-12 | 0.186 | pool = train ∪ COCONUT ∪ PubChemLite (1.07M), popularity prior, FPNet v2, parsimony fragmentation, weighted fusion |
| v4 | 2026-09-20 | 0.148 | train-only pool, FPNet, fragmentation, in-sample 22-feat reranker |
| v2 | 2026-09-18 | 0.152 | library + analog only |

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

Base = non-derived candidates ranked by the v8 ranker (no class-3 channel, like the
public recipes): 0.933 / 0.611 / 0.002. Inserting Channel 8 structures with
`casmi.overlay.merge_overlay`:

| Gate | Insert at | k | Gated | Class 1 | Class 2 | Class 3 | Δ public (fitted s3) | Δ (class 3 = ½) |
|---|---|---|---|---|---|---|---|---|
| lib_max < 0.8 | below rank 1 | 1 | 68% | 0.932 | 0.614 | **0.054** | **+0.005** | **+0.014** |
| always | below rank 1 | 1 | 100% | 0.929 | 0.612 | 0.061 | +0.005 | +0.015 |
| always | below rank 2 | 1 | 100% | 0.933 | 0.615 | 0.041 | +0.004 | +0.011 |
| lib_max < 0.8 | below rank 1 | 3 | 68% | 0.931 | 0.610 | 0.055 | +0.004 | +0.013 |

Replacing the base's rank 1 (insert at 0) never makes the top 15 under either weighting.
Reading: one structure directly below rank 1 is nearly free for classes 1–2 (−0.001 to
−0.004) and lifts class 3 to 0.05–0.06 (vs 0.10 when the v9 ranker may put it first).
On a 0.467 base that is ≈ +0.005…+0.015 → 0.47–0.48; the top 10 (0.484) needs better
Channel 8 recall (12% @25 now) on top of this.

## Built, not yet measured

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

## Open bets toward > 0.46 (ordered by expected gain per hour)

1. **Fork of the 0.462 public recipe** — pushed 2026-10-26 as
   `aagneye/casmi26-e006-challenger-fork` v1 (user-approved); submitted automatically when
   the kernel completes. Expected ≈ 0.46 (rank ≈ 780); it is the base for bet 3.
2. **Submit v9** (emulator 0.287, +0.035 over v8): confirms that Channel 8 transfers to
   the hidden test and pins the emulator's class-3 share.
3. **Channel 8 overlay on the fork**: the public recipes have no class-3 channel, so their
   class-3 MRR is ≈ 0; our derived candidates (0.10 offline) inserted for molecules whose
   engine evidence is weak is the one addition nobody else has. Needs the fork's
   per-molecule evidence exported (engine scores / entropy similarity) — see docs/12.
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
