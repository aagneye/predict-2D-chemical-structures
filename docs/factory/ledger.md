# Factory ledger — goal: public LB > 0.46

Newest first. Every entry: what changed, offline emulator number, Kaggle public score.

## Kaggle submissions

| Version | Date | Public | What changed |
|---|---|---|---|
| v8 | 2026-10-20 | **0.252** | evidence-based pruning, PubChem 106M tier channel, CV reranker (29 feat, OOF 0.66), popularity v2 |
| v7 | 2026-10-12 | 0.186 | pool = train ∪ COCONUT ∪ PubChemLite (1.07M), popularity prior, FPNet v2, parsimony fragmentation, weighted fusion |
| v4 | 2026-09-20 | 0.148 | train-only pool, FPNet, fragmentation, in-sample 22-feat reranker |
| v2 | 2026-09-18 | 0.152 | library + analog only |

Leaderboard reference (2026-10-19): 1st 0.495, 10th ≈ 0.464, best public notebook ≈ 0.458.

## Offline numbers (split v3, 2,238 molecules, grouped 5-fold OOF reranker)

| Config | Overall | Class 1 | Class 2 | Class 3 | Note |
|---|---|---|---|---|---|
| v8 stack | 0.6625 | 0.873 | 0.772 | 0.275 | class 3 inflated: PubChem channel could still find class-3 truths (fixed by `pubchem_blocklist`, 2026-10-21) |

CV → LB ratio so far ≈ 0.38 (0.66 → 0.252). Main suspects: real class 3 (~40–52% of
test) is in no database; validation spectra are mostly non-timsTOF; popularity prior
transfers ~7× weaker to the LB than to np-examples.

## Built, not yet measured

- 2026-10-21 **Channel 8, analog derivation** (`src/casmi/channels/derivation.py`):
  applies the biosynthetic edit implied by each analog's mass shift (31 reaction
  SMARTS: hydroxyl, methyl, sugars, acyls, redox, (de)hydration, decarboxylation; up to
  2 edits) to the analog, keeps products on the measured mass, ranks by FPNet.
  Features `derived`, `derivation_parent_sim`, `derivation_steps`, `derivation_support`
  (33 ranker features). Tests: every edit fires with its declared mass delta; a
  database-absent isorhamnetin is recovered from kaempferol and ranked first.
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

## Open bets toward > 0.46 (ordered by expected gain per hour)

1. **LB emulator** (`scripts/lb_emulator.py`): fit hidden-test class shares to our known
   (config → public score) pairs, so every change gets a predicted public score offline.
2. **Measure Channel 8** on split v3 with the PubChem blocklist; retrain the 33-feature ranker; submit v9.
3. **Public-lineage base + our class-3 overlay**: the fastest credible route
   past 0.46 — fork the best public notebook unchanged, add Channel 8 candidates only for
   molecules where its evidence is weak. Prize-eligibility risk: some lineage weights
   lack public training code (host: must be reproducible).
   Public notebook scores checked 2026-10-22 (public LB, ±0.01 noise):
   flexonafft/casmi26-closing-the-evidence-gap **0.462 best (V47)**, 0.457 latest (V49,
   Apache 2.0); ahmedberatozer/casmi26-v14-pubchemlite 0.455; guhongbin v461-e006-ccby-glob
   0.454; seyitkaangunes e006-n1a-chebi 0.452; mayankraaz "0.458+" 0.448 (title overstates);
   alexchilton v461-depth7 0.448. V49 = V47's challenger recipe (cells 0-6, writes
   `submission.csv`) + a cell-7 CE-ensemble treatment that overwrote it and cost 0.005, so
   V49 without cell 7 should reproduce 0.462. Fork built on the VM at
   `~/forks/ours_0462` (17 public inputs, pinned image); pushing it waits on the user's OK
   because it runs third-party code under our Kaggle account.
4. ICEBERG/GLACIER same-formula re-scoring in our own stack.
5. FPNet ensemble (merged-spectra variant + seeds) on a free GPU.
6. Hard validation split S2 (enveda-180 structures with no public spectra).
