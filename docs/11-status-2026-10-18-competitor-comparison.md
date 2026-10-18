# Status 2026-10-18 — our scores, what worked, what did not, how the leaders do it, and the plan to the top 10

Plain-language summary first, details after. Numbers are MRR@25 unless stated.

## 1. Where we stand

| | Public LB | Date | What it was |
|---|---|---|---|
| **Current best (v7)** | **0.186** | 2026-10-12 | pool = train ∪ COCONUT ∪ PubChemLite (1.07M), popularity prior (mu 0.15), FPNet v2, parsimony fragmentation, weighted fusion, no reranker |
| v4 | 0.148 | 2026-09-20 | train-only pool, FPNet + fragmentation + 22-feature GBM reranker (in-sample held-out 0.592) |
| v2 | 0.152 | 2026-09-18 | library + analog only, train-only pool |

Rank today: **2671 of 3535** teams. Leaderboard top is **0.495** (CHU1PC, pikachu), 3rd 0.493, MarvinTMB 0.484; **10th place is ~0.464**, and ranks 10–50 are a dense block of public-notebook forks at 0.44–0.46. The public LB is ~130 hidden molecules, so ±0.01 is noise.

Honest held-out on our split v2 (2,250 molecules): library+analog 0.308 → +popularity 0.322 → +mu blend 0.334. Class-2 recall@25 was only 0.54–0.58 (fixed on 10-12 by evidence-based pruning). The reranker's grouped 5-fold CV on `rank_train_v2` (all channels incl. the PubChem tier, split v3) is running on the VM as this is written; its number goes into §6 when it lands.

## 2. The best thing we have, and the gap

Our best *measured* configuration is v7. Its gain over v4 came from one thing: **coverage**. The train-only pool could not contain any class-2 molecule, so 45% of the test was unreachable; adding COCONUT and PubChemLite plus a documentation prior lifted 0.148 → 0.186.

What we have built since but have **not yet shipped** (code is on `origin/main`, artifacts partly on the VM):

- Evidence-based pruning (FPNet z-score + popularity, cap 150) instead of mass-only top-80.
- Channel 7, the full 106M-structure PubChem tier, scored by FPNet + popularity inside the window.
- Within-set z-scored FPNet in fusion (the raw dot product was ~1,500 for every candidate and either drowned everything or did nothing).
- A reranker on 29 features with grouped cross-validation (`scripts/cv_ranker.py`) instead of the in-sample 0.592.
- Split v3: identity by the scorer key (71k train rows had a different stored key), panel queries only from `enveda-np-examples` spectra.
- Parsimony fragmentation (+0.023 held-out in the community ablation).

So the "best model" today is not a single model. It is a retrieval stack whose ranking stage is missing. v8 = v7 + reranker + PubChem tier + pruning fix is what phase B produces.

## 3. What did not work (so we stop repeating it)

| What | Why it failed | Status |
|---|---|---|
| Treating the public LB as unusable ("leak") | the LB is scored on the real hidden test (31%); the leak only affects the editor run | corrected in docs/08 |
| Train-only candidate pool | class 2 (known structure, no spectrum) can never be a candidate | fixed (pool v2) |
| In-sample reranker number 0.592 | trained and scored on the same molecules | replaced by grouped CV |
| Weighted fusion with raw FPNet dot product | scale ~1,500 vs 1.0 for library; made the fusion FPNet-only | fixed (fpnet_z) |
| Mass-proximity pruning to 80 | threw the answer away for ~40% of class-2 molecules on a 1.1M pool | fixed |
| Split keyed by stored inchikey14 | 1.6% of structures leaked sibling spectra; class-3 floor 0.06 instead of 0 | fixed (split v3) |
| MolForge from FPNet bits as the class-3 plan | built on a 28% paper later shown to be a bug (corrected 10.7%); model lost in the VM wipe | parked |
| Keeping artifacts on the VM's `/mnt` | ephemeral disk, wiped twice (Sep 24, Oct 12) | everything valuable now in `~/casmi_checkpoints` |

## 4. How the leaders do it (fresh read of the forum and the top public code, 2026-10-18)

### 4.1 The public 0.45 lineage, as of this week

Every 0.44–0.458 public notebook is `E006` (ahmedberatozer v4 engine + prvsiyan engine 2 + ICEBERG/GLACIER + gated PubChem) with these increments, each measured on the public LB alone:

- PubChem promotion threshold `rel > 150` instead of 250 (+0.003–0.005).
- "Metric-identity engine-2 rank-2 insert": engine 2's top pick goes to rank 2 unless it is a stereo/tautomer twin of rank 1 (+0.005).
- CC-BY ChEBI + LIPID MAPS structures (sergiomalv) in engine 2 (+0.003).
- PubChemLite CC0 structures added to engine 2's pool (ahmedberatozer v14, 0.451+).
- N1A: molecules with no [M+H]+ spectrum get GLACIER scores from [M+NH4]+ / [M−H2O+H]+ spectra converted to [M+H]+ proxies (+0.006).
- Pin the Docker image: unpinned copies on Python 3.13 score 0.01–0.014 lower.

Honest commentary from the same people: "most of our small improvements on top of strong pipelines ended up slightly worse"; "the one change that moved the 0.437 pipeline at all was adding new candidates from an independent engine, merged by reciprocal-rank fusion with the source's top-1 protected" (0.437 → 0.438). The lineage is saturated; the private LB will reshuffle it.

### 4.2 What the real top (0.48–0.495) is *not* saying

MarvinTMB (0.484): "You do not need NIST or any additional data for a better score." No write-ups from the 0.49 teams yet. The gap between 0.458 (best public) and 0.495 is ~5 molecules out of 130; private-LB robustness will matter more than that gap.

### 4.3 Validation lessons from strong non-public teams

- **Bull (0.32 → ~0.33 predicted, discussion 748524):** hold out 1,250 structures (all 250 np-examples + 600 COCONUT-member NPs not in MassSpecGym + 400 non-COCONUT NPs), draw spectra per molecule to match the test's spectra-per-molecule distribution and the 10 test adducts, simulate class 3 by removing Tanimoto ≥ 0.7 neighbours from library *and* candidates. Fitting class shares to several LB scores gave **~28% class 1, ~20% class 2, ~52% class 3**. Their first weighted estimate (0.334) matched the LB (0.322). Also: PubChem candidates inserted at fixed ranks helped locally (+0.01–0.02) and *hurt* on the LB (0.295 → 0.288): "there appear to be almost no test molecules that exist only in PubChem."
- **Aryan Awasthi (0.451):** two splits, S1 = np-examples (all have public spectra elsewhere), S2 = 490 enveda-180 structures with *no* public spectra, answers reachable only via PubChem. **Ranked by popularity alone, the answer is #1 for 67% of S1 but ~0% of S2**; the popularity prior added +0.50 on S1 but only +0.075 on the LB. He now accepts a change only if it helps S2. LambdaRank + fragmentation + analog features: S2 0.50 → 0.66, LB +0.03–0.04 for the same steps.
- **Kino Lutor (0.44, discussion 750241):** on 130 validation molecules absent from every database, the pipeline ranks the answer first 7 times, but the analog-seeded *generator* could produce the correct structure for 100 of 130 when given suitable parents. Nobody on the public board has turned that into a score.
- **A Spanish post (750491):** leak autopsy identical to ours (self-Tanimoto 1.0 for train candidates, contaminated spectral bank); their honest score fell 0.133 → 0.094.

### 4.4 Host rulings this week that change what we may use

- **Non-commercial open data/weights are now allowed** (CC BY-NC etc.). NIST and anything trained on NIST stay banned.
- **FRIGID** checkpoints (Coley group, ICML 2026; MIST encoder → masked diffusion over SAFE strings, formula-conditioned, ICEBERG-based refinement; >18% top-1 on MassSpecGym) are allowed. TabPFN allowed. PubChem CID-SID/PMID/patent counts allowed as features. LOTUS and NPAtlas allowed.
- **The ICEBERG PubChem atlas** (predicted MS/MS spectra for 91.4M PubChem structures from a NIST-trained model, iceberg-ms.mit.edu) **may be used as an in-silico spectral library**, shipped as a private dataset with the query script.
- **Electron-mass correction:** the hidden test's positive-mode m/z were corrected; train.parquet was not. Subtract 0.000548 Da from positive-mode m/z in train to match (1.5 ppm at 350 Da; matters for ±5 ppm windows and 0.005 Da peak tolerances).
- Compliance check may cover *any* submission; third-party weights need public training code + permitted data.

## 5. Us vs them, component by component

| Component | Leaders (0.45–0.49) | Us (v7 shipped / v8 built) | Gap |
|---|---|---|---|
| Candidate pool | train ∪ COCONUT ∪ ChEBI/LMSD ∪ PubChemLite (~1.5M) + 106M PubChem tier channel | train ∪ COCONUT ∪ PubChemLite (1.07M) + 106M tier (built) | add ChEBI/LMSD (CC-BY); otherwise equal |
| Spectrum→fingerprint | 2–3 FPNets ensembled (single + merged spectra), one trained on *all* train | one FPNet v2 (cos 0.714) | ensemble + merged-spectrum variant |
| Library / analog | entropy similarity, adduct-shifted library matches, timsTOF representatives preferred, analogs top-80/200 | entropy similarity, analogs top-100, no adduct-shifted library match, no timsTOF preference | add both |
| Fragmentation | MetFrag-lite, adduct-aware carriers | parsimony MetFrag-lite (better per ablation), proton carriers only | add adduct carriers |
| Forward simulators | ICEBERG + GLACIER re-score same-formula groups, N1A adduct proxies | none | **biggest missing piece** |
| Prior | popularity 0.25 in PubChem window, 0.15 post-rank | popularity feature + 0.15 blend | equal; gate it on an S2-style split |
| Ranker | two engines: LightGBM over feature families + HistGBM 31 features, RRF fusion, rank-2 insert, top-1 protection | one HistGBM (29 features, CV'd), fusion fallback | second independent engine + RRF |
| Class 3 | nothing that scores (generator exists, 7/130) | nothing | open for everyone |
| Validation | S1/S2 splits, bootstrap CIs, seeds averaged | split v3 + 250-NP panel, grouped CV | add an S2 split (enveda-180, no public spectra) |

## 6. Plan to the top 10 (ordered; each step has a measurable gate)

Target: ≥0.47 public with a pipeline validated on an S2-style split, so the private LB does not take it back.

1. **Ship v8 this week** (reranker + PubChem tier + pruning fix + popularity v2). Gate: grouped-CV MRR on split v3 and the panel; expect LB 0.25–0.32. *Running.*
2. **Build the S2 split** (enveda-180 structures with no public spectra, answer only in PubChem/COCONUT) next to the NP panel; report both for every change. Apply the electron-mass correction to train. Gate: popularity prior's gain on S2, not S1.
3. **ICEBERG + GLACIER re-scoring** of same-formula groups among our top-60, with N1A adduct proxies; reuse the public offline packages (MIT, MassSpecGym weights, allowed). Gate: panel + S2 ≥ +0.02. Expected LB +0.02–0.04.
4. **Second engine + RRF**: a contrastive JESTR-style head on FPNet's encoder with same-formula hard negatives, fused by reciprocal rank with top-1 protection. Gate: S2 ≥ +0.02. Expected LB +0.02–0.03.
5. **Adduct-shifted library matching, timsTOF representatives, FPNet ensemble (single + merged), ChEBI/LMSD.** Cheap, each +0.005–0.01.
6. **ICEBERG atlas as an in-silico library** for class 2: pre-compute predicted spectra for our pool + PubChem window structures within the 9-h budget via a private dataset. Gate: S2 recall and MRR. Potentially the largest class-2 lever after the simulators.
7. **Class 3 is where the top 10 is decided** if Bull's 52% estimate is even half right. Two allowed routes: analog-seeded generation (MetGenX pattern: retrieve 12–30 spectrally similar *structures*, generate from their fingerprints with the mass-shell constraint, re-rank by template similarity) and **FRIGID** checkpoints (allowed) with ICEBERG refinement. Insert generated candidates only below database candidates and only when the reranker's calibrated confidence in the top candidate is low. Gate: Tanimoto ≥ 0.7-disjoint class-3 panel.
8. **Confidence gating** (Percolator-style calibrated probabilities) to decide PubChem/de-novo insertion depth per molecule instead of fixed slots; Bull's result shows fixed slots hurt.

Effort: steps 1–2 this week, 3–5 the week after, 6–8 the two weeks after that. Deadline 2026-12-14; there is time for two full iterations.

## 7. Sources read this round

Discussions 750330 (host: NC data allowed), 748270 (table of allowed data), 748882 (host answers: FRIGID, TabPFN, PubChem counts, electron mass, scorer), 745134 (ICEBERG atlas usable), 748524 (Bull's validation and class shares), 749251 (Kino/Aryan on what transfers), 749908 (pinned image), 750241 (class-3 generation potential), 750491 (leak autopsy), 746430 (FRIGID licence). Notebooks `ahmedberatozer/casmi26-v14-pubchemlite`, `seyitkaangunes/casmi26-e006-n1a-cc-by-chebi-lb-0-452`, `flexonafft/casmi26-closing-the-evidence-gap`. Papers: FRIGID arXiv 2604.16648; the rest in `docs/09-literature-review-2026-10-13.md`.
