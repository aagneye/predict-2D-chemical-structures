# 12 — Status 2026-10-26: where the score is lost, and the plan to the top 10

Plain-language companion to `docs/factory/ledger.md`. Read the ledger for numbers that
change daily; read this for *why* and *what next*.

## 1. Where we stand

| | Public score | Rank (of 4,059) |
|---|---|---|
| Our best submitted (v8) | 0.252 | 2,886 |
| v9 (Channel 8 + 33-feature ranker), emulator forecast | 0.287 | ≈ 2,750 |
| Best public notebook, copied unchanged (submitted today) | 0.467 expected | ≈ 66 |
| 10th place | 0.484 | 10 |
| 1st place | 0.504 | 1 |

The public leaderboard is the real hidden test (≈ 130 molecules). One molecule moved to
rank 1 is worth ≈ +0.0076; the whole gap between the best public notebook and 10th place is
**two to three molecules**.

## 2. What the LB emulator taught us (round 1, 2026-10-22)

We re-ran the exact code of our three scored submissions on a test-like split and fitted
`public ≈ α · (s1·m1 + s2·m2 + s3·m3)` to their Kaggle scores (RMSE 0.004):

* **Class 2 (in a database, no spectra) carries 2.7× the weight of class 1.** Fitted
  `α·s2 = 0.31` vs `α·s1 = 0.115`. Every 0.1 of class-2 MRR is worth 0.03 on the board.
* **Our class-1 proxy is too easy.** Offline class 1 = 0.90, but it buys little on the
  board: the hidden "has public spectra" molecules are measured on other instruments and
  our library match is weaker there than at home.
* **Class 3 is unmeasured.** All three calibration points had class 3 = 0, so the
  class-3 share is a free parameter until v9 (class 3 = 0.10 offline) is scored.
* **Half of every offline gain evaporates** (α ≈ 0.5): instrument and chemistry shift.
* **Ceiling of our own stack**: perfect class 1 + class 2 ≈ 0.42. Passing 0.46 with our
  own code needs class 3 *and* a smaller domain gap, not more of the same.

## 3. Where the score is lost (largest first)

1. **Class 2 ranking (worth ≈ 0.15 to us).** We rank database candidates with FPNet,
   parsimony fragmentation and a popularity prior. The public recipes add what we lack:
   a second ranking engine fused by reciprocal rank, PubChem promotion gated on
   popularity *and* margin, ICEBERG/GLACIER forward simulation to separate same-formula
   isomers, and a light neural re-rank of the final 25. Their class 2 is far above our 0.58.
2. **Domain shift (α = 0.5).** Validation carved from `train.parquet` is optimistic: the
   forum thread "Has anyone managed to improve their Class 3 score?" (Bull, Mansur,
   Tra Crew — ranks 466/225/40) reports the same: local class-3 gains do not move the
   board, and nobody has more than +0.003…0.004 from class 3.
3. **Class 3 (worth up to 0.5 × share).** *Corrected 2026-10-26:* the public recipes do
   generate database-absent structures — engine 1 runs `EngineCfg(generate=True)`, an
   analogue generator that edits close library relatives (the panel dataset's notes say it
   rebuilds answers removed from the pool). Our Channel 8 is the same idea built
   independently (0.10 offline, recall@25 12% → 25% after the site edits). v9 showed our
   version moves the board by ≈ +0.003, so class 3 is not where the gap is.
4. **Class 1 transfer.** A library match on a different instrument needs tolerant
   similarity (energy-bucketed entropy similarity, merged spectra) — the 0.462 lineage's
   "N1A collision-energy" work is this.

## 4. What the field does (forum + public notebooks, 2026-10-26)

* One lineage holds the public frontier: prvsiyan → ahmedberatozer v4 → guhongbin /
  sanpier / dmitriigluzdov "corrected 0.465 PubChem baseline" → sanpier neural re-rank
  (weight 0.25) → alexchilton RRF offset 1 = **0.467** (Apache 2.0, pinned image,
  7 datasets). Steps that repeatedly paid (+0.008 each) all came from giving the
  ICEBERG/GLACIER simulators evidence on more molecules; ~20 other knobs were flat.
* Aryan Awasthi's post names the "real class-3 lever": retraining engine 1's ranker with
  source flags, 10–40 CPU hours per setting — Kaggle-bound teams cannot afford it; our
  64-vCPU VM can.
* Host rulings: non-commercial open data is now allowed; NIST never; third-party weights
  need their original training code to be *prize*-eligible; compliance checks have only
  ever hit prize/gold positions. The 0.467 lineage's assets (fold-safe FPNet with
  training code, PubChem prior, natural-product tables, sanpier's spectrum model) are the
  cleanest in the field, but the embedded engines still carry weights without code.

## 5. The plan — three tracks

### Track A — today: public base + our class-3 overlay (fastest to ≈ 0.47–0.48)

*Update, same day:* emulator round 2 (v9 = 0.255) and the finding that the base already
has its own analogue generator cut this track's expected gain to 0 … +0.005; see the
ledger. Kept as a one-submission experiment.

* Base: unchanged copy of the 0.467 notebook (`aagneye/casmi26-lb-0-467-base-…`,
  submitted). Also the 0.462 flexonafft recipe as a second data point.
* Overlay (`src/casmi/overlay.py`, tested): run our pipeline in the same kernel after the
  base writes `submission.csv`; for every molecule whose best library similarity is
  below 0.8, insert our top Channel 8 structure **directly below the base's rank 1**.
  Offline simulation on 2,244 split-v3 molecules: class 1/2 cost −0.001/+0.003, class 3
  0.002 → 0.054; emulator delta **+0.005 (fitted class-3 share) to +0.014 (if class 3 is
  half the test)**. Replacing rank 1 never wins.
* Cost: one extra pipeline run per kernel (≈ 15 s/molecule on the VM → ≈ 2 h on Kaggle's
  4 cores for the hidden ≈ 420 molecules; base ≈ 30 min; budget 9 h).

### Track B — this week: make Channel 8 find more (recall 12% → 25%+)

Each bet is measured on the VM with `scripts/run_baseline.py --realistic-pool` on
`split_emu`, then `scripts/lb_emulator.py`, before any submission.

1. **Parents from the base's lists.** The 0.467 recipe's top-5 are better analogs than
   our library search; derive from them (they are in `frozen_candidate_evidence.parquet`
   inside the kernel).
2. **More edits, three steps.** Add rule families from BioTransformer / SyGMa / MetGenX
   (glycosyl transfer variants, acyl positions, oxidation ladders, ring closures,
   prenyl chains, halogenation for marine NPs) with their exact mass deltas; allow 3 edits
   when the mass shift demands it. Keep every product on the measured mass (±10 ppm).
3. **Score products with forward simulators.** ICEBERG/GLACIER (both in the base's
   inputs) predict a spectrum per candidate; entropy similarity to the query separates
   positional isomers that fingerprints cannot. Replace "FPNet only" ranking of derived
   structures with FPNet + simulator agreement.
4. **Compound-class gate.** NPClassifier pathway of the parent restricts which edits are
   plausible (no glucuronide on a terpene core, etc.).
5. **Fold-safe FPNet ensemble** (dmitriigluzdov, training code public) as the fingerprint
   evidence for derived candidates — stronger and prize-clean.

### Track C — in parallel: our own reproducible stack (prize path)

Class 2 is the lever (`α·s2 = 0.31`): ICEBERG/GLACIER same-formula re-scoring, a second
engine fused by RRF, candidate-level PubChem gate, engine-2-style rank-2 insertion. This is
slower but every weight has our training code; keep it as the final private-LB entry.

## 6. Literature review — targeted, each item ends in a measurable bet

| Topic | Papers / tools | Extract | Test |
|---|---|---|---|
| Biotransformation rules | BioTransformer 3.0 (Wishart lab), SyGMa, enviPath, MetGenX (template-guided generation) | rule SMARTS + mass deltas | Channel 8 recall on class-3 rows |
| Analog networks | GNPS molecular networking; ModiFinder (modification-site localisation); MS2Query analog search | parent choice, site localisation | derived-candidate rank |
| Forward simulation | ICEBERG, GLACIER (Coley lab ms-pred), FIORA (allowed), SCARF | candidate re-scoring | class-2/3 MRR on `split_emu` |
| De novo generation | DiffMS, MADGEN, MSNovelist, Spec2Mol; MassSpecGym audit (31% → 10.7%) | honest baselines; formula-conditioned decoding | only if > 5% top-1 on class 3 |
| Formula / class priors | SIRIUS & CSI:FingerID; CANOPUS / NPClassifier | class gates for edits | fewer wrong inserts |
| Re-ranking | Percolator / Prosit rescoring; the lineage's neural re-rank (weight 0.25) | semi-supervised rerank on test | class-2 MRR |
| Domain shift | instrument transfer (timsTOF vs Orbitrap/Q-TOF); forum #743395 (+0.5 mDa positive-mode offset) | tolerant similarity, mass correction | α in the emulator |

## 7. The testing centre

Nothing is submitted on a hunch. The chain is: build → tests on the VM →
`run_baseline.py --realistic-pool` on `split_emu` (600 molecules, 200 per class, class 3
blocked everywhere) → `lb_emulator.py` → ledger → Kaggle only if the forecast beats the
last score. New today: `scripts/simulate_overlay.py` for merge rules. Still to add: the
S2 split (enveda-180 structures with no public spectra) to tighten α.

## 8. Risks

* Prize eligibility of the public lineage (weights without training code). Track C exists
  for this; the public-base submissions are for leaderboard position.
* 5 submissions/day; 9 h kernel budget; dataset-version → kernel-push race (wait a few
  minutes, or pin the version).
* Class-3 transfer: every team reports local class-3 gains that vanish on the board; v9
  is our first real measurement.
