# CLAUDE.md — AI software factory protocol for this repo

## The goal (the only success criterion)

**Top 10 on the Kaggle public leaderboard** of Enveda CASMI 2026
(MRR@25, InChIKey14 after RDKit 2026.03.3 tautomer canonicalisation, stereo stripped).
10th place was 0.484 on 2026-10-26 and rises ≈ 0.01/week; the earlier milestone
> 0.46 ranked ≈ 780 that day, so it is a checkpoint, not the goal.
Final deadline 2026-12-14 23:59 UTC; entry/merge deadline 2026-12-07.

Score history lives in `docs/factory/ledger.md`. Read it first.

## Factory loop — do this at the start of every build session

1. **Read state**: `docs/factory/ledger.md` (scores, what was tried, what failed, open
   bets), then `git log --oneline -20`, then the latest `docs/1x-*.md` status doc.
2. **Measure, don't guess**: every change is judged by the offline LB emulator
   (`scripts/lb_emulator.py`, run on the VM) before any Kaggle submission. Kaggle
   submissions are only for confirming emulator gains (~130 public molecules,
   ±0.01 noise; 5 submissions/day).
3. **Pick the highest expected gain per hour** toward > 0.46 from the ledger's
   open bets. Research (papers, competitor notebooks, forum) when the next bet is unclear.
4. **Build with tests**, atomic commits (one logical change per commit), push.
5. **Train/evaluate on the VM**, never on the local PC (it is slow and the user
   does not want long local runs). Run the test suite on the VM.
6. **Update the ledger**: what changed, emulator number, Kaggle number if submitted,
   what to do next. Keep working until the goal is met; do not stop at a plan.

## Infrastructure

- VM `rogii-gpu` (`azureuser@20.51.160.183`, 64 vCPU, 432 GB, 4x T4; shared — check
  `nvidia-smi` and use a free GPU). `/mnt` is ephemeral and is wiped when the VM is
  deallocated (auto-shutdown 18:29 UTC); everything that must survive goes to
  `~/casmi_checkpoints/`. Repo clone at `/mnt/casmi/repo` (re-clone if missing).
- Kaggle: dataset `aagneye/casmi26-baseline-assets`, kernel
  `aagneye/casmi26-4-channel-retrieval-gbm-reranker`; CLI on the VM at `~/.local/bin/kaggle`.
  Submit from the VM with `~/submit_when_done.sh` (edit the version number).
- Local `.venv` (uv, Python 3.12, RDKit 2026.03.3) for editing and quick lint only.

## Hard rules

- No NIST data or NIST-trained weights (disqualification). Non-commercial open data is
  allowed (host ruling 2026-10-17). Third-party weights need public training code.
- Never tune on the visible `test.parquet` (it is a sample of train). The public LB is
  real (31% of the hidden test).
- Validation must be structure-disjoint by the *scorer* key; class-3 truths must be
  unreachable by every retrieval channel, including PubChem (`pubchem_blocklist`).
- Pin RDKit 2026.03.3; keep ranker pickles on scikit-learn 1.6.x (Kaggle's version).
