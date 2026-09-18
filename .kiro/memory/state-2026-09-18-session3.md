# Session State — 2026-09-18 (Session 3: Azure training + Kaggle submission)

## What happened this session

Continuation of session 2 (implementation). This session: pushed session 2's
code, provisioned the Azure 4x T4 box, downloaded the real competition data,
trained FPNet on it, ran the baseline pipeline against the real test set, and
submitted a notebook to the Kaggle leaderboard.

### 1. Pushed session 2's code

11 commits (`71351eb`..`9c39083`) pushed to `origin/main`. This was the first
push of actual implementation code — session 1's docs-only commit `9f19493`
was already on the remote; everything since is new.

### 2. Provisioned the Azure GPU box

Found via `az vm list -d`: `rogii-gpu` (resource group `ROGII-RG`,
`Standard_NC64as_T4_v3` = 4x Tesla T4 16GB, 64 cores, 432GB RAM), region
eastus, already running, IP `20.51.160.183`, user `azureuser`. SSH key
already present locally (`~/.ssh/id_ed25519`), passwordless sudo on the box.

**Root disk was at 97% full (8.8GB free)** — all work had to go on `/mnt`
(2.6TB, Azure's ephemeral/temp disk). System python was only 3.10; used `uv
venv --python 3.12` to get a matching interpreter. Installed torch 2.5.1+cu121
(system had no CUDA torch). Setup script: `scripts/azure_setup.sh`.

**IMPORTANT / RISK**: `/mnt` contents are lost if the VM is deallocated.
Mid-session this was caught and fixed: copied `fpnet_final.pt` (165MB),
`history.json`, `pool.npz` (104MB) and `split.npz` (55KB) to
`~/casmi_checkpoints/` on the persistent home directory. **Root disk is now
at 99% full (3GB free)** after that copy — this needs attention next session,
either by resizing the disk or moving less-critical data off it.

### 3. Downloaded the real competition data

`kaggle competitions download` on the Azure box (had `~/.kaggle/kaggle.json`
copied over with `chmod 600`). Confirmed real schema matches our docs exactly:

- `train.parquet`: 2,539,608 rows, 18 columns (docs/02-dataset.md said ~2.5M —
  correct)
- `test.parquet`: 1,213 rows / 400 molecules (docs said ~1,500 spectra; close
  enough, was always described as approximate)
- Both column lists match `docs/02-dataset.md` exactly, no surprises

### 4. Ran the leak-proof split on real data

`scripts/build_split.py` on real train.parquet: 275,810 unique structures,
2,000 held out (500/900/600 across classes 1/2/3), leakage gate passed. NP
weighting **measurably worked on real data**: hold-out composition was gnps
1,207 / enveda-180 251 (12.5%, vs the ~46% raw-count share it has in train) /
pluskal_ms2 150 / spectraverse 98 / mona 87 / drug_plus 72 / massbank 68 /
riken 44 / msdial 23. This is the single clearest piece of evidence the split
design (session 2) actually does what it was built to do.

### 5. Built the candidate pool on real data

`scripts/build_pool.py --train train.parquet` (no COCONUT source used this
session — only train structures). Result: 275,810 structures, 108MB
(`.npz`), fingerprinting parallelised across 64 cores (added this session,
see below) — took well under a minute instead of the ~20 min a single-core
loop would have needed.

### 6. Trained FPNet on real data (Azure 4x T4)

**Changes made to scripts/train_fpnet.py before the real run**:
- Parallelised fingerprinting via `multiprocessing.Pool` (RDKit generators
  aren't picklable, so each worker constructs its own `FingerprintCalculator`)
- Default batch size 64 -> 256 (64 under 4-way DataParallel was only 16/GPU,
  too small to keep T4s busy)
- Added `scripts/launch_training.sh` (detached via `setsid nohup`, survives
  SSH disconnect, refuses to double-launch)
- Added `scripts/training_status.py` (parses the log, reports throughput,
  loss/val trend, and ETA from a trailing-window rate estimate)

**Run**: 30,000 steps, batch 256, on 2,504,652 train-side rows (99.7% of
2,515,809 available after the split's leakage exclusions), held out 5,000 for
periodic validation. ~3.14 steps/sec sustained, all 4 GPUs at 86-94%
utilization throughout (compute-bound, not data-loader-bound). Total wall
time ~159 min (2h 39m).

**Results**:
- Loss: 0.8704 -> 0.2804
- Validation cosine similarity: 0.5115 (step 1000) -> 0.7110 (step 29,900,
  last of 29 evals) — still climbing at the end, so more steps would likely
  help further; 30k was not chosen from a convergence study, just a
  reasonable first budget.
- Checkpoints every 2000 steps in `/mnt/casmi/checkpoints/fpnet/`;
  `fpnet_final.pt` is 165MB (d=512, 6 layers, 8 heads, 10,407-bit output).
- **Copied to `~/casmi_checkpoints/fpnet_final.pt`** on persistent storage
  (see risk note above).

**NOT YET DONE**: this checkpoint is trained but **not wired into
`pipeline.py`**. `score_candidates()` exists and is unit-tested
(session 2), but nothing in the baseline pipeline calls it yet. The baseline
submission made this session used only Channels 1+2, no neural channel.

### 7. Ran the baseline pipeline against the REAL test.parquet (Azure box)

New script this session: `scripts/predict_test.py` (predicts on the actual
competition test set and writes a submission, with the same leakage
diagnostics as `run_baseline.py`).

**Result — independently reproduces the documented leak**:
```
best_library_sim   mean 1.0000  min 1.0000  max 1.0000
best_analog_sim    mean 0.9296
perfect library match for 100.0% of molecules
carried by library        100.0%
carried by analog/other   0.0%
```
Every one of the 400 visible test molecules has an exact spectral match in
train. This is our own pipeline, built independently, landing on the *exact*
same signature documented from the rank-1 public notebook in
`docs/05-community-intel.md` — strong independent confirmation that the
visible-test-set leak is real and not an artifact of someone else's code.

One example cross-check: our rank-1 prediction for molecule `m_005e53` was
byte-identical to the rank-1 public notebook's prediction for the same
molecule (same SMILES) — both pipelines are correctly finding the same leaked
exact answer via the same mechanism.

Submission validated locally: 400 rows, exactly 25 semicolon-separated SMILES
each, no duplicates, no empty cells, no filler-only rows.

### 8. Discovered and worked around: this is a NOTEBOOK-ONLY competition

`kaggle competitions submit -f submission.csv` returned **400 Bad Request**.
This matches what `docs/01-competition-rules.md` already said ("Submission is
through a Kaggle Notebook, committed, with Submit button") but hadn't been
operationally verified until this session. CSV-only submission is not
possible; a committed notebook is required.

**Environment probe** (`kaggle/probe/`): pushed a small notebook to discover
what Kaggle's scored-run image actually provides, since the rules require
every dependency to be pre-attached (no internet during scoring). Findings:
- Python 3.12.13, numpy 2.0.2, pandas 2.3.3, pyarrow 24.0.0, numba 0.60.0,
  sklearn 1.6.1, torch 2.10.0+cpu, scipy 1.16.3 — all present
- **RDKit is NOT present** — this was the critical unknown; our whole
  pipeline depends on it
- Only 4 CPU cores available (vs. 64 on the Azure box) — building the
  candidate pool inside the scored notebook would be wasteful; must be
  pre-built and attached
- Competition parquet files auto-mounted under
  `/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/`

**Built a Kaggle Dataset** (`aagneye/casmi26-baseline-assets`, ~140MB,
private) containing:
- `rdkit-2026.3.3-cp312-cp312-manylinux_2_28_x86_64.whl` (fetched via
  `pip download --no-deps` on the Azure box, matching our pinned version)
- `pool.npz` (the 275,810-structure candidate pool)
- `src/casmi/` (our package source, copied directly — not pip-packaged)

Assembly script: `scripts/build_kaggle_dataset.sh`.

**Submission notebook** (`kaggle/submission/casmi26-baseline.ipynb`,
kernel id `aagneye/casmi26-baseline-library-analog-propagation`): installs
RDKit from the wheel, adds our source to `sys.path`, loads the pool, runs
Channels 1+2 against the real test.parquet, writes and strictly validates
`submission.csv`, prints the same leakage diagnostics.

**First push failed**: bootstrap cell used a hardcoded dataset mount path
(`/kaggle/input/casmi26-baseline-assets`) that didn't match the actual mount,
so the RDKit install was silently skipped by an `if wheels:` guard and the
notebook died later on `import rdkit`. **Fixed** by searching all of
`/kaggle/input` recursively and raising immediately if the wheel or package
isn't found, instead of failing several cells later with a confusing error.

**Second push (version 2) succeeded**: ran to completion on Kaggle's servers.
- Setup (RDKit install + pool + library load): 52s
- Inference over 400 molecules / 1,213 spectra: 3,276s (~55 min)
- Total well within the 9h cap
- Same leakage diagnostics as the local Azure run (as expected, same code
  and data)

### 9. Submitted to the competition leaderboard

`kaggle competitions submissions` showed nothing after just running the
notebook — **running/completing a Kaggle notebook does NOT automatically
submit it**. Had to explicitly submit via:
```
kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra \
    -k aagneye/casmi26-baseline-library-analog-propagation \
    -f submission.csv -v 2 -m "<message>"
```//
(the `-k/--kernel` flag plus explicit `-f` filename and `-v` version number
were all required together; omitting any one produced an error).

**Confirmed accepted**: response was "3 submissions remaining today" (started
at 5/day per the user's plan), meaning the daily quota decremented, i.e. the
submission was accepted for scoring.

**Result as of end of session: submissions are still PENDING.**
```
ref        date                  description                              status
56335444   2026-09-18 16:49:33   Baseline: Channel1+Channel2, 275k pool    PENDING
56335438   2026-09-18 16:49:16   (no message)                             PENDING
```
Two submissions show as pending — the second (`56335438`, no message) was NOT
made by an action in this session's log; its timestamp (16:49:16, 17 seconds
before the one this session made) suggests either a duplicate side-effect of
the same `submit` call, or Kaggle queued something else. **Not yet
explained** — worth checking on the Kaggle site directly next session, since
neither the public nor private score was available before this session ended.
Given the diagnostics (100% perfect library match), expect a fairly high
public score once it resolves, but per docs/05-community-intel.md that score
would be **overwhelmingly leakage-driven** and not evidence the pipeline
generalizes — do not treat a high number here as validation of anything
beyond "Channel 1 works."

## Key numbers to remember without re-deriving

- FPNet: loss 0.8704->0.2804, val cosine 0.5115->0.7110, 30k steps, batch 256,
  ~3.14 steps/sec on 4x T4, 159 min wall time. Checkpoint at
  `~/casmi_checkpoints/fpnet_final.pt` on the Azure box (persistent) and
  `/mnt/casmi/checkpoints/fpnet/` (ephemeral, same content).
- Candidate pool: 275,810 structures (train-only this session, no COCONUT),
  108MB, 10,407-bit fingerprints.
- Split: 2,000 held out of 275,810 structures, 500/900/600 across
  classes 1/2/3, NP-weighted (verified: gnps 1,207 vs enveda-180 251 in the
  hold-out, a 4.8:1 ratio despite enveda-180 having ~4x more raw spectra
  overall).
- Real test.parquet: 400 molecules, 1,213 spectra (not ~1,500 as
  approximated in docs — minor, within the "approximately the same size"
  caveat).
- Kaggle scored-run environment: Python 3.12.13, 4 CPUs, NO rdkit, has
  numpy/pandas/pyarrow/numba/sklearn/torch.
- Submission mechanism: `kaggle competitions submit -k <kernel> -f
  submission.csv -v <version> -m <message>` — plain CSV upload via API does
  NOT work for this competition (confirmed 400 error).

## Azure box operational notes for next session

- IP `20.51.160.183`, user `azureuser`, resource group `ROGII-RG`, VM name
  `rogii-gpu`. SSH key already trusted locally.
- **Root disk 99% full (3GB free) as of session end** — needs attention.
  Either resize the OS disk via `az disk update`/`az vm resize`, or move
  large non-critical files off `/`. Do not put anything large on `/` until
  this is resolved; keep using `/mnt` for working data, but remember to copy
  anything valuable off `/mnt` before any VM deallocation.
- `/mnt/casmi/repo` has the git clone (synced to `origin/main` as of this
  session's last `git reset --hard`). `/mnt/casmi/data/raw/` has the real
  train.parquet (3GB) and test.parquet — no need to re-download.
- `~/casmi_checkpoints/` holds the persistent copies: `fpnet_final.pt`,
  `history.json`, `pool.npz`, `split.npz`. These survive VM deallocation;
  `/mnt/casmi/*` does not.
- The VM was already running (not started by us) — unclear who else uses it
  or what the billing arrangement is. `Standard_NC64as_T4_v3` is roughly
  $4-5/hr; flag to the user if it should be stopped between sessions.

## Not yet done / next-session TODOs

1. **Check the actual public score** once the two PENDING submissions
   resolve, and figure out what the second unexplained submission
   (`56335438`) was.
2. **Fix the Azure root disk** (99% full) before it causes an unrelated
   failure.
3. **Wire FPNet into `pipeline.py`** — the trained checkpoint exists and
   `score_candidates()` is tested, but the baseline pipeline still only uses
   Channels 1+2. This is the next real modeling step per
   `docs/06-implementation-plan.md` step 6.
4. **Build a submission scored on Channels 1+2+FPNet** and compare against
   the Channels-1+2-only baseline on the *local held-out split* (not the
   leaky public test) to see if the neural channel actually helps once
   Channel 1 can't just "look up the answer."
5. Steps 7-9 of `docs/06-implementation-plan.md` remain: MetFrag-lite
   fragmentation channel, learned GBM reranker, de novo generation for
   class 3 (still our intended differentiator — the neural-channel addition
   above does not address this).
6. Consider whether to add COCONUT to the candidate pool (this session's pool
   was train-structures-only; `04`/`06` plans call for COCONUT too, and
   `build_pool.py` already supports `--coconut`, it just wasn't given a
   source this session).
7. Re-run `scripts/build_split.py`-based validation (`run_baseline.py`) with
   the real data now that it's on the Azure box, to get an honest (non-leaky)
   MRR@25 number for the current Channels-1+2 baseline — this session only
   ran the leaky test.parquet path, not the honest local-split path, against
   real data.

## Orientation for whoever picks this up next

Read `docs/06-implementation-plan.md` for the plan, `docs/07-codebase-guide.md`
for what exists, then this file for exactly what happened operationally this
session (Azure box state, Kaggle submission mechanics, pending score). The
Azure box has real data and a trained FPNet checkpoint sitting on it right
now — no need to re-download or re-train from scratch.
