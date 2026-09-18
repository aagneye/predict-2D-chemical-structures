"""Predict on the competition test set and write a submission CSV.

Unlike ``run_baseline.py`` (which scores against our held-out split), this runs
against the real ``test.parquet`` and produces a submission.

**Read the diagnostics it prints.** The visible ``test.parquet`` is a sample of
``train.parquet``, so Channel 1 will find near-exact matches and the resulting
public leaderboard score will be high and largely meaningless — it measures
whether a library lookup was implemented, not whether unseen molecules can be
identified. The ``library`` contribution figure quantifies exactly how much of
the score comes from that leak.

Example:
    python scripts/predict_test.py --test data/raw/test.parquet \\
        --train data/raw/train.parquet --pool data/processed/pool.npz \\
        --sample data/raw/sample_submission.csv --out submissions/baseline.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from casmi.candidates.pool import CandidatePool
from casmi.data.loaders import load_query_molecules, load_spectral_library
from casmi.pipeline import run_pipeline, write_submission


def read_sample_ids(path: str) -> list[str]:
    """Molecule ids in the order the sample submission lists them."""
    with open(path, newline="", encoding="utf-8") as handle:
        return [row["molecule_id"] for row in csv.DictReader(handle)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True)
    parser.add_argument("--train", required=True, help="library for Channels 1 and 2")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--sample", required=True, help="sample_submission.csv, for row order")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ppm", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap molecules, for smoke tests")
    args = parser.parse_args()

    started = time.time()

    print("[INFO] loading spectral library (full train set)", flush=True)
    library = load_spectral_library(args.train)
    print(f"[OK] library: {library.n_spectra:,} spectra", flush=True)

    pool = CandidatePool.load(args.pool)
    print(f"[OK] pool: {len(pool):,} structures", flush=True)

    molecules = load_query_molecules(args.test)
    if args.limit:
        molecules = molecules[: args.limit]
    print(f"[OK] {len(molecules):,} test molecules", flush=True)

    config = None
    if args.ppm is not None:
        from dataclasses import replace

        from casmi.config import CFG

        config = replace(CFG, candidates=replace(CFG.candidates, ppm_window=args.ppm))

    result = run_pipeline(molecules, library, pool, config=config, progress_every=25)

    sample_ids = read_sample_ids(args.sample)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(result.predictions, sample_ids, str(out_path))

    # Diagnostics. A library contribution near 100% is the leakage signature.
    diagnostics = result.diagnostics
    contribution = result.channel_contribution()
    lib_sims = np.array([d.best_library_similarity for d in diagnostics])
    analog_sims = np.array([d.best_analog_similarity for d in diagnostics])
    n_cands = np.array([d.n_candidates for d in diagnostics])
    empty = sum(1 for m in molecules if not result.predictions.get(m.molecule_id))

    print("\n=== Test-set diagnostics ===")
    print(f"  molecules                {len(molecules):,}")
    print(f"  with no candidates       {empty:,}")
    print(f"  mean candidates/molecule {n_cands.mean():.1f}")
    print(f"  best_library_sim  mean {lib_sims.mean():.4f}  min {lib_sims.min():.4f}")
    print(f"  best_analog_sim   mean {analog_sims.mean():.4f}")
    print(f"  library_sim == 1.0 for  {(lib_sims >= 0.9999).mean():.1%} of molecules")
    print("\n=== Channel contribution ===")
    for name, value in contribution.items():
        print(f"  {name:<18} {value:>7.1%}")

    if (lib_sims >= 0.9999).mean() > 0.5:
        print(
            "\n[WARNING] More than half of test molecules have a PERFECT library match.\n"
            "          This is the documented leak in the visible test.parquet (it is a\n"
            "          sample of train). Any public LB score from this submission mostly\n"
            "          measures library lookup, not generalisation. Do not treat it as\n"
            "          progress; iterate against scripts/run_baseline.py instead."
        )

    print(f"\n[OK] wrote {out_path} in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
