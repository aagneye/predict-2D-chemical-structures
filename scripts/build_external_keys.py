"""Scorer keys of every structure in the external databases (COCONUT, PubChemLite...).

Used by ``run_baseline.py --realistic-pool`` / ``build_rank_train.py --realistic-pool``:
a held-out class-2 structure stays in the candidate pool only if an external
database contains it, as for the hidden test's class 2. Keys are recomputed from
SMILES (stereo-stripped, tautomer-canonical InChIKey14) because the stored keys in
these files disagree with the scorer for ~10% of COCONUT.

Example:
    python scripts/build_external_keys.py --csv external/coconut/coconut_candidates.csv \\
        --csv external/pubchemlite/pubchemlite_candidates.csv --out external_keys.txt --workers 48
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

from casmi.chem import inchikey14


def read_smiles(path: str) -> list[str]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        column = next(
            (c for c in ("smiles", "SMILES", "canonical_smiles") if c in (reader.fieldnames or [])),
            None,
        )
        if column is None:
            raise SystemExit(f"{path}: no SMILES column in {reader.fieldnames}")
        return [row[column] for row in reader if row.get(column)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", required=True, help="CSV with a SMILES column")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    smiles: list[str] = []
    for path in args.csv:
        rows = read_smiles(path)
        print(f"[INFO] {path}: {len(rows):,} SMILES", flush=True)
        smiles.extend(rows)
    with Pool(args.workers) as pool:
        keys = pool.map(inchikey14, smiles, chunksize=512)
    unique = sorted({k for k in keys if k})
    Path(args.out).write_text("\n".join(unique) + "\n", encoding="utf-8")
    print(f"[OK] wrote {args.out}: {len(unique):,} scorer keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
