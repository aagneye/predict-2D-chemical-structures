"""Scorer keys of every structure in the external databases (COCONUT, PubChemLite...).

Used by ``run_baseline.py --realistic-pool`` / ``build_rank_train.py --realistic-pool``:
a held-out class-2 structure stays in the candidate pool only if an external
database contains it, as for the hidden test's class 2. Keys are recomputed from
SMILES (stereo-stripped, tautomer-canonical InChIKey14) because the stored keys in
these files disagree with the scorer for ~10% of COCONUT.

``--csv PATH:FLAG`` keeps only rows whose ``FLAG`` column is non-zero (e.g. the
``np_coconut`` flag of ``pool_popularity.csv``); ``--keys PATH:COLUMN`` adds a
column of ready-made InChIKey14s from a CSV or parquet table as they are.

Example:
    python scripts/build_external_keys.py --csv pool_popularity.csv:np_coconut \
        --keys pubchemlite_annotations.parquet:ik14 --out external_keys.txt --workers 48
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

from casmi.chem import inchikey14


def split_spec(spec: str) -> tuple[str, str | None]:
    """``path[:column]``; a drive-letter colon is part of the path."""
    path, sep, column = spec.rpartition(":")
    if not sep or not path or "/" in column or "\\" in column:
        return spec, None
    return path, column


def read_smiles(path: str, flag: str | None = None) -> list[str]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        column = next((c for c in ("smiles", "SMILES", "canonical_smiles") if c in fields), None)
        if column is None:
            raise SystemExit(f"{path}: no SMILES column in {fields}")
        if flag is not None and flag not in fields:
            raise SystemExit(f"{path}: no column {flag!r} in {fields}")
        return [
            row[column] for row in reader
            if row.get(column) and (flag is None or row[flag] not in ("", "0", "0.0"))
        ]


def read_keys(path: str, column: str) -> set[str]:
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        values = pq.read_table(path, columns=[column]).column(column).to_pylist()
    else:
        with open(path, newline="", encoding="utf-8") as handle:
            values = [row[column] for row in csv.DictReader(handle)]
    return {v for v in values if v}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", default=[],
                        help="CSV with a SMILES column, optionally PATH:FLAG_COLUMN")
    parser.add_argument("--keys", action="append", default=[],
                        help="PATH:COLUMN of ready-made InChIKey14s (CSV or parquet)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not args.csv and not args.keys:
        parser.error("need at least one --csv or --keys")

    smiles: list[str] = []
    for spec in args.csv:
        path, flag = split_spec(spec)
        rows = read_smiles(path, flag)
        print(f"[INFO] {path}: {len(rows):,} SMILES" + (f" with {flag}" if flag else ""), flush=True)
        smiles.extend(rows)
    unique: set[str] = set()
    if smiles:
        with Pool(args.workers) as pool:
            unique |= {k for k in pool.map(inchikey14, smiles, chunksize=512) if k}
    for spec in args.keys:
        path, column = split_spec(spec)
        if column is None:
            parser.error(f"--keys needs PATH:COLUMN, got {spec!r}")
        keys = read_keys(path, column)
        print(f"[INFO] {path}: {len(keys):,} keys from {column}", flush=True)
        unique |= keys
    Path(args.out).write_text("\n".join(sorted(unique)) + "\n", encoding="utf-8")
    print(f"[OK] wrote {args.out}: {len(unique):,} keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
