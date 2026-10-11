"""Build and persist the leak-proof validation split.

Run this first. Everything downstream depends on the split, and the hard
leakage gate here is what makes later numbers trustworthy.

Example:
    python scripts/build_split.py --train data/raw/train.parquet \\
        --out data/processed/split.npz --n-holdout 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from casmi.config import SplitConfig
from casmi.data.split import make_split, verify_no_leakage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="path to train.parquet")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--n-holdout", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--database-keys",
        "--coconut-keys",
        dest="database_keys",
        default=None,
        help="structures a database can retrieve: a newline-delimited inchikey14 file or a "
        "pool .npz from scripts/build_pool.py; steers class 3 toward structures absent from it",
    )
    parser.add_argument(
        "--holdout-lib",
        action="append",
        default=[],
        help="hold out EVERY structure of this ingest_lib as class 2 (all spectra removed, "
        "structure kept in the pool), in addition to --n-holdout sampled structures. "
        "Repeatable. e.g. --holdout-lib enveda-np-examples builds the natural-product panel",
    )
    args = parser.parse_args()

    print(f"[INFO] reading structure columns from {args.train}")
    table = pq.read_table(args.train, columns=["inchikey14", "ingest_lib"])
    keys = np.asarray(table.column("inchikey14").to_pylist(), dtype=object)
    libs = np.asarray(table.column("ingest_lib").to_pylist(), dtype=object)
    print(f"[INFO] {len(keys):,} spectra, {len(set(keys)):,} unique structures")

    database_keys = None
    if args.database_keys:
        if args.database_keys.endswith(".npz"):
            pool_keys = np.load(args.database_keys, allow_pickle=False)["keys"]
            database_keys = {str(k) for k in pool_keys}
        else:
            database_keys = {
                line.strip()
                for line in Path(args.database_keys).read_text().splitlines()
                if line.strip()
            }
        print(f"[INFO] {len(database_keys):,} database keys loaded")

    forced = None
    if args.holdout_lib:
        wanted = set(args.holdout_lib)
        forced = {k for k, lib in zip(keys, libs, strict=True) if lib in wanted and k}
        print(f"[INFO] forcing {len(forced):,} structures from {sorted(wanted)} into class 2")

    config = SplitConfig(n_holdout=args.n_holdout, random_seed=args.seed)
    split = make_split(
        keys, libs, config, database_keys=database_keys, forced_class2_keys=forced
    )

    # Hard gate: refuse to emit a split that leaks.
    verify_no_leakage(split, keys)
    print("[OK] leakage check passed")

    summary = split.summary()
    for name, value in summary.items():
        print(f"  {name}: {value:,}")

    # Report the library composition of the hold-out, since an NP-weighted
    # sample is the whole point and a silent weighting regression would be
    # invisible in the aggregate MRR later.
    held = set(split.holdout_keys)
    composition: dict[str, int] = {}
    seen: set[str] = set()
    for key, lib in zip(keys, libs, strict=True):
        if key in held and key not in seen:
            seen.add(key)
            composition[lib] = composition.get(lib, 0) + 1
    print("[INFO] hold-out composition by source library:")
    for lib, count in sorted(composition.items(), key=lambda kv: -kv[1]):
        print(f"  {lib:<22} {count:>6,}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        train_mask=split.train_mask,
        val_mask=split.val_mask,
        holdout_keys=np.asarray(split.holdout_keys, dtype=str),
        novelty_class=np.asarray(
            [split.novelty_class[k] for k in split.holdout_keys], dtype=np.int8
        ),
        excluded_from_pool=np.asarray(sorted(split.excluded_from_pool), dtype=str),
    )
    print(f"[OK] wrote {out_path}")

    with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "holdout_composition": composition}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
