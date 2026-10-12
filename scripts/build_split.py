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
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from casmi.chem import inchikey14
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
        "--scorer-keys",
        action="store_true",
        help="split by the scorer key (stereo-stripped tautomer-canonical InChIKey14 of each "
        "structure's SMILES) instead of the stored inchikey14 column",
    )
    parser.add_argument(
        "--query-libs",
        action="append",
        default=[],
        help="for --holdout-lib structures, only spectra from these libraries are queries; "
        "their other spectra are dropped. Defaults to the --holdout-lib libraries",
    )
    parser.add_argument(
        "--reuse-holdout",
        default=None,
        help="an earlier split .npz whose hold-out structures (and classes) are kept, so a "
        "model trained against it stays leak-free; only the masks are recomputed",
    )
    parser.add_argument("--workers", type=int, default=8)
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
    columns = ["inchikey14", "ingest_lib"] + (["normalized_smiles"] if args.scorer_keys else [])
    table = pq.read_table(args.train, columns=columns)
    keys = np.asarray(table.column("inchikey14").to_pylist(), dtype=object)
    libs = np.asarray(table.column("ingest_lib").to_pylist(), dtype=object)
    print(f"[INFO] {len(keys):,} spectra, {len(set(keys)):,} unique structures")

    row_groups = None
    if args.scorer_keys:
        smiles = np.asarray(table.column("normalized_smiles").to_pylist(), dtype=object)
        unique = sorted({s for s in smiles if s})
        print(f"[INFO] computing scorer keys for {len(unique):,} SMILES "
              f"with {args.workers} workers", flush=True)
        with Pool(args.workers) as pool:
            scorer = pool.map(inchikey14, unique, chunksize=256)
        lookup = dict(zip(unique, scorer, strict=True))
        row_groups = np.asarray([lookup.get(s) or k for s, k in zip(smiles, keys, strict=True)],
                                dtype=object)
        changed = int(np.sum(row_groups != keys))
        print(f"[INFO] scorer key differs from the stored key on {changed:,} rows; "
              f"{len(set(row_groups)):,} scored identities")
        keys = row_groups

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
    query_libs = None
    if args.holdout_lib:
        wanted = set(args.holdout_lib)
        forced = {k for k, lib in zip(keys, libs, strict=True) if lib in wanted and k}
        query_libs = set(args.query_libs) if args.query_libs else wanted
        print(f"[INFO] forcing {len(forced):,} structures from {sorted(wanted)} into class 2; "
              f"queries from {sorted(query_libs)} only")

    fixed = None
    if args.reuse_holdout:
        prev = np.load(args.reuse_holdout, allow_pickle=False)
        fixed = dict(zip([str(k) for k in prev["holdout_keys"]],
                         [int(c) for c in prev["novelty_class"]], strict=True))
        if row_groups is not None:
            # Translate the earlier split's stored keys into scorer keys.
            stored_to_scorer = dict(zip(
                np.asarray(table.column("inchikey14").to_pylist(), dtype=object),
                row_groups, strict=True))
            fixed = {stored_to_scorer.get(k, k): c for k, c in fixed.items()}
        print(f"[INFO] reusing {len(fixed):,} hold-out structures from {args.reuse_holdout}")

    config = SplitConfig(n_holdout=args.n_holdout, random_seed=args.seed)
    split = make_split(
        keys, libs, config, database_keys=database_keys, forced_class2_keys=forced,
        query_libs=query_libs, row_groups=None, fixed_holdout=fixed,
    )
    if row_groups is not None:
        split.row_key = keys

    # Hard gate: refuse to emit a split that leaks.
    verify_no_leakage(split, keys)
    print("[OK] leakage check passed")

    summary = split.summary()
    for name, value in summary.items():
        print(f"  {name}: {value:,}")

    # Report the library composition of the hold-out, since an NP-weighted
    # sample is the whole point and a silent weighting regression would be
    # invisible in the aggregate MRR later.
    if split.dropped_mask is not None and split.dropped_mask.any():
        n_dropped = int(split.dropped_mask.sum())
        print(f"[INFO] dropped {n_dropped:,} non-query spectra of panel structures")
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
        **({"row_key": np.asarray(split.row_key, dtype=str)} if split.row_key is not None else {}),
    )
    print(f"[OK] wrote {out_path}")
    if forced:
        # The panel's keys, one per line, so run_baseline.py --cohort-keys can
        # report the forced library (e.g. enveda-np-examples) on its own.
        panel_path = out_path.with_suffix(".panel_keys.txt")
        panel_path.write_text("\n".join(sorted(forced)) + "\n", encoding="utf-8")
        print(f"[OK] wrote {panel_path} ({len(forced):,} panel keys)")

    with open(out_path.with_suffix(".summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "holdout_composition": composition}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
