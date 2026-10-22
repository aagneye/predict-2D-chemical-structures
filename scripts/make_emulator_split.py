"""Derive a test-like, class-balanced evaluation split readable by any pipeline version.

Starting from a split (e.g. split v3, keyed by the scorer key), this writes a split
whose *masks and pool exclusions alone* encode the hidden test's class definitions,
so even old versions of ``run_baseline.py`` (the code that produced v2/v7/v8 on
Kaggle) evaluate them correctly without new flags:

* only a class-stratified sample of held-out molecules keeps ``val_mask``; the other
  held-out rows go to neither side (they stay out of the library);
* ``excluded_from_pool`` = every class-3 key, plus every class-2 key that no external
  database holds (test-like class 2 is reachable only via COCONUT/PubChemLite/PubChem);
* ``holdout_keys``/``novelty_class`` list both the scorer key and every stored
  ``inchikey14`` alias, because old scripts group queries by the stored key.

Example:
    python scripts/make_emulator_split.py --split split_v3.npz --train train.parquet \\
        --external-keys external_keys.txt --per-class 200 --out split_emu.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from casmi.eval.realism import load_keys, stratified_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--external-keys", required=True)
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    split = np.load(args.split, allow_pickle=False)
    train_mask = split["train_mask"].copy()
    val_mask = split["val_mask"].copy()
    holdout = [str(k) for k in split["holdout_keys"]]
    novelty = dict(zip(holdout, [int(c) for c in split["novelty_class"]], strict=True))
    stored = np.asarray(
        pq.read_table(args.train, columns=["inchikey14"]).column("inchikey14").to_pylist(),
        dtype=object,
    )
    row_key = (
        np.asarray(split["row_key"], dtype=object) if "row_key" in split.files else stored
    )

    sample = set(stratified_sample(novelty, args.per_class, args.seed))
    keep_rows = np.array([k in sample for k in row_key]) & val_mask
    val_mask = keep_rows

    aliases: dict[str, set[str]] = {k: {k} for k in sample}
    for scorer, stored_key in zip(row_key[val_mask], stored[val_mask], strict=True):
        if stored_key:
            aliases[scorer].add(stored_key)

    external = load_keys(args.external_keys)
    excluded: set[str] = set()
    for key in sample:
        keys = aliases[key]
        if novelty[key] == 3 or (novelty[key] == 2 and not keys & external):
            excluded |= keys
    # Class-3 keys outside the sample must stay unreachable too.
    excluded |= {k for k, c in novelty.items() if c == 3}

    out_keys, out_classes = [], []
    for key in sorted(sample):
        for alias in sorted(aliases[key]):
            out_keys.append(alias)
            out_classes.append(novelty[key])

    np.savez_compressed(
        args.out,
        train_mask=train_mask,
        val_mask=val_mask,
        holdout_keys=np.asarray(out_keys, dtype=str),
        novelty_class=np.asarray(out_classes, dtype=np.int8),
        excluded_from_pool=np.asarray(sorted(excluded), dtype=str),
        row_key=np.asarray(row_key, dtype=str),
    )
    counts = {c: sum(1 for k in sample if novelty[k] == c) for c in (1, 2, 3)}
    reachable2 = sum(1 for k in sample if novelty[k] == 2 and aliases[k] & external)
    print(f"[OK] wrote {args.out}: {len(sample)} molecules {counts}; "
          f"class 2 reachable via external DBs: {reachable2}/{counts[2]}; "
          f"{len(excluded):,} keys excluded from the pool")
    Path(args.out).with_suffix(".sample.txt").write_text("\n".join(sorted(sample)) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
