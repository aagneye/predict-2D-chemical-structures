"""Fill popularity for pool structures the pool_popularity table lacks, from the PubChem tier.

The community ``pool_popularity.csv`` covers train ∪ COCONUT (710k). Our pool
also holds PubChemLite structures, which get 0 from that table and so look
undocumented to every ranker. The full tier's ``pc_ik14.npy`` (InChIKey14 of
the stereo-free SMILES, 106M rows) with ``pc_lsid``/``pc_lpmid`` lets us look
them up by key.

Example:
    python scripts/extend_popularity.py --pool pool_v2.npz --popularity popularity.npz \\
        --tier-popularity external/pubchem_pop --out popularity_v2.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from casmi.priors import PopularityPrior


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--popularity", required=True, help="existing prior (.npz or .csv)")
    parser.add_argument(
        "--tier-popularity", required=True, help="dir with pc_ik14.npy, pc_lsid.npy, pc_lpmid.npy"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    prior = PopularityPrior.load(args.popularity)
    pool_keys = np.load(args.pool, allow_pickle=False)["keys"].astype(str)
    missing = np.array(sorted({k for k in pool_keys if k not in prior}), dtype="S14")
    print(f"[INFO] pool {len(pool_keys):,} keys; {len(missing):,} without popularity")

    tier_dir = Path(args.tier_popularity)
    ik14 = np.load(tier_dir / "pc_ik14.npy", mmap_mode="r")
    lsid = np.load(tier_dir / "pc_lsid.npy", mmap_mode="r")
    lpmid = np.load(tier_dir / "pc_lpmid.npy", mmap_mode="r")
    print(f"[INFO] tier: {len(ik14):,} keys; sorting (one pass, a few minutes)", flush=True)
    keys = np.asarray(ik14).astype("S14")
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    del keys

    pos = np.searchsorted(sorted_keys, missing)
    pos = np.clip(pos, 0, len(sorted_keys) - 1)
    found = sorted_keys[pos] == missing
    rows = order[pos[found]]
    values = np.asarray(lsid[rows], dtype=np.float32) + np.asarray(lpmid[rows], dtype=np.float32)
    table = dict(prior._table)
    for key, value in zip(missing[found], values, strict=True):
        table[key.decode("ascii")] = float(value)
    print(f"[OK] matched {int(found.sum()):,} of {len(missing):,} missing keys in the tier")

    out = PopularityPrior(table)
    out.save(args.out)
    covered = sum(1 for k in pool_keys if k in out)
    share = covered / len(pool_keys)
    print(f"[OK] wrote {args.out}: {len(out):,} structures, covers {share:.1%} of the pool")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
