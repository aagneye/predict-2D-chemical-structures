"""Build the candidate structure pool and precompute fingerprints.

Sources are COCONUT (natural-product specific) and the training structures.
Blind PubChem expansion is intentionally not offered: it degraded ranking for
the reference solution by flooding the pool with decoys. ChEBI/LipidMaps can be
added via ``--extra-smiles`` but must earn inclusion on our own split first.

Run on the Azure box; the resulting ``.npz`` is small enough to attach to the
Kaggle notebook as a Dataset.

Example:
    python scripts/build_pool.py --train data/raw/train.parquet \\
        --coconut data/raw/coconut.csv --out data/processed/pool.npz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow.parquet as pq

from casmi.candidates.pool import build_pool, merge_pools
from casmi.chem import FingerprintCalculator


def read_smiles_column(path: str, column: str | None = None) -> list[str]:
    """Read SMILES from a CSV, auto-detecting a likely column name."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        name = column
        if name is None:
            for candidate in ("smiles", "SMILES", "canonical_smiles", "normalized_smiles"):
                if candidate in reader.fieldnames:
                    name = candidate
                    break
        if name is None:
            raise ValueError(f"no SMILES column found in {path}; columns={reader.fieldnames}")
        return [row[name] for row in reader if row.get(name)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=None, help="train.parquet, for its structures")
    parser.add_argument("--coconut", default=None, help="COCONUT CSV with a SMILES column")
    parser.add_argument("--smiles-column", default=None)
    parser.add_argument(
        "--extra-smiles",
        action="append",
        default=[],
        help="additional CSV sources (ChEBI, LipidMaps). Validate before relying on these.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress-every", type=int, default=20_000)
    args = parser.parse_args()

    if not args.train and not args.coconut and not args.extra_smiles:
        parser.error("supply at least one of --train, --coconut, --extra-smiles")

    calculator = FingerprintCalculator()
    print(f"[INFO] fingerprint layout: {calculator.n_bits} bits")
    pools = []

    # COCONUT first so it wins identity collisions: its SMILES are curated
    # natural-product representations.
    if args.coconut:
        smiles = read_smiles_column(args.coconut, args.smiles_column)
        print(f"[INFO] COCONUT: {len(smiles):,} SMILES")
        pools.append(build_pool(smiles, calculator, progress_every=args.progress_every))
        print(f"[OK] COCONUT pool: {len(pools[-1]):,} unique structures")

    for path in args.extra_smiles:
        smiles = read_smiles_column(path, args.smiles_column)
        print(f"[INFO] {path}: {len(smiles):,} SMILES")
        pools.append(build_pool(smiles, calculator, progress_every=args.progress_every))
        print(f"[OK] pool: {len(pools[-1]):,} unique structures")

    if args.train:
        table = pq.read_table(args.train, columns=["inchikey14", "normalized_smiles"])
        frame = table.to_pandas().dropna().drop_duplicates("inchikey14")
        print(f"[INFO] train structures: {len(frame):,}")
        pools.append(
            build_pool(
                frame["normalized_smiles"].tolist(),
                calculator,
                keys=frame["inchikey14"].tolist(),
                progress_every=args.progress_every,
            )
        )
        print(f"[OK] train pool: {len(pools[-1]):,} unique structures")

    pool = merge_pools(*pools) if len(pools) > 1 else pools[0]
    print(
        f"[OK] merged pool: {len(pool):,} structures, "
        f"mass {pool.mass.min():.2f}-{pool.mass.max():.2f} Da"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pool.save(out)
    print(f"[OK] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
