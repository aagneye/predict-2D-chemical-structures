"""Build the candidate structure pool and precompute fingerprints.

Sources, in precedence order: COCONUT (natural-product specific), any extra
CSVs (PubChemLite, ChEBI/LIPID MAPS; PubChemLite is what unlocks class-2
molecules that COCONUT lacks), then the training structures. Blind full-PubChem
isomer expansion is intentionally not offered here: it degraded ranking for
the reference solution by flooding the pool with decoys; a separate
PubChem-only channel is the right home for it.

Every structure is keyed by the **scorer key** (stereo-stripped,
tautomer-canonical InChIKey14, :func:`casmi.chem.inchikey14`) computed from its
SMILES, never a stored ``inchikey14`` column, which disagrees with the scorer
for ~1.6% of train structures and ~10% of COCONUT. Per-structure work runs in
a process pool: ~1.1M structures take minutes on 60 cores instead of hours.

Run on the Azure box; the resulting ``.npz`` is attached to the Kaggle
notebook as a Dataset.

Example:
    python scripts/build_pool.py --train data/raw/train.parquet \\
        --coconut external/coconut_candidates.csv \\
        --extra-smiles external/pubchemlite_candidates.csv \\
        --out data/processed/pool.npz --workers 60
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from casmi.candidates.pool import CandidatePool, merge_pools, pool_from_records
from casmi.chem import FingerprintCalculator, inchikey14

#: Per-worker calculator (RDKit generators are not picklable, so each process
#: builds its own once).
_CALC: FingerprintCalculator | None = None


def _init_worker() -> None:
    global _CALC
    _CALC = FingerprintCalculator()


def _record(smiles: str) -> tuple[str | None, np.ndarray | None, float]:
    """``(scorer_key, packed_fingerprint, mass)`` for one SMILES; failures -> None/nan."""
    assert _CALC is not None
    key = inchikey14(smiles)
    if key is None:
        return None, None, float("nan")
    result = _CALC.fingerprint_and_mass(smiles)
    if result is None:
        return None, None, float("nan")
    fp, mass = result
    return key, np.packbits(fp), mass


def build_pool_parallel(smiles: list[str], workers: int, progress_every: int = 0) -> CandidatePool:
    """Fingerprint + key every SMILES across ``workers`` processes, then assemble."""
    if workers <= 1:
        _init_worker()
        records = []
        for i, smi in enumerate(smiles):
            if progress_every and i and i % progress_every == 0:
                print(f"  processed {i:,}/{len(smiles):,}", flush=True)
            records.append(_record(smi))
    else:
        with Pool(workers, initializer=_init_worker) as pool:
            records = pool.map(_record, smiles, chunksize=512)
    keys = [r[0] for r in records]
    fps = [r[1] for r in records]
    masses = [r[2] for r in records]
    return pool_from_records(keys, smiles, fps, masses, n_bits=FingerprintCalculator().n_bits)


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
    parser.add_argument(
        "--workers", type=int, default=8, help="processes for fingerprinting/keying"
    )
    args = parser.parse_args()

    if not args.train and not args.coconut and not args.extra_smiles:
        parser.error("supply at least one of --train, --coconut, --extra-smiles")

    calculator = FingerprintCalculator()
    print(f"[INFO] fingerprint layout: {calculator.n_bits} bits, {args.workers} workers")
    pools = []

    # COCONUT first so it wins identity collisions: its SMILES are curated
    # natural-product representations.
    if args.coconut:
        smiles = read_smiles_column(args.coconut, args.smiles_column)
        print(f"[INFO] COCONUT: {len(smiles):,} SMILES", flush=True)
        pools.append(build_pool_parallel(smiles, args.workers, args.progress_every))
        print(f"[OK] COCONUT pool: {len(pools[-1]):,} unique structures", flush=True)

    for path in args.extra_smiles:
        smiles = read_smiles_column(path, args.smiles_column)
        print(f"[INFO] {path}: {len(smiles):,} SMILES", flush=True)
        pools.append(build_pool_parallel(smiles, args.workers, args.progress_every))
        print(f"[OK] pool: {len(pools[-1]):,} unique structures", flush=True)

    if args.train:
        table = pq.read_table(args.train, columns=["inchikey14", "normalized_smiles"])
        frame = table.to_pandas().dropna().drop_duplicates("inchikey14")
        print(f"[INFO] train structures: {len(frame):,}", flush=True)
        # Keys are recomputed from SMILES (scorer key), not taken from the
        # stored column; see the module docstring.
        pools.append(
            build_pool_parallel(
                frame["normalized_smiles"].tolist(), args.workers, args.progress_every
            )
        )
        print(f"[OK] train pool: {len(pools[-1]):,} unique structures", flush=True)

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
