"""Train FPNet on the split's train side. Run on the Azure 4x T4 box.

Trains only on ``train_mask`` rows. Training on unfiltered ``train.parquet``
would include held-out structures and inflate every validation number that
follows, so the split is a required argument rather than an option.

Example:
    python scripts/train_fpnet.py --train data/raw/train.parquet \\
        --split data/processed/split.npz --out checkpoints/fpnet \\
        --max-steps 20000 --batch-size 256
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from casmi.chem import FingerprintCalculator
from casmi.models.fpnet import FPNetConfig
from casmi.models.train import TrainingConfig, build_training_rows, train_fpnet

#: Per-worker calculator, initialised once per process. RDKit generators are not
#: picklable, so they are constructed in the worker rather than passed in.
_CALC: FingerprintCalculator | None = None


def _init_worker() -> None:
    global _CALC
    _CALC = FingerprintCalculator()


def _fingerprint(smiles: str):
    assert _CALC is not None
    return _CALC.from_smiles(smiles)


def fingerprint_structures(
    smiles: list[str], keys: list[str], workers: int
) -> dict[str, np.ndarray]:
    """Fingerprint unique structures, in parallel when workers > 1.

    Fingerprinting ~276k structures dominates startup, and the box has 64
    cores, so this is worth parallelising: it turns ~20 minutes of single-core
    work into well under a minute.
    """
    if workers <= 1:
        calculator = FingerprintCalculator()
        results = [calculator.from_smiles(s) for s in smiles]
    else:
        with Pool(workers, initializer=_init_worker) as pool:
            results = pool.map(_fingerprint, smiles, chunksize=256)
    return {k: fp for k, fp in zip(keys, results, strict=True) if fp is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--split", required=True, help="required: prevents training on hold-out")
    parser.add_argument("--out", default="checkpoints/fpnet")
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-rows", type=int, default=None, help="cap rows, for smoke tests")
    parser.add_argument("--val-rows", type=int, default=2_000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-multi-gpu", action="store_true")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument(
        "--fp-workers",
        type=int,
        default=8,
        help="processes for fingerprinting; set to core count on a big box",
    )
    args = parser.parse_args()

    split_data = np.load(args.split, allow_pickle=False)
    train_mask = split_data["train_mask"]
    print(f"[INFO] training on {train_mask.sum():,} of {train_mask.size:,} rows")

    columns = [
        "inchikey14",
        "normalized_smiles",
        "adduct",
        "ionization_mode",
        "instrument_type",
        "precursor_mz",
        "collision_energy_ev",
        "ms2_mzs",
        "ms2_normalized_intensities",
    ]
    available = set(pq.read_schema(args.train).names)
    table = pq.read_table(args.train, columns=[c for c in columns if c in available])

    # Fingerprint each unique structure once; a structure has many spectra and
    # re-fingerprinting per spectrum would dominate setup time.
    calculator = FingerprintCalculator()
    structures = (
        table.select(["inchikey14", "normalized_smiles"])
        .to_pandas()
        .dropna()
        .drop_duplicates("inchikey14")
    )
    print(
        f"[INFO] fingerprinting {len(structures):,} unique structures "
        f"with {args.fp_workers} workers",
        flush=True,
    )
    fingerprints = fingerprint_structures(
        structures["normalized_smiles"].tolist(),
        structures["inchikey14"].tolist(),
        workers=args.fp_workers,
    )
    print(f"[OK] {len(fingerprints):,} usable fingerprints, {calculator.n_bits} bits", flush=True)

    model_config = FPNetConfig(
        n_bits=calculator.n_bits, d_model=args.d_model, n_layers=args.layers
    )
    print("[INFO] building training rows")
    rows = build_training_rows(
        table,
        fingerprints.get,
        model_config,
        row_mask=train_mask,
        max_rows=args.max_rows,
    )
    print(f"[OK] {len(rows):,} training rows")
    if not rows:
        print("[ERROR] no training rows produced; check the split and input columns")
        return 1

    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    n_val = min(args.val_rows, max(1, len(rows) // 10))
    validation_rows, training_rows = rows[:n_val], rows[n_val:]
    print(f"[INFO] {len(training_rows):,} train / {len(validation_rows):,} val rows")

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    train_fpnet(
        training_rows,
        model_config,
        training_config,
        device=args.device,
        output_dir=args.out,
        validation_rows=validation_rows,
        multi_gpu=not args.no_multi_gpu,
    )
    print(f"[OK] checkpoints in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
