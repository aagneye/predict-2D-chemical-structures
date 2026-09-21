"""Train the de novo decoder on the split's train side. Run on the Azure box.

Trains only on ``train_mask`` structures, for the same leakage reason
``train_fpnet.py`` requires the split — this dataset's compounds recur across
all 11 libraries, and training on a held-out structure invalidates every
validation number that follows.

Unlike FPNet (which needs one row per *spectrum*, since the model consumes
peaks), this decoder needs only one row per *unique structure* — it learns
fingerprint -> SMILES, which does not depend on any particular spectrum.

Example:
    python scripts/train_denovo.py --train data/raw/train.parquet \\
        --split data/processed/split.npz --out checkpoints/denovo \\
        --max-steps 15000 --batch-size 128
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from casmi.chem import FingerprintCalculator
from casmi.models.denovo import DenovoConfig
from casmi.models.smiles_tokenizer import SmilesVocabulary
from casmi.models.train_denovo import DenovoTrainingConfig, build_denovo_training_rows, train_denovo

#: Per-worker calculator, initialised once per process (RDKit generators are
#: not picklable, matching train_fpnet.py's own pattern).
_CALC: FingerprintCalculator | None = None


def _init_worker() -> None:
    global _CALC
    _CALC = FingerprintCalculator()


def _fingerprint(smiles: str):
    assert _CALC is not None
    return _CALC.from_smiles(smiles)


def fingerprint_structures(smiles: list[str], workers: int) -> list[np.ndarray | None]:
    """Fingerprint structures, in parallel when ``workers > 1``."""
    if workers <= 1:
        calculator = FingerprintCalculator()
        return [calculator.from_smiles(s) for s in smiles]
    with Pool(workers, initializer=_init_worker) as pool:
        return pool.map(_fingerprint, smiles, chunksize=256)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--split", required=True, help="required: prevents training on hold-out")
    parser.add_argument("--out", default="checkpoints/denovo")
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--max-rows", type=int, default=None, help="cap structures, for smoke tests"
    )
    parser.add_argument("--val-rows", type=int, default=2_000)
    parser.add_argument("--max-length", type=int, default=120, help="max SMILES token length")
    parser.add_argument("--device", default=None)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-encoder-layers", type=int, default=6)
    parser.add_argument("--n-decoder-layers", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument(
        "--max-fingerprint-bits",
        type=int,
        default=512,
        help="cap on encoder input sequence length (number of set fingerprint bits)",
    )
    parser.add_argument("--bit-flip-rate", type=float, default=0.02)
    parser.add_argument(
        "--fp-workers",
        type=int,
        default=8,
        help="processes for fingerprinting; set to core count on a big box",
    )
    args = parser.parse_args()

    split_data = np.load(args.split, allow_pickle=False)
    train_mask = split_data["train_mask"]
    print(f"[INFO] split: {train_mask.sum():,} of {train_mask.size:,} rows on the train side")

    table = pq.read_table(args.train, columns=["inchikey14", "normalized_smiles"])
    frame = table.to_pandas()
    frame = frame[np.asarray(train_mask, dtype=bool)].dropna().drop_duplicates("inchikey14")
    if args.max_rows:
        frame = frame.iloc[: args.max_rows]
    smiles_list = frame["normalized_smiles"].tolist()
    print(f"[INFO] {len(smiles_list):,} unique train-side structures")

    print(f"[INFO] fingerprinting with {args.fp_workers} workers", flush=True)
    fingerprints = fingerprint_structures(smiles_list, workers=args.fp_workers)
    calculator = FingerprintCalculator()
    print(f"[OK] fingerprint width: {calculator.n_bits} bits", flush=True)

    print("[INFO] building vocabulary from the training corpus", flush=True)
    vocabulary = SmilesVocabulary.from_smiles_corpus(smiles_list)
    print(f"[OK] vocabulary size: {vocabulary.size} tokens", flush=True)

    fp_by_smiles = {
        s: fp for s, fp in zip(smiles_list, fingerprints, strict=True) if fp is not None
    }
    rows = build_denovo_training_rows(
        list(fp_by_smiles), fp_by_smiles.get, vocabulary, max_length=args.max_length
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

    model_config = DenovoConfig(
        n_bits=calculator.n_bits,
        vocab_size=vocabulary.size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        ffn_dim=args.ffn_dim,
        max_length=args.max_length,
        max_fingerprint_bits=args.max_fingerprint_bits,
    )
    training_config = DenovoTrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        bit_flip_rate=args.bit_flip_rate,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    train_denovo(
        training_rows,
        model_config,
        vocabulary,
        training_config,
        device=args.device,
        output_dir=args.out,
        validation_rows=validation_rows,
    )
    print(f"[OK] checkpoints in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
