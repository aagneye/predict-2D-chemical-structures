"""Fit the bagged GBM reranker from a training matrix and save it.

Example:
    python scripts/train_ranker.py --data data/processed/rank_train.npz \\
        --out checkpoints/ranker.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from casmi.channels.ranker import train_ranker
from casmi.config import CFG, RankerConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="output of scripts/build_rank_train.py")
    parser.add_argument("--out", required=True, help="output .pkl path")
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=None)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=False)
    x, y = data["X"], data["y"]
    print(f"[INFO] loaded {x.shape[0]:,} rows x {x.shape[1]} features, {y.sum():,} positive")

    base = CFG.ranker
    config = RankerConfig(
        max_depth=args.max_depth or base.max_depth,
        max_iter=args.max_iter or base.max_iter,
        learning_rate=args.learning_rate or base.learning_rate,
        min_samples_leaf=args.min_samples_leaf or base.min_samples_leaf,
        l2_regularization=base.l2_regularization,
        seeds=base.seeds,
        class1_priors=base.class1_priors,
    )
    print(
        f"[INFO] training {len(config.class1_priors)} priors x {len(config.seeds)} seeds "
        f"= {len(config.class1_priors) * len(config.seeds)} models"
    )

    reranker = train_ranker(x, y, config=config)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reranker.save(out_path)
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
