"""Honest reranker evaluation: grouped 5-fold cross-validation on rank_train.npz.

Training the reranker on the held-out molecules and then scoring the same
molecules (what the old 0.592 number did) is optimistic. This script trains
on four folds of molecules and ranks the fifth, so every molecule's
reciprocal rank comes from a model that never saw it, then fits the final
ensemble on all rows and saves it.

Reported MRR@25 is over the molecules present in ``rank_train.npz``; a
molecule whose true structure is not among its candidates scores 0 like in
the real metric. Molecules the pipeline produced no candidates for are not
in the file at all (build_rank_train.py prints how many), so compare with
``run_baseline.py`` numbers only after accounting for them.

Example:
    python scripts/cv_ranker.py --data data/processed/rank_train.npz \\
        --split data/processed/split.npz --out checkpoints/ranker.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from casmi.channels.ranker import train_ranker
from casmi.config import CFG, RankerConfig


def grouped_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold id per row such that every row of one molecule shares a fold."""
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    fold_of_group = {unique[g]: i % n_folds for i, g in enumerate(order)}
    return np.array([fold_of_group[g] for g in groups], dtype=np.int64)


def reciprocal_ranks(
    probabilities: np.ndarray, labels: np.ndarray, groups: np.ndarray, k: int = 25
) -> dict[str, float]:
    """Per-molecule reciprocal rank of the first positive within the top ``k``."""
    out: dict[str, float] = {}
    order = np.argsort(groups, kind="stable")
    groups_sorted = groups[order]
    boundaries = np.flatnonzero(np.diff(groups_sorted)) + 1
    for chunk in np.split(order, boundaries):
        molecule = str(groups[chunk[0]])
        ranked = chunk[np.argsort(-probabilities[chunk], kind="stable")][:k]
        hits = np.flatnonzero(labels[ranked] == 1)
        out[molecule] = 1.0 / (hits[0] + 1) if hits.size else 0.0
    return out


def summarise(rr: dict[str, float], classes: dict[str, int] | None) -> dict[str, dict]:
    def block(values: list[float]) -> dict:
        return {"n": len(values), "mrr": float(np.mean(values)) if values else 0.0,
                "recall": float(np.mean([v > 0 for v in values])) if values else 0.0,
                "top1": float(np.mean([v == 1.0 for v in values])) if values else 0.0}

    summary = {"overall": block(list(rr.values()))}
    if classes:
        for cls in (1, 2, 3):
            vals = [v for m, v in rr.items() if classes.get(m) == cls]
            if vals:
                summary[f"class{cls}"] = block(vals)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="output of scripts/build_rank_train.py")
    parser.add_argument("--out", required=True, help="final ranker .pkl (fit on all rows)")
    parser.add_argument("--split", default=None, help="split .npz for per-class reporting")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cv-seeds", type=int, default=2,
                        help="bagging seeds per fold (fewer than the final fit, for speed)")
    parser.add_argument("--report", default=None, help="write the CV summary json here")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=False)
    x, y, groups = data["X"], data["y"].astype(np.int64), data["molecule_id"].astype(str)
    print(f"[INFO] {x.shape[0]:,} rows x {x.shape[1]} features, "
          f"{len(np.unique(groups)):,} molecules, {int(y.sum()):,} positives")

    classes = None
    if args.split:
        split = np.load(args.split, allow_pickle=False)
        classes = dict(zip([str(k) for k in split["holdout_keys"]],
                           [int(c) for c in split["novelty_class"]], strict=True))

    base = CFG.ranker
    cv_config = RankerConfig(
        max_depth=base.max_depth, max_iter=base.max_iter, learning_rate=base.learning_rate,
        min_samples_leaf=base.min_samples_leaf, l2_regularization=base.l2_regularization,
        seeds=tuple(range(args.cv_seeds)), class1_priors=base.class1_priors,
    )
    folds = grouped_folds(groups, args.folds, args.seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for fold in range(args.folds):
        train_rows = folds != fold
        test_rows = ~train_rows
        model = train_ranker(x[train_rows], y[train_rows], config=cv_config)
        oof[test_rows] = model.predict_proba(x[test_rows])
        n_scored = int(test_rows.sum())
        print(f"[OK] fold {fold + 1}/{args.folds}: {n_scored:,} rows scored", flush=True)

    rr = reciprocal_ranks(oof, y, groups)
    summary = summarise(rr, classes)
    print("\n=== Out-of-fold reranker MRR@25 ===")
    header = f"{'cohort':<10}{'n':>7}{'MRR@25':>9}{'recall':>9}{'top1':>9}"
    print(header)
    print("-" * len(header))
    for name, m in summary.items():
        print(f"{name:<10}{m['n']:>7}{m['mrr']:>9.4f}{m['recall']:>9.4f}{m['top1']:>9.4f}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump({"folds": args.folds, "summary": summary}, handle, indent=2)

    print(f"\n[INFO] fitting the final ensemble on all rows "
          f"({len(base.class1_priors)} priors x {len(base.seeds)} seeds)")
    final = train_ranker(x, y, config=base)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    final.save(out)
    print(f"[OK] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
