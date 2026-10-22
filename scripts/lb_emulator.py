"""Predict the Kaggle public score of a configuration from offline per-class MRR.

Model: ``public ≈ alpha * (s1*m1 + s2*m2 + s3*m3)``, where ``m_c`` is the
configuration's MRR@25 on test-like novelty class ``c`` (from ``run_baseline.py
--realistic-pool``), ``s`` are the hidden test's class shares (a point on the
simplex) and ``alpha`` absorbs the remaining domain shift (instrument, chemistry).
``s`` and ``alpha`` are fitted to configurations whose public score we know, by
exhaustive search over the simplex (step 0.01) with the least-squares ``alpha``
for each candidate ``s``.

With 3 calibration points and 3 parameters the fit is exact-ish, so treat the
predicted score as a ranking signal between configurations first and an absolute
number second; every new Kaggle score added to ``--calibrate`` tightens it.

Example:
    python scripts/lb_emulator.py \\
        --calibrate runs/e_v2/summary.json=0.152 --calibrate runs/e_v7/summary.json=0.186 \\
        --calibrate runs/e_v8/summary.json=0.252 --predict runs/e_v9/summary.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def class_mrr(summary_path: str) -> np.ndarray:
    summary = json.load(open(summary_path, encoding="utf-8"))["summary"]
    return np.array([summary.get(f"class{c}", {}).get("mrr", 0.0) for c in (1, 2, 3)])


def fit(m: np.ndarray, y: np.ndarray, step: float = 0.01) -> tuple[np.ndarray, float, float]:
    """Best ``(shares, alpha, rmse)`` for ``y ≈ alpha * m @ shares``."""
    best = (np.array([1 / 3] * 3), 1.0, float("inf"))
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for s1 in grid:
        for s2 in grid:
            s3 = 1.0 - s1 - s2
            if s3 < -1e-9:
                continue
            shares = np.array([s1, s2, max(s3, 0.0)])
            x = m @ shares
            denom = float(x @ x)
            if denom <= 0:
                continue
            alpha = float(x @ y) / denom
            if not 0.1 <= alpha <= 1.5:
                continue
            rmse = float(np.sqrt(np.mean((alpha * x - y) ** 2)))
            if rmse < best[2]:
                best = (shares, alpha, rmse)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrate", action="append", default=[],
        help="summary.json=public_score for a configuration already scored on Kaggle",
    )
    parser.add_argument("--predict", action="append", default=[], help="summary.json to score")
    parser.add_argument("--shares", default=None,
                        help="fix class shares 's1,s2,s3' instead of fitting (alpha still fitted)")
    args = parser.parse_args()

    if not args.calibrate:
        parser.error("need at least one --calibrate summary.json=score")
    paths, scores = zip(*(item.rsplit("=", 1) for item in args.calibrate), strict=True)
    m = np.stack([class_mrr(p) for p in paths])
    y = np.array([float(s) for s in scores])

    if args.shares:
        shares = np.array([float(v) for v in args.shares.split(",")])
        shares = shares / shares.sum()
        x = m @ shares
        alpha = float(x @ y) / float(x @ x)
        rmse = float(np.sqrt(np.mean((alpha * x - y) ** 2)))
    else:
        shares, alpha, rmse = fit(m, y)

    print(f"class shares (1/2/3): {shares[0]:.2f} / {shares[1]:.2f} / {shares[2]:.2f}")
    print(f"domain factor alpha:  {alpha:.3f}    calibration RMSE: {rmse:.4f}")
    print(f"\n{'config':<44}{'m1':>7}{'m2':>7}{'m3':>7}{'pred':>8}{'public':>8}")
    for path, row, score in zip(paths, m, y, strict=True):
        print(f"{path[-44:]:<44}{row[0]:>7.3f}{row[1]:>7.3f}{row[2]:>7.3f}"
              f"{alpha * row @ shares:>8.3f}{score:>8.3f}")
    for path in args.predict:
        row = class_mrr(path)
        print(f"{path[-44:]:<44}{row[0]:>7.3f}{row[1]:>7.3f}{row[2]:>7.3f}"
              f"{alpha * row @ shares:>8.3f}{'?':>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
