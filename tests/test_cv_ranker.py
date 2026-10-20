"""cv_ranker helpers: string group ids and first-positive reciprocal rank."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "cv_ranker", Path(__file__).resolve().parents[1] / "scripts" / "cv_ranker.py"
)
cv_ranker = importlib.util.module_from_spec(spec)
sys.modules["cv_ranker"] = cv_ranker
spec.loader.exec_module(cv_ranker)


def test_reciprocal_ranks_with_string_groups():
    groups = np.array(["MOLB", "MOLA", "MOLA", "MOLB", "MOLC"], dtype=str)
    probs = np.array([0.1, 0.2, 0.9, 0.8, 0.5])
    labels = np.array([1, 1, 0, 0, 0])
    rr = cv_ranker.reciprocal_ranks(probs, labels, groups)
    assert rr == {"MOLA": 0.5, "MOLB": 0.5, "MOLC": 0.0}


def test_reciprocal_ranks_respects_top_k():
    groups = np.array(["M"] * 30, dtype=str)
    probs = np.linspace(1.0, 0.0, 30)
    labels = np.zeros(30, dtype=int)
    labels[27] = 1  # rank 28 > 25
    assert cv_ranker.reciprocal_ranks(probs, labels, groups, k=25) == {"M": 0.0}


def test_grouped_folds_keep_a_molecule_in_one_fold():
    groups = np.array([f"K{i % 7}" for i in range(50)], dtype=str)
    folds = cv_ranker.grouped_folds(groups, n_folds=3, seed=1)
    for g in np.unique(groups):
        assert len(np.unique(folds[groups == g])) == 1
