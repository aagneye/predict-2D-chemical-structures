"""Test-like pool exclusions, stratified sampling and the LB emulator fit."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from casmi.eval.realism import load_keys, realistic_pool_exclusions, stratified_sample


def test_class2_kept_only_when_an_external_database_has_it():
    novelty = {"A": 1, "B": 2, "C": 2, "D": 3}
    truth_keys = {"A": ("A", None), "B": ("B", "B2"), "C": ("C", None), "D": ("D", "D2")}
    drop = realistic_pool_exclusions(novelty, truth_keys, external_keys={"B2"})
    assert drop == {"C", "D", "D2"}  # B survives via its scorer key, A is class 1


def test_load_keys(tmp_path):
    path = tmp_path / "k.txt"
    path.write_text("AAA\n\nBBB\n", encoding="utf-8")
    assert load_keys(path) == {"AAA", "BBB"}


def test_stratified_sample_is_balanced_and_deterministic():
    novelty = {f"m{i}": 1 + i % 3 for i in range(90)}
    a = stratified_sample(novelty, per_class=10, seed=3)
    b = stratified_sample(novelty, per_class=10, seed=3)
    assert a == b and len(a) == 30
    assert sorted(novelty[m] for m in a).count(2) == 10


spec = importlib.util.spec_from_file_location(
    "lb_emulator", Path(__file__).resolve().parents[1] / "scripts" / "lb_emulator.py"
)
lb_emulator = importlib.util.module_from_spec(spec)
sys.modules["lb_emulator"] = lb_emulator
spec.loader.exec_module(lb_emulator)


def test_emulator_recovers_known_shares():
    true_shares, alpha = np.array([0.3, 0.2, 0.5]), 0.8
    m = np.array([[0.9, 0.0, 0.0], [0.9, 0.5, 0.0], [0.9, 0.7, 0.2], [0.8, 0.3, 0.1]])
    y = alpha * m @ true_shares
    shares, a, rmse = lb_emulator.fit(m, y)
    assert rmse < 1e-3
    assert a * m[2] @ shares == pytest.approx(y[2], abs=1e-3)


def test_class_mrr_reads_summary(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"summary": {"class1": {"mrr": 0.9}, "class3": {"mrr": 0.1}}}))
    assert lb_emulator.class_mrr(str(path)).tolist() == [0.9, 0.0, 0.1]
