"""Overlay merge: insertion position, identity de-duplication, caps and the gate."""

from __future__ import annotations

from casmi.overlay import OverlayConfig, merge_overlay, weak_evidence


def test_inserts_below_rank_one_by_default():
    cfg = OverlayConfig(max_insert=2)
    out = merge_overlay(["b1", "b2", "b3"], ["d1", "d2", "d3"], cfg)
    assert out == ["b1", "d1", "d2", "b2", "b3"]


def test_skips_structures_already_in_base_and_invalid_keys():
    cfg = OverlayConfig(insert_at=0, max_insert=5)
    key = {"b1": "K1", "d1": "K1", "d2": "K2", "d3": None, "d4": "K4"}.get
    out = merge_overlay(["b1"], ["d1", "d2", "d3", "d4"], cfg, key=key)
    assert out == ["d2", "d4", "b1"]


def test_result_is_cut_to_max_length():
    cfg = OverlayConfig(insert_at=1, max_insert=3, max_length=25)
    base = [f"b{i}" for i in range(25)]
    out = merge_overlay(base, ["d1", "d2", "d3"], cfg)
    assert len(out) == 25 and out[:4] == ["b0", "d1", "d2", "d3"] and out[-1] == "b21"


def test_insert_position_is_clamped_to_the_base_length():
    cfg = OverlayConfig(insert_at=10, max_insert=1)
    assert merge_overlay(["b1"], ["d1"], cfg) == ["b1", "d1"]
    assert merge_overlay([], ["d1"], cfg) == ["d1"]


def test_weak_evidence_gate():
    cfg = OverlayConfig(library_max_threshold=0.6)
    assert weak_evidence(0.3, cfg) and not weak_evidence(0.9, cfg)
