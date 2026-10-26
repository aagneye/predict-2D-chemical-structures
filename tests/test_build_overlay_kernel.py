"""build_overlay_kernel.py keeps the base untouched and appends our overlay cell."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "build_overlay_kernel",
    Path(__file__).resolve().parents[1] / "scripts" / "build_overlay_kernel.py",
)
bok = importlib.util.module_from_spec(spec)
sys.modules["build_overlay_kernel"] = bok
spec.loader.exec_module(bok)

BASE_NB = {"cells": [bok.markdown_cell("# base"), bok.code_cell("x = 1\n")],
           "metadata": {"kernelspec": {"name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
BASE_META = {"id": "someone/base", "id_no": 42, "code_file": "base.ipynb", "is_private": False,
             "enable_gpu": True, "docker_image": "gcr.io/x@sha256:abc",
             "dataset_sources": ["a/one", "b/two"], "kernel_sources": []}


def test_build_keeps_base_cells_and_merges_inputs():
    nb, meta = bok.build(BASE_NB, BASE_META, "print('overlay')\n", "me/overlay", "Overlay",
                         ["b/two", "me/assets"], "# attribution")
    assert nb["cells"][1:3] == BASE_NB["cells"]
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert nb["cells"][-1]["source"] == ["print('overlay')\n"]
    assert nb["metadata"] == BASE_NB["metadata"]
    assert meta["id"] == "me/overlay" and meta["code_file"] == "overlay.ipynb"
    assert meta["is_private"] is True and "id_no" not in meta
    assert meta["docker_image"] == "gcr.io/x@sha256:abc"
    assert meta["dataset_sources"] == ["a/one", "b/two", "me/assets"]
    assert BASE_META["dataset_sources"] == ["a/one", "b/two"]  # input not mutated


def test_overlay_cell_is_valid_python():
    code = bok.OVERLAY_CELL.read_text(encoding="utf-8")
    compile(code, str(bok.OVERLAY_CELL), "exec")
    assert "merge_overlay(" in code and "submission.csv" in code


def test_cli_pins_the_base_hash(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base"
    base.mkdir()
    (base / "base.ipynb").write_text(json.dumps(BASE_NB), encoding="utf-8")
    (base / "kernel-metadata.json").write_text(json.dumps(BASE_META), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["x", "--base-dir", str(base), "--base-sha256", "0" * 64,
                                      "--out", str(tmp_path / "out"), "--id", "me/o",
                                      "--title", "t"])
    try:
        bok.main()
    except SystemExit as exc:
        assert "sha256" in str(exc)
    else:
        raise AssertionError("a wrong base hash must be refused")
    monkeypatch.setattr(sys, "argv", ["x", "--base-dir", str(base), "--out",
                                      str(tmp_path / "out"), "--id", "me/o", "--title", "t",
                                      "--extra-dataset", "me/assets"])
    assert bok.main() == 0
    written = json.loads((tmp_path / "out" / "o.ipynb").read_text(encoding="utf-8"))
    assert len(written["cells"]) == 2 + 2 + 1
    assert "[OK]" in capsys.readouterr().out
