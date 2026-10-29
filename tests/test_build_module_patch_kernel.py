"""build_module_patch_kernel.py swaps one pinned module and its hash, nothing else."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "build_module_patch_kernel",
    Path(__file__).resolve().parents[1] / "scripts" / "build_module_patch_kernel.py",
)
bmp = importlib.util.module_from_spec(spec)
sys.modules["build_module_patch_kernel"] = bmp
spec.loader.exec_module(bmp)

OLD = "def f():\n    return 'v1'\n"
OLD_SHA = hashlib.sha256(OLD.encode()).hexdigest()
OTHER = "x = 1\n"
OTHER_SHA = hashlib.sha256(OTHER.encode()).hexdigest()


def notebook() -> dict:
    sources = (f"MODULES = {{}}\nMODULES['a.py'] = {OLD!r}\nMODULES['b.py'] = {OTHER!r}\n"
               f"MODULE_SHA256 = {{\n \"a.py\": \"{OLD_SHA}\",\n \"b.py\": \"{OTHER_SHA}\"\n}}")
    return {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# base"]},
                      {"cell_type": "code", "metadata": {}, "outputs": [],
                       "execution_count": None, "source": [sources]}],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5}


def run_cell(nb: dict) -> dict:
    scope: dict = {}
    exec("".join(nb["cells"][1]["source"]), scope)
    return scope


def test_patch_replaces_module_and_pin_only():
    new = "def f():\n    return 'ensemble'  # µ\n"
    nb, old_sha, new_sha = bmp.patch_module(notebook(), "a.py", new)
    scope = run_cell(nb)
    assert scope["MODULES"]["a.py"] == new and scope["MODULES"]["b.py"] == OTHER
    assert old_sha == OLD_SHA
    assert scope["MODULE_SHA256"]["a.py"] == new_sha == hashlib.sha256(new.encode()).hexdigest()
    assert scope["MODULE_SHA256"]["b.py"] == OTHER_SHA
    assert nb["cells"][0] == notebook()["cells"][0]


def test_patch_refuses_unknown_module():
    with pytest.raises(ValueError, match="no MODULES line"):
        bmp.patch_module(notebook(), "missing.py", "x")


def test_metadata_adds_datasets_and_goes_private():
    base = {"id": "someone/base", "id_no": 7, "code_file": "base.ipynb", "is_private": False,
            "docker_image": "img", "dataset_sources": ["a/one"]}
    meta = bmp.kernel_metadata(base, "me/fork", "Fork", ["a/one", "me/seeds"])
    assert meta["dataset_sources"] == ["a/one", "me/seeds"] and "id_no" not in meta
    assert meta["code_file"] == "fork.ipynb" and meta["is_private"] is True
    assert meta["docker_image"] == "img" and base["dataset_sources"] == ["a/one"]
