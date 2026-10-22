"""build_external_keys.py: flag-filtered SMILES tables and ready-made key columns."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location(
    "build_external_keys", Path(__file__).resolve().parents[1] / "scripts" / "build_external_keys.py"
)
bek = importlib.util.module_from_spec(spec)
sys.modules["build_external_keys"] = bek
spec.loader.exec_module(bek)


def test_split_spec_keeps_drive_letters():
    assert bek.split_spec("a.csv:np_coconut") == ("a.csv", "np_coconut")
    assert bek.split_spec("C:\data\a.csv") == ("C:\data\a.csv", None)
    assert bek.split_spec("a.csv") == ("a.csv", None)


def test_flag_filter_and_key_columns(tmp_path, monkeypatch):
    table = tmp_path / "pop.csv"
    table.write_text("inchikey14,smiles,np_coconut\nAAA,CCO,1\nBBB,CCN,0\n", encoding="utf-8")
    assert bek.read_smiles(str(table), "np_coconut") == ["CCO"]
    assert bek.read_smiles(str(table)) == ["CCO", "CCN"]

    parquet = tmp_path / "pcl.parquet"
    pq.write_table(pa.table({"ik14": ["KEY1", "", "KEY2"]}), parquet)
    assert bek.read_keys(str(parquet), "ik14") == {"KEY1", "KEY2"}

    out = tmp_path / "keys.txt"
    monkeypatch.setattr(sys, "argv", [
        "x", "--csv", f"{table}:np_coconut", "--keys", f"{parquet}:ik14",
        "--out", str(out), "--workers", "1",
    ])
    assert bek.main() == 0
    keys = out.read_text().split()
    assert {"KEY1", "KEY2"} <= set(keys) and len(keys) == 3  # + the scorer key of CCO
