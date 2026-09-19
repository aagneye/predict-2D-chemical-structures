"""End-to-end CLI smoke tests.

Runs the real scripts as subprocesses against a synthetic ``train.parquet``.
These are the only tests that exercise the wiring between modules the way an
actual run does — unit tests would not catch an argument name mismatch or a
missing column in a script's read list.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from casmi.adducts import PROTON_MASS
from casmi.chem import exact_mass, inchikey14
from tests.conftest import MOLECULES, synthetic_peaks

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

#: Libraries are varied so the split's NP weighting has something to prefer.
LIBRARIES = ["gnps", "riken", "enveda-180", "massbank", "enveda-np-examples"]


@pytest.fixture(scope="module")
def train_parquet(tmp_path_factory) -> str:
    """Synthetic train.parquet with every column the scripts read.

    Each structure gets 3 spectra so the split can hold one out while leaving
    siblings behind (the class-1 case).
    """
    import pyarrow as pa

    tmp = tmp_path_factory.mktemp("cli")
    rows: dict[str, list] = {
        "inchikey14": [],
        "normalized_smiles": [],
        "adduct": [],
        "ionization_mode": [],
        "instrument_type": [],
        "precursor_mz": [],
        "collision_energy_ev": [],
        "ms2_mzs": [],
        "ms2_normalized_intensities": [],
        "ingest_lib": [],
    }

    for i, (_name, smiles) in enumerate(MOLECULES):
        mass = exact_mass(smiles)
        precursor = mass + PROTON_MASS
        for spectrum in range(3):
            mz, intensity = synthetic_peaks(i * 7 + spectrum, precursor=max(precursor, 60.0))
            rows["inchikey14"].append(inchikey14(smiles))
            rows["normalized_smiles"].append(smiles)
            rows["adduct"].append("[M+H]+")
            rows["ionization_mode"].append("positive")
            rows["instrument_type"].append("timsTOF")
            rows["precursor_mz"].append(precursor)
            rows["collision_energy_ev"].append([20.0 + 10.0 * spectrum])
            rows["ms2_mzs"].append(mz.tolist())
            rows["ms2_normalized_intensities"].append(intensity.tolist())
            rows["ingest_lib"].append(LIBRARIES[i % len(LIBRARIES)])

    table = pa.table(
        {
            "inchikey14": pa.array(rows["inchikey14"], pa.string()),
            "normalized_smiles": pa.array(rows["normalized_smiles"], pa.string()),
            "adduct": pa.array(rows["adduct"], pa.string()),
            "ionization_mode": pa.array(rows["ionization_mode"], pa.string()),
            "instrument_type": pa.array(rows["instrument_type"], pa.string()),
            "precursor_mz": pa.array(rows["precursor_mz"], pa.float64()),
            "collision_energy_ev": pa.array(rows["collision_energy_ev"], pa.list_(pa.float64())),
            "ms2_mzs": pa.array(rows["ms2_mzs"], pa.list_(pa.float32())),
            "ms2_normalized_intensities": pa.array(
                rows["ms2_normalized_intensities"], pa.list_(pa.float32())
            ),
            "ingest_lib": pa.array(rows["ingest_lib"], pa.string()),
        }
    )
    path = tmp / "train.parquet"
    pq.write_table(table, path)
    return str(path)


@pytest.fixture(scope="module")
def coconut_csv(tmp_path_factory) -> str:
    """A COCONUT-like CSV covering most fixture molecules."""
    tmp = tmp_path_factory.mktemp("coconut")
    path = tmp / "coconut.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["identifier", "smiles"])
        for i, (_name, smiles) in enumerate(MOLECULES[:-2]):
            writer.writerow([f"CNP{i:05d}", smiles])
    return str(path)


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    """Invoke a script with the repo's src on the path."""
    command = [sys.executable, str(SCRIPTS / name), *args]
    env = {
        **dict(__import__("os").environ),
        "PYTHONPATH": str(REPO_ROOT / "src"),
    }
    return subprocess.run(
        command, capture_output=True, text=True, env=env, cwd=str(REPO_ROOT), timeout=900
    )


class TestBuildSplitScript:
    def test_runs_and_writes_split(self, train_parquet, tmp_path):
        out = tmp_path / "split.npz"
        result = run_script(
            "build_split.py",
            "--train", train_parquet,
            "--out", str(out),
            "--n-holdout", "6",
            "--seed", "0",
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()
        assert "leakage check passed" in result.stdout

    def test_split_contents_are_consistent(self, train_parquet, tmp_path):
        out = tmp_path / "split.npz"
        run_script(
            "build_split.py", "--train", train_parquet, "--out", str(out), "--n-holdout", "6"
        )
        data = np.load(out, allow_pickle=False)
        table = pq.read_table(train_parquet, columns=["inchikey14"])
        n_rows = table.num_rows

        assert data["train_mask"].size == n_rows
        assert data["val_mask"].size == n_rows
        assert not np.any(data["train_mask"] & data["val_mask"])
        assert np.all(data["train_mask"] | data["val_mask"])
        assert len(data["holdout_keys"]) == 6
        assert set(int(c) for c in data["novelty_class"]) <= {1, 2, 3}

    def test_writes_summary_json(self, train_parquet, tmp_path):
        out = tmp_path / "split.npz"
        run_script(
            "build_split.py", "--train", train_parquet, "--out", str(out), "--n-holdout", "6"
        )
        summary_path = out.with_suffix(".summary.json")
        assert summary_path.exists()
        payload = json.loads(summary_path.read_text())
        assert "summary" in payload
        assert "holdout_composition" in payload

    def test_reports_holdout_composition(self, train_parquet, tmp_path):
        result = run_script(
            "build_split.py",
            "--train", train_parquet,
            "--out", str(tmp_path / "s.npz"),
            "--n-holdout", "6",
        )
        assert "hold-out composition by source library" in result.stdout


class TestBuildPoolScript:
    def test_builds_from_coconut_and_train(self, train_parquet, coconut_csv, tmp_path):
        out = tmp_path / "pool.npz"
        result = run_script(
            "build_pool.py",
            "--coconut", coconut_csv,
            "--train", train_parquet,
            "--out", str(out),
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()

        from casmi.candidates.pool import CandidatePool

        pool = CandidatePool.load(out)
        # All fixture molecules: COCONUT covers all but two, train supplies those.
        assert len(pool) == len(MOLECULES)
        assert np.all(np.diff(pool.mass) >= 0)

    def test_coconut_only(self, coconut_csv, tmp_path):
        out = tmp_path / "pool.npz"
        result = run_script("build_pool.py", "--coconut", coconut_csv, "--out", str(out))
        assert result.returncode == 0, result.stderr

        from casmi.candidates.pool import CandidatePool

        assert len(CandidatePool.load(out)) == len(MOLECULES) - 2

    def test_requires_a_source(self, tmp_path):
        result = run_script("build_pool.py", "--out", str(tmp_path / "p.npz"))
        assert result.returncode != 0
        assert "at least one of" in result.stderr

    def test_rejects_csv_without_smiles_column(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("id,name\n1,foo\n", encoding="utf-8")
        result = run_script(
            "build_pool.py", "--coconut", str(bad), "--out", str(tmp_path / "p.npz")
        )
        assert result.returncode != 0
        assert "no SMILES column" in (result.stderr + result.stdout)


@pytest.fixture(scope="module")
def baseline_artifacts(train_parquet, coconut_csv, tmp_path_factory):
    """Split and pool built once, shared by the baseline tests."""
    tmp = tmp_path_factory.mktemp("baseline")
    split = tmp / "split.npz"
    pool = tmp / "pool.npz"
    assert run_script(
        "build_split.py", "--train", train_parquet, "--out", str(split),
        "--n-holdout", "6", "--seed", "0",
    ).returncode == 0
    assert run_script(
        "build_pool.py", "--coconut", coconut_csv, "--train", train_parquet,
        "--out", str(pool),
    ).returncode == 0
    return str(split), str(pool)


class TestRunBaselineScript:
    def test_runs_end_to_end(self, train_parquet, baseline_artifacts, tmp_path):
        split, pool = baseline_artifacts
        out = tmp_path / "run"
        result = run_script(
            "run_baseline.py",
            "--train", train_parquet,
            "--split", split,
            "--pool", pool,
            "--out", str(out),
            "--ppm", "50",
        )
        assert result.returncode == 0, result.stderr
        assert "Baseline MRR@25 on held-out split" in result.stdout
        assert "Channel contribution" in result.stdout

    def test_writes_summary_and_diagnostics(self, train_parquet, baseline_artifacts, tmp_path):
        split, pool = baseline_artifacts
        out = tmp_path / "run2"
        run_script(
            "run_baseline.py", "--train", train_parquet, "--split", split,
            "--pool", pool, "--out", str(out), "--ppm", "50",
        )
        summary = json.loads((out / "summary.json").read_text())
        assert "summary" in summary
        assert "channel_contribution" in summary
        assert "overall" in summary["summary"]
        assert 0.0 <= summary["summary"]["overall"]["mrr"] <= 1.0

        with open(out / "diagnostics.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert "best_library_sim" in rows[0]
        assert "best_analog_sim" in rows[0]

    def test_class1_outperforms_class3(self, train_parquet, baseline_artifacts, tmp_path):
        """Sanity check that the cohorts behave as constructed.

        Class 1 keeps sibling spectra in the library and should score well;
        class 3 is removed from the candidate pool entirely and must score 0.
        If this inverts, the split or the pool exclusion is broken.
        """
        split, pool = baseline_artifacts
        out = tmp_path / "run3"
        run_script(
            "run_baseline.py", "--train", train_parquet, "--split", split,
            "--pool", pool, "--out", str(out), "--ppm", "50",
        )
        summary = json.loads((out / "summary.json").read_text())["summary"]
        if "class3" in summary:
            assert summary["class3"]["mrr"] == 0.0, "class-3 structures must be unreachable"
        if "class1" in summary and "class3" in summary:
            assert summary["class1"]["mrr"] >= summary["class3"]["mrr"]


class TestTrainFPNetScript:
    def test_short_training_run(self, train_parquet, tmp_path):
        """Tiny run to prove the training CLI is wired correctly."""
        pytest.importorskip("torch", reason="requires the optional 'train' extra")
        split = tmp_path / "split.npz"
        assert run_script(
            "build_split.py", "--train", train_parquet, "--out", str(split), "--n-holdout", "4"
        ).returncode == 0

        result = run_script(
            "train_fpnet.py",
            "--train", train_parquet,
            "--split", str(split),
            "--out", str(tmp_path / "ckpt"),
            "--max-steps", "3",
            "--batch-size", "4",
            "--val-rows", "2",
            "--device", "cpu",
            "--no-multi-gpu",
            "--d-model", "32",
            "--layers", "1",
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "ckpt" / "fpnet_final.pt").exists()

    def test_split_argument_is_required(self, train_parquet, tmp_path):
        """The split must be mandatory, or training silently sees the hold-out."""
        result = run_script(
            "train_fpnet.py", "--train", train_parquet, "--out", str(tmp_path / "c")
        )
        assert result.returncode != 0
        assert "--split" in result.stderr


class TestRunBaselineChannelFlags:
    def test_fragmentation_flag_runs_end_to_end(self, train_parquet, baseline_artifacts, tmp_path):
        split, pool = baseline_artifacts
        out = tmp_path / "run_frag"
        result = run_script(
            "run_baseline.py",
            "--train", train_parquet,
            "--split", split,
            "--pool", pool,
            "--out", str(out),
            "--ppm", "50",
            "--fragmentation",
        )
        assert result.returncode == 0, result.stderr
        assert "fragmentation" in result.stdout

    def test_active_channels_reported_in_summary(self, train_parquet, baseline_artifacts, tmp_path):
        split, pool = baseline_artifacts
        out = tmp_path / "run_channels"
        run_script(
            "run_baseline.py", "--train", train_parquet, "--split", split,
            "--pool", pool, "--out", str(out), "--ppm", "50", "--fragmentation",
        )
        summary = json.loads((out / "summary.json").read_text())
        assert "active_channels" in summary
        assert "fragmentation" in summary["active_channels"]


class TestBuildRankTrainScript:
    def test_runs_and_writes_training_matrix(self, train_parquet, baseline_artifacts, tmp_path):
        split, pool = baseline_artifacts
        out = tmp_path / "rank_train.npz"
        result = run_script(
            "build_rank_train.py",
            "--train", train_parquet,
            "--split", split,
            "--pool", pool,
            "--out", str(out),
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()

        data = np.load(out, allow_pickle=False)
        assert data["X"].ndim == 2
        assert data["y"].ndim == 1
        assert data["X"].shape[0] == data["y"].shape[0]
        assert data["molecule_id"].shape[0] == data["y"].shape[0]

    def test_with_fragmentation_flag(self, train_parquet, baseline_artifacts, tmp_path):
        split, pool = baseline_artifacts
        out = tmp_path / "rank_train_frag.npz"
        result = run_script(
            "build_rank_train.py",
            "--train", train_parquet,
            "--split", split,
            "--pool", pool,
            "--out", str(out),
            "--fragmentation",
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()


class TestTrainRankerScript:
    def test_trains_and_saves_reranker(self, train_parquet, baseline_artifacts, tmp_path):
        split, pool = baseline_artifacts
        rank_train = tmp_path / "rank_train.npz"
        assert run_script(
            "build_rank_train.py",
            "--train", train_parquet,
            "--split", split,
            "--pool", pool,
            "--out", str(rank_train),
        ).returncode == 0

        out = tmp_path / "ranker.pkl"
        result = run_script(
            "train_ranker.py",
            "--data", str(rank_train),
            "--out", str(out),
            "--max-iter", "10",
        )
        assert result.returncode == 0, result.stderr
        assert out.exists()

        from casmi.channels.ranker import Reranker

        reranker = Reranker.load(out)
        assert len(reranker.models) > 0

    def test_ranker_usable_by_run_baseline(self, train_parquet, baseline_artifacts, tmp_path):
        """The trained reranker must be loadable and usable by run_baseline.py."""
        split, pool = baseline_artifacts
        rank_train = tmp_path / "rank_train2.npz"
        assert run_script(
            "build_rank_train.py", "--train", train_parquet, "--split", split,
            "--pool", pool, "--out", str(rank_train),
        ).returncode == 0

        ranker_path = tmp_path / "ranker2.pkl"
        assert run_script(
            "train_ranker.py", "--data", str(rank_train), "--out", str(ranker_path),
            "--max-iter", "10",
        ).returncode == 0

        out = tmp_path / "run_ranked"
        result = run_script(
            "run_baseline.py", "--train", train_parquet, "--split", split,
            "--pool", pool, "--out", str(out), "--ppm", "50",
            "--ranker", str(ranker_path),
        )
        assert result.returncode == 0, result.stderr
        assert "gbm_reranker" in result.stdout
