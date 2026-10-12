"""Channel 7 (PubChem tier) and the evidence-based pruning / z-scored FPNet fusion."""

from __future__ import annotations

import numpy as np
import pytest

from casmi.channels.fusion import DEFAULT_WEIGHTS, ScoredCandidate, fuse
from casmi.channels.pubchem import PubChemConfig, PubChemTier, pubchem_candidates
from casmi.chem import FingerprintCalculator, exact_mass, inchikey14

#: Same-formula isomers (C4H8O2) plus an unrelated structure at another mass.
ISOMERS = ["CCCC(=O)O", "CCOC(=O)C", "CC(C)C(=O)O", "COC(=O)CC"]
OTHER = "c1ccccc1O"


def write_tier(tmp_path, smiles: list[str], popularity: list[float] | None = None):
    """Write a tiny mass-sorted tier in the community layout."""
    masses = np.array([exact_mass(s) for s in smiles], dtype=np.float64)
    order = np.argsort(masses, kind="stable")
    smiles = [smiles[i] for i in order]
    masses = masses[order]
    encoded = [s.encode("ascii") for s in smiles]
    offsets = np.zeros(len(smiles) + 1, dtype=np.int64)
    for i, b in enumerate(encoded):
        offsets[i + 1] = offsets[i] + len(b)
    buffer = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    tier_dir = tmp_path / "tier"
    tier_dir.mkdir()
    np.save(tier_dir / "pc_mass.npy", masses)
    np.save(tier_dir / "pc_off.npy", offsets)
    np.save(tier_dir / "pc_smiles.npy", buffer)
    pop_dir = None
    if popularity is not None:
        pop_dir = tmp_path / "pop"
        pop_dir.mkdir()
        pop = np.array([popularity[i] for i in order], dtype=np.float16)
        np.save(pop_dir / "pc_lsid.npy", pop / 2)
        np.save(pop_dir / "pc_lpmid.npy", pop / 2)
    return tier_dir, pop_dir


@pytest.fixture(scope="module")
def calc():
    return FingerprintCalculator()


class TestTier:
    def test_window_and_smiles_roundtrip(self, tmp_path):
        tier_dir, _ = write_tier(tmp_path, ISOMERS + [OTHER])
        tier = PubChemTier(tier_dir)
        assert len(tier) == 5
        lo, hi = tier.window(exact_mass(ISOMERS[0]), ppm=10.0)
        assert hi - lo == 4
        assert set(tier.smiles(lo, hi)) == set(ISOMERS)
        assert not tier.has_popularity
        assert tier.popularity(lo, hi).tolist() == [0.0] * 4

    def test_popularity_arrays_are_aligned(self, tmp_path):
        tier_dir, pop_dir = write_tier(tmp_path, ISOMERS, popularity=[1.0, 10.0, 2.0, 3.0])
        tier = PubChemTier(tier_dir, pop_dir)
        assert tier.has_popularity
        lo, hi = tier.window(exact_mass(ISOMERS[0]), ppm=10.0)
        by_smiles = dict(zip(tier.smiles(lo, hi), tier.popularity(lo, hi).tolist(), strict=True))
        assert by_smiles["CCOC(=O)C"] == pytest.approx(10.0, abs=0.05)

    def test_empty_window(self, tmp_path):
        tier_dir, _ = write_tier(tmp_path, ISOMERS)
        tier = PubChemTier(tier_dir)
        assert tier.window(999.0, 10.0) == (0, 0) or tier.smiles(*tier.window(999.0, 10.0)) == []
        assert tier.window(float("nan"), 10.0) == (0, 0)


class TestCandidates:
    def test_true_fingerprint_ranks_its_structure_first(self, tmp_path, calc):
        tier_dir, _ = write_tier(tmp_path, ISOMERS + [OTHER])
        tier = PubChemTier(tier_dir)
        truth = "CC(C)C(=O)O"
        logits = calc.from_smiles(truth).astype(np.float32) * 4.0 - 2.0  # +2 on bits, -2 off
        out = pubchem_candidates(tier, exact_mass(truth), logits, calc, exclude_keys=set())
        assert out[0].inchikey14 == inchikey14(truth)
        assert len(out) == 4
        assert all(o.fingerprint.shape == (calc.n_bits,) for o in out)

    def test_excluded_keys_are_skipped(self, tmp_path, calc):
        tier_dir, _ = write_tier(tmp_path, ISOMERS)
        tier = PubChemTier(tier_dir)
        truth = "CC(C)C(=O)O"
        logits = calc.from_smiles(truth).astype(np.float32) * 4.0 - 2.0
        out = pubchem_candidates(
            tier, exact_mass(truth), logits, calc, exclude_keys={inchikey14(truth)}
        )
        assert inchikey14(truth) not in {o.inchikey14 for o in out}
        assert len(out) == 3

    def test_top_k_and_popularity_shift(self, tmp_path, calc):
        tier_dir, pop_dir = write_tier(tmp_path, ISOMERS, popularity=[0.0, 20.0, 0.0, 0.0])
        tier = PubChemTier(tier_dir, pop_dir)
        truth = "CC(C)C(=O)O"
        logits = calc.from_smiles(truth).astype(np.float32) * 0.1  # weak evidence
        out = pubchem_candidates(
            tier, exact_mass(truth), logits, calc, exclude_keys=set(),
            config=PubChemConfig(top_k=2, popularity_mu=1.0),
        )
        assert len(out) == 2
        assert out[0].smiles == "CCOC(=O)C"
        assert out[0].popularity == pytest.approx(20.0, abs=0.1)

    def test_max_window_keeps_most_documented(self, tmp_path, calc):
        tier_dir, pop_dir = write_tier(tmp_path, ISOMERS, popularity=[0.0, 5.0, 9.0, 0.0])
        tier = PubChemTier(tier_dir, pop_dir)
        logits = np.zeros(calc.n_bits, dtype=np.float32)
        out = pubchem_candidates(
            tier, exact_mass(ISOMERS[0]), logits, calc, exclude_keys=set(),
            config=PubChemConfig(max_window=2),
        )
        assert {o.smiles for o in out} == {"CCOC(=O)C", "CC(C)C(=O)O"}


class TestFusionZscore:
    def test_raw_fpnet_dot_no_longer_weighted(self):
        assert "fpnet_score" not in DEFAULT_WEIGHTS
        assert DEFAULT_WEIGHTS["fpnet_z"] > 0

    def test_fpnet_is_relative_within_the_set(self):
        a = ScoredCandidate("A" * 14, "CCO", 0.0, {"fpnet_normalised_score": 10.0})
        b = ScoredCandidate("B" * 14, "CCC", 0.0, {"fpnet_normalised_score": 12.0})
        c = ScoredCandidate("C" * 14, "CCCC", 0.0, {"fpnet_normalised_score": 8.0})
        ranked = fuse([a, b, c])
        assert [r.inchikey14 for r in ranked] == ["B" * 14, "A" * 14, "C" * 14]
        assert ranked[1].score == pytest.approx(0.0, abs=1e-9)

    def test_library_hit_without_fpnet_is_not_penalised(self):
        hit = ScoredCandidate("H" * 14, "CCO", 0.0, {"library_similarity": 0.9})
        a = ScoredCandidate("A" * 14, "CCC", 0.0, {"fpnet_normalised_score": 10.0})
        b = ScoredCandidate("B" * 14, "CCCC", 0.0, {"fpnet_normalised_score": 12.0})
        assert fuse([a, b, hit])[0].inchikey14 == "H" * 14

    def test_single_scored_candidate_gets_zero_z(self):
        a = ScoredCandidate("A" * 14, "CCO", 0.0, {"fpnet_normalised_score": 10.0})
        assert fuse([a])[0].score == pytest.approx(0.0)
