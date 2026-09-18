"""Tests for the candidate pool."""

import numpy as np
import pytest

from casmi.candidates.pool import CandidatePool, build_pool, merge_pools
from casmi.chem import FingerprintCalculator, exact_mass, inchikey14
from tests.conftest import MOLECULES

SMILES = [s for _, s in MOLECULES]


@pytest.fixture(scope="module")
def small_calc():
    """Narrow fingerprint layout to keep pool tests fast."""
    return FingerprintCalculator(bit_subset=np.arange(0, 10407, 40))


@pytest.fixture(scope="module")
def pool(small_calc):
    return build_pool(SMILES, calculator=small_calc)


class TestBuildPool:
    def test_contains_all_structures(self, pool):
        assert len(pool) == len(SMILES)

    def test_sorted_by_mass(self, pool):
        assert np.all(np.diff(pool.mass) >= 0)

    def test_masses_correct(self, pool):
        for i in range(len(pool)):
            assert pool.mass[i] == pytest.approx(exact_mass(pool.smiles[i]), abs=1e-6)

    def test_keys_are_inchikey14(self, pool):
        for i in range(len(pool)):
            assert pool.keys[i] == inchikey14(pool.smiles[i])

    def test_deduplicates_by_structure(self, small_calc):
        """The same skeleton twice must occupy one slot."""
        p = build_pool(["Oc1ccccc1", "c1ccccc1O", "CCO"], calculator=small_calc)
        assert len(p) == 2

    def test_skips_unparseable(self, small_calc):
        p = build_pool(["CCO", "!!garbage!!", "CCC"], calculator=small_calc)
        assert len(p) == 2

    def test_empty_input(self, small_calc):
        p = build_pool([], calculator=small_calc)
        assert len(p) == 0
        assert p.window(100.0, 10.0).size == 0

    def test_precomputed_keys_used(self, small_calc):
        keys = [inchikey14(s) for s in SMILES[:3]]
        p = build_pool(SMILES[:3], calculator=small_calc, keys=keys)
        assert set(p.keys) == set(keys)

    def test_key_length_mismatch_raises(self, small_calc):
        with pytest.raises(ValueError, match="same length"):
            build_pool(["CCO", "CCC"], calculator=small_calc, keys=["ONLYONE0000000"])


class TestWindow:
    def test_finds_exact_mass(self, pool):
        target = exact_mass(MOLECULES[0][1])
        idx = pool.window(target, ppm=5.0)
        assert inchikey14(MOLECULES[0][1]) in set(pool.keys[idx])

    def test_tight_window_excludes_others(self, pool):
        target = exact_mass(MOLECULES[0][1])
        idx = pool.window(target, ppm=0.1)
        for i in idx:
            assert abs(pool.mass[i] - target) < 0.001

    def test_wide_window_includes_more(self, pool):
        target = exact_mass(MOLECULES[0][1])
        assert len(pool.window(target, 100000.0)) >= len(pool.window(target, 5.0))

    def test_no_match(self, pool):
        assert pool.window(99999.0, 5.0).size == 0

    def test_nan_target(self, pool):
        assert pool.window(float("nan"), 5.0).size == 0


class TestFingerprints:
    def test_unpacked_width(self, pool, small_calc):
        fps = pool.fingerprints(np.array([0, 1]))
        assert fps.shape == (2, small_calc.n_bits)

    def test_roundtrip_matches_direct_calculation(self, pool, small_calc):
        for i in range(3):
            direct = small_calc.from_smiles(pool.smiles[i])
            assert np.array_equal(pool.fingerprints(np.array([i]))[0], direct)

    def test_empty_indices(self, pool, small_calc):
        assert pool.fingerprints(np.array([], dtype=np.int64)).shape == (0, small_calc.n_bits)

    def test_binary(self, pool):
        fps = pool.fingerprints(np.arange(len(pool)))
        assert set(np.unique(fps).tolist()) <= {0, 1}


class TestIndexOf:
    def test_finds_present_key(self, pool):
        key = inchikey14(MOLECULES[0][1])
        assert pool.keys[pool.index_of(key)] == key

    def test_absent_key_is_minus_one(self, pool):
        assert pool.index_of("NOTAREALKEY000") == -1


class TestSubsetAndExclude:
    def test_subset_keeps_masked_rows(self, pool):
        mask = np.zeros(len(pool), dtype=bool)
        mask[:3] = True
        assert len(pool.subset(mask)) == 3

    def test_exclude_removes_keys(self, pool):
        """Honouring excluded_from_pool is what makes class 3 measurable."""
        victim = inchikey14(MOLECULES[0][1])
        reduced = pool.exclude_keys({victim})
        assert len(reduced) == len(pool) - 1
        assert reduced.index_of(victim) == -1

    def test_exclude_nothing_returns_same_size(self, pool):
        assert len(pool.exclude_keys(set())) == len(pool)

    def test_subset_stays_sorted(self, pool):
        mask = np.zeros(len(pool), dtype=bool)
        mask[::2] = True
        assert np.all(np.diff(pool.subset(mask).mass) >= 0)


class TestValidation:
    def test_rejects_unsorted_mass(self):
        with pytest.raises(ValueError, match="sorted"):
            CandidatePool(
                mass=np.array([200.0, 100.0]),
                keys=np.asarray(["A" * 14, "B" * 14], dtype=object),
                smiles=np.asarray(["CCO", "CCC"], dtype=object),
                packed_fp=np.zeros((2, 2), dtype=np.uint8),
                n_bits=16,
            )

    def test_rejects_inconsistent_lengths(self):
        with pytest.raises(ValueError, match="inconsistent"):
            CandidatePool(
                mass=np.array([100.0]),
                keys=np.asarray(["A" * 14, "B" * 14], dtype=object),
                smiles=np.asarray(["CCO", "CCC"], dtype=object),
                packed_fp=np.zeros((2, 2), dtype=np.uint8),
                n_bits=16,
            )


class TestSaveLoad:
    def test_roundtrip(self, pool, tmp_path):
        path = tmp_path / "pool.npz"
        pool.save(path)
        loaded = CandidatePool.load(path)
        assert len(loaded) == len(pool)
        assert loaded.n_bits == pool.n_bits
        assert np.allclose(loaded.mass, pool.mass)
        assert list(loaded.keys) == list(pool.keys)
        assert np.array_equal(loaded.packed_fp, pool.packed_fp)


class TestMergePools:
    def test_merges_and_dedupes(self, small_calc):
        a = build_pool(SMILES[:5], calculator=small_calc)
        b = build_pool(SMILES[3:8], calculator=small_calc)
        merged = merge_pools(a, b)
        assert len(merged) == 8
        assert len(set(merged.keys)) == 8

    def test_stays_sorted(self, small_calc):
        merged = merge_pools(
            build_pool(SMILES[:4], calculator=small_calc),
            build_pool(SMILES[4:], calculator=small_calc),
        )
        assert np.all(np.diff(merged.mass) >= 0)

    def test_first_pool_wins_on_collision(self, small_calc):
        a = build_pool(["Oc1ccccc1"], calculator=small_calc)
        b = build_pool(["c1ccccc1O"], calculator=small_calc)
        merged = merge_pools(a, b)
        assert len(merged) == 1
        assert merged.smiles[0] == "Oc1ccccc1"

    def test_rejects_mismatched_widths(self, small_calc):
        a = build_pool(SMILES[:2], calculator=small_calc)
        b = build_pool(SMILES[2:4], calculator=FingerprintCalculator(bit_subset=np.arange(50)))
        with pytest.raises(ValueError, match="different fingerprint widths"):
            merge_pools(a, b)

    def test_all_empty_raises(self, small_calc):
        with pytest.raises(ValueError, match="no non-empty"):
            merge_pools(build_pool([], calculator=small_calc))
