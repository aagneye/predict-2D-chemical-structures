"""Tests for the leak-proof validation split.

The leakage checks here are the point of the module: a split that quietly
leaks produces optimistic numbers that look plausible, which is the single
most expensive failure mode available to this project.
"""

import numpy as np
import pytest

from casmi.config import SplitConfig
from casmi.data.split import (
    CLASS_DATABASE,
    CLASS_LIBRARY,
    CLASS_NOVEL,
    holdout_truth,
    make_split,
    verify_no_leakage,
)


def make_rows(n_structures: int = 40, spectra_each: int = 3, lib: str = "gnps"):
    """Synthetic row-level arrays: ``n_structures`` compounds, N spectra each."""
    keys, libs = [], []
    for i in range(n_structures):
        key = f"STRUCT{i:08d}AA"[:14]
        for _ in range(spectra_each):
            keys.append(key)
            libs.append(lib)
    return np.asarray(keys, dtype=object), np.asarray(libs, dtype=object)


class TestMakeSplit:
    def test_masks_partition_all_rows(self):
        keys, libs = make_rows()
        split = make_split(keys, libs, SplitConfig(n_holdout=10))
        assert np.all(split.train_mask | split.val_mask)
        assert not np.any(split.train_mask & split.val_mask)

    def test_holdout_size_respected(self):
        keys, libs = make_rows(n_structures=50)
        split = make_split(keys, libs, SplitConfig(n_holdout=12))
        assert len(split.holdout_keys) == 12

    def test_holdout_capped_at_available_structures(self):
        keys, libs = make_rows(n_structures=5)
        split = make_split(keys, libs, SplitConfig(n_holdout=100))
        assert len(split.holdout_keys) == 5

    def test_deterministic_for_fixed_seed(self):
        keys, libs = make_rows()
        a = make_split(keys, libs, SplitConfig(n_holdout=10, random_seed=7))
        b = make_split(keys, libs, SplitConfig(n_holdout=10, random_seed=7))
        assert a.holdout_keys == b.holdout_keys
        assert a.novelty_class == b.novelty_class

    def test_different_seed_changes_split(self):
        keys, libs = make_rows(n_structures=100)
        a = make_split(keys, libs, SplitConfig(n_holdout=20, random_seed=1))
        b = make_split(keys, libs, SplitConfig(n_holdout=20, random_seed=2))
        assert a.holdout_keys != b.holdout_keys

    def test_split_is_by_structure_not_by_spectrum(self):
        """Every spectrum of a class-2/3 structure must be held out together."""
        keys, libs = make_rows(n_structures=30, spectra_each=4)
        split = make_split(keys, libs, SplitConfig(n_holdout=9))
        train_keys = set(keys[split.train_mask])
        for key, cls in split.novelty_class.items():
            if cls in (CLASS_DATABASE, CLASS_NOVEL):
                assert key not in train_keys, f"{key} (class {cls}) leaked"

    def test_class1_keeps_sibling_spectra_in_train(self):
        """Class 1 means reference spectra exist, so siblings must remain."""
        keys, libs = make_rows(n_structures=30, spectra_each=3)
        split = make_split(keys, libs, SplitConfig(n_holdout=9))
        train_keys = set(keys[split.train_mask])
        class1 = [k for k, c in split.novelty_class.items() if c == CLASS_LIBRARY]
        assert class1, "fixture should produce at least one class-1 structure"
        for key in class1:
            assert key in train_keys

    def test_class1_holds_out_exactly_one_spectrum(self):
        keys, libs = make_rows(n_structures=30, spectra_each=3)
        split = make_split(keys, libs, SplitConfig(n_holdout=9))
        val_keys = list(keys[split.val_mask])
        for key, cls in split.novelty_class.items():
            if cls == CLASS_LIBRARY:
                assert val_keys.count(key) == 1

    def test_class_fractions_roughly_respected(self):
        keys, libs = make_rows(n_structures=200, spectra_each=3)
        split = make_split(
            keys, libs, SplitConfig(n_holdout=100, class_fractions=(0.25, 0.45, 0.30))
        )
        counts = split.summary()
        assert counts["class1_structures"] == pytest.approx(25, abs=3)
        assert counts["class2_structures"] == pytest.approx(45, abs=3)
        assert counts["class3_structures"] == pytest.approx(30, abs=6)

    def test_single_spectrum_structures_never_class1(self):
        """A structure with one spectrum cannot leave a sibling behind."""
        keys, libs = make_rows(n_structures=40, spectra_each=1)
        split = make_split(keys, libs, SplitConfig(n_holdout=20))
        assert all(c != CLASS_LIBRARY for c in split.novelty_class.values())

    def test_class3_excluded_from_pool(self):
        keys, libs = make_rows(n_structures=40)
        split = make_split(keys, libs, SplitConfig(n_holdout=20))
        class3 = {k for k, c in split.novelty_class.items() if c == CLASS_NOVEL}
        assert split.excluded_from_pool == class3

    def test_database_keys_steer_class3_to_absent_structures(self):
        """Class 3 should prefer structures a database cannot retrieve."""
        keys, libs = make_rows(n_structures=60, spectra_each=2)
        unique = sorted(set(keys))
        database = set(unique[:40])  # last 20 are "not in any database"
        split = make_split(
            keys,
            libs,
            SplitConfig(n_holdout=30, class_fractions=(0.2, 0.4, 0.4)),
            database_keys=database,
        )
        class3 = [k for k, c in split.novelty_class.items() if c == CLASS_NOVEL]
        absent_in_class3 = [k for k in class3 if k not in database]
        assert len(absent_in_class3) >= 1
        # No database-absent held-out structure should be labelled class 2,
        # since retrieval genuinely cannot reach it.
        class2 = [k for k, c in split.novelty_class.items() if c == CLASS_DATABASE]
        assert all(k in database for k in class2)

    def test_mismatched_input_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            make_split(np.array(["A"], dtype=object), np.array(["x", "y"], dtype=object))

    def test_no_structures_raises(self):
        with pytest.raises(ValueError, match="no structures"):
            make_split(np.array([], dtype=object), np.array([], dtype=object))


class TestLibraryWeighting:
    def test_np_libraries_oversampled(self):
        """Natural-product libraries must dominate the hold-out.

        enveda-180 is ~46% of real spectra but off-domain; a uniform split
        would make validation mostly synthetic screening chemistry.
        """
        keys, libs = [], []
        for i in range(100):
            key = f"NP{i:012d}"[:14]
            keys.append(key)
            libs.append("riken")
        for i in range(100):
            key = f"OD{i:012d}"[:14]
            keys.append(key)
            libs.append("enveda-180")
        keys = np.asarray(keys, dtype=object)
        libs = np.asarray(libs, dtype=object)

        split = make_split(keys, libs, SplitConfig(n_holdout=50, random_seed=0))
        np_held = sum(1 for k in split.holdout_keys if k.startswith("NP"))
        assert np_held > 35, f"expected NP-dominated hold-out, got {np_held}/50"

    def test_structure_in_both_libraries_uses_max_weight(self):
        """A compound in riken AND enveda-180 counts as in-domain."""
        keys = np.asarray(["SHARED00000000"] * 2, dtype=object)
        libs = np.asarray(["enveda-180", "riken"], dtype=object)
        split = make_split(keys, libs, SplitConfig(n_holdout=1))
        assert split.holdout_keys == ["SHARED00000000"]


class TestVerifyNoLeakage:
    def test_passes_for_valid_split(self):
        keys, libs = make_rows()
        split = make_split(keys, libs, SplitConfig(n_holdout=10))
        verify_no_leakage(split, keys)

    def test_detects_overlapping_masks(self):
        keys, libs = make_rows()
        split = make_split(keys, libs, SplitConfig(n_holdout=10))
        split.train_mask[:] = True
        split.val_mask[:] = True
        with pytest.raises(AssertionError, match="overlap"):
            verify_no_leakage(split, keys)

    def test_detects_uncovered_rows(self):
        keys, libs = make_rows()
        split = make_split(keys, libs, SplitConfig(n_holdout=10))
        split.train_mask[:] = False
        split.val_mask[:] = False
        with pytest.raises(AssertionError, match="neither side"):
            verify_no_leakage(split, keys)

    def test_detects_class2_leak(self):
        """The check must fail loudly if a class-2 structure stays in train."""
        keys, libs = make_rows(n_structures=30, spectra_each=3)
        split = make_split(keys, libs, SplitConfig(n_holdout=9))
        class2 = [k for k, c in split.novelty_class.items() if c == CLASS_DATABASE]
        assert class2
        leaked = class2[0]
        rows = np.flatnonzero(keys == leaked)
        split.train_mask[rows[0]] = True
        split.val_mask[rows[0]] = False
        with pytest.raises(AssertionError, match="leak into train"):
            verify_no_leakage(split, keys)

    def test_class1_siblings_allowed_but_can_be_forbidden(self):
        keys, libs = make_rows(n_structures=30, spectra_each=3)
        split = make_split(keys, libs, SplitConfig(n_holdout=9))
        verify_no_leakage(split, keys, allow_class1_siblings=True)
        if any(c == CLASS_LIBRARY for c in split.novelty_class.values()):
            with pytest.raises(AssertionError, match="leak into train"):
                verify_no_leakage(split, keys, allow_class1_siblings=False)


class TestHoldoutTruth:
    def test_builds_truth_and_class_maps(self):
        keys, libs = make_rows(n_structures=20, spectra_each=2)
        smiles = np.asarray(["CCO"] * len(keys), dtype=object)
        split = make_split(keys, libs, SplitConfig(n_holdout=6))
        truth, classes = holdout_truth(split, keys, smiles)
        assert set(truth) == set(split.holdout_keys)
        assert set(classes) == set(split.holdout_keys)
        assert all(c in (1, 2, 3) for c in classes.values())

    def test_uses_molecule_ids_when_given(self):
        keys, libs = make_rows(n_structures=20, spectra_each=1)
        smiles = np.asarray(["CCO"] * len(keys), dtype=object)
        molecule_ids = np.asarray([f"m_{i}" for i in range(len(keys))], dtype=object)
        split = make_split(keys, libs, SplitConfig(n_holdout=5))
        truth, _ = holdout_truth(split, keys, smiles, molecule_ids=molecule_ids)
        assert all(k.startswith("m_") for k in truth)

    def test_summary_counts(self):
        keys, libs = make_rows(n_structures=40, spectra_each=2)
        split = make_split(keys, libs, SplitConfig(n_holdout=10))
        summary = split.summary()
        assert summary["train_rows"] + summary["val_rows"] == len(keys)
        total = sum(summary[f"class{c}_structures"] for c in (1, 2, 3))
        assert total == 10
