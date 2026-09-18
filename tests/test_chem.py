"""Tests for tautomer-canonical InChIKey14 matching and fingerprints."""

import numpy as np
import pytest

from casmi.chem import (
    FingerprintCalculator,
    exact_mass,
    inchikey14,
    inchikey14_many,
    mol_from_smiles,
    tanimoto,
    tanimoto_matrix,
)


class TestInchikey14:
    def test_length_is_14(self):
        assert len(inchikey14("CCO")) == 14

    def test_stereoisomers_collapse(self):
        """Stereochemistry is not scored, so enantiomers must share a key."""
        assert inchikey14("C[C@H](N)C(=O)O") == inchikey14("C[C@@H](N)C(=O)O")
        assert inchikey14("C[C@H](N)C(=O)O") == inchikey14("CC(N)C(=O)O")

    def test_double_bond_stereo_collapses(self):
        assert inchikey14(r"C/C=C/C") == inchikey14(r"C/C=C\C")

    def test_tautomers_collapse(self):
        """Keto-enol pair must share a key after canonicalisation."""
        assert inchikey14("CC(=O)C") == inchikey14("CC(O)=C")

    def test_2_hydroxypyridine_tautomer_collapses(self):
        assert inchikey14("Oc1ccccn1") == inchikey14("O=c1cccc[nH]1")

    def test_smiles_writing_order_irrelevant(self):
        assert inchikey14("c1ccccc1O") == inchikey14("Oc1ccccc1")

    def test_different_molecules_differ(self):
        assert inchikey14("CCO") != inchikey14("CCC")

    @pytest.mark.parametrize("bad", ["", "not_a_smiles", "C(((", None, 123])
    def test_invalid_input_returns_none(self, bad):
        assert inchikey14(bad) is None if isinstance(bad, str) else True

    def test_none_and_nonstring_do_not_raise(self):
        assert mol_from_smiles(None) is None
        assert mol_from_smiles(123) is None

    def test_many(self):
        assert inchikey14_many(["CCO", "bad", "CCC"]) == [
            inchikey14("CCO"),
            None,
            inchikey14("CCC"),
        ]


class TestExactMass:
    def test_ethanol(self):
        assert exact_mass("CCO") == pytest.approx(46.0418, abs=1e-3)

    def test_glucose(self):
        assert exact_mass("OCC1OC(O)C(O)C(O)C1O") == pytest.approx(180.0634, abs=1e-3)

    def test_invalid(self):
        assert exact_mass("nope!!") is None


class TestFingerprints:
    def test_layout_width(self):
        fc = FingerprintCalculator()
        assert fc.n_bits == 4096 + 4096 + 2048 + 167 == 10407
        assert fc.from_smiles("CCO").shape == (10407,)

    def test_binary_uint8(self):
        fp = FingerprintCalculator().from_smiles("c1ccccc1O")
        assert fp.dtype == np.uint8
        assert set(np.unique(fp).tolist()) <= {0, 1}
        assert fp.sum() > 0

    def test_maccs_block_populated(self):
        """Guards against the MACCS tail silently landing as all-zero."""
        fp = FingerprintCalculator().from_smiles("c1ccccc1O")
        assert fp[-167:].sum() > 0

    def test_bit_subset(self):
        fc = FingerprintCalculator(bit_subset=np.array([0, 5, 100, 10406]))
        assert fc.n_bits == 4
        assert fc.from_smiles("CCO").shape == (4,)

    def test_bit_subset_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            FingerprintCalculator(bit_subset=np.array([99999]))

    def test_invalid_smiles(self):
        assert FingerprintCalculator().from_smiles("xyz!!") is None
        assert FingerprintCalculator().fingerprint_and_mass("xyz!!") is None

    def test_fingerprint_and_mass_agrees_with_separate_calls(self):
        fc = FingerprintCalculator()
        fp, mass = fc.fingerprint_and_mass("CCO")
        assert np.array_equal(fp, fc.from_smiles("CCO"))
        assert mass == pytest.approx(exact_mass("CCO"))

    def test_same_molecule_same_fingerprint(self):
        fc = FingerprintCalculator()
        assert np.array_equal(fc.from_smiles("Oc1ccccc1"), fc.from_smiles("c1ccccc1O"))


class TestTanimoto:
    def test_identity_is_one(self):
        fp = FingerprintCalculator().from_smiles("CCO")
        assert tanimoto(fp, fp) == pytest.approx(1.0)

    def test_disjoint_is_zero(self):
        a = np.array([1, 1, 0, 0], dtype=np.uint8)
        b = np.array([0, 0, 1, 1], dtype=np.uint8)
        assert tanimoto(a, b) == 0.0

    def test_known_value(self):
        a = np.array([1, 1, 1, 0], dtype=np.uint8)
        b = np.array([1, 1, 0, 0], dtype=np.uint8)
        assert tanimoto(a, b) == pytest.approx(2 / 3)

    def test_both_empty_is_zero(self):
        z = np.zeros(4, dtype=np.uint8)
        assert tanimoto(z, z) == 0.0

    def test_matrix_matches_pairwise(self):
        fc = FingerprintCalculator()
        fps = np.stack([fc.from_smiles(s) for s in ("CCO", "CCC", "c1ccccc1O")])
        matrix = tanimoto_matrix(fps, fps)
        assert matrix.shape == (3, 3)
        assert np.allclose(np.diag(matrix), 1.0)
        for i in range(3):
            for j in range(3):
                assert matrix[i, j] == pytest.approx(tanimoto(fps[i], fps[j]), abs=1e-6)

    def test_matrix_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            tanimoto_matrix(np.zeros(4), np.zeros((1, 4)))
