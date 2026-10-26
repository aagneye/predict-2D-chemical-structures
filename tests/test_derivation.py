"""Channel 8 (analog derivation): edits, mass bookkeeping, end-to-end recovery."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from casmi.channels.derivation import (
    TRANSFORMS,
    DerivationConfig,
    apply_transform,
    derive_candidates,
    derive_from_parent,
    formula_delta_mass,
)
from casmi.chem import FingerprintCalculator, exact_mass, inchikey14

#: Molecules that between them carry every motif a transform targets.
PROBES = [
    "Oc1ccccc1",                                   # phenol: aromatic CH, phenolic OH
    "COc1ccccc1O",                                 # guaiacol: O-methyl
    "CNCCc1ccccc1",                                # N-methyl amine, NH
    "OCC1OC(Oc2ccccc2)C(O)C(O)C1O",                # phenyl glucoside
    "CC1OC(Oc2ccccc2)C(O)C(O)C1O",                 # phenyl rhamnoside
    "OC1COC(Oc2ccccc2)C(O)C1O",                    # phenyl xyloside
    "CC(=O)OCCc1ccccc1",                           # acetate ester
    "CC(=O)CCC=CC",                                # ketone, C=C, CH2-CH2
    "OCCCC(O)CC",                                  # aliphatic alcohols
    "OC(=O)CCc1ccccc1",                            # carboxylic acid
    "CSCCC(N)C(=O)O",                              # thioether, primary amine
    "SCCO",                                        # thiol
    "CC(=O)NCc1ccccc1N",                           # amide NH, aniline NH2
    "O=CCc1ccccc1",                                # aldehyde
    "CC=NCC",                                      # imine
]


def test_formula_delta_counts_handles_mixed_signs():
    from casmi.channels.derivation import formula_delta_counts

    assert formula_delta_counts("+C6H10O5") == {"C": 6, "H": 10, "O": 5}
    assert formula_delta_counts("-C1H2") == {"C": -1, "H": -2}
    assert formula_delta_counts("+Cl1-H1") == {"Cl": 1, "H": -1}
    assert formula_delta_mass("+O1-H2") == pytest.approx(15.9949146 - 2 * 1.00782503, abs=1e-5)
    with pytest.raises(ValueError):
        formula_delta_counts("C1H2")


@pytest.fixture(scope="module")
def calc():
    return FingerprintCalculator()


class TestTransforms:
    def test_formula_delta_mass(self):
        assert formula_delta_mass("+O") == pytest.approx(15.9949146, abs=1e-6)
        assert formula_delta_mass("+C6H10O5") == pytest.approx(162.0528, abs=1e-4)
        assert formula_delta_mass("-C1H2") == pytest.approx(-14.01565, abs=1e-5)

    @pytest.mark.parametrize("index", range(len(TRANSFORMS)))
    def test_every_transform_fires_with_its_declared_delta(self, index):
        name, _, delta = TRANSFORMS[index]
        expected = formula_delta_mass(delta)
        fired = False
        for smiles in PROBES:
            parent = Chem.MolFromSmiles(smiles)
            for product in apply_transform(parent, index, limit=50):
                fired = True
                assert ExactMolWt(product) - ExactMolWt(parent) == pytest.approx(expected, abs=1e-3)
        assert fired, f"{name} never fired on the probe set"

    def test_acid_oh_is_not_glycosylated(self):
        index = [t[0] for t in TRANSFORMS].index("O_hexose")
        products = apply_transform(Chem.MolFromSmiles("OC(=O)CCC"), index, limit=10)
        assert products == []


class TestDeriveFromParent:
    def test_one_step_methylation_reaches_anisole(self):
        found = derive_from_parent("Oc1ccccc1", exact_mass("COc1ccccc1"))
        keys = {inchikey14(s) for s in found}
        assert inchikey14("COc1ccccc1") in keys
        assert inchikey14("Cc1ccccc1O") in keys  # C-methyl isomer, same formula
        assert all(steps == 1 for _, steps, _ in found.values())

    def test_two_step_hydroxylation_plus_methylation_reaches_guaiacol(self):
        found = derive_from_parent("Oc1ccccc1", exact_mass("COc1ccccc1O"))
        assert inchikey14("COc1ccccc1O") in {inchikey14(s) for s in found}

    def test_deglycosylation_recovers_the_aglycone(self):
        found = derive_from_parent("OCC1OC(Oc2ccccc2)C(O)C(O)C1O", exact_mass("Oc1ccccc1"))
        assert inchikey14("Oc1ccccc1") in {inchikey14(s) for s in found}

    def test_glycosylation_of_a_flavonol(self):
        kaempferol = "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12"
        astragalin = "O=c1c(OC2OC(CO)C(O)C(O)C2O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12"
        found = derive_from_parent(kaempferol, exact_mass(astragalin))
        assert inchikey14(astragalin) in {inchikey14(s) for s in found}

    def test_same_mass_parent_is_skipped(self):
        assert derive_from_parent("Oc1ccccc1", exact_mass("Oc1ccccc1")) == {}

    def test_no_edit_explains_the_delta(self):
        assert derive_from_parent("Oc1ccccc1", exact_mass("Oc1ccccc1") + 3.21) == {}


class TestDeriveCandidates:
    def test_ranked_by_fpnet_and_excludes_known_keys(self, calc):
        target = "COc1ccccc1"
        logits = calc.from_smiles(target).astype(np.float32) * 4.0 - 2.0
        out = derive_candidates(
            [("Oc1ccccc1", 0.8)], exact_mass(target), calc, logits, exclude_keys=set()
        )
        assert out[0].inchikey14 == inchikey14(target)
        assert out[0].parent_similarity == pytest.approx(0.8)
        assert out[0].steps == 1
        excluded = derive_candidates(
            [("Oc1ccccc1", 0.8)], exact_mass(target), calc, logits,
            exclude_keys={inchikey14(target)},
        )
        assert inchikey14(target) not in {c.inchikey14 for c in excluded}

    def test_support_counts_distinct_parents(self, calc):
        target = "COc1ccccc1O"  # guaiacol: catechol + CH2, or phenol + O + CH2
        out = derive_candidates(
            [("Oc1ccccc1O", 0.7), ("Oc1ccccc1", 0.6)], exact_mass(target), calc, None,
            exclude_keys=set(),
        )
        hit = next(c for c in out if c.inchikey14 == inchikey14(target))
        assert hit.support == 2
        assert hit.steps == 1
        assert hit.parent_similarity == pytest.approx(0.7)

    def test_low_similarity_parents_are_ignored(self, calc):
        out = derive_candidates(
            [("Oc1ccccc1", 0.1)], exact_mass("COc1ccccc1"), calc, None, exclude_keys=set(),
            config=DerivationConfig(min_parent_similarity=0.3),
        )
        assert out == []
