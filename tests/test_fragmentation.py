"""Tests for Channel 5 (MetFrag-lite in-silico fragmentation).

Key behavioural claims: fragmenting a real molecule always includes its whole
mass, breaking bonds actually shrinks fragment mass, and a spectrum built from
a molecule's own fragment masses scores higher than one built from unrelated
masses.
"""

from __future__ import annotations

import numpy as np
import pytest

from casmi.adducts import PROTON_MASS
from casmi.channels.fragmentation import (
    FragmentationConfig,
    explain_score,
    fragment_masses,
    fragmentation_scores,
)
from casmi.chem import exact_mass
from tests.conftest import MOLECULES


def smiles_of(name: str) -> str:
    return dict(MOLECULES)[name]


class TestFragmentMasses:
    def test_includes_whole_molecule_mass(self):
        smiles = smiles_of("aspirin")
        masses = fragment_masses(smiles)
        whole = exact_mass(smiles)
        assert masses.size > 0
        assert np.any(np.abs(masses - whole) < 1e-3)

    def test_unparseable_smiles_returns_empty(self):
        assert fragment_masses("not a smiles!!").size == 0

    def test_fragments_are_smaller_than_whole_molecule(self):
        """Every fragment other than the whole molecule must be lighter."""
        smiles = smiles_of("vanillin")
        whole = exact_mass(smiles)
        masses = fragment_masses(smiles)
        assert masses.max() == pytest.approx(whole, abs=1e-3)
        # At least one genuine (smaller) fragment should exist for a molecule
        # with several acyclic bonds.
        assert masses.min() < whole - 1.0

    def test_no_bonds_returns_single_mass(self):
        """A single heavy atom (e.g. methane) has no bonds to break."""
        masses = fragment_masses("C")
        assert masses.size == 1

    def test_deterministic(self):
        smiles = smiles_of("caffeine")
        a = fragment_masses(smiles)
        b = fragment_masses(smiles)
        np.testing.assert_array_equal(a, b)

    def test_double_break_superset_of_single_break(self):
        """max_breaks=2 must find every fragment max_breaks=1 finds, plus more."""
        smiles = smiles_of("quinoline")
        one_break = fragment_masses(smiles, FragmentationConfig(max_breaks=1))
        two_breaks = fragment_masses(smiles, FragmentationConfig(max_breaks=2))
        assert set(np.round(one_break, 4)).issubset(set(np.round(two_breaks, 4)))
        assert two_breaks.size >= one_break.size

    def test_bond_cap_falls_back_to_whole_mass(self):
        """Exceeding max_bonds must not crash, and returns just the whole mass."""
        smiles = smiles_of("naphthalene")
        masses = fragment_masses(smiles, FragmentationConfig(max_bonds=0))
        assert masses.size == 1
        assert masses[0] == pytest.approx(exact_mass(smiles), abs=1e-3)


class TestExplainScore:
    def test_empty_inputs_score_zero(self):
        assert explain_score(np.zeros(0), np.array([100.0]), np.array([1.0])) == 0.0
        assert explain_score(np.array([100.0]), np.zeros(0), np.zeros(0)) == 0.0

    def test_exact_fragment_ion_is_fully_explained(self):
        """A peak placed exactly at [fragment+H]+ must explain ~all intensity."""
        fragment = np.array([150.0])
        peak_mz = np.array([150.0 + PROTON_MASS])
        peak_intensity = np.array([1.0])
        score = explain_score(fragment, peak_mz, peak_intensity, positive_mode=True)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_unrelated_peaks_score_low(self):
        fragment = np.array([150.0])
        peak_mz = np.array([317.123, 402.456])
        peak_intensity = np.array([1.0, 1.0])
        score = explain_score(fragment, peak_mz, peak_intensity, positive_mode=True)
        assert score == 0.0

    def test_negative_mode_uses_deprotonation(self):
        fragment = np.array([150.0])
        peak_mz = np.array([150.0 - PROTON_MASS])
        peak_intensity = np.array([1.0])
        assert explain_score(
            fragment, peak_mz, peak_intensity, positive_mode=False
        ) == pytest.approx(1.0, abs=1e-6)
        # A peak far outside the +/- H_SHIFTS window under positive-mode
        # assumptions should not match at all (the [M-H]- ion for this
        # fragment sits ~2 * PROTON_MASS below the [M+H]+ ion, well beyond
        # the +/-2 hydrogen shifts considered).
        config = FragmentationConfig(mz_tolerance=0.001)
        assert explain_score(
            fragment, peak_mz, peak_intensity, positive_mode=True, config=config
        ) == 0.0

    def test_partial_intensity_explained(self):
        """One matching peak and one non-matching peak: ~half by intensity."""
        fragment = np.array([150.0])
        peak_mz = np.array([150.0 + PROTON_MASS, 999.0])
        peak_intensity = np.array([1.0, 1.0])
        score = explain_score(fragment, peak_mz, peak_intensity, positive_mode=True)
        assert 0.0 < score < 1.0

    def test_score_bounded_in_unit_interval(self):
        rng = np.random.default_rng(0)
        fragment = rng.uniform(50, 300, size=20)
        peak_mz = rng.uniform(50, 300, size=15)
        peak_intensity = rng.uniform(0.01, 1.0, size=15)
        score = explain_score(fragment, peak_mz, peak_intensity)
        assert 0.0 <= score <= 1.0


class TestFragmentationScores:
    def test_matching_molecule_outscores_unrelated_one(self):
        """A spectrum built from a molecule's own fragments should favour it
        over a same-length but chemically unrelated candidate."""
        true_smiles = smiles_of("aspirin")
        decoy_smiles = smiles_of("indole")

        fragments = fragment_masses(true_smiles)
        # Build a synthetic spectrum from a handful of the true molecule's own
        # fragment ions so this is a positive control.
        chosen = fragments[fragments > 0][:4]
        peak_mz = chosen + PROTON_MASS
        peak_intensity = np.ones(len(chosen))

        scores = fragmentation_scores(
            [true_smiles, decoy_smiles], [(peak_mz, peak_intensity)], positive_mode=True
        )
        assert scores[0] > scores[1]

    def test_empty_spectra_returns_zeros(self):
        scores = fragmentation_scores([smiles_of("phenol")], [], positive_mode=True)
        np.testing.assert_array_equal(scores, np.zeros(1, dtype=np.float32))

    def test_best_of_multiple_spectra(self):
        """A candidate should take its best score across the molecule's spectra."""
        smiles = smiles_of("toluene")
        fragments = fragment_masses(smiles)
        good_mz = fragments[fragments > 0][:3] + PROTON_MASS
        good_intensity = np.ones(len(good_mz))
        bad_mz = np.array([999.0, 888.0])
        bad_intensity = np.array([1.0, 1.0])

        scores_good_first = fragmentation_scores(
            [smiles], [(good_mz, good_intensity), (bad_mz, bad_intensity)], positive_mode=True
        )
        scores_bad_first = fragmentation_scores(
            [smiles], [(bad_mz, bad_intensity), (good_mz, good_intensity)], positive_mode=True
        )
        assert scores_good_first[0] == pytest.approx(scores_bad_first[0], abs=1e-6)
        assert scores_good_first[0] > 0.0

    def test_output_length_matches_candidates(self):
        smiles_list = [smiles_of("phenol"), smiles_of("glucose"), smiles_of("alanine")]
        scores = fragmentation_scores(
            smiles_list, [(np.array([100.0]), np.array([1.0]))], positive_mode=True
        )
        assert len(scores) == len(smiles_list)

    def test_unparseable_candidate_scores_zero(self):
        scores = fragmentation_scores(
            ["garbage!!", smiles_of("phenol")],
            [(np.array([95.05]), np.array([1.0]))],
            positive_mode=True,
        )
        assert scores[0] == 0.0
