"""Parsimony weighting in Channel 5: cleavage depth, linear intensity, tolerance."""

from __future__ import annotations

import numpy as np
import pytest

from casmi.adducts import PROTON_MASS
from casmi.channels.fragmentation import (
    FragmentationConfig,
    explain_score,
    fragment_masses,
    fragment_masses_with_depth,
    fragmentation_scores,
)
from tests.conftest import MOLECULES


def smiles_of(name: str) -> str:
    return dict(MOLECULES)[name]


class TestDefaults:
    def test_parsimony_defaults(self):
        cfg = FragmentationConfig()
        assert cfg.mz_tolerance == pytest.approx(0.005)
        assert cfg.intensity_power == pytest.approx(1.0)
        assert cfg.two_break_weight == pytest.approx(0.6)
        assert 3 in cfg.h_shifts and -3 not in cfg.h_shifts


class TestFragmentDepth:
    def test_depths_aligned_and_bounded(self):
        masses, depths = fragment_masses_with_depth(smiles_of("aspirin"))
        assert masses.shape == depths.shape
        assert set(np.unique(depths).tolist()) <= {0, 1, 2}
        assert np.all(np.diff(masses) >= 0)

    def test_whole_molecule_is_depth_zero_and_single_breaks_depth_one(self):
        masses, depths = fragment_masses_with_depth(smiles_of("vanillin"))
        assert depths[np.argmax(masses)] == 0
        assert (depths == 1).any()
        assert (depths == 2).any()

    def test_depth_free_view_matches(self):
        masses, _ = fragment_masses_with_depth(smiles_of("caffeine"))
        np.testing.assert_array_equal(masses, fragment_masses(smiles_of("caffeine")))

    def test_single_break_only_has_no_depth_two(self):
        _, depths = fragment_masses_with_depth(
            smiles_of("quinoline"), FragmentationConfig(max_breaks=1)
        )
        assert not (depths == 2).any()

    def test_unparseable_returns_empty_pair(self):
        masses, depths = fragment_masses_with_depth("not smiles!!")
        assert masses.size == 0 and depths.size == 0


class TestParsimonyCredit:
    def test_two_break_only_peak_earns_reduced_credit(self):
        fragments = np.array([100.0, 150.0])
        depths = np.array([2, 1])
        cfg = FragmentationConfig(two_break_weight=0.6)
        deep = explain_score(
            fragments, np.array([100.0 + PROTON_MASS]), np.array([1.0]),
            config=cfg, fragment_depth=depths,
        )
        shallow = explain_score(
            fragments, np.array([150.0 + PROTON_MASS]), np.array([1.0]),
            config=cfg, fragment_depth=depths,
        )
        assert deep == pytest.approx(0.6)
        assert shallow == pytest.approx(1.0)

    def test_cheapest_explanation_wins_when_both_match(self):
        """Two fragments at the same mass, one depth 2 one depth 1: full credit."""
        fragments = np.array([120.0, 120.0])
        depths = np.array([2, 1])
        score = explain_score(
            fragments, np.array([120.0 + PROTON_MASS]), np.array([1.0]), fragment_depth=depths
        )
        assert score == pytest.approx(1.0)

    def test_no_depth_means_full_credit(self):
        score = explain_score(np.array([100.0]), np.array([100.0 + PROTON_MASS]), np.array([1.0]))
        assert score == pytest.approx(1.0)

    def test_misaligned_depth_raises(self):
        with pytest.raises(ValueError):
            explain_score(
                np.array([1.0, 2.0]), np.array([3.0]), np.array([1.0]),
                fragment_depth=np.array([1]),
            )

    def test_linear_vs_sqrt_intensity_weighting(self):
        """A weak explained peak counts less under linear weighting than sqrt."""
        fragments = np.array([100.0])
        peak_mz = np.array([100.0 + PROTON_MASS, 500.0])
        peak_intensity = np.array([0.04, 1.0])
        linear = explain_score(
            fragments, peak_mz, peak_intensity, config=FragmentationConfig(intensity_power=1.0)
        )
        root = explain_score(
            fragments, peak_mz, peak_intensity, config=FragmentationConfig(intensity_power=0.5)
        )
        assert linear == pytest.approx(0.04 / 1.04)
        assert root == pytest.approx(0.2 / 1.2)
        assert linear < root

    def test_tolerance_is_respected(self):
        fragments = np.array([100.0])
        peak_mz = np.array([100.0 + PROTON_MASS + 0.008])
        tight = explain_score(fragments, peak_mz, np.array([1.0]),
                              config=FragmentationConfig(mz_tolerance=0.005))
        loose = explain_score(fragments, peak_mz, np.array([1.0]),
                              config=FragmentationConfig(mz_tolerance=0.01))
        assert tight == 0.0 and loose == pytest.approx(1.0)

    def test_plus_three_hydrogen_shift_matches(self):
        fragments = np.array([100.0])
        peak_mz = np.array([100.0 + 3 * 1.00782503207 + PROTON_MASS])
        assert explain_score(fragments, peak_mz, np.array([1.0])) == pytest.approx(1.0)
        legacy = FragmentationConfig(h_shifts=(-2, -1, 0, 1, 2))
        assert explain_score(fragments, peak_mz, np.array([1.0]), config=legacy) == 0.0


class TestChannelUsesDepth:
    def test_scores_use_parsimony_credit(self):
        """A candidate whose only explanation is a double cleavage scores < 1."""
        smiles = smiles_of("aspirin")
        masses, depths = fragment_masses_with_depth(smiles)
        shallow = masses[depths <= 1]
        # A deep fragment can share an ion mass with a shallow one at a different
        # hydrogen shift, so pick one at least 4 Da from every shallow fragment.
        isolated = [
            m for m in masses[depths == 2] if np.all(np.abs(shallow - m) > 4.0)
        ]
        assert isolated, "fixture molecule has no isolated two-break fragment"
        peak_mz = np.array([isolated[0] + PROTON_MASS])
        scores = fragmentation_scores([smiles], [(peak_mz, np.array([1.0]))], positive_mode=True)
        assert scores[0] == pytest.approx(0.6, abs=1e-6)
