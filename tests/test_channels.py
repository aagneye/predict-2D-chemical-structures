"""Tests for the evidence channels and their fusion.

The key behavioural tests are that Channel 1 recovers a structure whose exact
spectrum is present, and that Channel 2 recovers structural evidence when it is
*not* — that second case is what has to work on the real hidden test set.
"""

import numpy as np
import pytest

from casmi.candidates.pool import build_pool
from casmi.channels.analog import (
    Analog,
    AnalogIndex,
    find_analogs,
    propagate_to_candidates,
)
from casmi.channels.fusion import (
    ScoredCandidate,
    fuse,
    mass_error_penalty,
    to_smiles_list,
)
from casmi.channels.library import library_search
from casmi.chem import FingerprintCalculator, exact_mass, inchikey14
from casmi.config import CFG, AnalogConfig, CandidateConfig, SpectrumConfig
from casmi.data.loaders import library_from_arrays, query_molecules_from_table
from tests.conftest import MOLECULES, build_query_table, synthetic_peaks

PLAIN = SpectrumConfig(entropy_weighting=False)


@pytest.fixture(scope="module")
def small_calc():
    return FingerprintCalculator(bit_subset=np.arange(0, 10407, 40))


def single_molecule(smiles: str, seed: int = 0, n_spectra: int = 1):
    table = build_query_table([("m_test", smiles, seed, n_spectra)])
    return query_molecules_from_table(table)[0]


def library_with(entries):
    """Build a library from ``(smiles, seed)`` pairs using matching precursors."""
    from casmi.adducts import PROTON_MASS

    mzs, ints, keys, smis, precs, adducts = [], [], [], [], [], []
    for smiles, seed in entries:
        mass = exact_mass(smiles)
        precursor = mass + PROTON_MASS
        mz, inten = synthetic_peaks(seed, precursor=max(precursor, 60.0))
        mzs.append(mz)
        ints.append(inten)
        keys.append(inchikey14(smiles))
        smis.append(smiles)
        precs.append(precursor)
        adducts.append("[M+H]+")
    return library_from_arrays(mzs, ints, keys, smis, precs, adducts)


class TestLibrarySearch:
    def test_finds_exact_spectrum_match(self):
        """Same molecule, same seed: entropy similarity must be ~1.0."""
        smiles = MOLECULES[0][1]
        library = library_with([(smiles, 0)])
        molecule = single_molecule(smiles, seed=0)
        hits = library_search(molecule, library, spectrum_config=PLAIN)
        key = inchikey14(smiles)
        assert key in hits
        assert hits[key].similarity == pytest.approx(1.0, abs=1e-4)

    def test_no_hit_outside_mass_window(self):
        """A library holding only a different-mass molecule yields nothing."""
        library = library_with([(MOLECULES[3][1], 5)])
        molecule = single_molecule(MOLECULES[0][1], seed=0)
        hits = library_search(
            molecule,
            library,
            spectrum_config=PLAIN,
            candidate_config=CandidateConfig(ppm_window=1.0, ppm_fallback=1.0),
        )
        assert inchikey14(MOLECULES[0][1]) not in hits

    def test_different_spectrum_same_mass_scores_lower(self):
        """Mass match alone must not produce a high similarity."""
        smiles = MOLECULES[0][1]
        library = library_with([(smiles, 999)])  # same molecule, different peaks
        molecule = single_molecule(smiles, seed=0)
        hits = library_search(molecule, library, spectrum_config=PLAIN)
        assert hits[inchikey14(smiles)].similarity < 0.95

    def test_aggregates_best_across_multiple_spectra(self):
        """Per-molecule scoring keeps the best match over all its spectra."""
        smiles = MOLECULES[0][1]
        library = library_with([(smiles, 1)])
        molecule = single_molecule(smiles, seed=0, n_spectra=3)  # seeds 0,1,2
        hits = library_search(molecule, library, spectrum_config=PLAIN)
        assert hits[inchikey14(smiles)].similarity == pytest.approx(1.0, abs=1e-4)
        assert hits[inchikey14(smiles)].n_supporting_spectra >= 1

    def test_nan_target_mass_returns_empty(self):
        library = library_with([(MOLECULES[0][1], 0)])
        molecule = single_molecule(MOLECULES[0][1], seed=0)
        molecule.neutral_masses = [float("nan")]
        assert library_search(molecule, library, spectrum_config=PLAIN) == {}

    def test_empty_library(self):
        library = library_from_arrays([], [], [], [], [], [])
        molecule = single_molecule(MOLECULES[0][1])
        assert library_search(molecule, library, spectrum_config=PLAIN) == {}

    def test_returns_smiles_for_hits(self):
        smiles = MOLECULES[0][1]
        library = library_with([(smiles, 0)])
        hits = library_search(single_molecule(smiles, 0), library, spectrum_config=PLAIN)
        assert inchikey14(hits[inchikey14(smiles)].smiles) == inchikey14(smiles)


class TestAnalogIndex:
    def test_one_representative_per_structure(self):
        smiles = MOLECULES[0][1]
        library = library_with([(smiles, 0), (smiles, 1), (MOLECULES[1][1], 2)])
        index = AnalogIndex(library)
        assert len(index) == 2

    def test_representatives_sorted_by_mass(self):
        library = library_with([(s, i) for i, (_, s) in enumerate(MOLECULES)])
        index = AnalogIndex(library)
        assert np.all(np.diff(index.masses) >= 0)

    def test_window_narrows_selection(self):
        library = library_with([(s, i) for i, (_, s) in enumerate(MOLECULES)])
        index = AnalogIndex(library)
        target = exact_mass(MOLECULES[0][1])
        narrow = index.window(target, 1.0)
        wide = index.window(target, 500.0)
        assert (narrow.stop - narrow.start) <= (wide.stop - wide.start)

    def test_empty_library(self):
        index = AnalogIndex(library_from_arrays([], [], [], [], [], []))
        assert len(index) == 0


class TestFindAnalogs:
    def test_recovers_mass_shifted_relative(self):
        """Channel 2's core premise, and the one that must work on hidden test.

        A reference molecule whose fragments are the query's shifted by dM must
        be found even though its mass is far outside any ppm window.
        """
        from casmi.adducts import PROTON_MASS

        query_smiles = MOLECULES[0][1]
        query_mass = exact_mass(query_smiles)
        mz, inten = synthetic_peaks(0, precursor=query_mass + PROTON_MASS)

        shift = 14.0157  # one CH2
        analog_mass = query_mass - shift
        library = library_from_arrays(
            mz_lists=[mz - np.float32(shift)],
            intensity_lists=[inten],
            inchikey14=["ANALOGKEY00000"],
            smiles=[MOLECULES[1][1]],
            precursor_mz=[analog_mass + PROTON_MASS],
            adduct=["[M+H]+"],
        )
        molecule = single_molecule(query_smiles, seed=0)
        analogs = find_analogs(
            molecule,
            AnalogIndex(library),
            spectrum_config=PLAIN,
            analog_config=AnalogConfig(window_da=200.0),
        )
        assert analogs
        assert analogs[0].inchikey14 == "ANALOGKEY00000"
        assert analogs[0].similarity > 0.9
        assert analogs[0].mass_shift == pytest.approx(shift, abs=0.01)

    def test_respects_window(self):
        library = library_with([(MOLECULES[3][1], 5)])
        molecule = single_molecule(MOLECULES[0][1], seed=0)
        analogs = find_analogs(
            molecule,
            AnalogIndex(library),
            spectrum_config=PLAIN,
            analog_config=AnalogConfig(window_da=0.001),
        )
        assert analogs == []

    def test_caps_returned_analogs(self):
        library = library_with([(s, i) for i, (_, s) in enumerate(MOLECULES)])
        molecule = single_molecule(MOLECULES[0][1], seed=0)
        analogs = find_analogs(
            molecule,
            AnalogIndex(library),
            spectrum_config=PLAIN,
            analog_config=AnalogConfig(window_da=1000.0, n_analogs=3),
        )
        assert len(analogs) <= 3

    def test_sorted_best_first(self):
        library = library_with([(s, i) for i, (_, s) in enumerate(MOLECULES)])
        molecule = single_molecule(MOLECULES[0][1], seed=0)
        analogs = find_analogs(
            molecule,
            AnalogIndex(library),
            spectrum_config=PLAIN,
            analog_config=AnalogConfig(window_da=1000.0),
        )
        sims = [a.similarity for a in analogs]
        assert sims == sorted(sims, reverse=True)

    def test_nan_mass_returns_empty(self):
        library = library_with([(MOLECULES[0][1], 0)])
        molecule = single_molecule(MOLECULES[0][1], seed=0)
        molecule.neutral_masses = [float("nan")]
        assert find_analogs(molecule, AnalogIndex(library), spectrum_config=PLAIN) == []


class TestPropagateToCandidates:
    def test_identical_structure_scores_highest(self, small_calc):
        """A candidate identical to a strong analog must top the propagation."""
        pool = build_pool([s for _, s in MOLECULES], calculator=small_calc)
        target_key = pool.keys[0]
        analogs = [Analog(inchikey14=target_key, similarity=1.0, mass_shift=14.0)]
        indices = np.arange(len(pool))
        features = propagate_to_candidates(indices, pool, analogs)
        assert features["analog_power"][0] == pytest.approx(
            features["analog_power"].max()
        )
        assert features["analog_best_tanimoto"][0] == pytest.approx(1.0, abs=1e-5)

    def test_sim_power_sharpens_weighting(self, small_calc):
        """Higher p must suppress weak analogs relative to strong ones."""
        pool = build_pool([s for _, s in MOLECULES], calculator=small_calc)
        analogs = [Analog(pool.keys[0], 0.5, 0.0)]
        indices = np.arange(len(pool))
        p1 = propagate_to_candidates(indices, pool, analogs, AnalogConfig(sim_power=1.0))
        p4 = propagate_to_candidates(indices, pool, analogs, AnalogConfig(sim_power=4.0))
        assert p4["analog_power"][0] < p1["analog_power"][0]

    def test_no_analogs_yields_zeros(self, small_calc):
        pool = build_pool([s for _, s in MOLECULES], calculator=small_calc)
        features = propagate_to_candidates(np.arange(3), pool, [])
        for value in features.values():
            assert np.all(value == 0.0)

    def test_no_candidates_yields_empty(self, small_calc):
        pool = build_pool([s for _, s in MOLECULES], calculator=small_calc)
        features = propagate_to_candidates(
            np.array([], dtype=np.int64), pool, [Analog(pool.keys[0], 1.0, 0.0)]
        )
        assert all(v.size == 0 for v in features.values())

    def test_analog_absent_from_pool_ignored(self, small_calc):
        pool = build_pool([s for _, s in MOLECULES], calculator=small_calc)
        features = propagate_to_candidates(
            np.arange(3), pool, [Analog("NOTINPOOL00000", 1.0, 0.0)]
        )
        assert np.all(features["analog_power"] == 0.0)

    def test_all_expected_features_present(self, small_calc):
        pool = build_pool([s for _, s in MOLECULES], calculator=small_calc)
        features = propagate_to_candidates(
            np.arange(3), pool, [Analog(pool.keys[0], 0.9, 0.0)]
        )
        assert set(features) == {
            "analog_power",
            "analog_linear",
            "analog_best_tanimoto",
            "analog_top_tanimoto",
            "analog_mean",
        }


class TestMassErrorPenalty:
    def test_exact_match_scores_one(self):
        assert mass_error_penalty(np.array([300.0]), 300.0, 10.0)[0] == pytest.approx(1.0)

    def test_window_edge_scores_zero(self):
        target = 300.0
        edge = target + target * 10.0 / 1e6
        assert mass_error_penalty(np.array([edge]), target, 10.0)[0] == pytest.approx(0.0, abs=1e-6)

    def test_monotonic_decay(self):
        target = 300.0
        values = mass_error_penalty(
            np.array([300.0, 300.001, 300.002]), target, 20.0
        )
        assert values[0] > values[1] > values[2]

    def test_nan_target(self):
        assert mass_error_penalty(np.array([300.0]), float("nan"), 10.0)[0] == 0.0


class TestFusion:
    def test_ranks_by_weighted_score(self):
        candidates = [
            ScoredCandidate("KEYA00000000AA", "CCO", 0.0, {"library_similarity": 0.2}),
            ScoredCandidate("KEYB00000000BB", "CCC", 0.0, {"library_similarity": 0.9}),
        ]
        ranked = fuse(candidates)
        assert ranked[0].inchikey14 == "KEYB00000000BB"

    def test_library_outranks_analog_when_both_present(self):
        """A near-perfect library match must beat a moderate analog score."""
        candidates = [
            ScoredCandidate("LIB00000000000", "CCO", 0.0, {"library_similarity": 0.95}),
            ScoredCandidate("ANA00000000000", "CCC", 0.0, {"analog_power": 0.6}),
        ]
        assert fuse(candidates)[0].inchikey14 == "LIB00000000000"

    def test_analog_wins_when_library_silent(self):
        """Class-2 behaviour: analog evidence ranks when no library hit exists."""
        candidates = [
            ScoredCandidate("A00000000000AA", "CCO", 0.0, {"analog_power": 0.1}),
            ScoredCandidate("B00000000000BB", "CCC", 0.0, {"analog_power": 0.8}),
        ]
        assert fuse(candidates)[0].inchikey14 == "B00000000000BB"

    def test_dedupes_by_key_keeping_best(self):
        candidates = [
            ScoredCandidate("SAME0000000000", "CCO", 0.0, {"library_similarity": 0.2}),
            ScoredCandidate("SAME0000000000", "CCO", 0.0, {"library_similarity": 0.9}),
        ]
        ranked = fuse(candidates)
        assert len(ranked) == 1
        assert ranked[0].score == pytest.approx(0.9)

    def test_truncates_to_top_n(self):
        candidates = [
            ScoredCandidate(f"KEY{i:011d}", "CCO", 0.0, {"library_similarity": i / 100})
            for i in range(60)
        ]
        assert len(fuse(candidates, top_n=25)) == 25

    def test_custom_weights_change_order(self):
        candidates = [
            ScoredCandidate("LIB00000000000", "CCO", 0.0, {"library_similarity": 0.5}),
            ScoredCandidate("ANA00000000000", "CCC", 0.0, {"analog_power": 0.5}),
        ]
        assert fuse(candidates, weights={"analog_power": 10.0})[0].inchikey14 == "ANA00000000000"

    def test_empty_input(self):
        assert fuse([]) == []

    def test_to_smiles_list_skips_blanks(self):
        ranked = [
            ScoredCandidate("A00000000000AA", "CCO", 1.0),
            ScoredCandidate("B00000000000BB", "", 0.5),
        ]
        assert to_smiles_list(ranked) == ["CCO"]

    def test_default_weights_cover_expected_channels(self):
        from casmi.channels.fusion import DEFAULT_WEIGHTS

        assert "library_similarity" in DEFAULT_WEIGHTS
        assert "analog_power" in DEFAULT_WEIGHTS
        assert CFG.top_n == 25
