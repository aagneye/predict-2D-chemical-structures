"""Tests for spectral cleaning and entropy similarity."""

import numpy as np
import pytest

from casmi.config import SpectrumConfig
from casmi.spectra import (
    clean_spectrum,
    entropy_similarity,
    merge_spectra,
    search_library,
)

#: Cleaning config with entropy weighting off, so tests can reason about
#: probabilities directly as normalised intensities.
PLAIN = SpectrumConfig(entropy_weighting=False)


def make_flat_library(spectra):
    """Pack peak lists into the CSR-style layout ``search_library`` consumes."""
    offsets = np.zeros(len(spectra) + 1, dtype=np.int64)
    for i, (mz, _) in enumerate(spectra):
        offsets[i + 1] = offsets[i] + len(mz)
    all_mz = np.concatenate([np.asarray(m, np.float32) for m, _ in spectra])
    all_int = np.concatenate([np.asarray(i, np.float32) for _, i in spectra])
    return offsets, all_mz, all_int


class TestCleanSpectrum:
    def test_probabilities_sum_to_one(self):
        mz, p = clean_spectrum([100.0, 200.0, 300.0], [1.0, 0.5, 0.25], config=PLAIN)
        assert p.sum() == pytest.approx(1.0, abs=1e-6)

    def test_intensity_floor_drops_small_peaks(self):
        mz, p = clean_spectrum(
            [100.0, 200.0, 300.0], [1.0, 0.5, 0.0001], config=SpectrumConfig(
                intensity_floor=0.01, entropy_weighting=False
            )
        )
        assert 300.0 not in mz
        assert len(mz) == 2

    def test_max_peaks_keeps_most_intense(self):
        mz, p = clean_spectrum(
            [100.0, 200.0, 300.0, 400.0],
            [0.1, 1.0, 0.9, 0.2],
            config=SpectrumConfig(max_peaks=2, entropy_weighting=False),
        )
        assert sorted(mz.tolist()) == [200.0, 300.0]

    def test_output_sorted_by_mz(self):
        """Alignment in the similarity kernel assumes ascending m/z."""
        mz, _ = clean_spectrum(
            [300.0, 100.0, 200.0],
            [1.0, 0.9, 0.8],
            config=SpectrumConfig(max_peaks=2, entropy_weighting=False),
        )
        assert np.all(np.diff(mz) > 0)

    def test_precursor_filter_drops_heavy_peaks(self):
        mz, _ = clean_spectrum(
            [100.0, 200.0, 500.0], [1.0, 1.0, 1.0], config=PLAIN, precursor_mz=300.0
        )
        assert 500.0 not in mz

    def test_empty_input(self):
        mz, p = clean_spectrum([], [], config=PLAIN)
        assert mz.size == 0 and p.size == 0

    def test_all_zero_intensity(self):
        mz, p = clean_spectrum([100.0, 200.0], [0.0, 0.0], config=PLAIN)
        assert mz.size == 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shapes differ"):
            clean_spectrum([100.0, 200.0], [1.0], config=PLAIN)

    def test_entropy_weighting_changes_low_entropy_spectrum(self):
        """A spectrum dominated by one peak is re-weighted; the flag matters."""
        peaks = ([100.0, 200.0], [1.0, 0.01])
        _, plain = clean_spectrum(*peaks, config=SpectrumConfig(entropy_weighting=False))
        _, weighted = clean_spectrum(*peaks, config=SpectrumConfig(entropy_weighting=True))
        assert not np.allclose(plain, weighted)
        assert weighted.sum() == pytest.approx(1.0, abs=1e-6)


class TestEntropySimilarity:
    def test_identical_spectra_score_one(self):
        mz, p = clean_spectrum([100.0, 200.0, 300.0], [1.0, 0.6, 0.3], config=PLAIN)
        assert entropy_similarity(mz, p, mz, p) == pytest.approx(1.0, abs=1e-5)

    def test_disjoint_spectra_score_low(self):
        a_mz, a_p = clean_spectrum([100.0, 110.0], [1.0, 1.0], config=PLAIN)
        b_mz, b_p = clean_spectrum([500.0, 510.0], [1.0, 1.0], config=PLAIN)
        assert entropy_similarity(a_mz, a_p, b_mz, b_p) < 0.1

    def test_partial_overlap_is_intermediate(self):
        q_mz, q_p = clean_spectrum([100.0, 200.0, 300.0], [1.0, 1.0, 1.0], config=PLAIN)
        half_mz, half_p = clean_spectrum([100.0, 200.0, 900.0], [1.0, 1.0, 1.0], config=PLAIN)
        partial = entropy_similarity(q_mz, q_p, half_mz, half_p)
        assert 0.1 < partial < 1.0

    def test_empty_spectrum_scores_zero(self):
        mz, p = clean_spectrum([100.0], [1.0], config=PLAIN)
        assert entropy_similarity(mz, p, np.zeros(0), np.zeros(0)) == 0.0

    def test_tolerance_controls_alignment(self):
        a_mz, a_p = clean_spectrum([100.000], [1.0], config=PLAIN)
        b_mz, b_p = clean_spectrum([100.008], [1.0], config=PLAIN)
        assert entropy_similarity(a_mz, a_p, b_mz, b_p, tol=0.01) > 0.9
        assert entropy_similarity(a_mz, a_p, b_mz, b_p, tol=0.001) < 0.1

    def test_mass_shift_recovers_shifted_match(self):
        """Channel 2's premise: an analog's peaks align once shifted by dM."""
        q_mz, q_p = clean_spectrum([114.0, 214.0, 314.0], [1.0, 0.8, 0.6], config=PLAIN)
        a_mz, a_p = clean_spectrum([100.0, 200.0, 300.0], [1.0, 0.8, 0.6], config=PLAIN)
        unshifted = entropy_similarity(q_mz, q_p, a_mz, a_p)
        shifted = entropy_similarity(q_mz, q_p, a_mz, a_p, shift=14.0)
        assert shifted > 0.9
        assert shifted > unshifted

    def test_shift_returns_best_of_both(self):
        """Direct match must not be lost when a shift is also attempted."""
        mz, p = clean_spectrum([100.0, 200.0], [1.0, 1.0], config=PLAIN)
        assert entropy_similarity(mz, p, mz, p, shift=50.0) == pytest.approx(1.0, abs=1e-5)


class TestSearchLibrary:
    def test_finds_exact_match(self):
        library = [
            ([100.0, 200.0], [1.0, 0.5]),
            ([500.0, 600.0], [1.0, 0.5]),
        ]
        offsets, all_mz, all_int = make_flat_library(library)
        q_mz, q_p = clean_spectrum(*library[0], config=PLAIN)
        scores = search_library(
            q_mz, q_p, np.array([0, 1]), offsets, all_mz, all_int, config=PLAIN
        )
        assert scores[0] == pytest.approx(1.0, abs=1e-5)
        assert scores[1] < 0.1

    def test_empty_candidate_set(self):
        offsets, all_mz, all_int = make_flat_library([([100.0], [1.0])])
        q_mz, q_p = clean_spectrum([100.0], [1.0], config=PLAIN)
        assert search_library(
            q_mz, q_p, np.array([], dtype=np.int64), offsets, all_mz, all_int, config=PLAIN
        ).size == 0

    def test_shifted_search(self):
        library = [([100.0, 200.0, 300.0], [1.0, 0.8, 0.6])]
        offsets, all_mz, all_int = make_flat_library(library)
        q_mz, q_p = clean_spectrum([114.0, 214.0, 314.0], [1.0, 0.8, 0.6], config=PLAIN)
        scores = search_library(
            q_mz,
            q_p,
            np.array([0]),
            offsets,
            all_mz,
            all_int,
            config=PLAIN,
            shifts=np.array([14.0]),
        )
        assert scores[0] > 0.9

    def test_shift_length_mismatch_raises(self):
        offsets, all_mz, all_int = make_flat_library([([100.0], [1.0])])
        q_mz, q_p = clean_spectrum([100.0], [1.0], config=PLAIN)
        with pytest.raises(ValueError, match="!="):
            search_library(
                q_mz,
                q_p,
                np.array([0]),
                offsets,
                all_mz,
                all_int,
                config=PLAIN,
                shifts=np.array([1.0, 2.0]),
            )

    def test_handles_empty_library_row(self):
        """Rows with no peaks exist in train; they must score 0, not crash."""
        offsets, all_mz, all_int = make_flat_library([([], []), ([100.0], [1.0])])
        q_mz, q_p = clean_spectrum([100.0], [1.0], config=PLAIN)
        scores = search_library(
            q_mz, q_p, np.array([0, 1]), offsets, all_mz, all_int, config=PLAIN
        )
        assert scores[0] == 0.0
        assert scores[1] > 0.9


class TestMergeSpectra:
    def test_combines_distinct_peaks(self):
        mz, inten = merge_spectra([(np.array([100.0]), np.array([1.0])),
                                   (np.array([200.0]), np.array([1.0]))])
        assert mz.tolist() == [100.0, 200.0]

    def test_collapses_near_duplicates_keeping_max(self):
        """Peaks within tolerance collapse to one, keeping the stronger."""
        mz, inten = merge_spectra(
            [
                (np.array([100.000, 300.0]), np.array([0.4, 1.0])),
                (np.array([100.002, 300.0]), np.array([0.9, 1.0])),
            ]
        )
        assert len(mz) == 2
        # Both spectra are normalised by their own base peak (300.0 -> 1.0),
        # so the 100 Da peak keeps the larger relative intensity, 0.9.
        assert inten[0] == pytest.approx(0.9)

    def test_normalises_each_input_spectrum(self):
        """Raw scales differ across spectra; each is normalised before fusing."""
        mz, inten = merge_spectra(
            [(np.array([100.0]), np.array([1000.0])), (np.array([200.0]), np.array([0.5]))]
        )
        assert inten.max() == pytest.approx(1.0)
        assert len(mz) == 2

    def test_empty(self):
        mz, inten = merge_spectra([])
        assert mz.size == 0 and inten.size == 0

    def test_output_sorted(self):
        mz, _ = merge_spectra(
            [(np.array([300.0, 100.0]), np.array([1.0, 1.0])),
             (np.array([200.0]), np.array([1.0]))]
        )
        assert np.all(np.diff(mz) > 0)
