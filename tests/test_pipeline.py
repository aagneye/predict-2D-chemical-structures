"""End-to-end pipeline tests.

The important assertions are that a molecule whose spectrum is in the library
is recovered at rank 1 (class-1 behaviour), that a molecule whose spectrum is
absent can still be recovered via analog propagation (class-2 behaviour, the
regime that matters on the hidden test set), and that a molecule excluded from
the pool entirely is *not* recovered (class-3, confirming our synthetic class-3
cohort is genuinely unreachable rather than accidentally leaking).
"""

import csv

import numpy as np
import pytest

from casmi.adducts import PROTON_MASS
from casmi.candidates.pool import build_pool
from casmi.chem import FingerprintCalculator, exact_mass, inchikey14
from casmi.config import (
    AnalogConfig,
    CandidateConfig,
    Config,
    FingerprintConfig,
    SpectrumConfig,
)
from casmi.data.loaders import library_from_arrays, query_molecules_from_table
from casmi.eval.metrics import evaluate_predictions, mrr_at_k
from casmi.pipeline import run_pipeline, write_submission
from tests.conftest import MOLECULES, build_query_table, synthetic_peaks

#: Config with entropy weighting off and a generous window, so tests assert on
#: retrieval behaviour rather than on cleaning subtleties.
TEST_CFG = Config(
    spectrum=SpectrumConfig(entropy_weighting=False),
    candidates=CandidateConfig(ppm_window=50.0, ppm_fallback=200.0, max_candidates=80),
    analog=AnalogConfig(window_da=300.0, n_analogs=50),
    fingerprint=FingerprintConfig(),
)


@pytest.fixture(scope="module")
def small_calc():
    return FingerprintCalculator(bit_subset=np.arange(0, 10407, 40))


@pytest.fixture(scope="module")
def pool(small_calc):
    return build_pool([s for _, s in MOLECULES], calculator=small_calc)


def make_library(entries):
    """``entries`` = list of ``(smiles, seed)``."""
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


def make_queries(entries):
    """``entries`` = list of ``(molecule_id, smiles, seed, n_spectra)``."""
    return query_molecules_from_table(build_query_table(entries))


class TestClass1Behaviour:
    def test_exact_library_match_ranked_first(self, pool):
        """A molecule with a reference spectrum must be found at rank 1."""
        smiles = MOLECULES[0][1]
        library = make_library([(smiles, 0)])
        molecules = make_queries([("m_001", smiles, 0, 1)])
        result = run_pipeline(molecules, library, pool, config=TEST_CFG)
        predicted = result.predictions["m_001"]
        assert predicted
        assert inchikey14(predicted[0]) == inchikey14(smiles)

    def test_scores_perfect_mrr(self, pool):
        smiles = MOLECULES[0][1]
        library = make_library([(smiles, 0)])
        molecules = make_queries([("m_001", smiles, 0, 1)])
        result = run_pipeline(molecules, library, pool, config=TEST_CFG)
        scores = evaluate_predictions(result.predictions, {"m_001": smiles})
        assert mrr_at_k(scores) == pytest.approx(1.0)

    def test_diagnostics_report_strong_library_hit(self, pool):
        smiles = MOLECULES[0][1]
        library = make_library([(smiles, 0)])
        molecules = make_queries([("m_001", smiles, 0, 1)])
        result = run_pipeline(molecules, library, pool, config=TEST_CFG)
        diag = result.diagnostics[0]
        assert diag.best_library_similarity > 0.95
        assert diag.top_source == "library"
        assert diag.n_spectra == 1


class TestClass2Behaviour:
    def test_analog_propagation_recovers_without_exact_spectrum(self, small_calc):
        """The class-2 regime: no reference spectrum, but a shifted analog exists.

        The library contains only a *different* molecule whose fragments are the
        query's shifted by one CH2. Channel 1 cannot fire, so any recovery of
        the true structure must come from analog propagation.
        """
        true_smiles = MOLECULES[0][1]
        true_mass = exact_mass(true_smiles)
        mz, inten = synthetic_peaks(0, precursor=true_mass + PROTON_MASS)

        shift = 14.0157
        analog_smiles = MOLECULES[1][1]
        library = library_from_arrays(
            mz_lists=[mz - np.float32(shift)],
            intensity_lists=[inten],
            inchikey14=[inchikey14(analog_smiles)],
            smiles=[analog_smiles],
            precursor_mz=[true_mass - shift + PROTON_MASS],
            adduct=["[M+H]+"],
        )
        # Pool holds both, so the true structure is retrievable by mass.
        pool = build_pool([true_smiles, analog_smiles], calculator=small_calc)
        molecules = make_queries([("m_002", true_smiles, 0, 1)])

        result = run_pipeline(molecules, library, pool, config=TEST_CFG)
        diag = result.diagnostics[0]
        assert diag.best_analog_similarity > 0.9
        assert diag.n_analogs >= 1
        # The true structure must appear somewhere in the ranked list.
        keys = [inchikey14(s) for s in result.predictions["m_002"]]
        assert inchikey14(true_smiles) in keys

    def test_candidate_from_pool_even_without_any_library_hit(self, pool):
        """Mass-window retrieval alone must still populate candidates."""
        smiles = MOLECULES[5][1]
        library = make_library([(MOLECULES[3][1], 7)])
        molecules = make_queries([("m_003", smiles, 3, 1)])
        result = run_pipeline(molecules, library, pool, config=TEST_CFG)
        assert result.predictions["m_003"]


class TestClass3Behaviour:
    def test_structure_excluded_from_pool_is_not_predicted(self, pool):
        """Confirms the synthetic class-3 cohort is genuinely unreachable.

        If this failed, class-3 metrics would be silently inflated and the
        hardest cohort would look solved when it is not.
        """
        smiles = MOLECULES[0][1]
        key = inchikey14(smiles)
        reduced = pool.exclude_keys({key})
        library = make_library([(MOLECULES[3][1], 9)])
        molecules = make_queries([("m_004", smiles, 0, 1)])
        result = run_pipeline(molecules, library, reduced, config=TEST_CFG)
        keys = [inchikey14(s) for s in result.predictions["m_004"]]
        assert key not in keys

    def test_scores_zero_when_unreachable(self, pool):
        smiles = MOLECULES[0][1]
        reduced = pool.exclude_keys({inchikey14(smiles)})
        library = make_library([(MOLECULES[3][1], 9)])
        molecules = make_queries([("m_004", smiles, 0, 1)])
        result = run_pipeline(molecules, library, reduced, config=TEST_CFG)
        scores = evaluate_predictions(result.predictions, {"m_004": smiles})
        assert mrr_at_k(scores) == 0.0


class TestMultiSpectrumAggregation:
    def test_predictions_are_per_molecule_not_per_spectrum(self, pool):
        """Three spectra of one molecule yield exactly one ranked list."""
        smiles = MOLECULES[0][1]
        library = make_library([(smiles, 1)])
        molecules = make_queries([("m_005", smiles, 0, 3)])
        result = run_pipeline(molecules, library, pool, config=TEST_CFG)
        assert set(result.predictions) == {"m_005"}
        assert result.diagnostics[0].n_spectra == 3


class TestPipelineRobustness:
    def test_handles_empty_library(self, pool):
        library = library_from_arrays([], [], [], [], [], [])
        molecules = make_queries([("m_006", MOLECULES[0][1], 0, 1)])
        result = run_pipeline(molecules, library, pool, config=TEST_CFG)
        assert "m_006" in result.predictions

    def test_handles_molecule_with_unresolvable_mass(self, pool):
        library = make_library([(MOLECULES[0][1], 0)])
        molecules = make_queries([("m_007", MOLECULES[0][1], 0, 1)])
        molecules[0].neutral_masses = [float("nan")]
        result = run_pipeline(molecules, library, pool, config=TEST_CFG)
        assert result.predictions["m_007"] == []

    def test_respects_top_n_cap(self, pool):
        library = make_library([(s, i) for i, (_, s) in enumerate(MOLECULES)])
        molecules = make_queries([("m_008", MOLECULES[0][1], 0, 1)])
        result = run_pipeline(molecules, library, pool, config=TEST_CFG)
        assert len(result.predictions["m_008"]) <= TEST_CFG.top_n

    def test_no_duplicate_skeletons_in_output(self, pool):
        library = make_library([(s, i) for i, (_, s) in enumerate(MOLECULES)])
        molecules = make_queries([("m_009", MOLECULES[0][1], 0, 1)])
        result = run_pipeline(molecules, library, pool, config=TEST_CFG)
        keys = [inchikey14(s) for s in result.predictions["m_009"]]
        assert len(keys) == len(set(keys))

    def test_multiple_molecules(self, pool):
        library = make_library([(s, i) for i, (_, s) in enumerate(MOLECULES)])
        molecules = make_queries(
            [
                ("m_a", MOLECULES[0][1], 0, 1),
                ("m_b", MOLECULES[1][1], 10, 2),
                ("m_c", MOLECULES[2][1], 20, 1),
            ]
        )
        result = run_pipeline(molecules, library, pool, config=TEST_CFG)
        assert set(result.predictions) == {"m_a", "m_b", "m_c"}
        assert len(result.diagnostics) == 3


class TestChannelContribution:
    def test_reports_library_dominance(self, pool):
        """The leakage check: library-carried fraction across a cohort.

        Reproduces the diagnostic that revealed the public-LB leader's score
        was leakage-driven (100% of molecules at similarity 1.0).
        """
        library = make_library([(s, i * 10) for i, (_, s) in enumerate(MOLECULES[:3])])
        molecules = make_queries(
            [(f"m_{i}", s, i * 10, 1) for i, (_, s) in enumerate(MOLECULES[:3])]
        )
        result = run_pipeline(molecules, library, pool, config=TEST_CFG)
        contribution = result.channel_contribution()
        assert contribution["library"] == pytest.approx(1.0)
        assert contribution["analog_or_other"] == pytest.approx(0.0)

    def test_empty_diagnostics(self):
        from casmi.pipeline import PipelineResult

        assert PipelineResult().channel_contribution()["library"] == 0.0


class TestWriteSubmission:
    def test_format_is_valid(self, tmp_path):
        path = tmp_path / "submission.csv"
        write_submission({"m_1": ["CCO", "CCC"]}, ["m_1", "m_2"], str(path))

        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows[0] == ["molecule_id", "smiles"]
        assert len(rows) == 3
        for row in rows[1:]:
            assert len(row[1].split(";")) == 25

    def test_pads_short_lists(self, tmp_path):
        path = tmp_path / "s.csv"
        write_submission({"m_1": ["CCO"]}, ["m_1"], str(path))
        with open(path, encoding="utf-8") as handle:
            line = handle.readlines()[1]
        parts = line.strip().split(",")[1].split(";")
        assert parts[0] == "CCO"
        assert len(parts) == 25

    def test_truncates_long_lists(self, tmp_path):
        path = tmp_path / "s.csv"
        write_submission({"m_1": ["CCO"] * 40}, ["m_1"], str(path))
        with open(path, encoding="utf-8") as handle:
            line = handle.readlines()[1]
        assert len(line.strip().split(",")[1].split(";")) == 25

    def test_missing_molecule_gets_filler(self, tmp_path):
        """Every requested molecule must appear, even with no prediction."""
        path = tmp_path / "s.csv"
        write_submission({}, ["m_1"], str(path))
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows[1][0] == "m_1"
        assert len(rows[1][1].split(";")) == 25
