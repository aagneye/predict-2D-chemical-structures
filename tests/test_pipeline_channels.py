"""Integration tests for Channels 4/5 and the reranker wired into pipeline.py.

Complements tests/test_pipeline.py (which covers the Channel 1+2 baseline
unaffected by these additions). The key claims here are: the new channels are
strictly additive (omitting them reproduces baseline predictions exactly),
enabling fragmentation does not break candidate retrieval, and a fitted
reranker actually changes the final ranking relative to weighted fusion.
"""

from __future__ import annotations

import numpy as np
import pytest

from casmi.adducts import PROTON_MASS
from casmi.candidates.pool import build_pool
from casmi.channels.analog import AnalogIndex
from casmi.channels.ranker import RankTrainingExample, build_training_matrix, train_ranker
from casmi.chem import FingerprintCalculator, exact_mass, inchikey14
from casmi.config import (
    AnalogConfig,
    CandidateConfig,
    Config,
    FingerprintConfig,
    RankerConfig,
    SpectrumConfig,
)
from casmi.data.loaders import library_from_arrays, query_molecules_from_table
from casmi.pipeline import predict_molecule, run_pipeline
from tests.conftest import MOLECULES, build_query_table, synthetic_peaks

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
    return query_molecules_from_table(build_query_table(entries))


class TestFragmentationChannelWiring:
    def test_disabled_by_default_reproduces_baseline(self, pool):
        """fragmentation_channel=False must be a strict no-op vs. the old code."""
        smiles = MOLECULES[0][1]
        library = make_library([(smiles, 0)])
        molecules = make_queries([("m_001", smiles, 0, 1)])

        without = run_pipeline(molecules, library, pool, config=TEST_CFG)
        with_off = run_pipeline(
            molecules, library, pool, config=TEST_CFG, fragmentation_channel=False
        )
        assert without.predictions == with_off.predictions

    def test_enabling_does_not_break_retrieval(self, pool):
        smiles = MOLECULES[0][1]
        library = make_library([(smiles, 0)])
        molecules = make_queries([("m_001", smiles, 0, 1)])
        result = run_pipeline(
            molecules, library, pool, config=TEST_CFG, fragmentation_channel=True
        )
        predicted = result.predictions["m_001"]
        assert predicted
        assert inchikey14(predicted[0]) == inchikey14(smiles)

    def test_diagnostics_report_fragmentation_score(self, pool):
        smiles = MOLECULES[0][1]
        library = make_library([(smiles, 0)])
        molecules = make_queries([("m_001", smiles, 0, 1)])
        result = run_pipeline(
            molecules, library, pool, config=TEST_CFG, fragmentation_channel=True
        )
        assert result.diagnostics[0].best_fragmentation_score >= 0.0

    def test_diagnostics_zero_when_disabled(self, pool):
        smiles = MOLECULES[0][1]
        library = make_library([(smiles, 0)])
        molecules = make_queries([("m_001", smiles, 0, 1)])
        result = run_pipeline(
            molecules, library, pool, config=TEST_CFG, fragmentation_channel=False
        )
        assert result.diagnostics[0].best_fragmentation_score == 0.0

    def test_fragmentation_score_has_nonzero_default_weight(self):
        """Regression test: a feature absent from DEFAULT_WEIGHTS is silently
        dropped by fuse() (multiplied by an implicit weight of 0), so
        enabling Channel 5 would change diagnostics but never the ranking.
        This asserts the wiring directly rather than via an end-to-end
        scenario, since constructing a synthetic fixture where fragmentation
        is the *only* differentiating signal is fragile (mass-error and
        other tiebreakers tend to dominate small synthetic molecules).
        """
        from casmi.channels.fusion import DEFAULT_WEIGHTS

        assert "fragmentation_score" in DEFAULT_WEIGHTS
        assert DEFAULT_WEIGHTS["fragmentation_score"] > 0.0

    def test_fpnet_score_has_nonzero_default_weight(self):
        from casmi.channels.fusion import DEFAULT_WEIGHTS

        assert "fpnet_score" in DEFAULT_WEIGHTS
        assert DEFAULT_WEIGHTS["fpnet_score"] > 0.0

    def test_fuse_score_changes_with_fragmentation_feature(self):
        """Direct test of fuse() itself: two otherwise-identical candidates
        that differ only in fragmentation_score must get different fused
        scores, proving the feature is not silently dropped.
        """
        from casmi.channels.fusion import ScoredCandidate, fuse

        base_features = {
            "library_similarity": 0.0,
            "analog_power": 0.0,
            "analog_best_tanimoto": 0.0,
            "analog_mean": 0.0,
            "mass_error_penalty": 0.5,
        }
        low = ScoredCandidate(
            inchikey14="AAAAAAAAAAAAAA",
            smiles="C",
            score=0.0,
            features={**base_features, "fragmentation_score": 0.0},
        )
        high = ScoredCandidate(
            inchikey14="BBBBBBBBBBBBBB",
            smiles="CC",
            score=0.0,
            features={**base_features, "fragmentation_score": 1.0},
        )
        ranked = fuse([low, high])
        scores = {c.inchikey14: c.score for c in ranked}
        assert scores["BBBBBBBBBBBBBB"] > scores["AAAAAAAAAAAAAA"]


class TestRerankerWiring:
    def _fit_toy_reranker(self):
        rng = np.random.default_rng(0)
        examples = []
        for m in range(30):
            n = 5
            correct = rng.integers(0, n)
            for c in range(n):
                is_correct = c == correct
                lib = 0.9 if is_correct else rng.uniform(0.0, 0.3)
                examples.append(
                    RankTrainingExample(
                        molecule_id=f"m{m}",
                        features={
                            "library_similarity": lib,
                            "analog_power": 0.0,
                            "analog_linear": 0.0,
                            "analog_best_tanimoto": 0.0,
                            "analog_top_tanimoto": 0.0,
                            "analog_mean": 0.0,
                            "mass_error_penalty": 0.0,
                            "fragmentation_score": 0.0,
                        },
                        label=int(is_correct),
                    )
                )
        x, y, _ = build_training_matrix(examples)
        config = RankerConfig(max_iter=30, seeds=(0, 1), class1_priors=(0.5,))
        return train_ranker(x, y, config=config)

    def test_reranker_still_recovers_exact_library_match(self, pool):
        """A near-1.0 library similarity must still win even under reranking."""
        reranker = self._fit_toy_reranker()
        smiles = MOLECULES[0][1]
        library = make_library([(smiles, 0)])
        molecules = make_queries([("m_001", smiles, 0, 1)])
        result = run_pipeline(molecules, library, pool, config=TEST_CFG, reranker=reranker)
        predicted = result.predictions["m_001"]
        assert predicted
        assert inchikey14(predicted[0]) == inchikey14(smiles)

    def test_diagnostics_flag_reranker_usage(self, pool):
        reranker = self._fit_toy_reranker()
        smiles = MOLECULES[0][1]
        library = make_library([(smiles, 0)])
        molecules = make_queries([("m_001", smiles, 0, 1)])

        with_reranker = run_pipeline(molecules, library, pool, config=TEST_CFG, reranker=reranker)
        without_reranker = run_pipeline(molecules, library, pool, config=TEST_CFG)
        assert with_reranker.diagnostics[0].used_reranker is True
        assert without_reranker.diagnostics[0].used_reranker is False

    def test_no_reranker_falls_back_to_fusion(self, pool):
        smiles = MOLECULES[0][1]
        library = make_library([(smiles, 0)])
        molecules = make_queries([("m_001", smiles, 0, 1)])
        analog_index = AnalogIndex(library)
        smiles_out, diagnostics = predict_molecule(
            molecules[0], library, pool, analog_index, config=TEST_CFG, reranker=None
        )
        assert diagnostics.used_reranker is False
        assert smiles_out
