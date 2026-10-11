"""Popularity prior wiring: feature assembly, fusion weight, post-rank blend."""

from __future__ import annotations

import numpy as np
import pytest

from casmi.candidates.pool import build_pool
from casmi.channels.analog import AnalogIndex
from casmi.channels.fusion import DEFAULT_WEIGHTS, ScoredCandidate, fuse
from casmi.channels.ranker import FEATURE_NAMES, assemble_features
from casmi.chem import FingerprintCalculator, inchikey14
from casmi.pipeline import build_scored_candidates, predict_molecule
from casmi.priors import PopularityPrior
from tests.test_pipeline_channels import make_library, make_queries

#: Two constitutional isomers (C4H8O2, identical exact mass): the situation the
#: prior exists for. Nothing spectral distinguishes them in these fixtures.
BUTANOIC_ACID = "CCCC(=O)O"
ETHYL_ACETATE = "CCOC(=O)C"
#: A library molecule far outside the ±200 Da analog window, so no analog
#: evidence can break the tie either.
FAR_AWAY = "CCCCCCCCCCCCCCCCCCCCCC"


class TestFusionAndRankerFeatures:
    def test_popularity_has_nonzero_default_weight(self):
        """A feature absent from DEFAULT_WEIGHTS is silently zero-weighted (see
        the fragmentation/fpnet bug in the git history), so assert it is there."""
        assert DEFAULT_WEIGHTS.get("popularity", 0.0) > 0.0

    def test_popularity_breaks_an_otherwise_exact_tie(self):
        a = ScoredCandidate("AAAAAAAAAAAAAA", "CCO", 0.0, {"analog_power": 0.3, "popularity": 0.0})
        b = ScoredCandidate("BBBBBBBBBBBBBB", "CCC", 0.0, {"analog_power": 0.3, "popularity": 15.0})
        assert fuse([a, b])[0].inchikey14 == b.inchikey14

    def test_popularity_cannot_outrank_a_library_hit_under_fusion(self):
        hit = ScoredCandidate("AAAAAAAAAAAAAA", "CCO", 0.0, {"library_similarity": 0.95})
        famous = ScoredCandidate("BBBBBBBBBBBBBB", "CCC", 0.0, {"popularity": 20.0})
        assert fuse([famous, hit])[0].inchikey14 == hit.inchikey14

    def test_ranker_feature_columns_present_and_relative(self):
        for name in ("popularity", "popularity_rank", "popularity_z"):
            assert name in FEATURE_NAMES
        dicts = [{"popularity": 1.0}, {"popularity": 10.0}, {"popularity": 5.0}]
        matrix = assemble_features(dicts)
        col = {n: i for i, n in enumerate(FEATURE_NAMES)}
        assert matrix[:, col["popularity"]].tolist() == pytest.approx([1.0, 10.0, 5.0])
        assert matrix[1, col["popularity_rank"]] == pytest.approx(1.0)
        assert matrix[:, col["popularity_z"]].mean() == pytest.approx(0.0, abs=1e-6)


@pytest.fixture(scope="module")
def isomer_pool():
    calc = FingerprintCalculator(bit_subset=np.arange(0, 10407, 40))
    # Butanoic acid first so, on an exact tie, it wins by insertion order and
    # the prior has to do real work to promote ethyl acetate.
    return build_pool([BUTANOIC_ACID, ETHYL_ACETATE, "CCO"], calculator=calc)


@pytest.fixture(scope="module")
def setup(isomer_pool):
    library = make_library([(FAR_AWAY, 0)])
    molecule = make_queries([("m_1", BUTANOIC_ACID, 3, 2)])[0]
    prior = PopularityPrior({inchikey14(ETHYL_ACETATE): 18.0})
    return library, AnalogIndex(library), molecule, prior


#: Fusion weights with the prior switched off, to isolate the post-rank blend.
NO_PRIOR_WEIGHTS = {k: v for k, v in DEFAULT_WEIGHTS.items() if k != "popularity"}


class TestPipelineWiring:
    def test_both_isomers_are_candidates(self, isomer_pool, setup):
        library, index, molecule, _ = setup
        candidates, _ = build_scored_candidates(molecule, library, isomer_pool, index)
        keys = {c.inchikey14 for c in candidates}
        assert {inchikey14(BUTANOIC_ACID), inchikey14(ETHYL_ACETATE)} <= keys

    def test_feature_absent_without_prior(self, isomer_pool, setup):
        library, index, molecule, _ = setup
        candidates, _ = build_scored_candidates(molecule, library, isomer_pool, index)
        assert all(c.features.get("popularity", 0.0) == 0.0 for c in candidates)

    def test_feature_populated_with_prior(self, isomer_pool, setup):
        library, index, molecule, prior = setup
        candidates, _ = build_scored_candidates(
            molecule, library, isomer_pool, index, popularity=prior
        )
        by_key = {c.inchikey14: c.features["popularity"] for c in candidates}
        assert by_key[inchikey14(ETHYL_ACETATE)] == pytest.approx(18.0)
        assert by_key[inchikey14(BUTANOIC_ACID)] == 0.0

    def test_without_prior_the_tie_goes_to_pool_order(self, isomer_pool, setup):
        library, index, molecule, _ = setup
        ranked, _ = predict_molecule(molecule, library, isomer_pool, index)
        assert inchikey14(ranked[0]) == inchikey14(BUTANOIC_ACID)

    def test_prior_as_a_fusion_feature_promotes_the_documented_isomer(self, isomer_pool, setup):
        library, index, molecule, prior = setup
        ranked, _ = predict_molecule(molecule, library, isomer_pool, index, popularity=prior)
        assert inchikey14(ranked[0]) == inchikey14(ETHYL_ACETATE)

    def test_post_rank_blend_is_a_noop_at_mu_zero(self, isomer_pool, setup):
        library, index, molecule, prior = setup
        ranked, _ = predict_molecule(
            molecule, library, isomer_pool, index,
            weights=NO_PRIOR_WEIGHTS, popularity=prior, popularity_mu=0.0,
        )
        assert inchikey14(ranked[0]) == inchikey14(BUTANOIC_ACID)

    def test_post_rank_blend_promotes_the_documented_isomer(self, isomer_pool, setup):
        library, index, molecule, prior = setup
        ranked, _ = predict_molecule(
            molecule, library, isomer_pool, index,
            weights=NO_PRIOR_WEIGHTS, popularity=prior, popularity_mu=0.15,
        )
        assert inchikey14(ranked[0]) == inchikey14(ETHYL_ACETATE)

    def test_blend_keeps_the_same_candidate_set(self, isomer_pool, setup):
        library, index, molecule, prior = setup
        a, _ = predict_molecule(molecule, library, isomer_pool, index, popularity=prior)
        b, _ = predict_molecule(
            molecule, library, isomer_pool, index, popularity=prior, popularity_mu=0.15
        )
        assert sorted(a) == sorted(b)
        assert len(set(a)) == len(a)
