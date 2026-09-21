"""Tests for Channel 6's pipeline-facing bridge (casmi.channels.denovo).

Skipped entirely when torch is absent (the underlying decoder is optional).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the optional 'train' extra")

from casmi.channels.denovo import (  # noqa: E402
    DenovoChannelConfig,
    denovo_score_feature,
    generate_candidates,
)
from casmi.chem import exact_mass  # noqa: E402
from casmi.models.denovo import DenovoConfig, FingerprintToSmilesDecoder  # noqa: E402
from casmi.models.smiles_tokenizer import SmilesVocabulary  # noqa: E402

N_BITS = 32
CORPUS = ["CCO", "CCC", "c1ccccc1", "CCN", "CC(=O)O"]


@pytest.fixture
def vocabulary():
    return SmilesVocabulary.from_smiles_corpus(CORPUS)


@pytest.fixture
def model(vocabulary):
    config = DenovoConfig(
        n_bits=N_BITS,
        vocab_size=vocabulary.size,
        d_model=16,
        n_heads=2,
        n_encoder_layers=1,
        n_decoder_layers=1,
        ffn_dim=32,
        max_length=15,
        max_fingerprint_bits=16,
    )
    return FingerprintToSmilesDecoder(config)


def make_fingerprint(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=N_BITS).astype(np.float32)


class TestGenerateCandidates:
    def test_returns_list_of_denovo_candidates(self, model, vocabulary):
        fp = make_fingerprint(0)
        # A very loose mass tolerance and a nonsense target so nothing is
        # filtered out purely by chance in this smoke test.
        cfg = DenovoChannelConfig(beam_width=6, mass_tolerance_ppm=1e12)
        results = generate_candidates(fp, model, vocabulary, target_mass=100.0, config=cfg)
        for item in results:
            assert item.smiles
            assert item.inchikey14
            assert np.isfinite(item.log_probability)
            assert np.isfinite(item.mass)

    def test_mass_filter_rejects_incompatible_candidates(self, model, vocabulary):
        fp = make_fingerprint(1)
        # An impossibly tight tolerance around a mass nothing will match.
        cfg = DenovoChannelConfig(beam_width=8, mass_tolerance_ppm=0.0001)
        results = generate_candidates(fp, model, vocabulary, target_mass=1.0, config=cfg)
        assert results == []

    def test_infinite_tolerance_keeps_everything_parseable(self, model, vocabulary):
        fp = make_fingerprint(2)
        cfg = DenovoChannelConfig(beam_width=8, mass_tolerance_ppm=1e15)
        results = generate_candidates(fp, model, vocabulary, target_mass=500.0, config=cfg)
        # Every surviving candidate's mass really was computed and is sane.
        for item in results:
            recomputed = exact_mass(item.smiles)
            assert recomputed == pytest.approx(item.mass)

    def test_deduplicates_by_inchikey(self, model, vocabulary):
        fp = make_fingerprint(3)
        cfg = DenovoChannelConfig(beam_width=8, mass_tolerance_ppm=1e15)
        results = generate_candidates(fp, model, vocabulary, target_mass=500.0, config=cfg)
        keys = [c.inchikey14 for c in results]
        assert len(keys) == len(set(keys))

    def test_nan_target_mass_skips_mass_filtering(self, model, vocabulary):
        fp = make_fingerprint(4)
        cfg = DenovoChannelConfig(beam_width=6)
        # Must not raise despite an unusable target mass.
        results = generate_candidates(fp, model, vocabulary, target_mass=float("nan"), config=cfg)
        assert isinstance(results, list)

    def test_default_config_used_when_none_supplied(self, model, vocabulary):
        fp = make_fingerprint(5)
        results = generate_candidates(fp, model, vocabulary, target_mass=200.0)
        assert isinstance(results, list)


class TestDenovoScoreFeature:
    def test_zero_log_prob_maps_to_one(self):
        assert denovo_score_feature(0.0) == pytest.approx(1.0)

    def test_more_negative_log_prob_gives_smaller_score(self):
        high = denovo_score_feature(-1.0)
        low = denovo_score_feature(-10.0)
        assert high > low
        assert 0.0 < low < high <= 1.0

    def test_clips_extreme_negative_values(self):
        # Must not underflow to something that breaks downstream arithmetic.
        score = denovo_score_feature(-1e9)
        assert np.isfinite(score)
        assert score >= 0.0
