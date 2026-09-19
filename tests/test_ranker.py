"""Tests for the learned reranker (feature assembly, training, ensemble predict).

Key claims: feature assembly is deterministic and shape-stable regardless of
candidate-set size, the rank/z-score columns are actually relative to their
own molecule's candidate group (not global), and a trained ensemble assigns a
higher probability to an obviously-correct candidate than to noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from casmi.channels.ranker import (
    FEATURE_NAMES,
    N_FEATURES,
    RankTrainingExample,
    Reranker,
    assemble_features,
    build_training_matrix,
    train_ranker,
)
from casmi.config import RankerConfig


def make_feature_dict(**overrides) -> dict[str, float]:
    base = {
        "library_similarity": 0.0,
        "analog_power": 0.0,
        "analog_linear": 0.0,
        "analog_best_tanimoto": 0.0,
        "analog_top_tanimoto": 0.0,
        "analog_mean": 0.0,
        "mass_error_penalty": 0.0,
        "fragmentation_score": 0.0,
    }
    base.update(overrides)
    return base


class TestAssembleFeatures:
    def test_empty_input_has_correct_shape(self):
        matrix = assemble_features([])
        assert matrix.shape == (0, N_FEATURES)

    def test_shape_matches_candidate_count(self):
        dicts = [make_feature_dict(library_similarity=v) for v in (0.1, 0.5, 0.9)]
        matrix = assemble_features(dicts)
        assert matrix.shape == (3, N_FEATURES)

    def test_column_count_matches_feature_names(self):
        assert N_FEATURES == len(FEATURE_NAMES)

    def test_deterministic(self):
        dicts = [make_feature_dict(library_similarity=v) for v in (0.2, 0.8)]
        a = assemble_features(dicts)
        b = assemble_features(dicts)
        np.testing.assert_array_equal(a, b)

    def test_rank_feature_is_relative_to_its_own_group(self):
        """The best candidate in a group must get rank feature 1.0."""
        dicts = [make_feature_dict(library_similarity=v) for v in (0.1, 0.9, 0.5)]
        matrix = assemble_features(dicts)
        rank_col = FEATURE_NAMES.index("library_similarity_rank")
        best_row = 1  # library_similarity=0.9 is the max
        assert matrix[best_row, rank_col] == pytest.approx(1.0)

    def test_no_fpnet_scores_yields_zero_columns(self):
        dicts = [make_feature_dict()]
        matrix = assemble_features(dicts, fpnet_scores=None)
        fpnet_col = FEATURE_NAMES.index("fpnet_score")
        assert matrix[0, fpnet_col] == 0.0

    def test_fpnet_scores_populate_their_columns(self):
        dicts = [make_feature_dict(), make_feature_dict()]
        scores = np.array([1.5, -0.5])
        matrix = assemble_features(dicts, fpnet_scores=scores)
        fpnet_col = FEATURE_NAMES.index("fpnet_score")
        np.testing.assert_allclose(matrix[:, fpnet_col], scores)

    def test_single_candidate_group_rank_is_one(self):
        matrix = assemble_features([make_feature_dict(library_similarity=0.42)])
        rank_col = FEATURE_NAMES.index("library_similarity_rank")
        assert matrix[0, rank_col] == pytest.approx(1.0)

    def test_gap_feature_is_zero_for_the_best_candidate(self):
        dicts = [make_feature_dict(library_similarity=v) for v in (0.3, 0.9)]
        matrix = assemble_features(dicts)
        gap_col = FEATURE_NAMES.index("library_similarity_gap")
        assert matrix[1, gap_col] == pytest.approx(0.0)
        assert matrix[0, gap_col] < 0.0


class TestBuildTrainingMatrix:
    def test_empty_examples(self):
        x, y, ids = build_training_matrix([])
        assert x.shape == (0, N_FEATURES)
        assert y.shape == (0,)
        assert ids.shape == (0,)

    def test_groups_by_molecule_id(self):
        examples = [
            RankTrainingExample("m1", make_feature_dict(library_similarity=0.9), label=1),
            RankTrainingExample("m1", make_feature_dict(library_similarity=0.1), label=0),
            RankTrainingExample("m2", make_feature_dict(library_similarity=0.5), label=1),
        ]
        x, y, ids = build_training_matrix(examples)
        assert x.shape == (3, N_FEATURES)
        assert set(ids) == {"m1", "m2"}
        assert y.sum() == 2

    def test_rank_features_relative_within_molecule_group(self):
        """m2's single candidate must not be compared against m1's candidates."""
        examples = [
            RankTrainingExample("m1", make_feature_dict(library_similarity=0.9), label=1),
            RankTrainingExample("m1", make_feature_dict(library_similarity=0.1), label=0),
            RankTrainingExample("m2", make_feature_dict(library_similarity=0.01), label=1),
        ]
        x, y, ids = build_training_matrix(examples)
        rank_col = FEATURE_NAMES.index("library_similarity_rank")
        m2_row = np.where(ids == "m2")[0][0]
        # m2 has exactly one candidate in its own group, so despite a low raw
        # score it must still get the top rank (1.0) within its own group.
        assert x[m2_row, rank_col] == pytest.approx(1.0)


class TestTrainAndPredict:
    @pytest.fixture
    def toy_examples(self):
        rng = np.random.default_rng(0)
        examples = []
        for m in range(40):
            n_candidates = 5
            correct = rng.integers(0, n_candidates)
            for c in range(n_candidates):
                is_correct = c == correct
                lib = 0.9 + rng.normal(0, 0.02) if is_correct else rng.uniform(0.0, 0.3)
                examples.append(
                    RankTrainingExample(
                        molecule_id=f"m{m}",
                        features=make_feature_dict(library_similarity=max(lib, 0.0)),
                        label=int(is_correct),
                    )
                )
        return examples

    def test_trained_ranker_favours_high_library_similarity(self, toy_examples):
        x, y, _ = build_training_matrix(toy_examples)
        config = RankerConfig(max_iter=50, seeds=(0, 1), class1_priors=(0.5,))
        ranker = train_ranker(x, y, config=config)

        high_sim = assemble_features([make_feature_dict(library_similarity=0.95)])
        low_sim = assemble_features([make_feature_dict(library_similarity=0.05)])
        p_high = ranker.predict_proba(high_sim)[0]
        p_low = ranker.predict_proba(low_sim)[0]
        assert p_high > p_low

    def test_ensemble_size_matches_priors_times_seeds(self, toy_examples):
        x, y, _ = build_training_matrix(toy_examples)
        config = RankerConfig(max_iter=20, seeds=(0, 1, 2), class1_priors=(0.3, 0.6))
        ranker = train_ranker(x, y, config=config)
        assert len(ranker.models) == 6

    def test_predict_proba_empty_input(self, toy_examples):
        x, y, _ = build_training_matrix(toy_examples)
        config = RankerConfig(max_iter=10, seeds=(0,), class1_priors=(0.5,))
        ranker = train_ranker(x, y, config=config)
        empty = np.zeros((0, N_FEATURES), dtype=np.float32)
        assert ranker.predict_proba(empty).shape == (0,)

    def test_save_and_load_roundtrip(self, toy_examples, tmp_path):
        x, y, _ = build_training_matrix(toy_examples)
        config = RankerConfig(max_iter=10, seeds=(0,), class1_priors=(0.5,))
        ranker = train_ranker(x, y, config=config)
        path = tmp_path / "ranker.pkl"
        ranker.save(path)
        loaded = Reranker.load(path)

        sample = assemble_features([make_feature_dict(library_similarity=0.8)])
        np.testing.assert_allclose(ranker.predict_proba(sample), loaded.predict_proba(sample))

    def test_reranker_rejects_empty_model_list(self):
        with pytest.raises(ValueError, match="at least one"):
            Reranker(models=[])
