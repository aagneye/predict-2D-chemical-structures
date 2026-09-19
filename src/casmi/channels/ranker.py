"""Learned reranker replacing weighted fusion (step 8 of the implementation plan).

:mod:`casmi.channels.fusion` combines per-candidate features with fixed,
hand-chosen weights. That is a deliberate first step (see its module
docstring) but it cannot capture interactions between channels — for example,
that a mediocre library score combined with strong fragmentation support is
much stronger evidence than either alone. A gradient-boosted classifier can
learn that interaction directly from data.

Design:

* :func:`assemble_features` turns a candidate's ``ScoredCandidate.features``
  dict (plus a handful of derived rank/z-score variants, matching the
  reference solution's feature set) into a fixed-width numeric vector so the
  same function is used both to build a training matrix and to score
  candidates at inference time — the two must never diverge.
* :func:`train_ranker` fits a bagged ensemble of
  ``HistGradientBoostingClassifier`` over class-1 priors x seeds, exactly as
  :class:`casmi.config.RankerConfig` specifies. Bagging over both, rather than
  picking one setting, is what the reference solution's own ablations credit
  for removing a ~0.006 MRR noise floor.
* :class:`Reranker` wraps the fitted ensemble for use inside
  :mod:`casmi.pipeline`, exposing the same "rank candidates" shape as
  :func:`casmi.channels.fusion.fuse` so the pipeline can switch between them.

The reranker is optional throughout: it requires ``scikit-learn`` (already a
hard dependency, unlike torch) but a *fitted* model is itself an artifact that
must be trained offline against the held-out split and loaded here, so
:mod:`casmi.pipeline` falls back to weighted fusion whenever no ranker is
supplied.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from casmi.config import CFG, RankerConfig

#: Fixed feature order. Both training-set assembly and inference must agree on
#: this, so it is a module-level constant rather than inferred from a dict.
FEATURE_NAMES: tuple[str, ...] = (
    "library_similarity",
    "library_similarity_rank",
    "library_similarity_max",
    "library_similarity_gap",
    "library_hit",
    "analog_power",
    "analog_power_rank",
    "analog_power_max",
    "analog_power_gap",
    "analog_linear",
    "analog_best_tanimoto",
    "analog_top_tanimoto",
    "analog_mean",
    "mass_error_penalty",
    "fragmentation_score",
    "fragmentation_score_rank",
    "fragmentation_score_gap",
    "fpnet_score",
    "fpnet_score_z",
    "fpnet_score_rank",
    "fpnet_normalised_score",
    "log_n_candidates",
)
N_FEATURES = len(FEATURE_NAMES)


def _rank_normalise(values: np.ndarray) -> np.ndarray:
    """Map values to ``[0, 1]`` by descending rank (1.0 = best), ties broken stably.

    Rank features let the model reason about relative standing within one
    molecule's candidate set, which is scale-invariant across molecules in a
    way raw scores are not (a library similarity of 0.6 means very different
    things for different queries).
    """
    n = len(values)
    if n == 0:
        return values.astype(np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return 1.0 - ranks / (n - 1)


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return (values - values.mean()) / std


def assemble_features(
    feature_dicts: list[dict[str, float]],
    fpnet_scores: np.ndarray | None = None,
    fpnet_normalised: np.ndarray | None = None,
) -> np.ndarray:
    """Build the ``(n_candidates, N_FEATURES)`` matrix for one molecule.

    Args:
        feature_dicts: One ``ScoredCandidate.features`` dict per candidate, in
            the same order the caller wants scored.
        fpnet_scores: Optional Channel-4 dot-product scores, aligned with
            ``feature_dicts``. Omit entirely (rather than passing zeros) when
            no FPNet checkpoint is available, so the "no neural channel" case
            is distinguishable from "the neural channel scored this at zero".
        fpnet_normalised: Optional bit-count-normalised variant of the same
            scores (see :func:`casmi.models.fpnet.normalised_score`).

    Returns:
        A ``float32`` matrix with columns in :data:`FEATURE_NAMES` order.
    """
    n = len(feature_dicts)
    if n == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    def column(name: str) -> np.ndarray:
        return np.array([d.get(name, 0.0) for d in feature_dicts], dtype=np.float64)

    library_similarity = column("library_similarity")
    analog_power = column("analog_power")
    fragmentation_score = column("fragmentation_score")

    lib_max = float(library_similarity.max()) if n else 0.0
    ana_max = float(analog_power.max()) if n else 0.0
    frag_max = float(fragmentation_score.max()) if n else 0.0

    if fpnet_scores is not None:
        fpnet = np.asarray(fpnet_scores, dtype=np.float64)
        fpnet_z = _zscore(fpnet)
        fpnet_rank = _rank_normalise(fpnet)
    else:
        fpnet = np.zeros(n, dtype=np.float64)
        fpnet_z = np.zeros(n, dtype=np.float64)
        fpnet_rank = np.zeros(n, dtype=np.float64)
    fpnet_norm = (
        np.asarray(fpnet_normalised, dtype=np.float64)
        if fpnet_normalised is not None
        else np.zeros(n, dtype=np.float64)
    )

    columns = {
        "library_similarity": library_similarity,
        "library_similarity_rank": _rank_normalise(library_similarity),
        "library_similarity_max": np.full(n, lib_max),
        "library_similarity_gap": library_similarity - lib_max,
        "library_hit": (library_similarity > 0.0).astype(np.float64),
        "analog_power": analog_power,
        "analog_power_rank": _rank_normalise(analog_power),
        "analog_power_max": np.full(n, ana_max),
        "analog_power_gap": analog_power - ana_max,
        "analog_linear": column("analog_linear"),
        "analog_best_tanimoto": column("analog_best_tanimoto"),
        "analog_top_tanimoto": column("analog_top_tanimoto"),
        "analog_mean": column("analog_mean"),
        "mass_error_penalty": column("mass_error_penalty"),
        "fragmentation_score": fragmentation_score,
        "fragmentation_score_rank": _rank_normalise(fragmentation_score),
        "fragmentation_score_gap": fragmentation_score - frag_max,
        "fpnet_score": fpnet,
        "fpnet_score_z": fpnet_z,
        "fpnet_score_rank": fpnet_rank,
        "fpnet_normalised_score": fpnet_norm,
        "log_n_candidates": np.full(n, np.log1p(n)),
    }
    matrix = np.column_stack([columns[name] for name in FEATURE_NAMES])
    return matrix.astype(np.float32)


@dataclass
class RankTrainingExample:
    """One candidate row for reranker training.

    ``label`` is 1 for the candidate matching the molecule's true structure
    (by ``inchikey14``) and 0 for every other candidate considered for that
    molecule — standard pointwise learning-to-rank framing.
    """

    molecule_id: str
    features: dict[str, float]
    label: int
    fpnet_score: float = 0.0
    fpnet_normalised_score: float = 0.0


def build_training_matrix(
    examples: list[RankTrainingExample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble ``(X, y, molecule_ids)`` from per-molecule candidate groups.

    Features are assembled per molecule (so rank/z-score columns are relative
    to the right candidate set) and then stacked, rather than assembling all
    examples as one flat group.
    """
    if not examples:
        return (
            np.zeros((0, N_FEATURES), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=object),
        )

    groups: dict[str, list[int]] = {}
    for i, example in enumerate(examples):
        groups.setdefault(example.molecule_id, []).append(i)

    x_chunks, y_chunks, id_chunks = [], [], []
    for molecule_id, indices in groups.items():
        group_examples = [examples[i] for i in indices]
        feats = [e.features for e in group_examples]
        fpnet_scores = np.array([e.fpnet_score for e in group_examples], dtype=np.float64)
        fpnet_norm = np.array(
            [e.fpnet_normalised_score for e in group_examples], dtype=np.float64
        )
        x_chunks.append(assemble_features(feats, fpnet_scores, fpnet_norm))
        y_chunks.append(np.array([e.label for e in group_examples], dtype=np.int64))
        id_chunks.append(np.array([molecule_id] * len(group_examples), dtype=object))

    return (
        np.concatenate(x_chunks, axis=0),
        np.concatenate(y_chunks, axis=0),
        np.concatenate(id_chunks, axis=0),
    )


class Reranker:
    """Bagged ``HistGradientBoostingClassifier`` ensemble over priors x seeds.

    Prediction averages ``predict_proba`` across every ensemble member, which
    is what makes the bagging actually reduce variance rather than just
    picking one arbitrary member.
    """

    def __init__(self, models: list, config: RankerConfig | None = None) -> None:
        if not models:
            raise ValueError("Reranker requires at least one fitted model")
        self.models = models
        self.config = config or CFG.ranker

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Mean positive-class probability across ensemble members."""
        if features.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        return np.mean([m.predict_proba(features)[:, 1] for m in self.models], axis=0)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as handle:
            pickle.dump({"models": self.models, "config": self.config}, handle)

    @classmethod
    def load(cls, path: str | Path) -> Reranker:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        return cls(models=payload["models"], config=payload["config"])


def train_ranker(
    features: np.ndarray,
    labels: np.ndarray,
    config: RankerConfig | None = None,
) -> Reranker:
    """Fit the bagged reranker ensemble.

    One ``HistGradientBoostingClassifier`` is fit per ``(class1_prior, seed)``
    pair, sample-weighted so the positive class is worth ``prior`` and the
    negative class ``1 - prior`` — hedging over the prior is what the
    reference solution's config credits with a ~0.003 LB gain, on top of the
    ~0.006 seed-noise reduction from bagging seeds alone.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    cfg = config or CFG.ranker
    labels = np.asarray(labels)
    models = []
    for prior in cfg.class1_priors:
        sample_weight = np.where(labels == 1, prior, 1.0 - prior)
        for seed in cfg.seeds:
            model = HistGradientBoostingClassifier(
                max_depth=cfg.max_depth,
                max_iter=cfg.max_iter,
                learning_rate=cfg.learning_rate,
                min_samples_leaf=cfg.min_samples_leaf,
                l2_regularization=cfg.l2_regularization,
                random_state=seed,
            )
            model.fit(features, labels, sample_weight=sample_weight)
            models.append(model)
    return Reranker(models=models, config=cfg)
