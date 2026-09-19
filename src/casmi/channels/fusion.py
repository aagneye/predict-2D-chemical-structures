"""Candidate fusion into one ranked top-25 list.

The baseline uses transparent weighted fusion rather than a learned reranker.
That is deliberate: with a learned model it is hard to tell whether a score
change came from better evidence or from the reranker memorising something, and
the first thing this project needs is an honest measurement of each channel's
contribution. A GBM reranker replaces this once per-channel deltas are known
(step 8 of ``docs/06-implementation-plan.md``).

Fusion is score-based rather than rank-based because the channels' scores are
meaningfully calibrated against each other: a library similarity of 1.0 is
near-certain identification, whereas the best analog score is routinely ~0.5
for a correct answer. Reciprocal-rank fusion would discard that distinction and
let a weak-but-top-ranked analog outrank a perfect library match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from casmi.config import CFG


@dataclass
class ScoredCandidate:
    """A candidate structure with its per-channel evidence."""

    inchikey14: str
    smiles: str
    score: float
    #: Per-channel contributions, kept for diagnostics and reranker features.
    features: dict[str, float] = field(default_factory=dict)


#: Default channel weights for the baseline. Library search dominates when it
#: fires because a near-1.0 entropy similarity is close to proof of identity;
#: analog evidence is weaker but is the only signal available for class 2.
#:
#: ``fragmentation_score`` and ``fpnet_score`` are included so that enabling
#: Channels 4/5 has an effect even without a fitted reranker — otherwise a
#: feature absent from this dict is silently dropped by :func:`fuse` (it is
#: multiplied by an implicit weight of 0), and turning a channel on would
#: change nothing. Both are weighted modestly relative to library/analog
#: evidence: fragmentation is a plausibility prior that can coincidentally
#: match a wrong candidate, and the FPNet dot product is unnormalised across
#: molecules (see :func:`casmi.models.fpnet.normalised_score`), so neither
#: should be allowed to override a strong library or analog hit under plain
#: weighted fusion. The learned reranker (:mod:`casmi.channels.ranker`) is
#: what actually learns how much to trust each channel from data.
DEFAULT_WEIGHTS: dict[str, float] = {
    "library_similarity": 1.0,
    "analog_power": 0.55,
    "analog_best_tanimoto": 0.15,
    "analog_mean": 0.10,
    "mass_error_penalty": 0.05,
    "fragmentation_score": 0.20,
    "fpnet_score": 0.05,
}


def fuse(
    candidates: list[ScoredCandidate],
    weights: dict[str, float] | None = None,
    top_n: int | None = None,
) -> list[ScoredCandidate]:
    """Combine per-channel features into a final score and rank.

    Candidates are deduplicated by ``inchikey14`` (keeping the best-scoring
    entry) before truncation, so duplicate 2D skeletons never consume two of
    the 25 slots.
    """
    w = weights or DEFAULT_WEIGHTS
    limit = CFG.top_n if top_n is None else top_n

    best_by_key: dict[str, ScoredCandidate] = {}
    for candidate in candidates:
        score = sum(weight * candidate.features.get(name, 0.0) for name, weight in w.items())
        scored = ScoredCandidate(
            inchikey14=candidate.inchikey14,
            smiles=candidate.smiles,
            score=float(score),
            features=candidate.features,
        )
        existing = best_by_key.get(scored.inchikey14)
        if existing is None or scored.score > existing.score:
            best_by_key[scored.inchikey14] = scored

    ranked = sorted(best_by_key.values(), key=lambda c: -c.score)
    return ranked[:limit]


def mass_error_penalty(
    candidate_mass: np.ndarray, target_mass: float, ppm_window: float
) -> np.ndarray:
    """Feature rewarding candidates closest to the measured mass.

    Every candidate already passed the ppm window, so this is a tiebreaker
    within it: 1.0 at exact agreement decaying to 0.0 at the window edge. It
    carries real signal because instrument mass accuracy is far better than the
    window width, which is set wide enough to tolerate systematic offsets.
    """
    if not np.isfinite(target_mass) or target_mass <= 0:
        return np.zeros(len(candidate_mass), dtype=np.float32)
    tol = target_mass * ppm_window / 1e6
    if tol <= 0:
        return np.zeros(len(candidate_mass), dtype=np.float32)
    error = np.abs(np.asarray(candidate_mass, dtype=np.float64) - target_mass)
    return np.clip(1.0 - error / tol, 0.0, 1.0).astype(np.float32)


def to_smiles_list(ranked: list[ScoredCandidate], top_n: int | None = None) -> list[str]:
    """Extract the SMILES strings from a ranked list, best first."""
    limit = CFG.top_n if top_n is None else top_n
    return [c.smiles for c in ranked[:limit] if c.smiles]
