"""MRR@25 scoring — the metric every experiment is judged by.

The competition metric is the mean reciprocal rank of the *first* correct
candidate in a ranked list of at most 25 SMILES per molecule. A prediction is
correct when its tautomer-canonical InChIKey14 matches the truth's, so
stereochemistry and tautomer form are irrelevant (see :mod:`casmi.chem`).

Two consequences worth keeping in mind when reading these numbers:

* Rank 1 scores 1.0, rank 5 scores 0.2, rank 25 scores 0.04, absent scores 0.
  Getting *a* correct answer into the list matters far more than placing it
  exactly first, so recall@25 is reported alongside MRR.
* Scores must always be broken out by synthetic novelty class
  (:func:`summarise_by_class`). The public-LB-leading solution reported only an
  aggregate and consequently did not notice that three of its four evidence
  channels contributed nothing — see ``docs/05-community-intel.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from casmi.chem import inchikey14

#: The competition's list-length cap.
DEFAULT_K = 25


def reciprocal_rank(
    predicted_keys: Sequence[str | None], true_key: str, k: int = DEFAULT_K
) -> float:
    """Reciprocal rank of the first occurrence of ``true_key``, else 0.0.

    ``predicted_keys`` must already be InChIKey14 values, deduplicated in rank
    order by the caller. Only the first ``k`` entries are considered.
    """
    if not true_key:
        return 0.0
    for position, key in enumerate(predicted_keys[:k], start=1):
        if key is not None and key == true_key:
            return 1.0 / position
    return 0.0


@dataclass(frozen=True)
class MoleculeScore:
    """Per-molecule scoring outcome."""

    molecule_id: str
    #: Reciprocal rank of the first correct candidate (0.0 if none).
    rr: float
    #: 1-based rank of the first correct candidate, or ``None`` if absent.
    hit_rank: int | None
    #: Number of candidates submitted (after dedup, before the top-k cut).
    n_candidates: int
    #: Synthetic novelty class (1/2/3) when evaluating a local split.
    novelty_class: int | None = None

    @property
    def is_hit(self) -> bool:
        """Whether a correct candidate appeared anywhere in the scored list."""
        return self.hit_rank is not None


def dedupe_keys(keys: Iterable[str | None]) -> list[str | None]:
    """Drop repeated InChIKey14s, preserving first-seen order.

    Deduplication must happen *before* truncating to 25: two SMILES for the
    same 2D skeleton score identically, so keeping both wastes a slot.
    ``None`` (unparseable) entries are retained as placeholders so that a
    candidate list's length still reflects the slots consumed.
    """
    seen: set[str] = set()
    out: list[str | None] = []
    for key in keys:
        if key is None:
            out.append(None)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def score_molecule(
    molecule_id: str,
    predicted_smiles: Sequence[str],
    true_smiles: str,
    k: int = DEFAULT_K,
    novelty_class: int | None = None,
    predicted_keys: Sequence[str | None] | None = None,
    true_key: str | None = None,
) -> MoleculeScore:
    """Score one molecule's ranked SMILES list against its true structure.

    ``predicted_keys``/``true_key`` let a caller supply precomputed InChIKey14
    values, which matters because tautomer canonicalisation is the dominant
    cost when scoring thousands of molecules.
    """
    keys = (
        list(predicted_keys)
        if predicted_keys is not None
        else [inchikey14(s) for s in predicted_smiles]
    )
    keys = dedupe_keys(keys)
    truth = true_key if true_key is not None else inchikey14(true_smiles)

    rr = 0.0
    hit_rank: int | None = None
    if truth:
        for position, key in enumerate(keys[:k], start=1):
            if key == truth:
                rr = 1.0 / position
                hit_rank = position
                break
    return MoleculeScore(
        molecule_id=molecule_id,
        rr=rr,
        hit_rank=hit_rank,
        n_candidates=len(keys),
        novelty_class=novelty_class,
    )


def evaluate_predictions(
    predictions: Mapping[str, Sequence[str]],
    truth: Mapping[str, str],
    k: int = DEFAULT_K,
    novelty_classes: Mapping[str, int] | None = None,
    precomputed_keys: Mapping[str, Sequence[str | None]] | None = None,
    precomputed_truth_keys: Mapping[str, str] | None = None,
) -> list[MoleculeScore]:
    """Score every molecule in ``truth``.

    Molecules present in ``truth`` but missing from ``predictions`` score 0.0 —
    a missing submission row is a wrong answer, not an excluded sample.
    """
    scores: list[MoleculeScore] = []
    for molecule_id, true_smiles in truth.items():
        scores.append(
            score_molecule(
                molecule_id=molecule_id,
                predicted_smiles=predictions.get(molecule_id, ()),
                true_smiles=true_smiles,
                k=k,
                novelty_class=(novelty_classes or {}).get(molecule_id),
                predicted_keys=(precomputed_keys or {}).get(molecule_id),
                true_key=(precomputed_truth_keys or {}).get(molecule_id),
            )
        )
    return scores


def mrr_at_k(scores: Sequence[MoleculeScore]) -> float:
    """Mean reciprocal rank over ``scores`` (0.0 for an empty input)."""
    if not scores:
        return 0.0
    return sum(s.rr for s in scores) / len(scores)


def recall_at_k(scores: Sequence[MoleculeScore]) -> float:
    """Fraction of molecules with a correct candidate anywhere in the list.

    The ceiling MRR achievable by re-ranking alone: no amount of reordering
    helps a molecule whose correct structure never entered the candidate pool.
    Tracking this separately tells us whether to invest in retrieval (raise
    recall) or in ranking (convert recall into MRR).
    """
    if not scores:
        return 0.0
    return sum(1 for s in scores if s.is_hit) / len(scores)


def top1_accuracy(scores: Sequence[MoleculeScore]) -> float:
    """Fraction of molecules whose rank-1 candidate is correct."""
    if not scores:
        return 0.0
    return sum(1 for s in scores if s.hit_rank == 1) / len(scores)


def summarise(scores: Sequence[MoleculeScore]) -> dict[str, float]:
    """Aggregate metrics for one cohort of molecules."""
    return {
        "n": float(len(scores)),
        "mrr": mrr_at_k(scores),
        "recall": recall_at_k(scores),
        "top1": top1_accuracy(scores),
        "mean_candidates": (
            sum(s.n_candidates for s in scores) / len(scores) if scores else 0.0
        ),
    }


def summarise_by_class(scores: Sequence[MoleculeScore]) -> dict[str, dict[str, float]]:
    """Metrics overall and per synthetic novelty class.

    Always read these three cohorts rather than the aggregate: they answer
    different questions (library search working / retrieval working / de novo
    working) and an aggregate can stay flat while the mix underneath shifts.
    """
    out = {"overall": summarise(scores)}
    for novelty_class in (1, 2, 3):
        cohort = [s for s in scores if s.novelty_class == novelty_class]
        if cohort:
            out[f"class{novelty_class}"] = summarise(cohort)
    unlabelled = [s for s in scores if s.novelty_class is None]
    if unlabelled and len(unlabelled) != len(scores):
        out["unlabelled"] = summarise(unlabelled)
    return out


def format_summary(summary: Mapping[str, Mapping[str, float]]) -> str:
    """Render :func:`summarise_by_class` output as a fixed-width table."""
    header = f"{'cohort':<12}{'n':>7}{'MRR@25':>9}{'recall':>9}{'top1':>9}{'cands':>8}"
    lines = [header, "-" * len(header)]
    for cohort, metrics in summary.items():
        lines.append(
            f"{cohort:<12}{metrics['n']:>7.0f}{metrics['mrr']:>9.4f}"
            f"{metrics['recall']:>9.4f}{metrics['top1']:>9.4f}"
            f"{metrics['mean_candidates']:>8.1f}"
        )
    return "\n".join(lines)
