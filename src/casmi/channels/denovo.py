"""Channel 6 — de novo generation candidates from FPNet's fingerprint logits.

Bridges :mod:`casmi.models.denovo` (the sequence model) into the pipeline's
``ScoredCandidate`` shape. This is the only channel that can reach Class 3
(structures absent from every candidate database): Channels 1/2 need a
matching library spectrum, and the pool-lookup Channels 4/5 need the true
structure to already be a pool member. If it is neither, generation is the
only remaining path to recall.

Generated candidates are additive to, never a replacement for, the retrieval
pool: :func:`generate_candidates` is called once per molecule and its output
is merged into the same candidate list retrieval produces (see
``casmi.pipeline.build_scored_candidates``), deduplicated by ``inchikey14``
downstream by :func:`casmi.channels.fusion.fuse` the same way an out-of-pool
library hit already is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from casmi.chem import exact_mass, inchikey14
from casmi.models.denovo import FingerprintToSmilesDecoder, beam_search_generate
from casmi.models.smiles_tokenizer import SmilesVocabulary


@dataclass(frozen=True)
class DenovoChannelConfig:
    """Generation and mass-filtering parameters."""

    beam_width: int = 8
    max_length: int | None = None  # falls back to the model's own config
    #: A generated candidate must match the target neutral mass within this
    #: many ppm to be kept — cheap stand-in for full formula-constrained
    #: decoding (see docs/04-method-landscape.md Strategy C), reusing the
    #: same tolerance semantics as candidate-pool retrieval.
    mass_tolerance_ppm: float = 15.0


@dataclass(frozen=True)
class DenovoCandidate:
    """One generated structure, before mass filtering."""

    smiles: str
    inchikey14: str
    log_probability: float
    mass: float


def generate_candidates(
    fingerprint_logits: np.ndarray,
    model: FingerprintToSmilesDecoder,
    vocabulary: SmilesVocabulary,
    target_mass: float,
    config: DenovoChannelConfig | None = None,
    device: str = "cpu",
) -> list[DenovoCandidate]:
    """Generate, parse, and mass-filter de novo candidates for one molecule.

    Args:
        fingerprint_logits: FPNet's predicted logits for this molecule (see
            :func:`casmi.models.fpnet.predict_logits`) — the conditioning
            vector, fed directly rather than passing through a sigmoid, since
            the decoder was trained on true 0/1 fingerprints and the logit
            scale during training used the same convention (see
            :mod:`casmi.models.train_denovo`'s bit-flip augmentation, which
            operates on the same 0/1 representation).
        target_mass: The molecule's precursor-derived neutral mass.

    Returns:
        Mass-filtered, deduplicated (by ``inchikey14``) candidates, best
        (highest beam log-probability) first. Empty if nothing beam search
        produced parses or survives the mass filter — a de novo channel that
        finds nothing is a normal, expected outcome, not an error.
    """
    cfg = config or DenovoChannelConfig()
    raw = beam_search_generate(
        model,
        fingerprint_logits,
        vocabulary,
        beam_width=cfg.beam_width,
        max_length=cfg.max_length,
        device=device,
    )

    tolerance = (
        target_mass * cfg.mass_tolerance_ppm / 1e6
        if np.isfinite(target_mass) and target_mass > 0
        else float("inf")
    )

    seen_keys: set[str] = set()
    candidates: list[DenovoCandidate] = []
    for smiles, log_prob in raw:
        mass = exact_mass(smiles)
        if mass is None:
            continue
        if np.isfinite(target_mass) and target_mass > 0 and abs(mass - target_mass) > tolerance:
            continue
        key = inchikey14(smiles)
        if key is None or key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append(
            DenovoCandidate(smiles=smiles, inchikey14=key, log_probability=log_prob, mass=mass)
        )

    return candidates


def denovo_score_feature(
    log_probability: float, config: DenovoChannelConfig | None = None
) -> float:
    """Map a beam log-probability to a bounded ``[0, 1]``-ish fusion feature.

    Raw beam log-probabilities are unboundedly negative and scale with
    sequence length, which is not comparable to the other channels' ``[0, 1]``
    scores. ``exp(log_prob)`` (the sequence's actual probability) is a
    principled bounded rescaling: it is already in ``(0, 1]`` and preserves
    the beam's own ranking exactly.
    """
    del config  # kept for symmetry with the rest of the channel modules
    return float(np.exp(np.clip(log_probability, -50.0, 0.0)))
