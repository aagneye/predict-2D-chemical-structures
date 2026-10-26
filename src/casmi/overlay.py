"""Overlay: insert novel-structure candidates into a stronger base ranking.

The public 0.46+ recipes find database structures well; their own analogue generator
covers some structures that are in no database (hidden-test class 3). This module merges
our derived candidates (Channel 8) into such a base list as a second opinion for those
molecules. The base is better than us on
classes 1 and 2 and its rank 1 carries most of the score, so by default nothing is
inserted above rank 1 and insertion happens only when the base's own evidence is weak
(no close library match), which is where class 3 hides.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class OverlayConfig:
    insert_at: int = 1
    """0-based position of the first inserted structure; 1 = directly below rank 1."""
    max_insert: int = 3
    """At most this many new structures per molecule."""
    max_length: int = 25
    """Submission limit."""
    library_max_threshold: float = 0.6
    """Insert only when the molecule's best library similarity is below this."""


def weak_evidence(library_similarity_max: float, config: OverlayConfig) -> bool:
    """True when the base has no close library match, i.e. the molecule may be novel."""
    return float(library_similarity_max) < config.library_max_threshold


def merge_overlay(
    base: Sequence[str],
    inserts: Sequence[str],
    config: OverlayConfig,
    key: Callable[[str], str | None] = lambda s: s,
) -> list[str]:
    """Return ``base`` with up to ``max_insert`` structures from ``inserts`` placed at
    ``insert_at``; a structure whose ``key`` is None or already in ``base`` is skipped
    and the result is cut to ``max_length``."""
    seen = {key(s) for s in base}
    seen.discard(None)
    new: list[str] = []
    for smiles in inserts:
        k = key(smiles)
        if k is None or k in seen:
            continue
        seen.add(k)
        new.append(smiles)
        if len(new) == config.max_insert:
            break
    position = min(max(config.insert_at, 0), len(base))
    merged = list(base[:position]) + new + list(base[position:])
    return merged[: config.max_length]
