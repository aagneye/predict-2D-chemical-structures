"""Make a held-out split behave like the hidden test.

The hidden test's novelty classes are defined by *where the answer lives*:

* class 1 — the structure has public reference spectra;
* class 2 — no public spectra, but the structure is in PubChem or COCONUT;
* class 3 — in no structure database at all.

A plain structure hold-out from ``train.parquet`` gets class 2 wrong in the easy
direction: every held-out structure is still a *training structure*, and our
candidate pool contains all training structures, so "class 2" there means "in our
pool", which real class-2 molecules (absent from every public spectral library,
hence absent from train) almost never are. That mismatch is one reason a 0.66
offline score became 0.25 on the leaderboard.

:func:`realistic_pool_exclusions` returns the held-out keys that must leave the
pool so the proxy matches the real definitions: class 2 stays reachable only if
an *external* database (COCONUT, PubChemLite...) holds it — the PubChem tier
channel can still find it, as it would in the real test — and class 3 leaves the
pool entirely (and is blocklisted from PubChem by the eval scripts).

:func:`stratified_sample` draws a class-balanced subset so several pipeline
configurations can be compared in hours instead of a day.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np


def load_keys(path: str | Path) -> set[str]:
    """Newline-delimited InChIKey14 file -> set (blank lines ignored)."""
    return {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def realistic_pool_exclusions(
    novelty: Mapping[str, int],
    truth_keys: Mapping[str, Iterable[str | None]],
    external_keys: set[str],
) -> set[str]:
    """Keys to remove from the candidate pool for a test-like evaluation.

    Args:
        novelty: ``molecule_id -> class`` (1/2/3) of the held-out molecules.
        truth_keys: ``molecule_id -> keys of its true structure`` (stored and
            scorer key; ``None`` entries are ignored).
        external_keys: Scorer keys of structures present in external databases.

    Returns:
        Class-2 keys absent from ``external_keys`` plus every class-3 key.
    """
    drop: set[str] = set()
    for molecule_id, cls in novelty.items():
        keys = {k for k in truth_keys.get(molecule_id, ()) if k}
        if cls == 3:
            drop |= keys
        elif cls == 2 and not (keys & external_keys):
            drop |= keys
    return drop


def stratified_sample(
    novelty: Mapping[str, int], per_class: int, seed: int = 0
) -> list[str]:
    """Up to ``per_class`` molecule ids from each class, deterministic for a seed."""
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for cls in sorted(set(novelty.values())):
        ids = sorted(m for m, c in novelty.items() if c == cls)
        if len(ids) > per_class:
            ids = list(rng.choice(ids, size=per_class, replace=False))
        chosen.extend(ids)
    return sorted(chosen)
