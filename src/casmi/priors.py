"""Structure-level priors that do not depend on the spectrum.

Currently one prior: **PubChem popularity**, ``log1p(substance records) +
log1p(PubMed articles)`` per structure. The intuition is simple and, on this
competition's hidden test, very strong: the molecules Enveda isolates and
measures are overwhelmingly *documented* natural products, while a mass
window into a large structure database is full of obscure isomers nobody has
ever reported a spectrum for. Among same-formula candidates the well-known
one is far more often the answer. A community benchmark on 250 timsTOF
natural products measured a fingerprint model alone at 40% top-1 and the
same model plus ``0.25 * popularity`` at 93% top-1.

The prior enters the pipeline as one more per-candidate feature
(``popularity``), so weighted fusion gives it a modest fixed weight and the
learned reranker learns how much to trust it. An optional post-ranking blend
(:func:`popularity_blend`) reproduces the public lineage's
``z(ranker) + mu * popularity`` re-rank for direct comparison.

Structures absent from the table get 0.0, which is also the value a
genuinely undocumented structure would get (``log1p(0) + log1p(0)``), so
"unknown" and "nobody has reported it" are deliberately indistinguishable
here: both mean the prior has nothing good to say.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import numpy as np


class PopularityPrior:
    """``inchikey14 -> popularity`` lookup with a 0.0 default."""

    def __init__(self, table: dict[str, float]) -> None:
        self._table = table

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, key: str) -> bool:
        return key in self._table

    def get(self, key: str | None) -> float:
        if not key:
            return 0.0
        return float(self._table.get(key, 0.0))

    def get_many(self, keys: Iterable[str | None]) -> np.ndarray:
        return np.asarray([self.get(k) for k in keys], dtype=np.float32)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        key_column: str = "inchikey14",
        value_column: str = "pop",
    ) -> PopularityPrior:
        """Load from a CSV with an ``inchikey14`` and a ``pop`` column.

        Matches the layout of the community ``pool_popularity.csv``
        (``pop = log1p_substances + log1p_pubmed``). Duplicate keys keep the
        maximum, since stereoisomers that share a scored connectivity should
        pool their documentation.
        """
        table: dict[str, float] = {}
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            for needed in (key_column, value_column):
                if needed not in columns:
                    raise ValueError(f"{path} has no {needed!r} column; columns={columns}")
            for row in reader:
                key = (row.get(key_column) or "").strip()
                raw = row.get(value_column)
                if not key or raw in (None, ""):
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if not np.isfinite(value):
                    continue
                if value > table.get(key, -np.inf):
                    table[key] = value
        return cls(table)

    def save(self, path: str | Path) -> None:
        """Persist as a compact ``.npz`` (keys + float32 values) for Kaggle."""
        keys = np.asarray(list(self._table.keys()), dtype=str)
        values = np.asarray(list(self._table.values()), dtype=np.float32)
        np.savez_compressed(str(path), keys=keys, values=values)

    @classmethod
    def load(cls, path: str | Path) -> PopularityPrior:
        """Load a ``.npz`` written by :meth:`save` (or a ``.csv`` directly)."""
        path = Path(path)
        if path.suffix.lower() == ".csv":
            return cls.from_csv(path)
        data = np.load(str(path), allow_pickle=False)
        keys = [str(k) for k in data["keys"]]
        values = data["values"].astype(np.float64)
        return cls(dict(zip(keys, values.tolist(), strict=True)))


def popularity_blend(scores: np.ndarray, popularity: np.ndarray, mu: float) -> np.ndarray:
    """``z(scores) + mu * popularity`` within one molecule's candidate set.

    The z-score makes the ranker's output scale-free so ``mu`` means the same
    thing for weighted-fusion scores and reranker probabilities. With fewer
    than two candidates or a constant score vector the z-score is zero and
    the blend reduces to the prior alone.
    """
    s = np.asarray(scores, dtype=np.float64)
    p = np.asarray(popularity, dtype=np.float64)
    if s.shape != p.shape:
        raise ValueError(f"scores {s.shape} and popularity {p.shape} differ in shape")
    if s.size == 0:
        return s
    std = float(s.std())
    z = (s - s.mean()) / std if std > 1e-12 else np.zeros_like(s)
    return z + mu * p
