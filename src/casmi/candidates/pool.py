"""Mass-sorted candidate structure pool.

The pool is the set of structures retrieval can possibly return, so it bounds
recall@25: a structure absent here can never be predicted, no matter how good
the ranking. Built from COCONUT (natural-product specific) plus the training
structures.

Fingerprints are stored bit-packed (``np.packbits``) because the full pool is
~700k structures x 10,407 bits, which is ~7 GB unpacked but ~900 MB packed —
the difference between fitting in a Kaggle notebook's RAM and not.

Deliberately *not* included by default: blind PubChem isomer expansion. The
public-LB-leading solution found it degraded ranking by flooding the pool with
synthetic decoys, and shipped with its equivalent flag disabled. We keep the
option behind ``extra_sources`` and require it to earn its place against our
own held-out split.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from casmi.chem import FingerprintCalculator, inchikey14


@dataclass
class CandidatePool:
    """Structures searchable by neutral mass, with packed fingerprints.

    Attributes:
        mass: Ascending neutral monoisotopic masses.
        keys: ``inchikey14`` per structure, aligned with ``mass``.
        smiles: SMILES per structure, aligned with ``mass``.
        packed_fp: Bit-packed fingerprints, shape ``(n, ceil(n_bits / 8))``.
        n_bits: Unpacked fingerprint width.
    """

    mass: np.ndarray
    keys: np.ndarray
    smiles: np.ndarray
    packed_fp: np.ndarray
    n_bits: int

    def __post_init__(self) -> None:
        if not (len(self.mass) == len(self.keys) == len(self.smiles) == len(self.packed_fp)):
            raise ValueError("pool arrays have inconsistent lengths")
        if len(self.mass) > 1 and np.any(np.diff(self.mass) < 0):
            raise ValueError("pool must be sorted by ascending mass")
        self._key_to_index = {k: i for i, k in enumerate(self.keys)}

    def __len__(self) -> int:
        return len(self.mass)

    def index_of(self, key: str) -> int:
        """Row index for an ``inchikey14``, or -1 if absent."""
        return self._key_to_index.get(key, -1)

    def window(self, target_mass: float, ppm: float) -> np.ndarray:
        """Indices of structures within ±``ppm`` of ``target_mass``."""
        if not np.isfinite(target_mass) or len(self) == 0:
            return np.zeros(0, dtype=np.int64)
        tol = target_mass * ppm / 1e6
        lo = np.searchsorted(self.mass, target_mass - tol, "left")
        hi = np.searchsorted(self.mass, target_mass + tol, "right")
        return np.arange(lo, hi, dtype=np.int64)

    def fingerprints(self, indices: Sequence[int] | np.ndarray) -> np.ndarray:
        """Unpack fingerprints for ``indices`` as a ``(len(indices), n_bits)`` array."""
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size == 0:
            return np.zeros((0, self.n_bits), dtype=np.uint8)
        return np.unpackbits(self.packed_fp[idx], axis=1)[:, : self.n_bits]

    def subset(self, mask: np.ndarray) -> CandidatePool:
        """A new pool keeping only rows where ``mask`` is True.

        Used to honour a split's ``excluded_from_pool``, which is what makes
        the synthetic class-3 cohort genuinely unreachable by retrieval.
        """
        mask = np.asarray(mask, dtype=bool)
        return CandidatePool(
            mass=self.mass[mask],
            keys=self.keys[mask],
            smiles=self.smiles[mask],
            packed_fp=self.packed_fp[mask],
            n_bits=self.n_bits,
        )

    def exclude_keys(self, keys: Iterable[str]) -> CandidatePool:
        """Pool with the given ``inchikey14`` values removed."""
        excluded = set(keys)
        if not excluded:
            return self
        keep = np.array([k not in excluded for k in self.keys], dtype=bool)
        return self.subset(keep)

    def save(self, path: str | Path) -> None:
        """Persist to a single ``.npz``, for attaching as a Kaggle Dataset."""
        np.savez_compressed(
            str(path),
            mass=self.mass,
            keys=np.asarray(self.keys, dtype=str),
            smiles=np.asarray(self.smiles, dtype=str),
            packed_fp=self.packed_fp,
            n_bits=np.array([self.n_bits]),
        )

    @classmethod
    def load(cls, path: str | Path) -> CandidatePool:
        """Load a pool saved by :meth:`save`."""
        data = np.load(str(path), allow_pickle=False)
        return cls(
            mass=data["mass"],
            keys=np.asarray(data["keys"], dtype=object),
            smiles=np.asarray(data["smiles"], dtype=object),
            packed_fp=data["packed_fp"],
            n_bits=int(data["n_bits"][0]),
        )


def build_pool(
    smiles: Iterable[str],
    calculator: FingerprintCalculator | None = None,
    keys: Iterable[str] | None = None,
    progress_every: int = 0,
) -> CandidatePool:
    """Fingerprint structures and assemble a mass-sorted pool.

    Args:
        smiles: Candidate structures. Unparseable entries are skipped.
        calculator: Fingerprint calculator; a default layout is used if omitted.
        keys: Precomputed ``inchikey14`` values aligned with ``smiles``. Supply
            these when available — tautomer canonicalisation dominates build
            time for a ~700k-structure pool.
        progress_every: Print progress every N structures (0 disables).

    Returns:
        A pool deduplicated by ``inchikey14``, keeping the first SMILES seen
        for each 2D skeleton (any SMILES for a skeleton scores identically).
    """
    calc = calculator or FingerprintCalculator()
    smiles_list = list(smiles)
    key_list = list(keys) if keys is not None else [None] * len(smiles_list)
    if len(key_list) != len(smiles_list):
        raise ValueError("keys and smiles must be the same length")

    seen: set[str] = set()
    rows_fp: list[np.ndarray] = []
    rows_mass: list[float] = []
    rows_key: list[str] = []
    rows_smi: list[str] = []

    for i, (smi, key) in enumerate(zip(smiles_list, key_list, strict=True)):
        if progress_every and i and i % progress_every == 0:
            print(f"  fingerprinted {i:,}/{len(smiles_list):,}", flush=True)
        resolved_key = key if key else inchikey14(smi)
        if not resolved_key or resolved_key in seen:
            continue
        result = calc.fingerprint_and_mass(smi)
        if result is None:
            continue
        fp, mass = result
        if not np.isfinite(mass):
            continue
        seen.add(resolved_key)
        rows_fp.append(fp)
        rows_mass.append(mass)
        rows_key.append(resolved_key)
        rows_smi.append(smi)

    if not rows_fp:
        return CandidatePool(
            mass=np.zeros(0),
            keys=np.asarray([], dtype=object),
            smiles=np.asarray([], dtype=object),
            packed_fp=np.zeros((0, (calc.n_bits + 7) // 8), dtype=np.uint8),
            n_bits=calc.n_bits,
        )

    mass_arr = np.asarray(rows_mass, dtype=np.float64)
    order = np.argsort(mass_arr, kind="stable")
    return CandidatePool(
        mass=mass_arr[order],
        keys=np.asarray(rows_key, dtype=object)[order],
        smiles=np.asarray(rows_smi, dtype=object)[order],
        packed_fp=np.packbits(np.stack(rows_fp)[order], axis=1),
        n_bits=calc.n_bits,
    )


def merge_pools(*pools: CandidatePool) -> CandidatePool:
    """Concatenate pools, deduplicating by ``inchikey14`` and re-sorting by mass.

    Earlier pools take precedence on collision, so pass the more trusted
    source first (e.g. COCONUT before train structures).
    """
    pools = tuple(p for p in pools if len(p))
    if not pools:
        raise ValueError("no non-empty pools to merge")
    n_bits = pools[0].n_bits
    if any(p.n_bits != n_bits for p in pools):
        raise ValueError("cannot merge pools with different fingerprint widths")

    seen: set[str] = set()
    keep_idx: list[tuple[int, int]] = []
    for pool_i, pool in enumerate(pools):
        for row, key in enumerate(pool.keys):
            if key in seen:
                continue
            seen.add(key)
            keep_idx.append((pool_i, row))

    mass = np.array([pools[p].mass[r] for p, r in keep_idx], dtype=np.float64)
    order = np.argsort(mass, kind="stable")
    ordered = [keep_idx[i] for i in order]

    return CandidatePool(
        mass=mass[order],
        keys=np.asarray([pools[p].keys[r] for p, r in ordered], dtype=object),
        smiles=np.asarray([pools[p].smiles[r] for p, r in ordered], dtype=object),
        packed_fp=np.stack([pools[p].packed_fp[r] for p, r in ordered]),
        n_bits=n_bits,
    )
