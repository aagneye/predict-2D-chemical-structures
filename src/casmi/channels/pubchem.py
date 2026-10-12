"""Channel 7 — PubChem-only candidates from a mass-sorted 106M-structure tier.

The main candidate pool (train ∪ COCONUT ∪ PubChemLite, ~1.1M structures)
still misses many class-2 answers: "known structure, no reference spectrum"
means *in PubChem or COCONUT*, and PubChem is a hundred times larger than
the pool. This channel searches the full tier for every molecule and
proposes the few PubChem structures that are not already pool candidates.

The tier is three memory-mapped arrays (the community
``casmi26-pubchem-tier`` layout): ``pc_mass.npy`` (ascending neutral mass),
``pc_off.npy`` (byte offsets into) ``pc_smiles.npy`` (one ASCII SMILES per
structure). Optional per-row popularity arrays (``pc_lsid.npy``,
``pc_lpmid.npy``, float16 ``log1p`` counts, from
``casmi26-pubchem-popularity-prior``) are aligned row by row.

Scoring inside a mass window is the same Bayes dot product the pool uses
(:func:`casmi.models.fpnet.score_candidates`) on fingerprints computed on the
fly, z-scored within the window, plus ``mu * popularity``. On a 250-molecule
natural-product panel with the answer removed from the pool, FPNet alone
puts the answer first for 40% of molecules and FPNet plus the prior for 93%
(community benchmark): among a few hundred same-mass PubChem structures, the
documented one is the natural product.

The channel is only as good as the fingerprint model: it has no library or
analog evidence of its own, so its candidates carry ``pubchem_only = 1`` and
the reranker learns how much to trust them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from casmi.chem import FingerprintCalculator, inchikey14


@dataclass(frozen=True)
class PubChemConfig:
    """Window and shortlist parameters."""

    ppm_window: float = 10.0
    #: Structures proposed per molecule (after excluding pool keys and dedup).
    top_k: int = 25
    #: Windows larger than this are cut to the most documented structures
    #: before fingerprinting (fingerprinting is ~1-2 ms per structure).
    max_window: int = 4000
    #: Weight on popularity in the window score ``z(f.z) + mu * popularity``.
    popularity_mu: float = 0.25


@dataclass
class PubChemCandidate:
    smiles: str
    inchikey14: str
    mass: float
    fingerprint: np.ndarray
    fpnet_score: float
    fpnet_normalised_score: float
    popularity: float
    #: ``z(f.z) + mu * popularity`` inside the window, the channel's own rank score.
    window_score: float


class PubChemTier:
    """Memory-mapped access to the mass-sorted PubChem tier."""

    def __init__(self, tier_dir: str | Path, popularity_dir: str | Path | None = None) -> None:
        tier_dir = Path(tier_dir)
        self.mass = np.load(tier_dir / "pc_mass.npy", mmap_mode="r")
        self.offsets = np.load(tier_dir / "pc_off.npy", mmap_mode="r")
        self.smiles_buffer = np.load(tier_dir / "pc_smiles.npy", mmap_mode="r")
        if len(self.offsets) != len(self.mass) + 1:
            raise ValueError("pc_off.npy must have one more entry than pc_mass.npy")
        self.lsid = self.lpmid = None
        if popularity_dir is not None:
            popularity_dir = Path(popularity_dir)
            lsid = popularity_dir / "pc_lsid.npy"
            lpmid = popularity_dir / "pc_lpmid.npy"
            if lsid.exists() and lpmid.exists():
                self.lsid = np.load(lsid, mmap_mode="r")
                self.lpmid = np.load(lpmid, mmap_mode="r")
                if len(self.lsid) != len(self.mass) or len(self.lpmid) != len(self.mass):
                    raise ValueError("popularity arrays are not aligned with the tier")

    def __len__(self) -> int:
        return int(len(self.mass))

    @property
    def has_popularity(self) -> bool:
        return self.lsid is not None

    def window(self, target_mass: float, ppm: float) -> tuple[int, int]:
        """``(start, stop)`` row range within ±``ppm`` of ``target_mass``."""
        if not np.isfinite(target_mass) or target_mass <= 0 or len(self) == 0:
            return 0, 0
        tol = target_mass * ppm / 1e6
        lo = int(np.searchsorted(self.mass, target_mass - tol, "left"))
        hi = int(np.searchsorted(self.mass, target_mass + tol, "right"))
        return lo, hi

    def smiles(self, start: int, stop: int) -> list[str]:
        """Decode the SMILES of rows ``start:stop`` with one buffer read."""
        if stop <= start:
            return []
        off = np.asarray(self.offsets[start : stop + 1]).astype(np.int64)
        raw = bytes(self.smiles_buffer[off[0] : off[-1]])
        base = int(off[0])
        return [raw[off[i] - base : off[i + 1] - base].decode("ascii") for i in range(stop - start)]

    def popularity(self, start: int, stop: int) -> np.ndarray:
        """``log1p(substances) + log1p(pubmed)`` per row, zeros if unavailable."""
        if stop <= start:
            return np.zeros(0, dtype=np.float32)
        if self.lsid is None:
            return np.zeros(stop - start, dtype=np.float32)
        return (
            np.asarray(self.lsid[start:stop], dtype=np.float32)
            + np.asarray(self.lpmid[start:stop], dtype=np.float32)
        )


def pubchem_candidates(
    tier: PubChemTier,
    target_mass: float,
    fpnet_logits: np.ndarray,
    calculator: FingerprintCalculator,
    exclude_keys: set[str],
    config: PubChemConfig | None = None,
) -> list[PubChemCandidate]:
    """Propose up to ``top_k`` PubChem-only structures for one molecule.

    Args:
        target_mass: The molecule's precursor-derived neutral mass.
        fpnet_logits: FPNet's predicted fingerprint logits for this molecule.
        calculator: Fingerprint layout matching ``fpnet_logits``.
        exclude_keys: Scorer keys already proposed by other channels (the pool
            window, library hits); PubChem adds only what they lack.

    Returns:
        Candidates best first by ``window_score``; empty if the window is
        empty or nothing parses.
    """
    cfg = config or PubChemConfig()
    start, stop = tier.window(target_mass, cfg.ppm_window)
    if stop <= start:
        return []

    smiles = tier.smiles(start, stop)
    popularity = tier.popularity(start, stop)
    rows = np.arange(len(smiles))
    if len(smiles) > cfg.max_window:
        # Keep the most documented structures: in this competition's domain the
        # answer is better documented than almost all of its mass neighbours.
        rows = np.argsort(-popularity, kind="stable")[: cfg.max_window]
        smiles = [smiles[i] for i in rows]
        popularity = popularity[rows]

    fps: list[np.ndarray] = []
    masses: list[float] = []
    keep: list[int] = []
    for i, smi in enumerate(smiles):
        result = calculator.fingerprint_and_mass(smi)
        if result is None:
            continue
        fp, mass = result
        fps.append(fp)
        masses.append(mass)
        keep.append(i)
    if not fps:
        return []

    from casmi.models.fpnet import normalised_score, score_candidates

    fp_matrix = np.stack(fps)
    logits = np.asarray(fpnet_logits, dtype=np.float32)
    raw = score_candidates(fp_matrix, logits).astype(np.float64)
    norm = normalised_score(fp_matrix, logits).astype(np.float64)
    std = float(norm.std())
    z = (norm - norm.mean()) / std if std > 1e-9 else np.zeros_like(norm)
    pop = popularity[keep].astype(np.float64)
    window_score = z + cfg.popularity_mu * pop

    out: list[PubChemCandidate] = []
    seen: set[str] = set(exclude_keys)
    for j in np.argsort(-window_score, kind="stable"):
        smi = smiles[keep[j]]
        key = inchikey14(smi)
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(
            PubChemCandidate(
                smiles=smi,
                inchikey14=key,
                mass=float(masses[j]),
                fingerprint=fp_matrix[j],
                fpnet_score=float(raw[j]),
                fpnet_normalised_score=float(norm[j]),
                popularity=float(pop[j]),
                window_score=float(window_score[j]),
            )
        )
        if len(out) >= cfg.top_k:
            break
    return out
