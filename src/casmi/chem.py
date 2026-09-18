"""RDKit primitives: tautomer-canonical InChIKey14 matching and fingerprints.

Correctness here underpins every number the project produces. The competition
scores a prediction correct when its **tautomer-canonicalised InChIKey first
block** (``InChIKey14``, the 2D skeleton hash) equals the truth's. Stereo-
chemistry and tautomer form are not scored, so :func:`inchikey14` deliberately
collapses both.

RDKit is pinned to 2026.3.3 in ``pyproject.toml`` to match the scoring
environment — tautomer canonicalisation output is version-sensitive.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.MolStandardize import rdMolStandardize

from casmi.config import CFG, FingerprintConfig

# RDKit chatters on stderr for every unparseable SMILES; we handle failures by
# returning None and don't want the noise.
RDLogger.DisableLog("rdApp.*")

#: Length of an InChIKey's first block (the 2D skeleton hash).
INCHIKEY14_LEN = 14


@lru_cache(maxsize=1)
def _tautomer_enumerator() -> rdMolStandardize.TautomerEnumerator:
    """Shared enumerator. Construction is not free, and it is stateless in use."""
    return rdMolStandardize.TautomerEnumerator()


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse SMILES, returning ``None`` on failure rather than raising.

    Candidate databases and model output both contain unparseable strings; a
    pipeline that raises on the first bad SMILES is useless.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def canonical_tautomer(mol: Chem.Mol) -> Chem.Mol:
    """Return the canonical tautomer, falling back to the input on failure.

    Canonicalisation can fail or time out on pathological structures; degrading
    to the original molecule keeps the pipeline running (the resulting key may
    simply not match, which is the correct failure mode).
    """
    try:
        return _tautomer_enumerator().Canonicalize(mol)
    except Exception:
        return mol


@lru_cache(maxsize=200_000)
def inchikey14(smiles: str) -> str | None:
    """Tautomer-canonical InChIKey first block for ``smiles``.

    This is *the* matching key for scoring. Returns ``None`` if the SMILES
    cannot be parsed or keyed.

    Cached because the baseline pipeline keys the same candidate structures
    repeatedly across molecules, and tautomer canonicalisation dominates its
    cost.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(canonical_tautomer(mol))
    except Exception:
        return None
    if not key:
        return None
    return key[:INCHIKEY14_LEN]


def inchikey14_many(smiles_list) -> list[str | None]:
    """Vectorised :func:`inchikey14` over an iterable of SMILES."""
    return [inchikey14(s) for s in smiles_list]


def exact_mass(smiles: str) -> float | None:
    """Monoisotopic mass of the neutral molecule, or ``None`` if unparseable."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        return float(ExactMolWt(mol))
    except Exception:
        return None


class FingerprintCalculator:
    """Computes the concatenated fingerprint defined by :class:`FingerprintConfig`.

    Layout is ``[Morgan r2 | Morgan r3 | RDKit path FP | MACCS]``. The order and
    widths are load-bearing — a trained FPNet checkpoint and any precomputed
    candidate-pool fingerprint array are only valid for the exact layout they
    were built with, so :attr:`n_bits` is asserted against saved artifacts.

    An optional ``bit_subset`` selects a fixed subset of bits (the reference
    solution trains on 6,930 of the 10,407 available bits, dropping bits that
    are constant across its corpus).
    """

    def __init__(
        self,
        config: FingerprintConfig | None = None,
        bit_subset: np.ndarray | None = None,
    ) -> None:
        self.config = config or CFG.fingerprint
        self._morgan2 = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=self.config.morgan2_bits
        )
        self._morgan3 = rdFingerprintGenerator.GetMorganGenerator(
            radius=3, fpSize=self.config.morgan3_bits
        )
        self._rdkit = rdFingerprintGenerator.GetRDKitFPGenerator(
            fpSize=self.config.rdkit_bits, maxPath=self.config.rdkit_max_path
        )
        if bit_subset is not None:
            bit_subset = np.asarray(bit_subset, dtype=np.int64)
            if bit_subset.size and (
                bit_subset.min() < 0 or bit_subset.max() >= self.config.total_bits
            ):
                raise ValueError(
                    f"bit_subset out of range for layout of {self.config.total_bits} bits"
                )
        self.bit_subset = bit_subset

    @property
    def n_bits(self) -> int:
        """Width of the vectors this calculator emits."""
        if self.bit_subset is not None:
            return int(self.bit_subset.size)
        return self.config.total_bits

    def from_mol(self, mol: Chem.Mol) -> np.ndarray | None:
        """Fingerprint a parsed molecule as a ``uint8`` 0/1 vector."""
        try:
            fp = np.concatenate(
                [
                    self._morgan2.GetFingerprintAsNumPy(mol).astype(np.uint8),
                    self._morgan3.GetFingerprintAsNumPy(mol).astype(np.uint8),
                    self._rdkit.GetFingerprintAsNumPy(mol).astype(np.uint8),
                    np.frombuffer(
                        bytes(MACCSkeys.GenMACCSKeys(mol).ToBitString(), "ascii"), dtype=np.uint8
                    )
                    - ord("0"),
                ]
            )
        except Exception:
            return None
        if fp.size != self.config.total_bits:
            return None
        if self.bit_subset is not None:
            fp = fp[self.bit_subset]
        return fp

    def from_smiles(self, smiles: str) -> np.ndarray | None:
        """Fingerprint a SMILES string, or ``None`` if unparseable."""
        mol = mol_from_smiles(smiles)
        if mol is None:
            return None
        return self.from_mol(mol)

    def fingerprint_and_mass(self, smiles: str) -> tuple[np.ndarray, float] | None:
        """Both fingerprint and neutral monoisotopic mass in one parse.

        Used when building candidate pools, where parsing twice would double
        the dominant cost.
        """
        mol = mol_from_smiles(smiles)
        if mol is None:
            return None
        fp = self.from_mol(mol)
        if fp is None:
            return None
        try:
            mass = float(ExactMolWt(mol))
        except Exception:
            return None
        return fp, mass


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto similarity between two binary fingerprint vectors."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    inter = float(a @ b)
    union = float(a.sum() + b.sum() - inter)
    return inter / union if union > 0 else 0.0


def tanimoto_matrix(fps_a: np.ndarray, fps_b: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto between two stacks of binary fingerprints.

    Returns shape ``(len(fps_a), len(fps_b))``. Computed as one matmul so that
    Channel 2 can score a full candidate set against all retained analogs at
    once.
    """
    a = np.asarray(fps_a, dtype=np.float32)
    b = np.asarray(fps_b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("expected 2-D fingerprint stacks")
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)
