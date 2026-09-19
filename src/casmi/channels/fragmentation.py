"""Channel 5 — MetFrag-lite in-silico fragmentation plausibility.

Targets a different failure mode than Channels 1/2: when no library spectrum
or spectral analog is close enough to help, the observed MS2 peaks themselves
still constrain which candidate structure is correct, because a structure can
only produce fragment ions that its own bonds can actually yield.

The approach (a simplified MetFrag, hence "-lite"):

1. Enumerate every fragment reachable by breaking 1 or 2 acyclic bonds of the
   candidate molecule (breaking a bond inside a ring does not disconnect
   anything, so ring bonds contribute no extra fragments beyond the whole
   molecule — the connected-components check below handles this for free).
2. For each fragment, compute the neutral mass of the disconnected piece
   (atoms plus their explicit/implicit hydrogens).
3. Score how much of the observed spectrum's intensity is "explained": a peak
   is explained if it lands within tolerance of a fragment mass adjusted by a
   small hydrogen rearrangement (protonation/deprotonation and a few extra H
   transfers, since bond cleavage in MS2 rarely preserves the exact valence).

This is a *prior*, not a primary identification channel: an incorrect
candidate can still explain some peaks by coincidence, but the correct
structure should explain a systematically higher fraction of the observed
intensity than a randomly chosen isomer at the same mass. It becomes useful
exactly where Channels 1/2 have nothing to say — a genuinely novel structure
with no close spectral relative in the training data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from casmi.adducts import PROTON_MASS
from casmi.chem import mol_from_smiles

#: Hydrogen shifts considered when matching a fragment mass to an observed
#: peak. MS2 bond cleavage commonly transfers 1-2 extra hydrogens beyond the
#: nominal fragment, in either direction, so a fixed set of shifts is checked
#: rather than assuming exact valence preservation.
H_SHIFTS: tuple[int, ...] = (-2, -1, 0, 1, 2)

#: Molecules with more acyclic bonds than this are skipped for the 2-break
#: enumeration (combinatorial cost grows as O(bonds^2)); the whole-molecule
#: mass is still returned so such candidates aren't zeroed out entirely.
MAX_BONDS_FOR_DOUBLE_BREAK = 40


@dataclass(frozen=True)
class FragmentationConfig:
    """Parameters for :func:`fragment_masses` / :func:`explain_score`."""

    max_breaks: int = 2
    max_bonds: int = MAX_BONDS_FOR_DOUBLE_BREAK
    mz_tolerance: float = 0.01
    h_shifts: tuple[int, ...] = H_SHIFTS


def _atom_hydrogen_mass(mol: Chem.Mol) -> np.ndarray | None:
    """Per-atom mass including that atom's attached hydrogens.

    Distributing H mass onto the heavy atom it's bonded to means a fragment's
    total mass is just the sum over its atom indices, with no separate
    hydrogen-bookkeeping needed by the caller.
    """
    from rdkit.Chem import GetPeriodicTable

    table = GetPeriodicTable()
    masses = np.zeros(mol.GetNumAtoms(), dtype=np.float64)
    h_mass = table.GetMostCommonIsotopeMass(1)
    for atom in mol.GetAtoms():
        try:
            base = table.GetMostCommonIsotopeMass(atom.GetAtomicNum())
        except Exception:
            return None
        masses[atom.GetIdx()] = base + atom.GetTotalNumHs() * h_mass
    return masses


def _connected_components(
    n_atoms: int, bonds: list[tuple[int, int]], dropped: set[int]
) -> list[list[int]]:
    """Connected components of the atom graph with ``dropped`` bond indices removed."""
    adjacency: list[list[int]] = [[] for _ in range(n_atoms)]
    for bond_index, (a, b) in enumerate(bonds):
        if bond_index in dropped:
            continue
        adjacency[a].append(b)
        adjacency[b].append(a)

    seen = np.zeros(n_atoms, dtype=bool)
    components: list[list[int]] = []
    for start in range(n_atoms):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        component = [start]
        while stack:
            node = stack.pop()
            for neighbour in adjacency[node]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
                    component.append(neighbour)
        components.append(component)
    return components


def fragment_masses(
    smiles: str, config: FragmentationConfig | None = None
) -> np.ndarray:
    """Neutral masses of all fragments reachable by breaking 1-2 acyclic bonds.

    Returns an empty array for unparseable SMILES, and the whole-molecule mass
    alone (as a single-element array) when the bond count exceeds the
    double-break cap or the molecule has no bonds at all.
    """
    cfg = config or FragmentationConfig()
    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.zeros(0, dtype=np.float64)

    atom_masses = _atom_hydrogen_mass(mol)
    if atom_masses is None:
        return np.zeros(0, dtype=np.float64)

    n_atoms = mol.GetNumAtoms()
    whole_mass = float(atom_masses.sum())
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    n_bonds = len(bonds)

    if n_bonds == 0 or n_bonds > cfg.max_bonds:
        return np.array([whole_mass], dtype=np.float64)

    masses = {whole_mass}
    for i in range(n_bonds):
        for component in _connected_components(n_atoms, bonds, {i}):
            masses.add(float(atom_masses[component].sum()))

    if cfg.max_breaks >= 2:
        for i in range(n_bonds):
            for j in range(i + 1, n_bonds):
                for component in _connected_components(n_atoms, bonds, {i, j}):
                    masses.add(float(atom_masses[component].sum()))

    return np.array(sorted(masses), dtype=np.float64)


def explain_score(
    fragment_mass: np.ndarray,
    peak_mz: np.ndarray,
    peak_intensity: np.ndarray,
    positive_mode: bool = True,
    config: FragmentationConfig | None = None,
) -> float:
    """Fraction of observed peak intensity explained by ``fragment_mass``.

    Each fragment mass is checked as an ion at every configured hydrogen
    shift, protonated or deprotonated according to ``positive_mode``. A peak
    counts as explained if any resulting ion mass falls within
    ``config.mz_tolerance``. The score is the intensity-weighted (square-root
    scaled, matching the spectral-cleaning convention elsewhere in this
    package) fraction of the spectrum's total intensity that is explained —
    0.0 if there is nothing to explain or nothing explains it.
    """
    cfg = config or FragmentationConfig()
    fragment_mass = np.asarray(fragment_mass, dtype=np.float64)
    peak_mz = np.asarray(peak_mz, dtype=np.float64)
    peak_intensity = np.asarray(peak_intensity, dtype=np.float64)
    if fragment_mass.size == 0 or peak_mz.size == 0:
        return 0.0

    charge_mass = PROTON_MASS if positive_mode else -PROTON_MASS
    ions = np.concatenate(
        [fragment_mass + shift * 1.00782503207 + charge_mass for shift in cfg.h_shifts]
    )
    ions = np.sort(ions)

    weights = np.sqrt(np.clip(peak_intensity, 0.0, None))
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0

    insertion = np.searchsorted(ions, peak_mz)
    explained = np.zeros(peak_mz.size, dtype=bool)
    for offset in (-1, 0):
        candidate_idx = np.clip(insertion + offset, 0, ions.size - 1)
        explained |= np.abs(ions[candidate_idx] - peak_mz) <= cfg.mz_tolerance

    return float(weights[explained].sum() / total)


def fragmentation_scores(
    candidate_smiles: list[str],
    spectra: list[tuple[np.ndarray, np.ndarray]],
    positive_mode: bool = True,
    config: FragmentationConfig | None = None,
) -> np.ndarray:
    """MetFrag-lite explain-ratio per candidate, best-of a molecule's spectra.

    Args:
        candidate_smiles: One SMILES per candidate structure.
        spectra: The query molecule's ``(mz, intensity)`` pairs. A candidate
            takes the best score across spectra, matching how Channels 1/2
            aggregate evidence per molecule rather than per spectrum.
        positive_mode: Ionisation polarity, shared across the molecule's
            spectra (adducts within one molecule share polarity).
        config: Fragmentation parameters.

    Returns:
        ``(len(candidate_smiles),)`` scores in ``[0, 1]``.
    """
    cfg = config or FragmentationConfig()
    scores = np.zeros(len(candidate_smiles), dtype=np.float32)
    if not spectra:
        return scores

    fragment_cache: dict[str, np.ndarray] = {}
    for i, smiles in enumerate(candidate_smiles):
        fragments = fragment_cache.get(smiles)
        if fragments is None:
            fragments = fragment_masses(smiles, config=cfg)
            fragment_cache[smiles] = fragments
        if fragments.size == 0:
            continue
        best = 0.0
        for mz, intensity in spectra:
            score = explain_score(fragments, mz, intensity, positive_mode=positive_mode, config=cfg)
            if score > best:
                best = score
        scores[i] = best
    return scores
