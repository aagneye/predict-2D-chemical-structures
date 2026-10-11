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

**Parsimony matters more than cleavage chemistry.** A community ablation on
the competition's own natural-product panel (1,000 held-out molecules, four
query sets) found that hand-written bond-propensity rules did nothing, while
"how few cleavages does this candidate need" carried real signal: credit a
peak that only a two-bond cleavage explains less than one a single cleavage
explains (x0.6), weight intensities linearly rather than by square root,
tighten the tolerance to 0.005 Da, and allow hydrogen shifts up to +3. The
four together were worth +0.023 MRR@25 out of sample. Those are the defaults
in :class:`FragmentationConfig`; the previous behaviour is one config away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from casmi.adducts import PROTON_MASS
from casmi.chem import mol_from_smiles

#: Hydrogen shifts considered when matching a fragment mass to an observed
#: peak. MS2 bond cleavage commonly transfers extra hydrogens beyond the
#: nominal fragment, in either direction, so a fixed set of shifts is checked
#: rather than assuming exact valence preservation. +3 earned its place in
#: the held-out ablation (see module docstring); -3 did not.
H_SHIFTS: tuple[int, ...] = (-2, -1, 0, 1, 2, 3)

#: Monoisotopic hydrogen mass used for the shifts.
_H_MASS = 1.00782503207

#: Molecules with more acyclic bonds than this are skipped for the 2-break
#: enumeration (combinatorial cost grows as O(bonds^2)); the whole-molecule
#: mass is still returned so such candidates aren't zeroed out entirely.
MAX_BONDS_FOR_DOUBLE_BREAK = 40


@dataclass(frozen=True)
class FragmentationConfig:
    """Parameters for :func:`fragment_masses` / :func:`explain_score`.

    Defaults are the parsimony-tuned values (module docstring). The legacy
    behaviour is ``FragmentationConfig(mz_tolerance=0.01, h_shifts=(-2, -1, 0, 1, 2),
    intensity_power=0.5, two_break_weight=1.0)``.
    """

    max_breaks: int = 2
    max_bonds: int = MAX_BONDS_FOR_DOUBLE_BREAK
    mz_tolerance: float = 0.005
    h_shifts: tuple[int, ...] = H_SHIFTS
    #: Exponent applied to peak intensities before weighting (1.0 linear,
    #: 0.5 square root).
    intensity_power: float = 1.0
    #: Credit for a peak that only a two-bond cleavage can explain, relative
    #: to a peak a single cleavage (or the intact molecule) explains.
    two_break_weight: float = 0.6


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


def fragment_masses_with_depth(
    smiles: str, config: FragmentationConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fragment masses plus the fewest bond breaks needed to reach each one.

    Depth 0 is the intact molecule, 1 a single cleavage, 2 a double cleavage.
    A fragment reachable both ways keeps the smaller depth: parsimony credits
    the cheapest explanation. Returns two empty arrays for unparseable SMILES,
    and the whole-molecule mass alone (depth 0) when the bond count exceeds
    the double-break cap or the molecule has no bonds at all.
    """
    cfg = config or FragmentationConfig()
    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int8)

    atom_masses = _atom_hydrogen_mass(mol)
    if atom_masses is None:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int8)

    n_atoms = mol.GetNumAtoms()
    whole_mass = float(atom_masses.sum())
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    n_bonds = len(bonds)

    if n_bonds == 0 or n_bonds > cfg.max_bonds:
        return np.array([whole_mass], dtype=np.float64), np.zeros(1, dtype=np.int8)

    # mass -> minimum depth. Masses are rounded to 1e-6 Da so the same fragment
    # reached via two different breaks collapses to one key.
    depth: dict[float, int] = {round(whole_mass, 6): 0}
    for i in range(n_bonds):
        for component in _connected_components(n_atoms, bonds, {i}):
            m = round(float(atom_masses[component].sum()), 6)
            if depth.get(m, 99) > 1:
                depth[m] = 1

    if cfg.max_breaks >= 2:
        for i in range(n_bonds):
            for j in range(i + 1, n_bonds):
                for component in _connected_components(n_atoms, bonds, {i, j}):
                    m = round(float(atom_masses[component].sum()), 6)
                    if m not in depth:
                        depth[m] = 2

    order = sorted(depth)
    return (
        np.array(order, dtype=np.float64),
        np.array([depth[m] for m in order], dtype=np.int8),
    )


def fragment_masses(
    smiles: str, config: FragmentationConfig | None = None
) -> np.ndarray:
    """Neutral masses of all fragments reachable by breaking 1-2 acyclic bonds.

    Depth-free view of :func:`fragment_masses_with_depth`: an empty array for
    unparseable SMILES, the whole-molecule mass alone when the bond count
    exceeds the double-break cap or the molecule has no bonds at all.
    """
    masses, _ = fragment_masses_with_depth(smiles, config=config)
    return masses


def explain_score(
    fragment_mass: np.ndarray,
    peak_mz: np.ndarray,
    peak_intensity: np.ndarray,
    positive_mode: bool = True,
    config: FragmentationConfig | None = None,
    fragment_depth: np.ndarray | None = None,
) -> float:
    """Parsimony-weighted fraction of observed intensity explained by the fragments.

    Each fragment mass is checked as an ion at every configured hydrogen
    shift, protonated or deprotonated according to ``positive_mode``. A peak
    is explained if any resulting ion mass falls within ``config.mz_tolerance``;
    it earns full credit when the intact molecule or a single cleavage
    explains it and ``config.two_break_weight`` when only a double cleavage
    does. The score is the (``intensity_power``-scaled) intensity-weighted
    credit over the spectrum's total intensity; 0.0 if there is nothing to
    explain or nothing explains it.

    Args:
        fragment_depth: Per-fragment minimum cleavage count aligned with
            ``fragment_mass`` (see :func:`fragment_masses_with_depth`). When
            omitted every fragment counts as depth 1 (full credit), which
            reproduces the depth-free score.
    """
    cfg = config or FragmentationConfig()
    fragment_mass = np.asarray(fragment_mass, dtype=np.float64)
    peak_mz = np.asarray(peak_mz, dtype=np.float64)
    peak_intensity = np.asarray(peak_intensity, dtype=np.float64)
    if fragment_mass.size == 0 or peak_mz.size == 0:
        return 0.0

    if fragment_depth is None:
        credit = np.ones(fragment_mass.size, dtype=np.float64)
    else:
        fragment_depth = np.asarray(fragment_depth)
        if fragment_depth.shape != fragment_mass.shape:
            raise ValueError("fragment_depth must align with fragment_mass")
        credit = np.where(fragment_depth >= 2, cfg.two_break_weight, 1.0).astype(np.float64)

    charge_mass = PROTON_MASS if positive_mode else -PROTON_MASS
    ions = np.concatenate(
        [fragment_mass + shift * _H_MASS + charge_mass for shift in cfg.h_shifts]
    )
    ion_credit = np.concatenate([credit for _ in cfg.h_shifts])
    order = np.argsort(ions, kind="stable")
    ions = ions[order]
    ion_credit = ion_credit[order]

    weights = np.power(np.clip(peak_intensity, 0.0, None), cfg.intensity_power)
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0

    lo = np.searchsorted(ions, peak_mz - cfg.mz_tolerance, side="left")
    hi = np.searchsorted(ions, peak_mz + cfg.mz_tolerance, side="right")
    explained = np.zeros(peak_mz.size, dtype=np.float64)
    for k in range(peak_mz.size):
        if hi[k] > lo[k]:
            explained[k] = float(ion_credit[lo[k] : hi[k]].max())

    return float((weights * explained).sum() / total)


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

    fragment_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, smiles in enumerate(candidate_smiles):
        cached = fragment_cache.get(smiles)
        if cached is None:
            cached = fragment_masses_with_depth(smiles, config=cfg)
            fragment_cache[smiles] = cached
        fragments, depths = cached
        if fragments.size == 0:
            continue
        best = 0.0
        for mz, intensity in spectra:
            score = explain_score(
                fragments,
                mz,
                intensity,
                positive_mode=positive_mode,
                config=cfg,
                fragment_depth=depths,
            )
            if score > best:
                best = score
        scores[i] = best
    return scores
