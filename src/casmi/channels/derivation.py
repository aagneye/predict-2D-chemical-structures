"""Channel 8 — analog derivation: propose novel structures as modified analogs.

The hidden test is "real or hypothesized natural products and their analogs"
measured by Enveda, and an estimated 40-50% of it is in no structure database
at all (class 3). No retrieval channel can reach those, and generic de novo
generators score 2-11% top-1 on honest benchmarks. But a novel natural product
is very often a *known* natural product plus one or two biosynthetic edits: a
hydroxyl, a methyl, a sugar, an acyl group. Its MS/MS spectrum is then similar
to the known parent's after a mass shift, which is exactly what Channel 2
(mass-shifted analog search) finds.

This channel closes that loop. For each spectral analog of the query, the
mass difference between the query and the analog says *which* edit happened
(``+15.995`` is an oxygen, ``+162.053`` a hexose, ``+14.016`` a methyl...);
the channel enumerates every placement of that edit (or pair of edits) on the
analog's structure with RDKit reactions, keeps the products that land on the
measured mass, and ranks them with the same evidence the pool uses (FPNet,
fragmentation, analog relations). It is the MetGenX "structure-to-structure"
idea (Nat. Commun. 2026: retrieve spectrally similar templates, generate from
them; 21.7% database-free top-1) made explicit and chemically constrained, and
it is what one competitor measured as able to produce the correct structure
for 100 of 130 database-absent molecules.

Every edit is a reaction SMARTS plus its declared formula delta. A product is
kept only if its exact mass equals parent + declared delta (this catches any
SMARTS whose hydrogen bookkeeping is wrong) and lands within the ppm window of
the query's neutral mass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt

from casmi.adducts import ATOMIC_MASS
from casmi.chem import inchikey14

RDLogger.DisableLog("rdApp.*")

#: A hydroxyl that is not part of a carboxylic acid (esters/glycosides of acids exist,
#: but acylating an acid OH makes anhydrides, which natural products rarely are).
_OH = "[O;H1;!$(OC=O):1]-[#6:2]"

#: (name, reaction SMARTS, formula delta). Deltas use +/- element counts.
TRANSFORMS: tuple[tuple[str, str, str], ...] = (
    ("hydroxylation_aromatic", "[c;H1:1]>>[c:1]O", "+O"),
    ("hydroxylation_aliphatic", "[C;X4;!H0:1]>>[C:1]O", "+O"),
    ("methoxylation_aromatic", "[c;H1:1]>>[c:1]OC", "+C1H2O1"),
    ("O_methylation", _OH + ">>[#6:2]-[O:1]C", "+C1H2"),
    ("N_methylation", "[N;H1,H2;!$(NC=O);!$(N-a):1]>>[N:1]C", "+C1H2"),
    ("C_methylation_aromatic", "[c;H1:1]>>[c:1]C", "+C1H2"),
    ("O_demethylation", "[#6:2]-[O;X2:1]-[CH3]>>[#6:2]-[O:1]", "-C1H2"),
    ("N_demethylation", "[#6:2]-[N:1]-[CH3]>>[#6:2]-[N:1]", "-C1H2"),
    ("O_hexose", _OH + ">>[#6:2]-[O:1]C1OC(CO)C(O)C(O)C1O", "+C6H10O5"),
    ("O_deoxyhexose", _OH + ">>[#6:2]-[O:1]C1OC(C)C(O)C(O)C1O", "+C6H10O4"),
    ("O_pentose", _OH + ">>[#6:2]-[O:1]C1OCC(O)C(O)C1O", "+C5H8O4"),
    ("O_glucuronide", _OH + ">>[#6:2]-[O:1]C1OC(C(=O)O)C(O)C(O)C1O", "+C6H8O6"),
    (
        "de_hexose",
        "[#6:2]-[O:1]-[C;R1]1-[O;R1]-[C;R1](-[CH2]-[OH1])-[C;R1](-[OH1])"
        "-[C;R1](-[OH1])-[C;R1]1-[OH1]>>[#6:2]-[O:1]",
        "-C6H10O5",
    ),
    (
        "de_deoxyhexose",
        "[#6:2]-[O:1]-[C;R1]1-[O;R1]-[C;R1](-[CH3])-[C;R1](-[OH1])"
        "-[C;R1](-[OH1])-[C;R1]1-[OH1]>>[#6:2]-[O:1]",
        "-C6H10O4",
    ),
    (
        "de_pentose",
        "[#6:2]-[O:1]-[C;R1]1-[O;R1]-[CH2;R1]-[C;R1](-[OH1])"
        "-[C;R1](-[OH1])-[C;R1]1-[OH1]>>[#6:2]-[O:1]",
        "-C5H8O4",
    ),
    ("O_acetylation", _OH + ">>[#6:2]-[O:1]C(C)=O", "+C2H2O1"),
    ("O_deacetylation", "[#6:2]-[O:1]-C(=O)-[CH3]>>[#6:2]-[O:1]", "-C2H2O1"),
    ("O_malonylation", _OH + ">>[#6:2]-[O:1]C(=O)CC(=O)O", "+C3H2O3"),
    ("O_coumaroylation", _OH + ">>[#6:2]-[O:1]C(=O)C=Cc1ccc(O)cc1", "+C9H6O2"),
    ("O_caffeoylation", _OH + ">>[#6:2]-[O:1]C(=O)C=Cc1ccc(O)c(O)c1", "+C9H6O3"),
    ("O_feruloylation", _OH + ">>[#6:2]-[O:1]C(=O)C=Cc1ccc(O)c(OC)c1", "+C10H8O3"),
    ("O_galloylation", _OH + ">>[#6:2]-[O:1]C(=O)c1cc(O)c(O)c(O)c1", "+C7H4O4"),
    ("O_sulfation", "[O;H1;$(Oc):1]-[c:2]>>[c:2]-[O:1]S(=O)(=O)O", "+O3S1"),
    ("C_prenylation_aromatic", "[c;H1:1]>>[c:1]CC=C(C)C", "+C5H8"),
    ("hydrogenation", "[C:1]=[C:2]>>[C:1]-[C:2]", "+H2"),
    ("dehydrogenation", "[C;H1,H2:1]-[C;H1,H2:2]>>[C:1]=[C:2]", "-H2"),
    ("alcohol_oxidation", "[C;H1,H2:1]-[O;H1:2]>>[C:1]=[O:2]", "-H2"),
    ("ketone_reduction", "[#6:3]-[C:1](=[O:2])-[#6:4]>>[#6:3]-[C:1](-[O:2])-[#6:4]", "+H2"),
    ("dehydration", "[C;!H0:1]-[C:2]-[O;H1]>>[C:1]=[C:2]", "-H2O1"),
    ("hydration", "[C:1]=[C:2]>>[C:1]-[C:2]O", "+H2O1"),
    ("decarboxylation", "[#6:1]-C(=O)-[O;H1]>>[#6:1]", "-C1O2"),
)

_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_delta_mass(delta: str) -> float:
    """Monoisotopic mass of a signed formula delta like ``+C6H10O5`` or ``-C1H2``."""
    sign = -1.0 if delta.startswith("-") else 1.0
    mass = 0.0
    for element, count in _ELEMENT.findall(delta.lstrip("+-")):
        mass += ATOMIC_MASS[element] * (int(count) if count else 1)
    return sign * mass


@dataclass(frozen=True)
class DerivationConfig:
    """Search limits. Defaults keep the channel to a few seconds per molecule."""

    #: Spectral analogs used as parents, best first.
    n_parents: int = 15
    #: Ignore analogs below this mass-shifted spectral similarity.
    min_parent_similarity: float = 0.3
    #: Up to this many edits per product (1 or 2).
    max_steps: int = 2
    #: Products kept per (parent, edit path) before the mass check.
    max_products_per_path: int = 300
    #: Hard cap on unique products scored per molecule.
    max_products: int = 1500
    #: Generated candidates returned per molecule (after FPNet ranking).
    top_k: int = 30
    #: Product mass must be within this many ppm of the query neutral mass.
    ppm: float = 10.0
    #: Absolute floor on the delta-matching tolerance (Da).
    min_tol_da: float = 0.003


@dataclass
class DerivedCandidate:
    smiles: str
    inchikey14: str
    mass: float
    fingerprint: np.ndarray
    parent_similarity: float
    steps: int
    support: int
    fpnet_score: float = 0.0
    fpnet_normalised_score: float = 0.0
    paths: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def _compiled() -> tuple[tuple[str, object, float], ...]:
    out = []
    for name, smarts, delta in TRANSFORMS:
        rxn = AllChem.ReactionFromSmarts(smarts)
        rxn.Initialize()
        out.append((name, rxn, formula_delta_mass(delta)))
    return tuple(out)


def _edit_paths(delta: float, tol: float, max_steps: int) -> list[tuple[int, ...]]:
    """Index sequences of 1..max_steps transforms whose deltas sum to ``delta``."""
    transforms = _compiled()
    paths: list[tuple[int, ...]] = []
    for steps in range(1, max_steps + 1):
        for combo in combinations_with_replacement(range(len(transforms)), steps):
            if abs(sum(transforms[i][2] for i in combo) - delta) <= tol:
                paths.append(combo)
    return paths


def apply_transform(mol: Chem.Mol, index: int, limit: int) -> list[Chem.Mol]:
    """All single-site products of transform ``index`` on ``mol`` (sanitised, mass-checked)."""
    _, rxn, delta = _compiled()[index]
    parent_mass = ExactMolWt(mol)
    seen: set[str] = set()
    products: list[Chem.Mol] = []
    try:
        outcomes = rxn.RunReactants((mol,), maxProducts=limit * 4)
    except Exception:
        return products
    for outcome in outcomes:
        product = outcome[0]
        try:
            Chem.SanitizeMol(product)
            smiles = Chem.MolToSmiles(product)
        except Exception:
            continue
        if smiles in seen or "." in smiles:
            continue
        clean = Chem.MolFromSmiles(smiles)
        if clean is None or abs(ExactMolWt(clean) - parent_mass - delta) > 1e-3:
            continue
        seen.add(smiles)
        products.append(clean)
        if len(products) >= limit:
            break
    return products


def derive_from_parent(
    parent_smiles: str, target_mass: float, config: DerivationConfig | None = None
) -> dict[str, tuple[Chem.Mol, int, str]]:
    """Products of ``parent_smiles`` reaching ``target_mass``: ``smiles -> (mol, steps, path)``."""
    cfg = config or DerivationConfig()
    parent = Chem.MolFromSmiles(parent_smiles)
    if parent is None or not np.isfinite(target_mass):
        return {}
    delta = target_mass - ExactMolWt(parent)
    tol = max(cfg.min_tol_da, target_mass * cfg.ppm / 1e6)
    if abs(delta) <= tol:
        return {}  # same mass: an isomer, which retrieval already covers
    names = [t[0] for t in _compiled()]
    found: dict[str, tuple[Chem.Mol, int, str]] = {}
    for path in _edit_paths(delta, tol, cfg.max_steps):
        frontier = [parent]
        for index in path:
            nxt: list[Chem.Mol] = []
            for mol in frontier:
                nxt.extend(apply_transform(mol, index, cfg.max_products_per_path))
            frontier = nxt[: cfg.max_products_per_path]
            if not frontier:
                break
        label = "+".join(names[i] for i in path)
        for mol in frontier:
            mass = ExactMolWt(mol)
            if abs(mass - target_mass) > target_mass * cfg.ppm / 1e6:
                continue
            smiles = Chem.MolToSmiles(mol)
            if smiles not in found or found[smiles][1] > len(path):
                found[smiles] = (mol, len(path), label)
    return found


def derive_candidates(
    parents: list[tuple[str, float]],
    target_mass: float,
    calculator,
    fpnet_logits: np.ndarray | None,
    exclude_keys: set[str],
    config: DerivationConfig | None = None,
) -> list[DerivedCandidate]:
    """Generate, key, fingerprint and rank derived candidates for one molecule.

    Args:
        parents: ``(smiles, spectral_similarity)`` of the query's spectral analogs.
        calculator: :class:`casmi.chem.FingerprintCalculator` matching ``fpnet_logits``.
        fpnet_logits: FPNet's predicted logits; candidates are ranked by the
            bit-count-normalised dot product. Without logits, by parent similarity.
        exclude_keys: Scorer keys other channels already propose.
    """
    cfg = config or DerivationConfig()
    usable = [(s, sim) for s, sim in parents if s and sim >= cfg.min_parent_similarity]
    usable = sorted(usable, key=lambda p: -p[1])[: cfg.n_parents]
    by_key: dict[str, DerivedCandidate] = {}
    for parent_smiles, similarity in usable:
        for smiles, (mol, steps, label) in derive_from_parent(
            parent_smiles, target_mass, cfg
        ).items():
            key = inchikey14(smiles)
            if key is None or key in exclude_keys:
                continue
            existing = by_key.get(key)
            if existing is None:
                if len(by_key) >= cfg.max_products:
                    continue
                fp = calculator.from_mol(mol)
                if fp is None:
                    continue
                by_key[key] = DerivedCandidate(
                    smiles=smiles, inchikey14=key, mass=float(ExactMolWt(mol)),
                    fingerprint=fp, parent_similarity=float(similarity),
                    steps=steps, support=1, paths=[label],
                )
            else:
                existing.support += 1
                existing.parent_similarity = max(existing.parent_similarity, float(similarity))
                existing.steps = min(existing.steps, steps)
                if label not in existing.paths:
                    existing.paths.append(label)
    candidates = list(by_key.values())
    if not candidates:
        return []
    if fpnet_logits is not None:
        from casmi.models.fpnet import normalised_score, score_candidates

        fps = np.stack([c.fingerprint for c in candidates])
        raw = score_candidates(fps, fpnet_logits)
        norm = normalised_score(fps, fpnet_logits)
        for c, r, n in zip(candidates, raw, norm, strict=True):
            c.fpnet_score = float(r)
            c.fpnet_normalised_score = float(n)
        candidates.sort(key=lambda c: (-c.fpnet_normalised_score, -c.parent_similarity))
    else:
        candidates.sort(key=lambda c: (-c.parent_similarity, c.steps))
    return candidates[: cfg.top_k]
