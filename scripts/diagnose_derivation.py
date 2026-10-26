"""Why does Channel 8 miss a class-3 truth? Per held-out novel molecule, compare the
truth's formula with each spectral analog's (the derivation parents) and classify:

* ``generated``   — the truth is among ``derive_from_parent`` products (recall);
* ``reachable_1/2/3`` — the formula delta is a sum of 1/2/3 transform deltas but the
  product was not generated (site chemistry, caps, or 3 steps needed);
* ``zero_delta``  — the truth is an isomer of a parent (no transform changes formula);
* ``unreachable`` — no combination of our transform deltas gives the formula;
* ``no_parent``   — no spectral analog above the similarity floor.

The top missing deltas (with example truth/parent pairs) say which edits to add.

Example:
    python scripts/diagnose_derivation.py --train train.parquet --split split_v3.npz \\
        --pool pool_v2.npz --limit 400 --out diag_derivation.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from itertools import combinations_with_replacement

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from casmi.candidates.pool import CandidatePool
from casmi.channels.analog import AnalogIndex, find_analogs
from casmi.channels.derivation import (
    TRANSFORMS,
    DerivationConfig,
    derive_from_parent,
    formula_delta_counts,
)
from casmi.chem import inchikey14
from casmi.config import CFG
from casmi.data.loaders import load_spectral_library, query_molecules_from_table

_ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*)")
_ORDER = ("C", "H", "N", "O", "P", "S")


def formula_counts(formula: str, sign: int = 1) -> Counter:
    """``C6H10O5`` -> Counter; ``sign`` -1 negates (for ``-C1H2`` style deltas)."""
    out: Counter = Counter()
    for element, number in _ELEMENT.findall(formula):
        out[element] += sign * (int(number) if number else 1)
    return out


def delta_counts(delta: str) -> Counter:
    return Counter(formula_delta_counts(delta))


def vector(counts: Counter) -> tuple[tuple[str, int], ...]:
    items = {k: v for k, v in counts.items() if v}
    order = sorted(items, key=lambda e: (_ORDER.index(e) if e in _ORDER else 99, e))
    return tuple((e, items[e]) for e in order)


def label(vec: tuple[tuple[str, int], ...]) -> str:
    return "".join(f"{e}{n:+d}" for e, n in vec) or "0"


def add(*vecs):
    total: Counter = Counter()
    for v in vecs:
        total.update(dict(v))
    return vector(total)


def smiles_counts(smiles: str) -> Counter | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return formula_counts(CalcMolFormula(mol).replace("+", "").replace("-", ""))


def site_signature(small: Chem.Mol, large: Chem.Mol) -> str | None:
    """If ``small`` is a substructure of ``large``: the extra fragment of ``large`` and the
    atom types it attaches to (``a`` aromatic / ``A`` aliphatic, symbol, H count)."""
    match = large.GetSubstructMatch(small)
    if not match:
        return None
    matched = set(match)
    extra = [a.GetIdx() for a in large.GetAtoms() if a.GetIdx() not in matched]
    if not extra:
        return None
    attach = {
        f"{'a' if nb.GetIsAromatic() else 'A'}{nb.GetSymbol()}H{nb.GetTotalNumHs()}"
        for idx in extra
        for nb in large.GetAtomWithIdx(idx).GetNeighbors()
        if nb.GetIdx() in matched
    }
    fragment = Chem.MolFragmentToSmiles(large, atomsToUse=extra)
    return f"{fragment} @ {','.join(sorted(attach))}"


def edit_signature(parent_smiles: str, truth_smiles: str, vec) -> str:
    """``+frag @ site`` for a pure addition, ``-frag @ site`` for a pure removal that is a
    substructure relation; ``rearranged`` when the formula fits but the scaffold differs."""
    parent, truth = Chem.MolFromSmiles(parent_smiles), Chem.MolFromSmiles(truth_smiles)
    if parent is None or truth is None:
        return "unparseable"
    signs = {n > 0 for _, n in vec}
    if signs == {True}:
        sig = site_signature(parent, truth)
        return f"+{sig}" if sig else "rearranged"
    if signs == {False}:
        sig = site_signature(truth, parent)
        return f"-{sig}" if sig else "rearranged"
    return "mixed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=DerivationConfig.max_steps)
    parser.add_argument("--max-products-per-path", type=int,
                        default=DerivationConfig.max_products_per_path)
    parser.add_argument("--min-parent-similarity", type=float,
                        default=DerivationConfig.min_parent_similarity)
    parser.add_argument("--n-parents", type=int, default=DerivationConfig.n_parents)
    parser.add_argument("--only-keys", default=None,
                        help="text file of scorer keys: restrict to these class-3 molecules "
                             "(e.g. natural-product-flagged ones, the hidden test's chemistry)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    split = np.load(args.split, allow_pickle=False)
    train_mask, val_mask = split["train_mask"], split["val_mask"]
    novelty = dict(
        zip(split["holdout_keys"].tolist(), split["novelty_class"].tolist(), strict=True)
    )
    library = load_spectral_library(args.train, keep_rows=train_mask)
    index = AnalogIndex(library)
    pool = CandidatePool.load(args.pool)
    print(f"[OK] library {library.n_spectra:,} spectra, pool {len(pool):,}", flush=True)

    columns = ["inchikey14", "normalized_smiles", "adduct", "ionization_mode", "instrument_type",
               "precursor_mz", "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"]
    available = set(pq.read_schema(args.train).names)
    table = pq.read_table(args.train, columns=[c for c in columns if c in available])
    frame = table.filter(pa.array(val_mask)).to_pandas()
    frame["molecule_id"] = np.asarray(split["row_key"], dtype=str)[val_mask]
    frame = frame[frame["molecule_id"].map(lambda m: novelty.get(m) == 3)]
    if args.only_keys:
        only = {line.strip() for line in open(args.only_keys, encoding="utf-8") if line.strip()}
        frame = frame[frame["molecule_id"].isin(only)]
        print(f"[INFO] restricted to {frame['molecule_id'].nunique()} class-3 molecules "
              f"in {args.only_keys}", flush=True)
    ids = sorted(frame["molecule_id"].unique())
    rng = np.random.default_rng(args.seed)
    keep = set(rng.choice(ids, size=min(args.limit, len(ids)), replace=False).tolist())
    frame = frame[frame["molecule_id"].isin(keep)]
    truth = frame.drop_duplicates("molecule_id").set_index("molecule_id")["normalized_smiles"]
    molecules = query_molecules_from_table(pa.Table.from_pandas(frame, preserve_index=False))
    print(f"[INFO] {len(molecules)} class-3 molecules", flush=True)

    singles = {vector(delta_counts(d)): name for name, _, d in TRANSFORMS}
    steps: dict[tuple, tuple[int, str]] = {}
    for n in (1, 2, 3):
        for combo in combinations_with_replacement(sorted(singles), n):
            vec = add(*combo)
            steps.setdefault(vec, (n, "+".join(singles[c] for c in combo)))
    cfg = DerivationConfig(
        max_steps=args.max_steps, max_products_per_path=args.max_products_per_path,
        min_parent_similarity=args.min_parent_similarity, n_parents=args.n_parents,
    )
    print(f"[INFO] {cfg}", flush=True)
    categories: Counter = Counter()
    missing: Counter = Counter()
    examples: dict[str, list] = {}
    not_generated: Counter = Counter()
    sites: Counter = Counter()
    for i, molecule in enumerate(molecules):
        if i and i % 50 == 0:
            print(f"  {i}/{len(molecules)}  {dict(categories)}", flush=True)
        truth_smiles = truth[molecule.molecule_id]
        truth_key = inchikey14(truth_smiles)
        tc = smiles_counts(truth_smiles)
        analogs = find_analogs(molecule, index, CFG.spectrum, CFG.analog)
        parents = []
        for analog in analogs:
            row = pool.index_of(analog.inchikey14)
            if row >= 0 and analog.similarity >= cfg.min_parent_similarity:
                parents.append((pool.smiles[row], float(analog.similarity)))
        parents = sorted(parents, key=lambda p: -p[1])[: cfg.n_parents]
        if not parents or tc is None:
            categories["no_parent"] += 1
            continue
        best = ("unreachable", None, None)
        rank = {"generated": 0, "reachable_1": 1, "reachable_2": 2, "reachable_3": 3,
                "zero_delta": 4, "unreachable": 5}
        for parent_smiles, similarity in parents:
            pc = smiles_counts(parent_smiles)
            if pc is None:
                continue
            diff: Counter = Counter(tc)
            diff.subtract(pc)
            vec = vector(diff)
            if not vec:
                cat, path = "zero_delta", None
            elif vec in steps:
                n, path = steps[vec]
                cat = f"reachable_{n}"
                if n <= cfg.max_steps:
                    products = derive_from_parent(parent_smiles, molecule.target_mass, cfg)
                    if any(inchikey14(s) == truth_key for s in products):
                        cat = "generated"
            else:
                cat, path = "unreachable", None
            if rank[cat] < rank[best[0]]:
                best = (cat, vec, (parent_smiles, similarity, path))
            if cat == "generated":
                break
        cat, vec, info = best
        if info is None:  # every parent failed to parse
            categories["no_parent"] += 1
            continue
        categories[cat] += 1
        if cat == "unreachable":
            missing[label(vec)] += 1
            examples.setdefault(label(vec), []).append(
                dict(truth=truth_smiles, parent=info[0], similarity=round(info[1], 3)))
        elif cat.startswith("reachable"):
            not_generated[f"{label(vec)} via {info[2]}"] += 1
            examples.setdefault(f"{label(vec)} via {info[2]}", []).append(
                dict(truth=truth_smiles, parent=info[0], similarity=round(info[1], 3)))
            if cat == "reachable_1":
                sig = edit_signature(info[0], truth_smiles, vec)
                sites[sig] += 1
                examples.setdefault(f"site {sig}", []).append(
                    dict(truth=truth_smiles, parent=info[0], similarity=round(info[1], 3)))

    total = sum(categories.values())
    print(f"\n=== Channel 8 miss diagnosis on {total} class-3 molecules ===")
    for cat, n in categories.most_common():
        print(f"{cat:<14}{n:>6}  {n / total:6.1%}")
    print("\nTop missing formula deltas (unreachable with our transforms):")
    for key, n in missing.most_common(25):
        ex = examples[key][0]
        print(f"{n:>4}  {key:<22} e.g. parent {ex['parent'][:50]} -> truth {ex['truth'][:50]}")
    print("\nReachable by formula but not generated (site/caps/steps):")
    for key, n in not_generated.most_common(15):
        print(f"{n:>4}  {key}")
    print("\nOne-step misses by edit site (what a single-site SMARTS would have to do):")
    for key, n in sites.most_common(30):
        print(f"{n:>4}  {key}")
    json.dump({"categories": dict(categories), "missing": missing.most_common(),
               "not_generated": not_generated.most_common(), "sites": sites.most_common(),
               "examples": {k: v[:5] for k, v in examples.items()}},
              open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
