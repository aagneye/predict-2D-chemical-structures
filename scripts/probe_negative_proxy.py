"""Does a negative-mode spectrum carry evidence for positive-mode spectrum matching?

About 7% of test molecules have no [M+H]+ spectrum, only [M-H]- (or formate), so the
public pipeline's GLACIER simulator, which predicts [M+H]+ spectra, never scores them.
Idea (E2): shift every [M-H]- peak by two protons, so each fragment keeps its neutral loss
from the precursor, and compare this proxy with [M+H]+ spectra.

This probe measures the ceiling of that idea with measured spectra only (no model, no
leakage): for molecules measured as both [M+H]+ and [M-H]-, rank the same-formula
molecules by similarity of the query's proxy to their measured [M+H]+ spectra. Rows:

* ``proxy``   — [M-H]- shifted by +2 protons vs [M+H]+ (the E2 idea);
* ``raw_neg`` — [M-H]- unshifted vs [M+H]+ (control: should be near chance);
* ``pos2``    — a second [M+H]+ spectrum of the query vs [M+H]+ (ceiling of any proxy);
* ``chance_mrr`` — expected MRR of a random order.

Example:
    python scripts/probe_negative_proxy.py --train train.parquet --queries 3000 --out probe.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from casmi.spectra import clean_spectrum, entropy_similarity

PROTON = 1.007276
COLUMNS = ["inchikey14", "molecular_formula", "adduct", "precursor_mz", "ms2_mzs",
           "ms2_normalized_intensities"]


def load(path: str, adduct: str):
    table = pq.read_table(path, columns=COLUMNS, filters=[("adduct", "==", adduct)])
    table = table.filter(pc.greater_equal(pc.list_value_length(table["ms2_mzs"]), 3))
    return table.to_pydict()


def by_key(rows: dict) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, key in enumerate(rows["inchikey14"]):
        out.setdefault(key, []).append(i)
    return out


def spectrum(rows: dict, i: int, shift: float = 0.0):
    mz, p = clean_spectrum(rows["ms2_mzs"][i], rows["ms2_normalized_intensities"][i],
                           precursor_mz=rows["precursor_mz"][i])
    return (mz + np.float32(shift), p) if shift else (mz, p)


def rank_of_truth(scores: np.ndarray, truth: int) -> float:
    """1-based rank of ``scores[truth]``; ties count half (the expected rank of a random
    tie-break)."""
    better = float((scores > scores[truth]).sum())
    ties = float((scores == scores[truth]).sum()) - 1.0
    return 1.0 + better + ties / 2.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train", required=True)
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--max-candidates", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pos = load(args.train, "[M+H]+")
    neg = load(args.train, "[M-H]-")
    pos_by, neg_by = by_key(pos), by_key(neg)
    formula = {k: pos["molecular_formula"][ix[0]] for k, ix in pos_by.items()}
    isomers: dict[str, list[str]] = {}
    for key, f in formula.items():
        isomers.setdefault(f, []).append(key)
    queries = [k for k in neg_by if k in pos_by and len(isomers[formula[k]]) >= 2]
    rng.shuffle(queries)
    queries = queries[: args.queries]
    # One fixed [M+H]+ reference spectrum per molecule: what a simulator would stand in for.
    reference = {k: int(rng.choice(ix)) for k, ix in pos_by.items()}

    ranks: dict[str, list[float]] = {"proxy": [], "raw_neg": [], "pos2": []}
    n_cands = []
    for key in queries:
        others = [k for k in isomers[formula[key]] if k != key]
        if len(others) > args.max_candidates - 1:
            others = list(rng.choice(others, args.max_candidates - 1, replace=False))
        cands = [key] + others
        refs = [spectrum(pos, reference[k]) for k in cands]
        qi = int(rng.choice(neg_by[key]))
        variants = {"proxy": spectrum(neg, qi, 2 * PROTON), "raw_neg": spectrum(neg, qi)}
        second = [i for i in pos_by[key] if i != reference[key]]
        if second:
            variants["pos2"] = spectrum(pos, int(rng.choice(second)))
        for name, (qmz, qp) in variants.items():
            scores = np.array([entropy_similarity(qmz, qp, mz, p) for mz, p in refs])
            ranks[name].append(rank_of_truth(scores, 0))
        n_cands.append(len(cands))

    chance = [sum(1.0 / r for r in range(1, n + 1)) / n for n in n_cands]
    report = {"queries": len(queries), "mean_candidates": float(np.mean(n_cands)),
              "chance_mrr": float(np.mean(chance))}
    for name, r in ranks.items():
        r = np.asarray(r)
        report[name] = {"n": int(r.size), "mrr": float(np.mean(1.0 / r)),
                        "top1": float(np.mean(r <= 1.0)), "top5": float(np.mean(r <= 5.0))}
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
