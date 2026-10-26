"""Offline proxy for the class-3 overlay on ranker training rows.

Base = each molecule's non-derived candidates ranked by the legacy 29-feature ranker
(v8, no class-3 channel, like the public recipes). Overlay = Channel 8 structures
ranked by the 33-feature ranker and inserted with ``casmi.overlay.merge_overlay``
under several gates and positions. Reports MRR@25 per novelty class for every
variant and the public-score delta implied by the round-1 LB emulator
(``public ≈ 0.115·m1 + 0.306·m2 + α·s3·m3``; α·s3 = 0.08 fitted, 0.25 if class 3 is
half of the test).

Example:
    python scripts/simulate_overlay.py --data rank_train_v3.npz --split split_v3.npz \\
        --base-ranker ranker_v8.pkl --insert-ranker ranker_v9.pkl --out overlay_sim.json
"""

from __future__ import annotations

import argparse
import json
from itertools import product

import numpy as np

from casmi.channels.ranker import FEATURE_NAMES, Reranker
from casmi.overlay import OverlayConfig, merge_overlay, weak_evidence

WEIGHTS = {"fitted": (0.115, 0.306, 0.080), "class3_half": (0.115, 0.306, 0.250)}


def reciprocal_rank(labels: list[int]) -> float:
    for rank, label in enumerate(labels[:25], start=1):
        if label:
            return 1.0 / rank
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--base-ranker", required=True)
    parser.add_argument("--insert-ranker", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=False)
    x, y, mol = data["X"], data["y"], data["molecule_id"]
    split = np.load(args.split, allow_pickle=False)
    novelty = dict(
        zip(split["holdout_keys"].tolist(), split["novelty_class"].tolist(), strict=True)
    )
    base_scores = Reranker.load(args.base_ranker).predict_proba(x)
    insert_scores = Reranker.load(args.insert_ranker).predict_proba(x)
    col = {name: i for i, name in enumerate(FEATURE_NAMES)}

    order = np.argsort(mol, kind="stable")
    groups = np.split(order, np.flatnonzero(mol[order][1:] != mol[order][:-1]) + 1)
    molecules = []
    for idx in groups:
        derived = x[idx, col["derived"]] > 0.5
        base = idx[~derived][np.argsort(-base_scores[idx[~derived]], kind="stable")]
        inserts = idx[derived][np.argsort(-insert_scores[idx[derived]], kind="stable")]
        molecules.append({
            "class": int(novelty.get(str(mol[idx[0]]), 0)),
            "base": [str(i) for i in base],
            "inserts": [str(i) for i in inserts],
            "library_max": float(x[idx[0], col["library_similarity_max"]]),
            "top_prob": float(base_scores[base[0]]) if len(base) else 0.0,
        })
    labels = {str(i): int(y[i]) for i in range(len(y))}

    def per_class(lists: list[list[str]]) -> dict[str, float]:
        out = {}
        for c in (1, 2, 3):
            rows = [reciprocal_rank([labels[k] for k in lst])
                    for lst, m in zip(lists, molecules, strict=True) if m["class"] == c]
            out[f"class{c}"] = float(np.mean(rows)) if rows else 0.0
        return out

    gates = {
        "always": lambda m, cfg: True,
        "lib_max<0.8": lambda m, cfg: weak_evidence(m["library_max"], cfg),
        "lib_max<0.6": lambda m, cfg: weak_evidence(m["library_max"], cfg),
        "lib_max<0.4": lambda m, cfg: weak_evidence(m["library_max"], cfg),
        "top_prob<0.5": lambda m, cfg: m["top_prob"] < 0.5,
        "top_prob<0.3": lambda m, cfg: m["top_prob"] < 0.3,
    }
    thresholds = {"lib_max<0.8": 0.8, "lib_max<0.6": 0.6, "lib_max<0.4": 0.4}
    base_mrr = per_class([m["base"] for m in molecules])
    results = []
    for gate_name, insert_at, k in product(gates, (1, 2, 5), (1, 3, 5)):
        cfg = OverlayConfig(insert_at=insert_at, max_insert=k,
                            library_max_threshold=thresholds.get(gate_name, 0.6))
        lists, n_gated = [], 0
        for m in molecules:
            if gates[gate_name](m, cfg):
                n_gated += 1
                lists.append(merge_overlay(m["base"], m["inserts"], cfg))
            else:
                lists.append(m["base"])
        mrr = per_class(lists)
        delta = {name: sum(w * (mrr[f"class{c}"] - base_mrr[f"class{c}"])
                           for w, c in zip(ws, (1, 2, 3), strict=True))
                 for name, ws in WEIGHTS.items()}
        results.append({"gate": gate_name, "insert_at": insert_at, "max_insert": k,
                        "gated_fraction": n_gated / len(molecules), **mrr, "delta": delta})
    results.sort(key=lambda r: -r["delta"]["fitted"])

    print(f"molecules: {len(molecules)}  base MRR per class: "
          f"{base_mrr['class1']:.3f} / {base_mrr['class2']:.3f} / {base_mrr['class3']:.3f}")
    print(f"{'gate':<14}{'at':>4}{'k':>3}{'gated':>7}{'m1':>7}{'m2':>7}{'m3':>7}"
          f"{'Δfit':>8}{'Δhalf':>8}")
    for r in results[:15] + [r for r in results if r["gate"] == "always"][:3]:
        print(f"{r['gate']:<14}{r['insert_at']:>4}{r['max_insert']:>3}{r['gated_fraction']:>7.2f}"
              f"{r['class1']:>7.3f}{r['class2']:>7.3f}{r['class3']:>7.3f}"
              f"{r['delta']['fitted']:>+8.4f}{r['delta']['class3_half']:>+8.4f}")
    json.dump({"base": base_mrr, "variants": results}, open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
