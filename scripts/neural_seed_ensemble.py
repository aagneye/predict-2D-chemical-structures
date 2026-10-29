"""Seed ensemble of the public spectrum-candidate scorer (sanpier/casmi-spectrum-model-v1).

The 0.466 → 0.477 public gains all came from re-ranking the final 25 with this small
model (3-layer transformer, ≈ 5 GPU-minutes of training). A longer-trained replacement lost
0.019 on the board, so this keeps its exact recipe and length and only averages several
independently seeded copies: same bias, less variance.

The model code is the bundle's own (``--bundle`` is the dataset folder with ``train.py``,
``prepare.py``, ``model.py``, ``features.py``, ``config.json``); nothing here changes it.

    prepare  — build the training cache once (holdout split fixed by the config seed)
    train    — one seed: holdout stage (best_validation_model.pt) + full refit (final_model.pt)
    eval     — diagnostic MRR@25 of each holdout model and of their average, on the same
               held-out molecules and candidates (random molecular holdout: relative only)

Example:
    python scripts/neural_seed_ensemble.py prepare --bundle B --work W --train train.parquet
    python scripts/neural_seed_ensemble.py train --bundle B --work W --out runs/s1 --seed 1
    python scripts/neural_seed_ensemble.py eval --bundle B --work W --models runs/s1 runs/s2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def config(args: argparse.Namespace, **overrides) -> dict:
    sys.path.insert(0, str(Path(args.bundle).resolve()))
    cfg = json.loads((Path(args.bundle) / "config.json").read_text())
    cfg.update(work=str(Path(args.work).resolve()), **overrides)
    return cfg


def prepare(args: argparse.Namespace) -> None:
    cfg = config(args, train_path=str(Path(args.train).resolve()), external_paths=[])
    import prepare as bundle_prepare

    # The competition file names the column `normalized_smiles` (the bundle's data_audit.json
    # lists exactly these columns, so its author mapped it outside prepare.py).
    bundle_prepare.ALIASES["smiles"].append("normalized_smiles")
    bundle_prepare.prepare(cfg)


def train(args: argparse.Namespace) -> None:
    cfg = config(args, output=str(Path(args.out).resolve()), seed=args.seed)
    from train import run

    print(json.dumps(run(cfg), indent=2))


def evaluate(args: argparse.Namespace) -> None:
    import torch

    cfg = config(args)
    from model import SpectrumCandidateModel
    from train import Data, candidate_sets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data(cfg)
    candidates = candidate_sets(data, cfg)
    models = []
    for d in args.models:
        model = SpectrumCandidateModel(cfg).to(device)
        ck = torch.load(Path(d) / args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ck["state_dict"])
        models.append(model.eval())

    ranks = [[] for _ in range(len(models) + 1)]  # each model, then the average
    with torch.inference_mode():
        for i, ids in candidates.items():
            sr = data.by_mol[i]
            sr = sr[data.spec.source.values[sr] == 0]
            if not len(sr):
                continue
            fp = torch.tensor(np.asarray(data.fp[ids]), device=device, dtype=torch.float32)
            per_model = []
            for model in models:
                z, p = model.encode_spectra(*data.batch(sr, device))
                z, p = z.mean(0), p.mean(0)
                scores = ((model.encode_molecules(fp) @ z) / cfg["temperature"]
                          + cfg["fp_score_weight"] * (fp @ p) / cfg["fp_bits"])
                per_model.append(scores.cpu().numpy())
            for k, scores in enumerate(per_model + [np.mean(per_model, axis=0)]):
                # Pessimistic ties, as the bundle's own validation: gold is at index zero.
                ranks[k].append(1 + int(np.sum(scores[1:] >= scores[0])))

    report = {}
    for k, r in enumerate(ranks):
        r = np.asarray(r)
        name = args.models[k] if k < len(models) else f"average of {len(models)}"
        report[name] = dict(n=int(r.size), mrr25=float(np.mean(np.where(r <= 25, 1.0 / r, 0.0))),
                            top1=float(np.mean(r == 1)))
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--bundle", required=True)
        p.add_argument("--work", required=True)
        if name == "prepare":
            p.add_argument("--train", required=True)
        if name == "train":
            p.add_argument("--out", required=True)
            p.add_argument("--seed", type=int, required=True)
        if name == "eval":
            p.add_argument("--models", nargs="+", required=True)
            p.add_argument("--checkpoint", default="best_validation_model.pt")
            p.add_argument("--out")
    args = ap.parse_args()
    {"prepare": prepare, "train": train, "eval": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
