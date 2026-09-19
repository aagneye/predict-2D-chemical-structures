"""Run the pipeline against the held-out split and report MRR@25 by class.

This produces *the* baseline number for the project, and — when the optional
channel flags are passed — the honest incremental MRR@25 each addition earns
on the same held-out split, which is what step 6-8 of
``docs/06-implementation-plan.md`` calls for. It reads the split's masks so
the library it searches physically excludes held-out structures, and it
removes the class-3 cohort from the candidate pool, so each cohort measures
what it claims to.

Also prints the channel-contribution diagnostic. If ``library`` approaches
1.00 here, the split is leaking and the MRR is meaningless — that is exactly
the pattern visible in the public-LB-leading notebook's own output.

Example:
    # Channel 1+2 baseline (original behaviour, no flags needed)
    python scripts/run_baseline.py --train data/raw/train.parquet \\
        --split data/processed/split.npz --pool data/processed/pool.npz

    # Full pipeline: + fragmentation, + FPNet, + learned reranker
    python scripts/run_baseline.py --train data/raw/train.parquet \\
        --split data/processed/split.npz --pool data/processed/pool.npz \\
        --fragmentation --fpnet-checkpoint checkpoints/fpnet/fpnet_final.pt \\
        --ranker checkpoints/ranker.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from casmi.candidates.pool import CandidatePool
from casmi.data.loaders import load_spectral_library, query_molecules_from_table
from casmi.eval.metrics import evaluate_predictions, format_summary, summarise_by_class
from casmi.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--split", required=True, help="output of scripts/build_split.py")
    parser.add_argument("--pool", required=True, help="output of scripts/build_pool.py")
    parser.add_argument("--out", default="runs/baseline")
    parser.add_argument("--limit", type=int, default=None, help="cap molecules, for smoke tests")
    parser.add_argument("--ppm", type=float, default=None, help="override candidate ppm window")
    parser.add_argument(
        "--fragmentation", action="store_true", help="enable Channel 5 (MetFrag-lite)"
    )
    parser.add_argument(
        "--fpnet-checkpoint", default=None, help="enable Channel 4 with this checkpoint"
    )
    parser.add_argument("--fpnet-device", default="cpu")
    parser.add_argument(
        "--ranker", default=None, help="path to a fitted Reranker (.pkl); replaces weighted fusion"
    )
    args = parser.parse_args()

    split_data = np.load(args.split, allow_pickle=False)
    train_mask = split_data["train_mask"]
    val_mask = split_data["val_mask"]
    holdout_keys = [str(k) for k in split_data["holdout_keys"]]
    novelty = dict(
        zip(holdout_keys, [int(c) for c in split_data["novelty_class"]], strict=True)
    )
    excluded = {str(k) for k in split_data["excluded_from_pool"]}
    print(f"[INFO] split: {train_mask.sum():,} train rows, {val_mask.sum():,} held-out rows")

    # The library contains only the train side. This is enforced at load time
    # rather than by filtering later, so a forgotten filter cannot leak.
    print("[INFO] loading library (train side only)")
    library = load_spectral_library(args.train, keep_rows=train_mask)
    print(f"[OK] library: {library.n_spectra:,} spectra")

    pool = CandidatePool.load(args.pool)
    print(f"[OK] pool: {len(pool):,} structures")
    # Class 3 must be unreachable by retrieval, or its metric is fiction.
    pool = pool.exclude_keys(excluded)
    print(f"[OK] pool after removing {len(excluded):,} class-3 structures: {len(pool):,}")

    # Build queries from the held-out rows, one synthetic molecule per structure.
    columns = [
        "inchikey14",
        "normalized_smiles",
        "adduct",
        "ionization_mode",
        "instrument_type",
        "precursor_mz",
        "collision_energy_ev",
        "ms2_mzs",
        "ms2_normalized_intensities",
    ]
    available = set(pq.read_schema(args.train).names)
    table = pq.read_table(args.train, columns=[c for c in columns if c in available])
    table = table.filter(pa.array(val_mask))
    frame = table.to_pandas()
    frame["molecule_id"] = frame["inchikey14"]

    truth = (
        frame.drop_duplicates("molecule_id").set_index("molecule_id")["normalized_smiles"].to_dict()
    )
    if args.limit:
        keep_ids = list(truth)[: args.limit]
        frame = frame[frame["molecule_id"].isin(keep_ids)]
        truth = {k: truth[k] for k in keep_ids}

    molecules = query_molecules_from_table(pa.Table.from_pandas(frame, preserve_index=False))
    print(f"[INFO] evaluating {len(molecules):,} held-out molecules")

    config = None
    if args.ppm is not None:
        from dataclasses import replace

        from casmi.config import CFG

        config = replace(CFG, candidates=replace(CFG.candidates, ppm_window=args.ppm))

    fpnet_model = None
    fpnet_config = None
    if args.fpnet_checkpoint:
        from casmi.models.train import load_checkpoint

        fpnet_model, fpnet_config = load_checkpoint(args.fpnet_checkpoint, device=args.fpnet_device)
        print(f"[OK] loaded FPNet checkpoint: {args.fpnet_checkpoint}")

    reranker = None
    if args.ranker:
        from casmi.channels.ranker import Reranker

        reranker = Reranker.load(args.ranker)
        print(f"[OK] loaded reranker: {args.ranker} ({len(reranker.models)} models)")

    result = run_pipeline(
        molecules,
        library,
        pool,
        config=config,
        progress_every=25,
        fragmentation_channel=args.fragmentation,
        fpnet_model=fpnet_model,
        fpnet_config=fpnet_config,
        fpnet_device=args.fpnet_device,
        reranker=reranker,
    )

    scores = evaluate_predictions(
        result.predictions,
        truth,
        novelty_classes={m: novelty.get(m) for m in truth if novelty.get(m)},
    )
    summary = summarise_by_class(scores)

    active_channels = ["library", "analog"]
    if args.fragmentation:
        active_channels.append("fragmentation")
    if args.fpnet_checkpoint:
        active_channels.append("fpnet")
    if args.ranker:
        active_channels.append("gbm_reranker")
    print(f"\n=== Active channels: {', '.join(active_channels)} ===")
    print("=== Baseline MRR@25 on held-out split ===")
    print(format_summary(summary))

    contribution = result.channel_contribution()
    print("\n=== Channel contribution ===")
    for name, value in contribution.items():
        print(f"  {name:<18} {value:>7.1%}")
    if contribution["library"] > 0.95:
        print(
            "\n[WARNING] library search carries >95% of molecules. On a held-out\n"
            "          split this indicates LEAKAGE, not a strong pipeline. Check\n"
            "          the split before trusting the MRR above."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "active_channels": active_channels,
                "summary": summary,
                "channel_contribution": contribution,
                "n_molecules": len(molecules),
            },
            handle,
            indent=2,
        )
    with open(out_dir / "diagnostics.csv", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "molecule_id,target_mass,n_spectra,n_candidates,"
            "best_library_sim,best_analog_sim,n_analogs,top_score,top_source,"
            "best_fragmentation_score,best_fpnet_score,used_reranker\n"
        )
        for d in result.diagnostics:
            handle.write(
                f"{d.molecule_id},{d.target_mass:.5f},{d.n_spectra},{d.n_candidates},"
                f"{d.best_library_similarity:.4f},{d.best_analog_similarity:.4f},"
                f"{d.n_analogs},{d.top_score:.4f},{d.top_source},"
                f"{d.best_fragmentation_score:.4f},{d.best_fpnet_score:.4f},{d.used_reranker}\n"
            )
    print(f"\n[OK] wrote {out_dir}/summary.json and diagnostics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
