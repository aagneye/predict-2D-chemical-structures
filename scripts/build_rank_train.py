"""Build the reranker's training set from the held-out split.

For each held-out molecule, runs Channels 1/2 (and optionally 4/5) exactly as
the pipeline does, then labels every candidate 1 if its ``inchikey14``
matches the molecule's true structure and 0 otherwise. This is the standard
pointwise learning-to-rank framing :func:`casmi.channels.ranker.train_ranker`
expects.

Must be run against the split's *train*-side library only (never the full
unfiltered ``train.parquet``), for the same leakage reason ``run_baseline.py``
and ``train_fpnet.py`` both enforce it.

Example:
    python scripts/build_rank_train.py --train data/raw/train.parquet \\
        --split data/processed/split.npz --pool data/processed/pool.npz \\
        --out data/processed/rank_train.npz \\
        --fragmentation --fpnet-checkpoint checkpoints/fpnet/fpnet_final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from casmi.candidates.pool import CandidatePool
from casmi.channels.analog import AnalogIndex
from casmi.channels.ranker import RankTrainingExample, build_training_matrix
from casmi.chem import inchikey14
from casmi.data.loaders import load_spectral_library, query_molecules_from_table
from casmi.pipeline import build_scored_candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--split", required=True, help="output of scripts/build_split.py")
    parser.add_argument("--pool", required=True, help="output of scripts/build_pool.py")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--limit", type=int, default=None, help="cap molecules, for smoke tests")
    parser.add_argument(
        "--realistic-pool",
        default=None,
        help="external_keys.txt from build_external_keys.py: held-out class-2 structures stay "
        "in the pool only if an external database has them (test-like class 2)",
    )
    parser.add_argument(
        "--fragmentation", action="store_true", help="enable Channel 5 features"
    )
    parser.add_argument(
        "--fpnet-checkpoint", default=None, help="path to a trained FPNet checkpoint"
    )
    parser.add_argument("--fpnet-device", default="cpu")
    parser.add_argument(
        "--denovo-checkpoint", default=None, help="path to a trained de novo decoder checkpoint"
    )
    parser.add_argument("--denovo-device", default="cpu")
    parser.add_argument(
        "--denovo-beam-width", type=int, default=8, help="beam search width for de novo generation"
    )
    parser.add_argument(
        "--progress-every", type=int, default=25, help="print progress every N molecules"
    )
    parser.add_argument(
        "--popularity",
        default=None,
        help="PubChem popularity prior (.npz or pool_popularity.csv); adds the popularity feature",
    )
    parser.add_argument(
        "--pubchem-tier",
        default=None,
        help="directory with pc_mass.npy/pc_off.npy/pc_smiles.npy (Channel 7, needs FPNet)",
    )
    parser.add_argument(
        "--pubchem-popularity",
        default=None,
        help="directory with pc_lsid.npy/pc_lpmid.npy aligned with the tier",
    )
    parser.add_argument("--pubchem-top-k", type=int, default=25)
    parser.add_argument(
        "--derivation",
        action="store_true",
        help="enable Channel 8 (analog derivation of novel structures; needs FPNet)",
    )
    args = parser.parse_args()

    split_data = np.load(args.split, allow_pickle=False)
    train_mask = split_data["train_mask"]
    val_mask = split_data["val_mask"]
    excluded = {str(k) for k in split_data["excluded_from_pool"]}
    holdout_keys = [str(k) for k in split_data["holdout_keys"]]
    novelty = dict(
        zip(holdout_keys, [int(c) for c in split_data["novelty_class"]], strict=True)
    )

    print(f"[INFO] split: {train_mask.sum():,} train rows, {val_mask.sum():,} held-out rows")
    library = load_spectral_library(args.train, keep_rows=train_mask)
    print(f"[OK] library: {library.n_spectra:,} spectra")

    pool = CandidatePool.load(args.pool)
    pool = pool.exclude_keys(excluded)
    print(f"[OK] pool: {len(pool):,} structures (class-3 excluded)")

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
    if "row_key" in split_data.files:
        # The split was built on scorer keys: group queries by the same identity.
        frame["molecule_id"] = np.asarray(split_data["row_key"], dtype=str)[val_mask]
    else:
        frame["molecule_id"] = frame["inchikey14"]

    truth = (
        frame.drop_duplicates("molecule_id")
        .set_index("molecule_id")["normalized_smiles"]
        .to_dict()
    )
    # Same scorer-key exclusion as run_baseline.py: the pool is keyed by the
    # recomputed scorer key, the split by the stored column.
    scorer_excluded = {
        k for m, smi in truth.items() if novelty.get(m) == 3 and (k := inchikey14(smi))
    }
    pool = pool.exclude_keys(scorer_excluded - excluded)
    print(f"[OK] pool after scorer-key class-3 exclusion: {len(pool):,} structures")
    if args.realistic_pool:
        from casmi.eval.realism import load_keys, realistic_pool_exclusions

        external = load_keys(args.realistic_pool)
        truth_keys = {m: (m, inchikey14(smi)) for m, smi in truth.items()}
        drop = realistic_pool_exclusions(novelty, truth_keys, external)
        before = len(pool)
        pool = pool.exclude_keys(drop)
        scorer_excluded = scorer_excluded | {
            k for m, keys in truth_keys.items() if novelty.get(m) == 3 for k in keys if k
        }
        print(f"[OK] realistic pool: removed {before - len(pool):,} held-out structures "
              f"not in an external database ({len(external):,} external keys)")
    if args.limit:
        keep_ids = list(truth)[: args.limit]
        frame = frame[frame["molecule_id"].isin(keep_ids)]
        truth = {k: truth[k] for k in keep_ids}

    molecules = query_molecules_from_table(pa.Table.from_pandas(frame, preserve_index=False))
    print(f"[INFO] building training rows for {len(molecules):,} held-out molecules")

    fpnet_model = None
    fpnet_config = None
    if args.fpnet_checkpoint:
        from casmi.models.train import load_checkpoint

        fpnet_model, fpnet_config = load_checkpoint(args.fpnet_checkpoint, device=args.fpnet_device)
        print(f"[OK] loaded FPNet checkpoint: {args.fpnet_checkpoint}")

    denovo_model = None
    denovo_vocabulary = None
    denovo_config = None
    if args.denovo_checkpoint:
        from casmi.channels.denovo import DenovoChannelConfig
        from casmi.models.train_denovo import load_denovo_checkpoint

        denovo_model, _, denovo_vocabulary = load_denovo_checkpoint(
            args.denovo_checkpoint, device=args.denovo_device
        )
        denovo_config = DenovoChannelConfig(beam_width=args.denovo_beam_width)
        print(f"[OK] loaded de novo checkpoint: {args.denovo_checkpoint}")

    popularity = None
    if args.popularity:
        from casmi.priors import PopularityPrior

        popularity = PopularityPrior.load(args.popularity)
        print(f"[OK] loaded popularity prior: {len(popularity):,} structures")

    derivation_config = None
    if args.derivation:
        from casmi.channels.derivation import DerivationConfig

        derivation_config = DerivationConfig()
        print("[OK] Channel 8 (analog derivation) enabled")

    pubchem_tier = None
    pubchem_config = None
    if args.pubchem_tier:
        from casmi.channels.pubchem import PubChemConfig, PubChemTier

        pubchem_tier = PubChemTier(args.pubchem_tier, args.pubchem_popularity)
        pubchem_config = PubChemConfig(top_k=args.pubchem_top_k)
        print(
            f"[OK] PubChem tier: {len(pubchem_tier):,} structures, "
            f"popularity={'yes' if pubchem_tier.has_popularity else 'no'}"
        )

    analog_index = AnalogIndex(library)
    examples: list[RankTrainingExample] = []
    n_with_positive = 0

    for i, molecule in enumerate(molecules):
        if args.progress_every and i and i % args.progress_every == 0:
            print(f"  {i}/{len(molecules)} molecules", flush=True)

        true_smiles = truth.get(molecule.molecule_id)
        true_key = inchikey14(true_smiles) if true_smiles else None

        candidates, _ = build_scored_candidates(
            molecule,
            library,
            pool,
            analog_index,
            fragmentation_channel=args.fragmentation,
            fpnet_model=fpnet_model,
            fpnet_config=fpnet_config,
            fpnet_device=args.fpnet_device,
            denovo_model=denovo_model,
            denovo_vocabulary=denovo_vocabulary,
            denovo_config=denovo_config,
            denovo_device=args.denovo_device,
            popularity=popularity,
            pubchem_tier=pubchem_tier,
            pubchem_config=pubchem_config,
            # Only class-3 truths are absent from PubChem in the real test; class 2 may
            # legitimately be found there even when the pool no longer holds it.
            pubchem_blocklist={k for k, c in novelty.items() if c == 3} | scorer_excluded,
            derivation_config=derivation_config,
        )
        if not candidates:
            continue

        has_positive = any(c.inchikey14 == true_key for c in candidates)
        if has_positive:
            n_with_positive += 1

        for candidate in candidates:
            examples.append(
                RankTrainingExample(
                    molecule_id=molecule.molecule_id,
                    features=candidate.features,
                    label=int(candidate.inchikey14 == true_key),
                    fpnet_score=candidate.features.get("fpnet_score", 0.0),
                    fpnet_normalised_score=candidate.features.get("fpnet_normalised_score", 0.0),
                )
            )

    print(
        f"[INFO] {n_with_positive:,}/{len(molecules):,} molecules had their true "
        "structure among scored candidates (an upper bound on achievable recall)"
    )

    x, y, molecule_ids = build_training_matrix(examples)
    print(f"[OK] assembled {x.shape[0]:,} rows x {x.shape[1]} features, {y.sum():,} positive")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=x, y=y, molecule_id=np.asarray(molecule_ids, dtype=str))
    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
