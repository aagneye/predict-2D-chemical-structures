"""Baseline pipeline: query molecules to ranked top-25 SMILES.

Assembles Channel 1 (library search), Channel 2 (analog propagation), and
optionally Channel 5 (MetFrag-lite fragmentation, :mod:`casmi.channels.
fragmentation`) and Channel 4 (FPNet spectrum->fingerprint scoring,
:mod:`casmi.models.fpnet`) over a mass-filtered candidate pool, then fuses
them — by default with transparent weighted fusion
(:mod:`casmi.channels.fusion`), or with a learned reranker
(:mod:`casmi.channels.ranker`) when one is supplied.

Channels 4 and 5 are both optional and additive: omitting an ``fpnet_model``
or setting ``fragmentation_channel=False`` reproduces the original Channel
1+2 baseline exactly, so this module still serves as the honest floor
described in ``docs/06-implementation-plan.md`` when run with defaults, while
also being the place later additions plug into.

The diagnostics emitted per molecule are as important as the predictions. The
public-LB-leading solution's own output showed ``best_library_sim == 1.0`` for
100% of visible test molecules with 0% of molecules carried by its other
channels, which is how we know its score is leakage-driven rather than
modelling-driven. :class:`MoleculeDiagnostics` records the same quantities so
the equivalent check is always one glance away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from casmi.candidates.pool import CandidatePool
from casmi.channels.analog import AnalogIndex, find_analogs, propagate_to_candidates
from casmi.channels.fragmentation import FragmentationConfig, fragmentation_scores
from casmi.channels.fusion import (
    DEFAULT_WEIGHTS,
    ScoredCandidate,
    fuse,
    mass_error_penalty,
)
from casmi.channels.library import library_search
from casmi.config import CFG, Config
from casmi.data.loaders import QueryMolecule, SpectralLibrary

if TYPE_CHECKING:
    from casmi.channels.ranker import Reranker


@dataclass
class MoleculeDiagnostics:
    """Per-molecule evidence summary.

    ``best_library_similarity`` near 1.0 across an entire cohort is the
    signature of test-set leakage, not of a strong pipeline — see the module
    docstring.
    """

    molecule_id: str
    target_mass: float
    n_spectra: int
    n_candidates: int
    best_library_similarity: float
    best_analog_similarity: float
    n_analogs: int
    top_score: float
    #: Which channel supplied the top-ranked candidate.
    top_source: str
    #: Best MetFrag-lite explain ratio across candidates (0.0 if the channel
    #: was not run).
    best_fragmentation_score: float = 0.0
    #: Best FPNet dot-product score across candidates (0.0 if no model was
    #: supplied).
    best_fpnet_score: float = 0.0
    #: Whether a learned reranker (rather than weighted fusion) produced the
    #: final ranking.
    used_reranker: bool = False


@dataclass
class PipelineResult:
    """Predictions plus diagnostics for one run."""

    predictions: dict[str, list[str]] = field(default_factory=dict)
    diagnostics: list[MoleculeDiagnostics] = field(default_factory=list)

    def channel_contribution(self, threshold: float = 0.85) -> dict[str, float]:
        """Fraction of molecules carried by each channel.

        Args:
            threshold: Library similarity above which a molecule counts as
                library-identified.

        Returns:
            Fractions keyed by ``library`` and ``analog_or_other``. If
            ``library`` approaches 1.0 on a supposedly held-out cohort, the
            split is leaking and the run's score is meaningless.
        """
        if not self.diagnostics:
            return {"library": 0.0, "analog_or_other": 0.0}
        strong = sum(1 for d in self.diagnostics if d.best_library_similarity > threshold)
        total = len(self.diagnostics)
        return {
            "library": strong / total,
            "analog_or_other": (total - strong) / total,
        }


def predict_molecule(
    molecule: QueryMolecule,
    library: SpectralLibrary,
    pool: CandidatePool,
    analog_index: AnalogIndex,
    config: Config | None = None,
    weights: dict[str, float] | None = None,
    fragmentation_channel: bool = False,
    fpnet_model: Any | None = None,
    fpnet_config: Any | None = None,
    fpnet_device: str = "cpu",
    reranker: Reranker | None = None,
) -> tuple[list[str], MoleculeDiagnostics]:
    """Produce a ranked SMILES list and diagnostics for one molecule.

    Args:
        fragmentation_channel: Enable Channel 5 (MetFrag-lite). Off by default
            since it is the most expensive per-candidate channel (one RDKit
            fragment enumeration per candidate SMILES).
        fpnet_model: A loaded :class:`casmi.models.fpnet.FPNet` in eval mode,
            or ``None`` to skip Channel 4 entirely (the default — this keeps
            the baseline import-clean of torch, matching the rest of the
            codebase's "torch is optional" pattern).
        fpnet_config: The model's :class:`casmi.models.fpnet.FPNetConfig`,
            required alongside ``fpnet_model``.
        fpnet_device: Device the model lives on.
        reranker: A fitted :class:`casmi.channels.ranker.Reranker`. When
            supplied, final ranking uses its predicted probabilities instead
            of :func:`casmi.channels.fusion.fuse`'s weighted sum; the fused
            score is still computed first so every candidate keeps a
            deterministic tiebreaker and diagnostics stay comparable across
            runs with and without a reranker.
    """
    cfg = config or CFG
    target = molecule.target_mass

    library_hits = library_search(
        molecule,
        library,
        spectrum_config=cfg.spectrum,
        candidate_config=cfg.candidates,
    )
    analogs = find_analogs(
        molecule, analog_index, spectrum_config=cfg.spectrum, analog_config=cfg.analog
    )

    candidate_indices = pool.window(target, cfg.candidates.ppm_window)
    if candidate_indices.size == 0:
        candidate_indices = pool.window(target, cfg.candidates.ppm_fallback)

    # Prune to a scoreable number before the expensive fingerprint work, using
    # cheap signals only: library similarity dominates, mass proximity breaks ties.
    if candidate_indices.size > cfg.candidates.max_candidates:
        coarse = np.array(
            [library_hits[pool.keys[i]].similarity if pool.keys[i] in library_hits else 0.0
             for i in candidate_indices],
            dtype=np.float64,
        )
        coarse = coarse * 100.0 - np.abs(pool.mass[candidate_indices] - target)
        keep = np.argsort(-coarse)[: cfg.candidates.max_candidates]
        candidate_indices = candidate_indices[np.sort(keep)]

    analog_features = propagate_to_candidates(
        candidate_indices, pool, analogs, analog_config=cfg.analog
    )
    mass_feature = mass_error_penalty(
        pool.mass[candidate_indices], target, cfg.candidates.ppm_window
    )

    # Channel 5: MetFrag-lite fragmentation plausibility. Scored once per
    # candidate over the molecule's spectra, before the per-candidate loop
    # below so it can be looked up by position like the other array features.
    fragmentation_feature = np.zeros(candidate_indices.size, dtype=np.float32)
    if fragmentation_channel and candidate_indices.size:
        fragmentation_feature = fragmentation_scores(
            [pool.smiles[i] for i in candidate_indices],
            molecule.spectra,
            positive_mode=molecule.is_positive,
            config=FragmentationConfig(),
        )

    # Channel 4: FPNet spectrum -> fingerprint dot-product score. One forward
    # pass per molecule (not per candidate), then a single matmul against the
    # candidate fingerprints.
    fpnet_feature = np.zeros(candidate_indices.size, dtype=np.float32)
    fpnet_norm_feature = np.zeros(candidate_indices.size, dtype=np.float32)
    if fpnet_model is not None and candidate_indices.size:
        from casmi.models.fpnet import normalised_score, score_candidates
        from casmi.models.train import predict_logits

        logits = predict_logits(
            fpnet_model,
            spectra=molecule.spectra,
            precursor_mz=molecule.precursor_mz,
            adducts=molecule.adducts,
            instruments=molecule.instrument_types,
            collision_energies=molecule.collision_energies,
            ionization_modes=molecule.ionization_modes,
            model_config=fpnet_config,
            device=fpnet_device,
        )
        candidate_fp = pool.fingerprints(candidate_indices)
        fpnet_feature = score_candidates(candidate_fp, logits).astype(np.float32)
        fpnet_norm_feature = normalised_score(candidate_fp, logits).astype(np.float32)

    candidates: list[ScoredCandidate] = []
    for position, index in enumerate(candidate_indices):
        key = pool.keys[index]
        hit = library_hits.get(key)
        candidates.append(
            ScoredCandidate(
                inchikey14=key,
                smiles=pool.smiles[index],
                score=0.0,
                features={
                    "library_similarity": hit.similarity if hit else 0.0,
                    "analog_power": float(analog_features["analog_power"][position]),
                    "analog_linear": float(analog_features["analog_linear"][position]),
                    "analog_best_tanimoto": float(
                        analog_features["analog_best_tanimoto"][position]
                    ),
                    "analog_top_tanimoto": float(
                        analog_features["analog_top_tanimoto"][position]
                    ),
                    "analog_mean": float(analog_features["analog_mean"][position]),
                    "mass_error_penalty": float(mass_feature[position]),
                    "mass_error_ppm": (
                        abs(float(pool.mass[index]) - target) / target * 1e6
                        if np.isfinite(target) and target > 0
                        else float("nan")
                    ),
                    "fragmentation_score": float(fragmentation_feature[position]),
                    "fpnet_score": float(fpnet_feature[position]),
                    "fpnet_normalised_score": float(fpnet_norm_feature[position]),
                },
            )
        )

    # Library hits outside the pool still deserve a slot: a structure with a
    # matching reference spectrum is strong evidence even if the pool lacks it.
    pooled_keys = {pool.keys[i] for i in candidate_indices}
    for key, hit in library_hits.items():
        if key not in pooled_keys and hit.smiles:
            candidates.append(
                ScoredCandidate(
                    inchikey14=key,
                    smiles=hit.smiles,
                    score=0.0,
                    features={"library_similarity": hit.similarity},
                )
            )

    fused_weights = weights or DEFAULT_WEIGHTS
    ranked = fuse(candidates, weights=fused_weights, top_n=cfg.top_n)

    used_reranker = False
    if reranker is not None and ranked:
        from casmi.channels.ranker import assemble_features

        feature_dicts = [c.features for c in ranked]
        fpnet_scores = np.array([d.get("fpnet_score", 0.0) for d in feature_dicts])
        fpnet_norm = np.array([d.get("fpnet_normalised_score", 0.0) for d in feature_dicts])
        matrix = assemble_features(feature_dicts, fpnet_scores, fpnet_norm)
        probabilities = reranker.predict_proba(matrix)
        order = np.argsort(-probabilities)
        ranked = [ranked[i] for i in order]
        used_reranker = True

    best_library = max((h.similarity for h in library_hits.values()), default=0.0)
    best_analog = analogs[0].similarity if analogs else 0.0
    if ranked:
        top = ranked[0]
        top_source = (
            "library"
            if top.features.get("library_similarity", 0.0) > 0.0
            else ("analog" if top.features.get("analog_power", 0.0) > 0.0 else "mass_only")
        )
    else:
        top_source = "none"

    diagnostics = MoleculeDiagnostics(
        molecule_id=molecule.molecule_id,
        target_mass=target,
        n_spectra=molecule.n_spectra,
        n_candidates=len(ranked),
        best_library_similarity=float(best_library),
        best_analog_similarity=float(best_analog),
        n_analogs=len(analogs),
        top_score=ranked[0].score if ranked else 0.0,
        top_source=top_source,
        best_fragmentation_score=(
            float(fragmentation_feature.max()) if fragmentation_feature.size else 0.0
        ),
        best_fpnet_score=float(fpnet_feature.max()) if fpnet_feature.size else 0.0,
        used_reranker=used_reranker,
    )
    return [c.smiles for c in ranked if c.smiles], diagnostics


def run_pipeline(
    molecules: list[QueryMolecule],
    library: SpectralLibrary,
    pool: CandidatePool,
    config: Config | None = None,
    weights: dict[str, float] | None = None,
    progress_every: int = 0,
    fragmentation_channel: bool = False,
    fpnet_model: Any | None = None,
    fpnet_config: Any | None = None,
    fpnet_device: str = "cpu",
    reranker: Reranker | None = None,
) -> PipelineResult:
    """Run the baseline over many molecules.

    The analog index is built once and shared, since deriving representative
    spectra per query would dominate runtime. See :func:`predict_molecule` for
    the meaning of the optional-channel and reranker arguments.
    """
    cfg = config or CFG
    analog_index = AnalogIndex(library)
    result = PipelineResult()

    for i, molecule in enumerate(molecules):
        if progress_every and i and i % progress_every == 0:
            print(f"  {i}/{len(molecules)} molecules", flush=True)
        smiles, diagnostics = predict_molecule(
            molecule,
            library,
            pool,
            analog_index,
            config=cfg,
            weights=weights,
            fragmentation_channel=fragmentation_channel,
            fpnet_model=fpnet_model,
            fpnet_config=fpnet_config,
            fpnet_device=fpnet_device,
            reranker=reranker,
        )
        result.predictions[molecule.molecule_id] = smiles
        result.diagnostics.append(diagnostics)

    return result


def write_submission(
    predictions: dict[str, list[str]],
    molecule_ids: list[str],
    path: str,
    top_n: int | None = None,
    filler: str = "CCO",
) -> None:
    """Write a competition-format submission CSV.

    Every row is padded to exactly ``top_n`` semicolon-separated SMILES.
    Padding with a filler rather than emitting short rows keeps the format
    strictly valid; a wrong guess and a missing guess both score zero, so
    padding costs nothing.
    """
    limit = CFG.top_n if top_n is None else top_n
    lines = ["molecule_id,smiles"]
    for molecule_id in molecule_ids:
        smiles = [s for s in predictions.get(molecule_id, []) if s][:limit]
        if len(smiles) < limit:
            smiles = smiles + [filler] * (limit - len(smiles))
        lines.append(f"{molecule_id},{';'.join(smiles)}")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
