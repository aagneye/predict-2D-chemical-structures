"""Baseline pipeline: query molecules to ranked top-25 SMILES.

Assembles Channel 1 (library search) and Channel 2 (analog propagation) over a
mass-filtered candidate pool, then fuses them. This is the baseline defined in
``docs/06-implementation-plan.md`` — deliberately without the neural channel or
a learned reranker, so that later additions have an honest floor to beat.

The diagnostics emitted per molecule are as important as the predictions. The
public-LB-leading solution's own output showed ``best_library_sim == 1.0`` for
100% of visible test molecules with 0% of molecules carried by its other
channels, which is how we know its score is leakage-driven rather than
modelling-driven. :class:`MoleculeDiagnostics` records the same quantities so
the equivalent check is always one glance away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from casmi.candidates.pool import CandidatePool
from casmi.channels.analog import AnalogIndex, find_analogs, propagate_to_candidates
from casmi.channels.fusion import (
    DEFAULT_WEIGHTS,
    ScoredCandidate,
    fuse,
    mass_error_penalty,
)
from casmi.channels.library import library_search
from casmi.config import CFG, Config
from casmi.data.loaders import QueryMolecule, SpectralLibrary


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
) -> tuple[list[str], MoleculeDiagnostics]:
    """Produce a ranked SMILES list and diagnostics for one molecule."""
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

    ranked = fuse(candidates, weights=weights or DEFAULT_WEIGHTS, top_n=cfg.top_n)

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
    )
    return [c.smiles for c in ranked if c.smiles], diagnostics


def run_pipeline(
    molecules: list[QueryMolecule],
    library: SpectralLibrary,
    pool: CandidatePool,
    config: Config | None = None,
    weights: dict[str, float] | None = None,
    progress_every: int = 0,
) -> PipelineResult:
    """Run the baseline over many molecules.

    The analog index is built once and shared, since deriving representative
    spectra per query would dominate runtime.
    """
    cfg = config or CFG
    analog_index = AnalogIndex(library)
    result = PipelineResult()

    for i, molecule in enumerate(molecules):
        if progress_every and i and i % progress_every == 0:
            print(f"  {i}/{len(molecules)} molecules", flush=True)
        smiles, diagnostics = predict_molecule(
            molecule, library, pool, analog_index, config=cfg, weights=weights
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
