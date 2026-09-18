"""Channel 1 — direct spectral library search.

Targets novelty class 1: the molecule has public reference spectra, so a
near-exact spectral match against the training library identifies it directly.
When it fires, this is the strongest single piece of evidence available.

Evidence is aggregated **per molecule**, not per spectrum: a molecule has 1-16
spectra at different adducts and collision energies, and the competition scores
one ranked list per molecule. Each candidate structure keeps its best score
across the molecule's spectra, since a match at any collision energy is
positive evidence and a poor match at one energy does not refute it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from casmi.config import CFG, CandidateConfig, SpectrumConfig
from casmi.data.loaders import QueryMolecule, SpectralLibrary
from casmi.spectra import clean_spectrum, search_library


@dataclass
class LibraryHit:
    """One structure matched by spectral similarity."""

    inchikey14: str
    smiles: str
    similarity: float
    #: Number of the molecule's spectra that matched this structure at all.
    n_supporting_spectra: int


def library_search(
    molecule: QueryMolecule,
    library: SpectralLibrary,
    spectrum_config: SpectrumConfig | None = None,
    candidate_config: CandidateConfig | None = None,
    ppm: float | None = None,
) -> dict[str, LibraryHit]:
    """Search ``library`` for spectra matching ``molecule``.

    Returns a mapping of ``inchikey14`` to its best :class:`LibraryHit`. An
    empty mapping means no library spectrum fell inside the mass window, which
    is the expected outcome for class-2 and class-3 molecules.
    """
    spec_cfg = spectrum_config or CFG.spectrum
    cand_cfg = candidate_config or CFG.candidates
    tolerance = cand_cfg.ppm_window if ppm is None else ppm

    target = molecule.target_mass
    if not np.isfinite(target):
        return {}

    rows = library.rows_in_mass_window(target, tolerance)
    if rows.size == 0:
        # Widen once rather than returning nothing: a systematic calibration
        # offset would otherwise silently zero out every candidate.
        rows = library.rows_in_mass_window(target, cand_cfg.ppm_fallback)
    if rows.size == 0:
        return {}

    best: dict[str, float] = {}
    support: dict[str, int] = {}
    for mz, intensity in molecule.spectra:
        query_mz, query_p = clean_spectrum(mz, intensity, config=spec_cfg)
        if query_mz.size == 0:
            continue
        scores = search_library(
            query_mz,
            query_p,
            rows,
            library.offsets,
            library.all_mz,
            library.all_intensity,
            config=spec_cfg,
        )
        for row, score in zip(rows, scores, strict=True):
            key = library.inchikey14[row]
            if not key:
                continue
            value = float(score)
            if value <= 0.0:
                continue
            if value > best.get(key, -1.0):
                best[key] = value
            support[key] = support.get(key, 0) + 1

    structure_smiles = library.structure_smiles()
    return {
        key: LibraryHit(
            inchikey14=key,
            smiles=structure_smiles.get(key, ""),
            similarity=score,
            n_supporting_spectra=support.get(key, 0),
        )
        for key, score in best.items()
    }
