"""Channel 2 — mass-shifted analog propagation.

Targets novelty class 2: no reference spectrum exists for the molecule itself,
but a *structurally related* natural product has one. Related natural products
commonly differ by interpretable modifications (±CH2, ±OH, ±hexose), and the
fragments that survive such a modification appear in the analog's spectrum
shifted by exactly the mass difference.

The propagation has two steps:

1. Find spectral analogs in a wide (±200 Da) mass window, scoring each with
   the *mass-shifted* entropy similarity so shifted fragment series still
   align.
2. Score each candidate structure by how similar it is to those analogs'
   structures, weighted by how spectrally similar each analog was.

The analog similarity is raised to a power before weighting. Intuition: a
weakly-matching analog carries almost no structural information, and linear
weighting lets a crowd of mediocre analogs outvote one strong one. The
reference solution measured p=1 -> 0.498, p=3 -> 0.521, p=4 -> 0.525 class-2
MRR — but those sweeps ran against a leaked test set, so the default here is a
starting point to re-sweep, not a settled optimum.

Why this matters for our data problem specifically: it converts the thin
natural-product slice of the training set into useful signal for molecules that
are *not* in it, which is exactly what the off-domain-heavy library mix
otherwise fails to provide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from casmi.candidates.pool import CandidatePool
from casmi.chem import tanimoto_matrix
from casmi.config import CFG, AnalogConfig, SpectrumConfig
from casmi.data.loaders import QueryMolecule, SpectralLibrary
from casmi.spectra import clean_spectrum, search_library


@dataclass
class Analog:
    """A spectrally similar reference structure at a different mass."""

    inchikey14: str
    similarity: float
    mass_shift: float


def _representative_rows(library: SpectralLibrary) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One richest-spectrum row per structure, sorted by neutral mass.

    Searching every spectrum of every structure would repeat near-identical
    work across collision energies; the peak-richest spectrum carries the most
    structural information, so it stands in for the structure.
    """
    peaks_per_row = np.diff(library.offsets)
    best_row: dict[str, int] = {}
    for row, key in enumerate(library.inchikey14):
        if not key or not np.isfinite(library.neutral_mass[row]):
            continue
        current = best_row.get(key)
        if current is None or peaks_per_row[row] > peaks_per_row[current]:
            best_row[key] = row

    if not best_row:
        return (
            np.zeros(0, dtype=np.int64),
            np.asarray([], dtype=object),
            np.zeros(0, dtype=np.float64),
        )

    rows = np.fromiter(best_row.values(), dtype=np.int64, count=len(best_row))
    keys = np.asarray(list(best_row.keys()), dtype=object)
    masses = library.neutral_mass[rows]
    order = np.argsort(masses, kind="stable")
    return rows[order], keys[order], masses[order]


class AnalogIndex:
    """Precomputed representative-spectrum index for analog search.

    Built once per library and reused across molecules — recomputing the
    representative set per query would dominate runtime.
    """

    def __init__(self, library: SpectralLibrary) -> None:
        self.library = library
        self.rows, self.keys, self.masses = _representative_rows(library)

    def __len__(self) -> int:
        return len(self.rows)

    def window(self, target_mass: float, window_da: float) -> slice:
        """Slice of representatives within ±``window_da`` of ``target_mass``."""
        lo = int(np.searchsorted(self.masses, target_mass - window_da, "left"))
        hi = int(np.searchsorted(self.masses, target_mass + window_da, "right"))
        return slice(lo, hi)


def find_analogs(
    molecule: QueryMolecule,
    index: AnalogIndex,
    spectrum_config: SpectrumConfig | None = None,
    analog_config: AnalogConfig | None = None,
) -> list[Analog]:
    """Find the most spectrally similar reference structures at shifted masses.

    Returns up to ``analog_config.n_analogs`` analogs, best first.
    """
    spec_cfg = spectrum_config or CFG.spectrum
    ana_cfg = analog_config or CFG.analog

    target = molecule.target_mass
    if not np.isfinite(target) or len(index) == 0:
        return []

    span = index.window(target, ana_cfg.window_da)
    rows = index.rows[span]
    keys = index.keys[span]
    if rows.size == 0:
        return []

    # Shift needed to bring each reference's fragments onto the query's scale.
    shifts = (target - index.masses[span]).astype(np.float32)

    best: dict[str, tuple[float, float]] = {}
    for mz, intensity in molecule.spectra:
        query_mz, query_p = clean_spectrum(mz, intensity, config=spec_cfg)
        if query_mz.size == 0:
            continue
        scores = search_library(
            query_mz,
            query_p,
            rows,
            index.library.offsets,
            index.library.all_mz,
            index.library.all_intensity,
            config=spec_cfg,
            shifts=shifts,
        )
        for key, score, shift in zip(keys, scores, shifts, strict=True):
            value = float(score)
            if value <= 0.0:
                continue
            current = best.get(key)
            if current is None or value > current[0]:
                best[key] = (value, float(shift))

    analogs = [
        Analog(inchikey14=key, similarity=score, mass_shift=shift)
        for key, (score, shift) in best.items()
    ]
    analogs.sort(key=lambda a: -a.similarity)
    return analogs[: ana_cfg.n_analogs]


def propagate_to_candidates(
    candidate_indices: np.ndarray,
    pool: CandidatePool,
    analogs: list[Analog],
    analog_config: AnalogConfig | None = None,
) -> dict[str, np.ndarray]:
    """Score candidates by structural similarity to spectral analogs.

    Returns a dict of per-candidate feature arrays, each aligned with
    ``candidate_indices``:

    * ``analog_power``: max over analogs of ``Tanimoto * similarity^p`` — the
      primary propagated score.
    * ``analog_linear``: the same with ``p = 1``, kept as a separate feature so
      a reranker can learn how much sharpening actually helps.
    * ``analog_best_tanimoto``: max Tanimoto ignoring spectral similarity,
      which flags candidates near *some* reference structure regardless of
      spectral support.
    * ``analog_top_tanimoto``: Tanimoto to the single best-matching analog.
    * ``analog_mean``: similarity-weighted mean, less spiky than the max.
    """
    ana_cfg = analog_config or CFG.analog
    n = len(candidate_indices)
    empty = np.zeros(n, dtype=np.float32)
    if n == 0 or not analogs:
        return {
            "analog_power": empty,
            "analog_linear": empty.copy(),
            "analog_best_tanimoto": empty.copy(),
            "analog_top_tanimoto": empty.copy(),
            "analog_mean": empty.copy(),
        }

    # Only analogs whose structures are in the pool have fingerprints available.
    analog_rows, analog_sims = [], []
    for analog in analogs:
        row = pool.index_of(analog.inchikey14)
        if row >= 0:
            analog_rows.append(row)
            analog_sims.append(analog.similarity)
    if not analog_rows:
        return {
            "analog_power": empty,
            "analog_linear": empty.copy(),
            "analog_best_tanimoto": empty.copy(),
            "analog_top_tanimoto": empty.copy(),
            "analog_mean": empty.copy(),
        }

    candidate_fp = pool.fingerprints(candidate_indices)
    analog_fp = pool.fingerprints(np.asarray(analog_rows, dtype=np.int64))
    tanimoto = tanimoto_matrix(candidate_fp, analog_fp)

    sims = np.asarray(analog_sims, dtype=np.float32)
    powered = np.clip(sims, 0.0, None) ** ana_cfg.sim_power
    powered_sum = float(powered.sum())

    return {
        "analog_power": (tanimoto * powered[None, :]).max(axis=1).astype(np.float32),
        "analog_linear": (tanimoto * sims[None, :]).max(axis=1).astype(np.float32),
        "analog_best_tanimoto": tanimoto.max(axis=1).astype(np.float32),
        "analog_top_tanimoto": tanimoto[:, 0].astype(np.float32),
        "analog_mean": (
            (tanimoto * powered[None, :]).sum(axis=1) / max(powered_sum, 1e-9)
        ).astype(np.float32),
    }
