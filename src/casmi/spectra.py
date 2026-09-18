"""Spectral cleaning and similarity.

Implements **spectral entropy similarity** (Li, Fleming et al., *Nature
Methods* 2021) rather than plain cosine. Entropy similarity down-weights
spectra dominated by a few peaks, which are common in this dataset and which
cosine over-rewards.

The hot loops are Numba-JIT compiled because Channel 1 scores a query against
every train spectrum inside a mass window, and Channel 2 does the same across
a ±200 Da window — tens of millions of pairwise comparisons per run in pure
Python is not viable.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from casmi.config import CFG, SpectrumConfig

#: Normalising constant in the entropy-similarity definition.
_LOG4 = np.log(4.0)
#: Entropy below which Li et al. apply intensity re-weighting.
_ENTROPY_THRESHOLD = 3.0


@njit(cache=True, fastmath=True)
def _clean_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    floor: float,
    max_peaks: int,
    power: float,
    entropy_weighting: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter, truncate and probability-normalise one peak list.

    Returns ``(mz, probability)`` sorted by m/z, where probabilities sum to 1.
    Entropy weighting (Li et al. 2021) raises intensities to ``0.25 + 0.25*S``
    when Shannon entropy ``S < 3``, sharpening uninformative spectra.
    """
    n = mz.shape[0]
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)

    max_int = 0.0
    for i in range(n):
        if intensity[i] > max_int:
            max_int = intensity[i]
    if max_int <= 0.0:
        return np.empty(0, np.float32), np.empty(0, np.float32)

    # The similarity kernel merges two peak lists assuming ascending m/z.
    # Parquet spectra arrive sorted, but relying on that silently would turn a
    # single unsorted input into a wrong-but-plausible score, so enforce it.
    # The check is O(n) and the sort only runs when actually needed.
    is_sorted = True
    for i in range(1, n):
        if mz[i] < mz[i - 1]:
            is_sorted = False
            break
    if not is_sorted:
        order = np.argsort(mz)
        mz = mz[order]
        intensity = intensity[order]

    threshold = floor * max_int
    kept = 0
    for i in range(n):
        if intensity[i] >= threshold:
            kept += 1
    if kept == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)

    idx = np.empty(kept, np.int64)
    j = 0
    for i in range(n):
        if intensity[i] >= threshold:
            idx[j] = i
            j += 1

    # Keep only the most intense `max_peaks`, then restore m/z order so the
    # downstream merge-style alignment stays valid.
    if kept > max_peaks:
        vals = np.empty(kept, np.float32)
        for i in range(kept):
            vals[i] = intensity[idx[i]]
        order = np.argsort(vals)[kept - max_peaks :]
        top = np.empty(max_peaks, np.int64)
        for i in range(max_peaks):
            top[i] = idx[order[i]]
        top.sort()
        idx = top
        kept = max_peaks

    out_mz = np.empty(kept, np.float32)
    out_p = np.empty(kept, np.float32)
    total = 0.0
    for i in range(kept):
        out_mz[i] = mz[idx[i]]
        v = intensity[idx[i]] ** power
        out_p[i] = v
        total += v
    if total <= 0.0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    for i in range(kept):
        out_p[i] /= total

    if entropy_weighting:
        entropy = 0.0
        for i in range(kept):
            if out_p[i] > 0.0:
                entropy -= out_p[i] * np.log(out_p[i])
        if entropy < _ENTROPY_THRESHOLD:
            w = 0.25 + 0.25 * entropy
            total2 = 0.0
            for i in range(kept):
                out_p[i] = out_p[i] ** w
                total2 += out_p[i]
            if total2 > 0.0:
                for i in range(kept):
                    out_p[i] /= total2

    return out_mz, out_p


@njit(cache=True, fastmath=True)
def _entropy_similarity(
    qmz: np.ndarray, qp: np.ndarray, cmz: np.ndarray, cp: np.ndarray, tol: float
) -> float:
    """Entropy similarity of two cleaned, probability-normalised spectra.

    ``1 - (2*S_AB - S_A - S_B) / ln(4)``, where ``S_AB`` is the entropy of the
    merged peak list. Ranges 0..1 for comparable spectra.
    """
    n = qmz.shape[0]
    m = cmz.shape[0]
    if n == 0 or m == 0:
        return 0.0

    s_a = 0.0
    for i in range(n):
        if qp[i] > 0.0:
            s_a -= qp[i] * np.log(qp[i])
    s_b = 0.0
    for i in range(m):
        if cp[i] > 0.0:
            s_b -= cp[i] * np.log(cp[i])

    # Merge the two sorted peak lists, summing intensities of aligned peaks.
    merged = np.empty(n + m, np.float64)
    count = 0
    i = 0
    j = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol:
            merged[count] = qp[i]
            i += 1
        elif d > tol:
            merged[count] = cp[j]
            j += 1
        else:
            merged[count] = qp[i] + cp[j]
            i += 1
            j += 1
        count += 1
    while i < n:
        merged[count] = qp[i]
        i += 1
        count += 1
    while j < m:
        merged[count] = cp[j]
        j += 1
        count += 1

    total = 0.0
    for x in range(count):
        total += merged[x]
    if total <= 0.0:
        return 0.0

    s_ab = 0.0
    for x in range(count):
        v = merged[x] / total
        if v > 0.0:
            s_ab -= v * np.log(v)

    return 1.0 - (2.0 * s_ab - s_a - s_b) / _LOG4


@njit(cache=True, fastmath=True)
def _entropy_similarity_shifted(
    qmz: np.ndarray,
    qp: np.ndarray,
    cmz: np.ndarray,
    cp: np.ndarray,
    tol: float,
    shift: float,
) -> float:
    """Best of direct and mass-shifted similarity.

    Channel 2 compares a query against an analog whose precursor differs by
    ``shift`` Da. Fragments that retain the modified moiety appear shifted by
    that amount, while fragments that lost it align directly — taking the max
    captures whichever set of fragments is more informative for this pair.
    """
    direct = _entropy_similarity(qmz, qp, cmz, cp, tol)
    if -1e-3 < shift < 1e-3:
        return direct
    shifted_mz = np.empty(cmz.shape[0], np.float32)
    for i in range(cmz.shape[0]):
        shifted_mz[i] = cmz[i] + shift
    shifted = _entropy_similarity(qmz, qp, shifted_mz, cp, tol)
    return direct if direct > shifted else shifted


@njit(cache=True, fastmath=True, parallel=True)
def _search_many(
    qmz: np.ndarray,
    qp: np.ndarray,
    candidate_rows: np.ndarray,
    offsets: np.ndarray,
    all_mz: np.ndarray,
    all_intensity: np.ndarray,
    tol: float,
    floor: float,
    max_peaks: int,
    power: float,
    entropy_weighting: bool,
    shifts: np.ndarray,
    use_shift: bool,
) -> np.ndarray:
    """Score one query against many library rows held in a flat CSR-style array.

    ``offsets`` delimits each row's slice of ``all_mz``/``all_intensity``; this
    layout comes straight from Arrow's list columns, so no per-row Python
    objects are ever materialised.
    """
    out = np.zeros(candidate_rows.shape[0], np.float32)
    for k in prange(candidate_rows.shape[0]):
        row = candidate_rows[k]
        start = offsets[row]
        end = offsets[row + 1]
        if end <= start:
            continue
        cmz, cp = _clean_peaks(
            all_mz[start:end], all_intensity[start:end], floor, max_peaks, power, entropy_weighting
        )
        if cmz.shape[0] == 0:
            continue
        if use_shift:
            out[k] = _entropy_similarity_shifted(qmz, qp, cmz, cp, tol, shifts[k])
        else:
            out[k] = _entropy_similarity(qmz, qp, cmz, cp, tol)
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def clean_spectrum(
    mz, intensity, config: SpectrumConfig | None = None, precursor_mz: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Clean one spectrum into ``(mz, probability)`` arrays.

    If ``precursor_mz`` is given, peaks above it (plus the configured margin)
    are dropped first — a fragment cannot outweigh its singly-charged
    precursor, so those peaks are co-isolated contaminants.
    """
    cfg = config or CFG.spectrum
    mz_arr = np.ascontiguousarray(np.asarray(mz, dtype=np.float32))
    int_arr = np.ascontiguousarray(np.asarray(intensity, dtype=np.float32))
    if mz_arr.shape != int_arr.shape:
        raise ValueError(f"mz and intensity shapes differ: {mz_arr.shape} vs {int_arr.shape}")
    if precursor_mz is not None:
        keep = mz_arr <= (float(precursor_mz) + cfg.precursor_margin)
        mz_arr = np.ascontiguousarray(mz_arr[keep])
        int_arr = np.ascontiguousarray(int_arr[keep])
    return _clean_peaks(
        mz_arr,
        int_arr,
        cfg.intensity_floor,
        cfg.max_peaks,
        cfg.intensity_power,
        cfg.entropy_weighting,
    )


def entropy_similarity(
    query_mz,
    query_p,
    other_mz,
    other_p,
    tol: float | None = None,
    shift: float = 0.0,
) -> float:
    """Entropy similarity between two already-cleaned spectra.

    ``shift`` (Da) enables the mass-shifted comparison used by Channel 2.
    """
    t = CFG.spectrum.mz_tol if tol is None else tol
    q_mz = np.ascontiguousarray(np.asarray(query_mz, dtype=np.float32))
    q_p = np.ascontiguousarray(np.asarray(query_p, dtype=np.float32))
    o_mz = np.ascontiguousarray(np.asarray(other_mz, dtype=np.float32))
    o_p = np.ascontiguousarray(np.asarray(other_p, dtype=np.float32))
    if shift:
        return float(_entropy_similarity_shifted(q_mz, q_p, o_mz, o_p, t, float(shift)))
    return float(_entropy_similarity(q_mz, q_p, o_mz, o_p, t))


def search_library(
    query_mz: np.ndarray,
    query_p: np.ndarray,
    candidate_rows: np.ndarray,
    offsets: np.ndarray,
    all_mz: np.ndarray,
    all_intensity: np.ndarray,
    config: SpectrumConfig | None = None,
    shifts: np.ndarray | None = None,
) -> np.ndarray:
    """Score a cleaned query spectrum against selected library rows.

    Pass ``shifts`` (one mass shift per candidate row, Da) to run the
    mass-shifted comparison used by analog propagation.
    """
    cfg = config or CFG.spectrum
    rows = np.ascontiguousarray(np.asarray(candidate_rows, dtype=np.int64))
    if rows.size == 0:
        return np.zeros(0, np.float32)
    use_shift = shifts is not None
    shift_arr = (
        np.ascontiguousarray(np.asarray(shifts, dtype=np.float32))
        if use_shift
        else np.zeros(rows.size, np.float32)
    )
    if use_shift and shift_arr.size != rows.size:
        raise ValueError(f"shifts size {shift_arr.size} != candidate count {rows.size}")
    return _search_many(
        np.ascontiguousarray(np.asarray(query_mz, dtype=np.float32)),
        np.ascontiguousarray(np.asarray(query_p, dtype=np.float32)),
        rows,
        np.ascontiguousarray(np.asarray(offsets, dtype=np.int64)),
        np.ascontiguousarray(np.asarray(all_mz, dtype=np.float32)),
        np.ascontiguousarray(np.asarray(all_intensity, dtype=np.float32)),
        cfg.mz_tol,
        cfg.intensity_floor,
        cfg.max_peaks,
        cfg.intensity_power,
        cfg.entropy_weighting,
        shift_arr,
        use_shift,
    )


def merge_spectra(
    spectra: list[tuple[np.ndarray, np.ndarray]], mz_tol: float = 0.005
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse a molecule's several spectra into one peak list.

    Each molecule has 1-16 spectra at different adducts/collision energies and
    predictions are made per molecule, so a fused view is one of the two ways
    to aggregate evidence (the other being per-spectrum scoring then a max).
    Intensities are max-pooled across near-identical m/z values.
    """
    if not spectra:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mz = np.concatenate([np.asarray(m, dtype=np.float64) for m, _ in spectra])
    inten = np.concatenate(
        [
            np.asarray(i, dtype=np.float64) / max(float(np.max(i)), 1e-9)
            if len(i)
            else np.zeros(0)
            for _, i in spectra
        ]
    )
    if mz.size == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    order = np.argsort(mz)
    mz, inten = mz[order], inten[order]
    keep_mz: list[float] = []
    keep_int: list[float] = []
    for k in range(mz.size):
        if keep_mz and (mz[k] - keep_mz[-1]) < mz_tol:
            if inten[k] > keep_int[-1]:
                keep_mz[-1] = mz[k]
                keep_int[-1] = inten[k]
        else:
            keep_mz.append(float(mz[k]))
            keep_int.append(float(inten[k]))
    return (
        np.asarray(keep_mz, dtype=np.float32),
        np.asarray(keep_int, dtype=np.float32),
    )
