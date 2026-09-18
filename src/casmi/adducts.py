"""Adduct arithmetic: measured precursor m/z to neutral monoisotopic mass.

Every retrieval channel keys off the *neutral* mass, so this conversion sits
upstream of the whole pipeline and an error here silently poisons all
candidate windows. Masses are IUPAC monoisotopic values; the electron mass is
carried explicitly because at a ±8.5 ppm window it is not negligible (one
electron is ~1.2 ppm of a 450 Da molecule).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Monoisotopic atomic masses (Da).
ATOMIC_MASS: dict[str, float] = {
    "C": 12.0,
    "H": 1.00782503207,
    "N": 14.0030740048,
    "O": 15.9949146196,
    "P": 30.97376163,
    "S": 31.97207100,
    "F": 18.99840322,
    "Cl": 34.96885268,
    "Br": 78.9183371,
    "I": 126.904473,
    "Na": 22.9897692809,
    "K": 38.96370668,
    "Si": 27.9769265325,
    "B": 11.0093054,
    "Se": 79.9165213,
}

ELECTRON_MASS = 0.00054857990
PROTON_MASS = ATOMIC_MASS["H"] - ELECTRON_MASS
H2O = 2 * ATOMIC_MASS["H"] + ATOMIC_MASS["O"]
NH3 = ATOMIC_MASS["N"] + 3 * ATOMIC_MASS["H"]
NH4 = ATOMIC_MASS["N"] + 4 * ATOMIC_MASS["H"]
FORMIC_ACID = ATOMIC_MASS["C"] + 2 * ATOMIC_MASS["H"] + 2 * ATOMIC_MASS["O"]
ACETIC_ACID = 2 * ATOMIC_MASS["C"] + 4 * ATOMIC_MASS["H"] + 2 * ATOMIC_MASS["O"]
METHANOL = ATOMIC_MASS["C"] + 4 * ATOMIC_MASS["H"] + ATOMIC_MASS["O"]
ACETONITRILE = 2 * ATOMIC_MASS["C"] + 3 * ATOMIC_MASS["H"] + ATOMIC_MASS["N"]


@dataclass(frozen=True)
class Adduct:
    """An adduct's mass relationship to the neutral molecule.

    The measured ion satisfies ``mz * charge = n_mer * neutral_mass + delta``,
    so ``neutral_mass = (mz * charge - delta) / n_mer``.
    """

    n_mer: int
    charge: int
    delta: float
    polarity: int  # +1 positive, -1 negative


def _pos(n_mer: int, charge: int, delta: float) -> Adduct:
    return Adduct(n_mer, charge, delta, +1)


def _neg(n_mer: int, charge: int, delta: float) -> Adduct:
    return Adduct(n_mer, charge, delta, -1)


#: Adduct table. Covers the ten adducts used by the test set plus the wider
#: set appearing in train (needed because library/analog channels search train
#: spectra, whose adducts are more varied than the test set's).
ADDUCTS: dict[str, Adduct] = {
    # --- positive mode, monomer -------------------------------------------
    "[M+H]+": _pos(1, 1, PROTON_MASS),
    "[M+NH4]+": _pos(1, 1, NH4 - ELECTRON_MASS),
    "[M+Na]+": _pos(1, 1, ATOMIC_MASS["Na"] - ELECTRON_MASS),
    "[M+K]+": _pos(1, 1, ATOMIC_MASS["K"] - ELECTRON_MASS),
    "[M-H2O+H]+": _pos(1, 1, PROTON_MASS - H2O),
    "[M-2H2O+H]+": _pos(1, 1, PROTON_MASS - 2 * H2O),
    "[M-H2O]+": _pos(1, 1, -ELECTRON_MASS - H2O),
    "[M-NH3+H]+": _pos(1, 1, PROTON_MASS - NH3),
    "[M]+": _pos(1, 1, -ELECTRON_MASS),
    "[M+CH3OH+H]+": _pos(1, 1, PROTON_MASS + METHANOL),
    "[M+CH3CN+H]+": _pos(1, 1, PROTON_MASS + ACETONITRILE),
    "[M+2H]2+": _pos(1, 2, 2 * PROTON_MASS),
    # --- negative mode, monomer -------------------------------------------
    "[M-H]-": _neg(1, 1, -PROTON_MASS),
    "[M-H2O-H]-": _neg(1, 1, -PROTON_MASS - H2O),
    "[M+CH2O2-H]-": _neg(1, 1, FORMIC_ACID - PROTON_MASS),
    "[M+C2H4O2-H]-": _neg(1, 1, ACETIC_ACID - PROTON_MASS),
    "[M+Cl]-": _neg(1, 1, ATOMIC_MASS["Cl"] + ELECTRON_MASS),
    "[M+Na-2H]-": _neg(1, 1, ATOMIC_MASS["Na"] - 2 * PROTON_MASS),
    "[M]-": _neg(1, 1, ELECTRON_MASS),
    "[M-2H]-": _neg(1, 2, -2 * PROTON_MASS),
    # --- multimers (train only) -------------------------------------------
    "[2M+H]+": _pos(2, 1, PROTON_MASS),
    "[2M+Na]+": _pos(2, 1, ATOMIC_MASS["Na"] - ELECTRON_MASS),
    "[2M+NH4]+": _pos(2, 1, NH4 - ELECTRON_MASS),
    "[2M+K]+": _pos(2, 1, ATOMIC_MASS["K"] - ELECTRON_MASS),
    "[2M-H]-": _neg(2, 1, -PROTON_MASS),
    "[2M+CH2O2-H]-": _neg(2, 1, FORMIC_ACID - PROTON_MASS),
    "[2M+C2H4O2-H]-": _neg(2, 1, ACETIC_ACID - PROTON_MASS),
    "[2M+Na-2H]-": _neg(2, 1, ATOMIC_MASS["Na"] - 2 * PROTON_MASS),
    "[3M+H]+": _pos(3, 1, PROTON_MASS),
    "[3M-H]-": _neg(3, 1, -PROTON_MASS),
}

#: The ten adducts the test set is restricted to (docs/02-dataset.md).
TEST_ADDUCTS: tuple[str, ...] = (
    "[M+H]+",
    "[M+NH4]+",
    "[M-H2O+H]+",
    "[M-2H2O+H]+",
    "[M+Na]+",
    "[M+K]+",
    "[M-H]-",
    "[M-H2O-H]-",
    "[M+CH2O2-H]-",
    "[M+Cl]-",
)


def neutral_mass(mz: float, adduct: str) -> float:
    """Neutral monoisotopic mass implied by one ``(mz, adduct)`` pair.

    Returns ``nan`` for unknown adducts rather than raising: train contains
    dozens of adduct spellings and dropping those rows is the caller's choice.
    """
    spec = ADDUCTS.get(adduct)
    if spec is None:
        return float("nan")
    return (float(mz) * spec.charge - spec.delta) / spec.n_mer


def neutral_mass_array(mz: np.ndarray, adduct: np.ndarray) -> np.ndarray:
    """Vectorised :func:`neutral_mass` over aligned arrays.

    Groups by adduct string so each distinct adduct costs one masked vector
    operation instead of a per-row Python call — this runs over all ~2.5M train
    rows, where a row-wise loop is measurably slow.
    """
    mz = np.asarray(mz, dtype=np.float64)
    adduct = np.asarray(adduct, dtype=object)
    if mz.shape != adduct.shape:
        raise ValueError(f"mz and adduct shapes differ: {mz.shape} vs {adduct.shape}")
    out = np.full(mz.shape, np.nan, dtype=np.float64)
    for name, spec in ADDUCTS.items():
        mask = adduct == name
        if mask.any():
            out[mask] = (mz[mask] * spec.charge - spec.delta) / spec.n_mer
    return out


def ppm_window(mass: float, ppm: float) -> tuple[float, float]:
    """Inclusive ``(low, high)`` mass bounds for a ±``ppm`` tolerance."""
    tol = mass * ppm / 1e6
    return mass - tol, mass + tol


def ppm_error(observed: float, expected: float) -> float:
    """Signed ppm error of ``observed`` relative to ``expected``."""
    if expected == 0:
        return float("nan")
    return (observed - expected) / expected * 1e6


def is_positive_mode(adduct: str) -> bool | None:
    """Polarity implied by the adduct, or ``None`` if unknown."""
    spec = ADDUCTS.get(adduct)
    if spec is None:
        return None
    return spec.polarity > 0
