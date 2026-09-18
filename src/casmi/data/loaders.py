"""Parquet loading and the in-memory spectral library layout.

Spectra are held in a flat CSR-style layout (one long ``mz`` array plus row
offsets) rather than as per-row numpy objects. Arrow's list columns already
have exactly this shape, so loading is close to zero-copy and the Numba search
kernels can read it directly — materialising 2.5M small arrays instead would
cost far more memory than the peak data itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from casmi.adducts import neutral_mass_array

#: Columns needed from train.parquet for library search and analog propagation.
LIBRARY_COLUMNS = (
    "inchikey14",
    "normalized_smiles",
    "adduct",
    "precursor_mz",
    "ms2_mzs",
    "ms2_normalized_intensities",
)

#: Columns needed from test.parquet (no labels present).
QUERY_COLUMNS = (
    "molecule_id",
    "spectrum_id",
    "adduct",
    "ionization_mode",
    "instrument_type",
    "precursor_mz",
    "collision_energy_ev",
    "ms2_mzs",
    "ms2_normalized_intensities",
)


@dataclass
class SpectralLibrary:
    """Flat, mass-sorted spectral library backing Channels 1 and 2.

    Attributes:
        offsets: Row ``i`` occupies ``all_mz[offsets[i]:offsets[i + 1]]``.
        all_mz: Concatenated fragment m/z values across all spectra.
        all_intensity: Concatenated intensities, aligned with ``all_mz``.
        neutral_mass: Per-row neutral mass, ``nan`` where the adduct is unknown.
        inchikey14: Per-row structure key.
        smiles: Per-row SMILES.
        mass_order: Row indices sorted by ascending neutral mass, finite first.
        sorted_mass: ``neutral_mass[mass_order]`` restricted to finite entries.
        n_finite: Number of rows with a resolvable neutral mass.
    """

    offsets: np.ndarray
    all_mz: np.ndarray
    all_intensity: np.ndarray
    neutral_mass: np.ndarray
    inchikey14: np.ndarray
    smiles: np.ndarray
    mass_order: np.ndarray
    sorted_mass: np.ndarray
    n_finite: int

    @property
    def n_spectra(self) -> int:
        return len(self.offsets) - 1

    def peaks(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """Raw ``(mz, intensity)`` for one row."""
        start, end = self.offsets[row], self.offsets[row + 1]
        return self.all_mz[start:end], self.all_intensity[start:end]

    def rows_in_mass_window(self, target: float, ppm: float) -> np.ndarray:
        """Row indices whose neutral mass lies within ±``ppm`` of ``target``.

        Binary search over the mass-sorted view, so this is O(log n) rather
        than a full scan of 2.5M rows per query.
        """
        tol = target * ppm / 1e6
        lo = np.searchsorted(self.sorted_mass, target - tol, "left")
        hi = np.searchsorted(self.sorted_mass, target + tol, "right")
        return self.mass_order[lo:hi]

    def rows_in_shift_window(self, target: float, window_da: float) -> np.ndarray:
        """Row indices within ±``window_da`` absolute mass of ``target``.

        Channel 2's wide analog window, expressed in Da rather than ppm
        because it is searching for chemical modifications of known size
        (±CH2, ±OH, ±hexose), not for measurement error.
        """
        lo = np.searchsorted(self.sorted_mass, target - window_da, "left")
        hi = np.searchsorted(self.sorted_mass, target + window_da, "right")
        return self.mass_order[lo:hi]

    def structure_smiles(self) -> dict[str, str]:
        """First-seen SMILES per ``inchikey14``.

        Used to turn library hits into submittable structures; any SMILES for
        a given 2D skeleton scores identically, so the first is as good as any.
        """
        out: dict[str, str] = {}
        for key, smi in zip(self.inchikey14, self.smiles, strict=True):
            if key and smi and key not in out:
                out[key] = smi
        return out


def _flat_list_column(table: pa.Table, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract an Arrow list column as ``(offsets, values)`` without per-row objects."""
    column = table.column(name).combine_chunks()
    if isinstance(column, pa.ChunkedArray):  # single-chunk after combine
        column = column.chunk(0)
    offsets = column.offsets.to_numpy().astype(np.int64)
    values = column.values.to_numpy(zero_copy_only=False).astype(np.float32)
    return offsets, values


def _string_column(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table.column(name).cast(pa.string()).to_pylist(), dtype=object)


def load_spectral_library(
    path: str | Path,
    columns: tuple[str, ...] = LIBRARY_COLUMNS,
    keep_rows: np.ndarray | None = None,
) -> SpectralLibrary:
    """Load ``train.parquet`` into a :class:`SpectralLibrary`.

    ``keep_rows`` (a boolean mask over the file's rows) restricts the library
    to the training side of a split. This is how validation leak-proofing is
    enforced at the data layer: the held-out molecules' spectra are physically
    absent from the library the channels search, so a leak cannot happen by
    forgetting a filter downstream.
    """
    table = pq.read_table(str(path), columns=list(columns))
    if keep_rows is not None:
        mask = np.asarray(keep_rows, dtype=bool)
        if mask.size != table.num_rows:
            raise ValueError(f"keep_rows size {mask.size} != table rows {table.num_rows}")
        table = table.filter(pa.array(mask))

    offsets, all_mz = _flat_list_column(table, "ms2_mzs")
    int_offsets, all_intensity = _flat_list_column(table, "ms2_normalized_intensities")
    if not np.array_equal(offsets, int_offsets):
        raise ValueError("ms2_mzs and ms2_normalized_intensities have misaligned offsets")

    precursor = table.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64)
    adduct = _string_column(table, "adduct")
    keys = _string_column(table, "inchikey14")
    smiles = _string_column(table, "normalized_smiles")

    masses = neutral_mass_array(precursor, adduct)
    finite = np.isfinite(masses)
    # Push non-finite masses to the end of the sort so the searchable prefix is
    # contiguous and binary search never lands on a nan.
    order = np.argsort(np.where(finite, masses, np.inf), kind="stable")
    n_finite = int(finite.sum())

    return SpectralLibrary(
        offsets=offsets,
        all_mz=all_mz,
        all_intensity=all_intensity,
        neutral_mass=masses,
        inchikey14=keys,
        smiles=smiles,
        mass_order=order,
        sorted_mass=masses[order][:n_finite],
        n_finite=n_finite,
    )


def library_from_arrays(
    mz_lists: list[np.ndarray],
    intensity_lists: list[np.ndarray],
    inchikey14: list[str],
    smiles: list[str],
    precursor_mz: list[float],
    adduct: list[str],
) -> SpectralLibrary:
    """Build a :class:`SpectralLibrary` from in-memory lists.

    Exists so the pipeline can be exercised on synthetic fixtures without the
    3 GB parquet files present.
    """
    n = len(mz_lists)
    if not (len(intensity_lists) == len(inchikey14) == len(smiles) == len(precursor_mz) == n):
        raise ValueError("all input lists must have equal length")

    offsets = np.zeros(n + 1, dtype=np.int64)
    for i, mz in enumerate(mz_lists):
        offsets[i + 1] = offsets[i] + len(mz)
    all_mz = (
        np.concatenate([np.asarray(m, np.float32) for m in mz_lists])
        if n
        else np.zeros(0, np.float32)
    )
    all_intensity = (
        np.concatenate([np.asarray(i, np.float32) for i in intensity_lists])
        if n
        else np.zeros(0, np.float32)
    )

    masses = neutral_mass_array(
        np.asarray(precursor_mz, dtype=np.float64), np.asarray(adduct, dtype=object)
    )
    finite = np.isfinite(masses)
    order = np.argsort(np.where(finite, masses, np.inf), kind="stable")
    n_finite = int(finite.sum())

    return SpectralLibrary(
        offsets=offsets,
        all_mz=all_mz,
        all_intensity=all_intensity,
        neutral_mass=masses,
        inchikey14=np.asarray(inchikey14, dtype=object),
        smiles=np.asarray(smiles, dtype=object),
        mass_order=order,
        sorted_mass=masses[order][:n_finite],
        n_finite=n_finite,
    )


@dataclass
class QueryMolecule:
    """One molecule's spectra, the unit predictions are made for.

    A molecule has 1-16 spectra at different adducts and collision energies.
    The competition scores per molecule, so evidence must be aggregated across
    all of them — never scored against a single spectrum in isolation.
    """

    molecule_id: str
    spectra: list[tuple[np.ndarray, np.ndarray]]
    precursor_mz: list[float]
    adducts: list[str]
    neutral_masses: list[float]
    ionization_modes: list[str]
    instrument_types: list[str]
    collision_energies: list[float]

    @property
    def n_spectra(self) -> int:
        return len(self.spectra)

    @property
    def target_mass(self) -> float:
        """Consensus neutral mass across this molecule's spectra.

        The median is used rather than the mean because a single mis-assigned
        adduct produces an outlier mass that would drag a mean out of the
        ±8.5 ppm candidate window entirely.
        """
        finite = [m for m in self.neutral_masses if np.isfinite(m)]
        return float(np.median(finite)) if finite else float("nan")

    @property
    def is_positive(self) -> bool:
        positives = sum(1 for m in self.ionization_modes if m == "positive")
        return positives >= (len(self.ionization_modes) - positives)


def _as_float_list(values) -> list[float]:
    """Coerce a collision-energy cell (scalar, list, or null) to floats.

    ``collision_energy_ev`` is a list because merged multi-energy acquisitions
    exist; nulls are common where the source library recorded nothing.
    """
    out: list[float] = []
    for v in values:
        if v is None:
            out.append(float("nan"))
            continue
        arr = np.atleast_1d(np.asarray(v, dtype=np.float64))
        arr = arr[np.isfinite(arr)]
        out.append(float(arr.mean()) if arr.size else float("nan"))
    return out


def load_query_molecules(
    path: str | Path, columns: tuple[str, ...] = QUERY_COLUMNS
) -> list[QueryMolecule]:
    """Load ``test.parquet`` grouped into :class:`QueryMolecule` objects."""
    available = set(pq.read_schema(str(path)).names)
    table = pq.read_table(str(path), columns=[c for c in columns if c in available])
    return query_molecules_from_table(table)


def query_molecules_from_table(table: pa.Table) -> list[QueryMolecule]:
    """Group an Arrow table of spectra into per-molecule query objects."""
    frame = table.to_pandas()
    if "molecule_id" not in frame.columns:
        raise ValueError("table must contain molecule_id")

    molecules: list[QueryMolecule] = []
    for molecule_id, group in frame.groupby("molecule_id", sort=True):
        precursors = group["precursor_mz"].astype(float).tolist()
        adducts = group["adduct"].astype(object).tolist()
        masses = neutral_mass_array(
            np.asarray(precursors, dtype=np.float64), np.asarray(adducts, dtype=object)
        ).tolist()
        molecules.append(
            QueryMolecule(
                molecule_id=str(molecule_id),
                spectra=[
                    (
                        np.asarray(mz, dtype=np.float32),
                        np.asarray(it, dtype=np.float32),
                    )
                    for mz, it in zip(
                        group["ms2_mzs"], group["ms2_normalized_intensities"], strict=True
                    )
                ],
                precursor_mz=precursors,
                adducts=adducts,
                neutral_masses=masses,
                ionization_modes=(
                    group["ionization_mode"].astype(object).tolist()
                    if "ionization_mode" in group
                    else ["positive"] * len(group)
                ),
                instrument_types=(
                    group["instrument_type"].astype(object).tolist()
                    if "instrument_type" in group
                    else ["timsTOF"] * len(group)
                ),
                collision_energies=(
                    _as_float_list(group["collision_energy_ev"])
                    if "collision_energy_ev" in group
                    else [float("nan")] * len(group)
                ),
            )
        )
    return molecules
