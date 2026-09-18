"""Shared synthetic fixtures.

No competition data is present locally (``data/raw/`` is empty and the files
are 3 GB), so the whole pipeline is exercised against synthetic spectra whose
correct answers are known by construction. These fixtures deliberately
reproduce the awkward properties of the real data: compounds recurring across
libraries, variable spectra-per-molecule, unknown adducts, and empty peak
lists.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from casmi.adducts import PROTON_MASS
from casmi.chem import exact_mass, inchikey14

#: Small, real, structurally varied molecules. Using real SMILES (rather than
#: random strings) means RDKit fingerprints and InChIKeys behave realistically.
MOLECULES: list[tuple[str, str]] = [
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("phenol", "Oc1ccccc1"),
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("glucose", "OCC1OC(O)C(O)C(O)C1O"),
    ("alanine", "CC(N)C(=O)O"),
    ("benzoic_acid", "OC(=O)c1ccccc1"),
    ("toluene", "Cc1ccccc1"),
    ("naphthalene", "c1ccc2ccccc2c1"),
    ("indole", "c1ccc2[nH]ccc2c1"),
    ("vanillin", "COc1cc(C=O)ccc1O",),
    ("quinoline", "c1ccc2ncccc2c1"),
    ("anisole", "COc1ccccc1"),
]


def synthetic_peaks(seed: int, n_peaks: int = 12, precursor: float = 200.0):
    """Deterministic peak list below ``precursor``.

    Intensities are drawn on a wide dynamic range so intensity-floor and
    top-N truncation logic is actually exercised.
    """
    rng = np.random.default_rng(seed)
    mz = np.sort(rng.uniform(50.0, precursor - 1.0, size=n_peaks)).astype(np.float32)
    intensity = rng.uniform(0.01, 1.0, size=n_peaks).astype(np.float32)
    intensity[rng.integers(0, n_peaks)] = 1.0  # guarantee a base peak
    return mz, intensity


def shifted_peaks(mz: np.ndarray, intensity: np.ndarray, shift: float):
    """Peak list shifted by ``shift`` Da, simulating a mass-shifted analog."""
    return (np.asarray(mz, np.float32) + np.float32(shift), np.asarray(intensity, np.float32))


@pytest.fixture(scope="session")
def molecules() -> list[tuple[str, str]]:
    return MOLECULES


@pytest.fixture(scope="session")
def molecule_keys() -> dict[str, str]:
    """``name -> inchikey14`` for the fixture molecules."""
    return {name: inchikey14(smiles) for name, smiles in MOLECULES}


def build_library_table(
    entries: list[tuple[str, str, int]],
    adduct: str = "[M+H]+",
    ingest_lib: str = "gnps",
) -> pa.Table:
    """Build a train-shaped Arrow table.

    Args:
        entries: ``(name, smiles, seed)`` triples, one per spectrum. Repeat a
            name to simulate the same compound measured several times.
        adduct: Adduct string applied to every row.
        ingest_lib: Source library label applied to every row.
    """
    mzs, intensities, keys, smiles_out, precursors, libs = [], [], [], [], [], []
    for _name, smiles, seed in entries:
        mass = exact_mass(smiles)
        precursor = mass + PROTON_MASS
        mz, inten = synthetic_peaks(seed, precursor=max(precursor, 60.0))
        mzs.append(mz.tolist())
        intensities.append(inten.tolist())
        keys.append(inchikey14(smiles))
        smiles_out.append(smiles)
        precursors.append(precursor)
        libs.append(ingest_lib)

    return pa.table(
        {
            "inchikey14": pa.array(keys, pa.string()),
            "normalized_smiles": pa.array(smiles_out, pa.string()),
            "adduct": pa.array([adduct] * len(entries), pa.string()),
            "precursor_mz": pa.array(precursors, pa.float64()),
            "ms2_mzs": pa.array(mzs, pa.list_(pa.float32())),
            "ms2_normalized_intensities": pa.array(intensities, pa.list_(pa.float32())),
            "ingest_lib": pa.array(libs, pa.string()),
        }
    )


def build_query_table(entries: list[tuple[str, str, int, int]]) -> pa.Table:
    """Build a test-shaped Arrow table.

    Args:
        entries: ``(molecule_id, smiles, seed, n_spectra)`` tuples; each
            produces ``n_spectra`` rows sharing one ``molecule_id``, matching
            the real test set's 1-16 spectra per molecule.
    """
    (
        molecule_ids,
        spectrum_ids,
        mzs,
        intensities,
        precursors,
        adducts,
        modes,
        instruments,
        energies,
    ) = ([], [], [], [], [], [], [], [], [])

    for molecule_id, smiles, seed, n_spectra in entries:
        mass = exact_mass(smiles)
        precursor = mass + PROTON_MASS
        for s in range(n_spectra):
            mz, inten = synthetic_peaks(seed + s, precursor=max(precursor, 60.0))
            molecule_ids.append(molecule_id)
            spectrum_ids.append(f"{molecule_id}_s{s}")
            mzs.append(mz.tolist())
            intensities.append(inten.tolist())
            precursors.append(precursor)
            adducts.append("[M+H]+")
            modes.append("positive")
            instruments.append("timsTOF")
            energies.append([20.0 + 10.0 * s])

    return pa.table(
        {
            "molecule_id": pa.array(molecule_ids, pa.string()),
            "spectrum_id": pa.array(spectrum_ids, pa.string()),
            "ms2_mzs": pa.array(mzs, pa.list_(pa.float32())),
            "ms2_normalized_intensities": pa.array(intensities, pa.list_(pa.float32())),
            "precursor_mz": pa.array(precursors, pa.float64()),
            "adduct": pa.array(adducts, pa.string()),
            "ionization_mode": pa.array(modes, pa.string()),
            "instrument_type": pa.array(instruments, pa.string()),
            "collision_energy_ev": pa.array(energies, pa.list_(pa.float64())),
        }
    )


@pytest.fixture
def library_table() -> pa.Table:
    """Library where the first three molecules each have two spectra."""
    entries = []
    for i, (name, smiles) in enumerate(MOLECULES):
        entries.append((name, smiles, i * 10))
        if i < 3:
            entries.append((name, smiles, i * 10 + 1))
    return build_library_table(entries)


@pytest.fixture
def library_parquet(tmp_path, library_table) -> str:
    import pyarrow.parquet as pq

    path = tmp_path / "train.parquet"
    pq.write_table(library_table, path)
    return str(path)


@pytest.fixture
def query_parquet(tmp_path) -> str:
    import pyarrow.parquet as pq

    table = build_query_table(
        [
            ("m_001", MOLECULES[0][1], 0, 3),
            ("m_002", MOLECULES[1][1], 10, 1),
            ("m_003", MOLECULES[2][1], 20, 2),
        ]
    )
    path = tmp_path / "test.parquet"
    pq.write_table(table, path)
    return str(path)
