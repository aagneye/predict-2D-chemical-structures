"""Tests for parquet loading and the flat spectral-library layout."""

import numpy as np
import pytest

from casmi.adducts import PROTON_MASS
from casmi.chem import exact_mass, inchikey14
from casmi.data.loaders import (
    library_from_arrays,
    load_query_molecules,
    load_spectral_library,
    query_molecules_from_table,
)
from tests.conftest import MOLECULES, build_library_table, build_query_table


class TestLoadSpectralLibrary:
    def test_row_count_and_offsets(self, library_parquet, library_table):
        lib = load_spectral_library(library_parquet)
        assert lib.n_spectra == library_table.num_rows
        assert lib.offsets[0] == 0
        assert lib.offsets[-1] == lib.all_mz.size

    def test_peaks_roundtrip(self, library_parquet, library_table):
        lib = load_spectral_library(library_parquet)
        expected = library_table.column("ms2_mzs")[0].as_py()
        mz, inten = lib.peaks(0)
        assert mz.tolist() == pytest.approx(expected)
        assert len(mz) == len(inten)

    def test_neutral_mass_matches_structure(self, library_parquet):
        """Derived neutral mass must equal the labelled structure's exact mass."""
        lib = load_spectral_library(library_parquet)
        for row in range(lib.n_spectra):
            expected = exact_mass(lib.smiles[row])
            assert lib.neutral_mass[row] == pytest.approx(expected, abs=1e-4)

    def test_mass_sorted_view_is_ascending(self, library_parquet):
        lib = load_spectral_library(library_parquet)
        assert np.all(np.diff(lib.sorted_mass) >= 0)
        assert lib.n_finite == lib.n_spectra

    def test_mass_window_finds_target(self, library_parquet):
        lib = load_spectral_library(library_parquet)
        target = exact_mass(MOLECULES[0][1])
        rows = lib.rows_in_mass_window(target, ppm=10.0)
        assert len(rows) >= 1
        found = {lib.inchikey14[r] for r in rows}
        assert inchikey14(MOLECULES[0][1]) in found

    def test_mass_window_excludes_distant_masses(self, library_parquet):
        lib = load_spectral_library(library_parquet)
        rows = lib.rows_in_mass_window(exact_mass(MOLECULES[0][1]), ppm=1.0)
        for r in rows:
            assert abs(lib.neutral_mass[r] - exact_mass(MOLECULES[0][1])) < 0.01

    def test_empty_window(self, library_parquet):
        lib = load_spectral_library(library_parquet)
        assert lib.rows_in_mass_window(9999.0, ppm=5.0).size == 0

    def test_shift_window_is_wider(self, library_parquet):
        lib = load_spectral_library(library_parquet)
        target = exact_mass(MOLECULES[0][1])
        tight = lib.rows_in_mass_window(target, ppm=8.5)
        wide = lib.rows_in_shift_window(target, window_da=200.0)
        assert len(wide) > len(tight)

    def test_keep_rows_filters(self, library_parquet, library_table):
        """keep_rows is how held-out spectra are physically removed."""
        mask = np.zeros(library_table.num_rows, dtype=bool)
        mask[:4] = True
        lib = load_spectral_library(library_parquet, keep_rows=mask)
        assert lib.n_spectra == 4

    def test_keep_rows_wrong_size_raises(self, library_parquet):
        with pytest.raises(ValueError, match="!="):
            load_spectral_library(library_parquet, keep_rows=np.ones(3, dtype=bool))

    def test_structure_smiles_map(self, library_parquet):
        lib = load_spectral_library(library_parquet)
        mapping = lib.structure_smiles()
        assert len(mapping) == len(MOLECULES)
        key = inchikey14(MOLECULES[0][1])
        assert inchikey14(mapping[key]) == key


class TestLibraryFromArrays:
    def test_builds_equivalent_library(self):
        smiles = MOLECULES[0][1]
        mass = exact_mass(smiles)
        lib = library_from_arrays(
            mz_lists=[np.array([100.0, 150.0], np.float32)],
            intensity_lists=[np.array([1.0, 0.5], np.float32)],
            inchikey14=[inchikey14(smiles)],
            smiles=[smiles],
            precursor_mz=[mass + PROTON_MASS],
            adduct=["[M+H]+"],
        )
        assert lib.n_spectra == 1
        assert lib.neutral_mass[0] == pytest.approx(mass, abs=1e-4)

    def test_unknown_adduct_excluded_from_searchable_prefix(self):
        lib = library_from_arrays(
            mz_lists=[np.array([100.0], np.float32), np.array([100.0], np.float32)],
            intensity_lists=[np.array([1.0], np.float32), np.array([1.0], np.float32)],
            inchikey14=["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"],
            smiles=["CCO", "CCC"],
            precursor_mz=[100.0, 100.0],
            adduct=["[M+H]+", "[M+Unknown]+"],
        )
        assert lib.n_finite == 1
        assert lib.sorted_mass.size == 1
        assert np.isnan(lib.neutral_mass).sum() == 1

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="equal length"):
            library_from_arrays([np.array([1.0])], [], ["K"], ["CCO"], [100.0], ["[M+H]+"])

    def test_empty(self):
        lib = library_from_arrays([], [], [], [], [], [])
        assert lib.n_spectra == 0


class TestLoadQueryMolecules:
    def test_groups_spectra_per_molecule(self, query_parquet):
        molecules = load_query_molecules(query_parquet)
        assert len(molecules) == 3
        by_id = {m.molecule_id: m for m in molecules}
        assert by_id["m_001"].n_spectra == 3
        assert by_id["m_002"].n_spectra == 1
        assert by_id["m_003"].n_spectra == 2

    def test_target_mass_matches_structure(self, query_parquet):
        molecules = load_query_molecules(query_parquet)
        by_id = {m.molecule_id: m for m in molecules}
        assert by_id["m_001"].target_mass == pytest.approx(
            exact_mass(MOLECULES[0][1]), abs=1e-4
        )

    def test_target_mass_uses_median_to_resist_outliers(self):
        """One bad adduct assignment must not drag the consensus mass away."""
        table = build_query_table([("m_x", MOLECULES[0][1], 0, 3)])
        molecules = query_molecules_from_table(table)
        mol = molecules[0]
        good = mol.target_mass
        mol.neutral_masses[0] = 9999.0  # simulate a mis-assigned adduct
        assert mol.target_mass == pytest.approx(good, abs=1e-6)

    def test_target_mass_all_nan(self):
        table = build_query_table([("m_x", MOLECULES[0][1], 0, 1)])
        molecules = query_molecules_from_table(table)
        molecules[0].neutral_masses[0] = float("nan")
        assert np.isnan(molecules[0].target_mass)

    def test_collision_energy_averaged(self, query_parquet):
        molecules = load_query_molecules(query_parquet)
        by_id = {m.molecule_id: m for m in molecules}
        assert by_id["m_001"].collision_energies == pytest.approx([20.0, 30.0, 40.0])

    def test_ionization_mode(self, query_parquet):
        molecules = load_query_molecules(query_parquet)
        assert molecules[0].is_positive is True

    def test_missing_molecule_id_raises(self):
        import pyarrow as pa

        with pytest.raises(ValueError, match="molecule_id"):
            query_molecules_from_table(pa.table({"precursor_mz": pa.array([1.0])}))


class TestOffsetMisalignment:
    def test_misaligned_peak_columns_raise(self, tmp_path):
        """Guards against silently pairing the wrong m/z with the wrong intensity."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = build_library_table([("phenol", MOLECULES[1][1], 0)])
        bad = table.set_column(
            table.schema.get_field_index("ms2_normalized_intensities"),
            "ms2_normalized_intensities",
            pa.array([[1.0, 0.5]], pa.list_(pa.float32())),
        )
        path = tmp_path / "bad.parquet"
        pq.write_table(bad, path)
        with pytest.raises(ValueError, match="misaligned offsets"):
            load_spectral_library(str(path))
