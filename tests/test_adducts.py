"""Tests for adduct mass arithmetic.

Expected neutral masses are derived from published monoisotopic masses, not
from this module's own output, so these are genuine checks rather than
regression snapshots.
"""

import numpy as np
import pytest

from casmi.adducts import (
    ADDUCTS,
    ELECTRON_MASS,
    PROTON_MASS,
    TEST_ADDUCTS,
    is_positive_mode,
    neutral_mass,
    neutral_mass_array,
    ppm_error,
    ppm_window,
)

#: Glucose, monoisotopic 180.06339 Da.
GLUCOSE = 180.0633881


class TestConstants:
    def test_proton_mass(self):
        assert PROTON_MASS == pytest.approx(1.007276, abs=1e-6)

    def test_electron_mass(self):
        assert ELECTRON_MASS == pytest.approx(0.000549, abs=1e-6)

    def test_all_test_adducts_present(self):
        """The ten test-set adducts must all be handled."""
        for adduct in TEST_ADDUCTS:
            assert adduct in ADDUCTS, adduct


class TestNeutralMass:
    def test_protonated(self):
        assert neutral_mass(GLUCOSE + PROTON_MASS, "[M+H]+") == pytest.approx(GLUCOSE, abs=1e-6)

    def test_deprotonated(self):
        assert neutral_mass(GLUCOSE - PROTON_MASS, "[M-H]-") == pytest.approx(GLUCOSE, abs=1e-6)

    def test_sodiated(self):
        mz = GLUCOSE + 22.9897692809 - ELECTRON_MASS
        assert neutral_mass(mz, "[M+Na]+") == pytest.approx(GLUCOSE, abs=1e-6)

    def test_ammoniated(self):
        # NH4 = N + 4H = 14.0030740 + 4(1.0078250) = 18.0343741
        mz = GLUCOSE + 18.0343741 - ELECTRON_MASS
        assert neutral_mass(mz, "[M+NH4]+") == pytest.approx(GLUCOSE, abs=1e-5)

    def test_water_loss(self):
        mz = GLUCOSE + PROTON_MASS - 18.0105646
        assert neutral_mass(mz, "[M-H2O+H]+") == pytest.approx(GLUCOSE, abs=1e-5)

    def test_double_water_loss(self):
        mz = GLUCOSE + PROTON_MASS - 2 * 18.0105646
        assert neutral_mass(mz, "[M-2H2O+H]+") == pytest.approx(GLUCOSE, abs=1e-5)

    def test_chloride(self):
        mz = GLUCOSE + 34.96885268 + ELECTRON_MASS
        assert neutral_mass(mz, "[M+Cl]-") == pytest.approx(GLUCOSE, abs=1e-6)

    def test_formate(self):
        mz = GLUCOSE + 46.0054793 - PROTON_MASS
        assert neutral_mass(mz, "[M+CH2O2-H]-") == pytest.approx(GLUCOSE, abs=1e-5)

    def test_doubly_charged(self):
        """Charge 2 means the measured m/z is roughly half the neutral mass."""
        mz = (GLUCOSE + 2 * PROTON_MASS) / 2
        assert neutral_mass(mz, "[M+2H]2+") == pytest.approx(GLUCOSE, abs=1e-6)

    def test_dimer(self):
        """A 2M adduct's ion carries two copies of the molecule."""
        mz = 2 * GLUCOSE + PROTON_MASS
        assert neutral_mass(mz, "[2M+H]+") == pytest.approx(GLUCOSE, abs=1e-6)

    def test_trimer(self):
        mz = 3 * GLUCOSE + PROTON_MASS
        assert neutral_mass(mz, "[3M+H]+") == pytest.approx(GLUCOSE, abs=1e-6)

    def test_unknown_adduct_is_nan(self):
        assert np.isnan(neutral_mass(400.0, "[M+Unobtainium]+"))

    def test_roundtrip_all_adducts(self):
        """Every table entry must invert its own forward calculation."""
        for name, spec in ADDUCTS.items():
            mz = (GLUCOSE * spec.n_mer + spec.delta) / spec.charge
            assert neutral_mass(mz, name) == pytest.approx(GLUCOSE, abs=1e-6), name


class TestNeutralMassArray:
    def test_matches_scalar(self):
        mz = np.array([181.0706, 179.0561, 203.0526])
        adducts = np.array(["[M+H]+", "[M-H]-", "[M+Na]+"], dtype=object)
        got = neutral_mass_array(mz, adducts)
        for i in range(len(mz)):
            assert got[i] == pytest.approx(neutral_mass(mz[i], adducts[i]), abs=1e-9)

    def test_unknown_adducts_nan(self):
        got = neutral_mass_array(
            np.array([181.07, 181.07]), np.array(["[M+H]+", "???"], dtype=object)
        )
        assert np.isfinite(got[0])
        assert np.isnan(got[1])

    def test_empty(self):
        assert neutral_mass_array(np.array([]), np.array([], dtype=object)).size == 0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shapes differ"):
            neutral_mass_array(np.array([1.0, 2.0]), np.array(["[M+H]+"], dtype=object))


class TestPpm:
    def test_window_width(self):
        low, high = ppm_window(1000.0, 10.0)
        assert low == pytest.approx(999.99)
        assert high == pytest.approx(1000.01)

    def test_error_roundtrip(self):
        assert ppm_error(1000.01, 1000.0) == pytest.approx(10.0, abs=1e-6)

    def test_error_sign(self):
        assert ppm_error(999.99, 1000.0) < 0

    def test_error_zero_expected_is_nan(self):
        assert np.isnan(ppm_error(1.0, 0.0))


class TestPolarity:
    def test_positive(self):
        assert is_positive_mode("[M+H]+") is True

    def test_negative(self):
        assert is_positive_mode("[M-H]-") is False

    def test_unknown(self):
        assert is_positive_mode("???") is None
