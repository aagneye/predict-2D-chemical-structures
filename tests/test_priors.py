"""Tests for structure-level priors (PubChem popularity)."""

import numpy as np
import pytest

from casmi.priors import PopularityPrior, popularity_blend


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "pool_popularity.csv"
    path.write_text(
        "pool_row,inchikey14,pubchem_cid,pop,np_coconut\n"
        "0,RYYVLZVUVIJVGH,2519,18.43,1\n"
        "1,LPHGQDQBBGAPDZ,1326,7.25,1\n"
        "2,RYYVLZVUVIJVGH,99,3.0,0\n"  # duplicate key: keep the max
        "3,,5,9.9,0\n"  # empty key skipped
        "4,BADVALUE000000,6,nan,0\n"  # non-finite skipped
        "5,NOVALUE0000000,7,,0\n",
        encoding="utf-8",
    )
    return path


class TestPopularityPrior:
    def test_from_csv_reads_pop_column(self, csv_path):
        prior = PopularityPrior.from_csv(csv_path)
        assert prior.get("LPHGQDQBBGAPDZ") == pytest.approx(7.25)

    def test_duplicate_keys_keep_max(self, csv_path):
        prior = PopularityPrior.from_csv(csv_path)
        assert prior.get("RYYVLZVUVIJVGH") == pytest.approx(18.43)

    def test_bad_rows_are_skipped(self, csv_path):
        prior = PopularityPrior.from_csv(csv_path)
        assert len(prior) == 2
        assert prior.get("BADVALUE000000") == 0.0

    def test_unknown_and_none_default_to_zero(self, csv_path):
        prior = PopularityPrior.from_csv(csv_path)
        assert prior.get("ZZZZZZZZZZZZZZ") == 0.0
        assert prior.get(None) == 0.0
        assert prior.get("") == 0.0

    def test_get_many_is_aligned(self, csv_path):
        prior = PopularityPrior.from_csv(csv_path)
        out = prior.get_many(["LPHGQDQBBGAPDZ", "nope", "RYYVLZVUVIJVGH"])
        assert out.dtype == np.float32
        assert out.tolist() == pytest.approx([7.25, 0.0, 18.43])

    def test_save_and_load_npz_roundtrip(self, csv_path, tmp_path):
        prior = PopularityPrior.from_csv(csv_path)
        out = tmp_path / "popularity.npz"
        prior.save(out)
        loaded = PopularityPrior.load(out)
        assert len(loaded) == len(prior)
        assert loaded.get("RYYVLZVUVIJVGH") == pytest.approx(18.43, abs=1e-5)

    def test_load_accepts_csv_directly(self, csv_path):
        assert len(PopularityPrior.load(csv_path)) == 2

    def test_missing_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="inchikey14"):
            PopularityPrior.from_csv(path)


class TestPopularityBlend:
    def test_zero_mu_is_pure_zscore(self):
        s = np.array([1.0, 2.0, 3.0])
        out = popularity_blend(s, np.array([10.0, 0.0, 5.0]), mu=0.0)
        assert out.mean() == pytest.approx(0.0)
        assert out.argmax() == 2

    def test_prior_can_reorder_close_scores(self):
        s = np.array([0.51, 0.50])
        pop = np.array([0.0, 18.0])
        assert popularity_blend(s, pop, mu=0.0).argmax() == 0
        assert popularity_blend(s, pop, mu=0.15).argmax() == 1

    def test_constant_scores_fall_back_to_prior(self):
        out = popularity_blend(np.array([0.5, 0.5, 0.5]), np.array([1.0, 3.0, 2.0]), mu=0.1)
        assert out.argmax() == 1

    def test_empty_and_shape_mismatch(self):
        assert popularity_blend(np.array([]), np.array([]), 0.1).size == 0
        with pytest.raises(ValueError):
            popularity_blend(np.array([1.0]), np.array([1.0, 2.0]), 0.1)
