"""Channel 8 wired into the pipeline: a database-absent molecule becomes reachable."""

from __future__ import annotations

import numpy as np
import pytest

from casmi.candidates.pool import build_pool
from casmi.channels.analog import Analog, AnalogIndex
from casmi.channels.derivation import DerivationConfig
from casmi.channels.ranker import FEATURE_NAMES, assemble_features
from casmi.chem import FingerprintCalculator, exact_mass, inchikey14
from casmi.pipeline import build_scored_candidates, predict_molecule
from tests.test_pipeline_channels import make_library, make_queries

PARENT = "O=c1c(O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12"           # kaempferol (known)
TRUTH = "COc1cc(-c2oc3cc(O)cc(O)c3c(=O)c2O)ccc1O"            # isorhamnetin: kaempferol + OCH2
FAR = "CCCCCCCCCCCCCCCCCCCCCCCCCC"                           # unrelated library entry


@pytest.fixture(scope="module")
def setup():
    calc = FingerprintCalculator()
    # Pool knows the parent and an unrelated molecule, never the truth (class 3).
    pool = build_pool([PARENT, FAR, "CCO"], calculator=calc)
    library = make_library([(FAR, 0)])
    molecule = make_queries([("m_novel", TRUTH, 5, 2)])[0]
    truth_logits = calc.from_smiles(TRUTH).astype(np.float32) * 4.0 - 2.0
    return calc, pool, library, AnalogIndex(library), molecule, truth_logits


@pytest.fixture
def patched(monkeypatch, setup):
    calc, pool, library, index, molecule, truth_logits = setup
    monkeypatch.setattr("casmi.models.train.predict_logits", lambda *a, **k: truth_logits)
    monkeypatch.setattr(
        "casmi.pipeline.find_analogs",
        lambda *a, **k: [Analog(inchikey14=inchikey14(PARENT), similarity=0.85,
                                mass_shift=exact_mass(TRUTH) - exact_mass(PARENT))],
    )
    return setup


class TestDerivationWiring:
    def test_truth_unreachable_without_channel_8(self, patched):
        _, pool, library, index, molecule, _ = patched
        candidates, ctx = build_scored_candidates(
            molecule, library, pool, index, fpnet_model=object()
        )
        assert inchikey14(TRUTH) not in {c.inchikey14 for c in candidates}
        assert ctx["derived_candidates"] == []

    def test_truth_generated_with_channel_8(self, patched):
        _, pool, library, index, molecule, _ = patched
        candidates, ctx = build_scored_candidates(
            molecule, library, pool, index, fpnet_model=object(),
            fragmentation_channel=True, derivation_config=DerivationConfig(),
        )
        by_key = {c.inchikey14: c for c in candidates}
        hit = by_key.get(inchikey14(TRUTH))
        assert hit is not None, "derivation should have produced isorhamnetin"
        assert hit.features["derived"] == 1.0
        assert hit.features["derivation_parent_sim"] == pytest.approx(0.85)
        assert hit.features["derivation_steps"] in (1.0, 2.0)
        assert hit.features["fpnet_normalised_score"] > 0
        assert ctx["derived_candidates"]
        # every derived candidate sits on the query's mass
        for c in candidates:
            if c.features.get("derived"):
                assert c.features["mass_error_ppm"] < 10.0

    def test_truth_ranked_first_when_fpnet_agrees(self, patched):
        _, pool, library, index, molecule, _ = patched
        ranked, diag = predict_molecule(
            molecule, library, pool, index, fpnet_model=object(),
            derivation_config=DerivationConfig(),
        )
        assert inchikey14(ranked[0]) == inchikey14(TRUTH)
        assert diag.n_derived_candidates > 0
        # the overlay reads derived structures in final rank order from the diagnostics
        assert diag.derived_smiles and inchikey14(diag.derived_smiles[0]) == inchikey14(TRUTH)
        assert all(s in ranked for s in diag.derived_smiles)

    def test_ranker_features_include_derivation_columns(self):
        for name in ("derived", "derivation_parent_sim", "derivation_steps", "derivation_support"):
            assert name in FEATURE_NAMES
        m = assemble_features([{"derived": 1.0, "derivation_steps": 2.0}, {}])
        col = {n: i for i, n in enumerate(FEATURE_NAMES)}
        assert m[0, col["derived"]] == 1.0 and m[1, col["derived"]] == 0.0
        assert m[0, col["derivation_steps"]] == 2.0
