"""Tests for the MRR@25 scorer.

Correctness here gates every experiment, so the metric is checked against
hand-computed values rather than the implementation's own output.
"""

import pytest

from casmi.chem import inchikey14
from casmi.eval.metrics import (
    dedupe_keys,
    evaluate_predictions,
    format_summary,
    mrr_at_k,
    recall_at_k,
    reciprocal_rank,
    score_molecule,
    summarise,
    summarise_by_class,
    top1_accuracy,
)

ETHANOL = "CCO"
PROPANE = "CCC"
BENZENE = "c1ccccc1"
PHENOL = "Oc1ccccc1"


class TestReciprocalRank:
    def test_rank_one(self):
        assert reciprocal_rank(["A", "B"], "A") == 1.0

    def test_rank_two(self):
        assert reciprocal_rank(["A", "B"], "B") == pytest.approx(0.5)

    def test_rank_five(self):
        assert reciprocal_rank(list("ABCDE"), "E") == pytest.approx(0.2)

    def test_rank_25_is_lowest_nonzero(self):
        keys = [f"K{i}" for i in range(25)]
        assert reciprocal_rank(keys, "K24") == pytest.approx(1 / 25)

    def test_beyond_k_scores_zero(self):
        """Position 26 is outside the competition's 25-slot list."""
        keys = [f"K{i}" for i in range(30)]
        assert reciprocal_rank(keys, "K25") == 0.0

    def test_absent_scores_zero(self):
        assert reciprocal_rank(["A", "B"], "Z") == 0.0

    def test_empty_prediction(self):
        assert reciprocal_rank([], "A") == 0.0

    def test_empty_truth(self):
        assert reciprocal_rank(["A"], "") == 0.0

    def test_first_occurrence_wins(self):
        assert reciprocal_rank(["A", "B", "A"], "A") == 1.0

    def test_none_entries_skipped(self):
        assert reciprocal_rank([None, "A"], "A") == pytest.approx(0.5)

    def test_custom_k(self):
        assert reciprocal_rank(["A", "B", "C"], "C", k=2) == 0.0


class TestDedupeKeys:
    def test_removes_repeats_preserving_order(self):
        assert dedupe_keys(["A", "B", "A", "C"]) == ["A", "B", "C"]

    def test_keeps_none_placeholders(self):
        assert dedupe_keys(["A", None, "A", None]) == ["A", None, None]

    def test_empty(self):
        assert dedupe_keys([]) == []


class TestScoreMolecule:
    def test_correct_first_guess(self):
        s = score_molecule("m1", [ETHANOL, PROPANE], ETHANOL)
        assert s.rr == 1.0
        assert s.hit_rank == 1
        assert s.is_hit

    def test_correct_second_guess(self):
        s = score_molecule("m1", [PROPANE, ETHANOL], ETHANOL)
        assert s.rr == pytest.approx(0.5)
        assert s.hit_rank == 2

    def test_miss(self):
        s = score_molecule("m1", [PROPANE, BENZENE], ETHANOL)
        assert s.rr == 0.0
        assert s.hit_rank is None
        assert not s.is_hit

    def test_stereochemistry_ignored(self):
        """A prediction differing only in stereo must count as correct."""
        s = score_molecule("m1", ["C[C@@H](N)C(=O)O"], "C[C@H](N)C(=O)O")
        assert s.rr == 1.0

    def test_tautomer_ignored(self):
        s = score_molecule("m1", ["CC(O)=C"], "CC(=O)C")
        assert s.rr == 1.0

    def test_equivalent_smiles_spelling(self):
        s = score_molecule("m1", [PHENOL], "c1ccccc1O")
        assert s.rr == 1.0

    def test_duplicate_predictions_do_not_waste_slots(self):
        """Dedup happens before truncation, so the 26th unique entry can score."""
        preds = [PROPANE] * 25 + [ETHANOL]
        assert score_molecule("m1", preds, ETHANOL).rr == pytest.approx(0.5)

    def test_unparseable_predictions_tolerated(self):
        s = score_molecule("m1", ["!!bad!!", ETHANOL], ETHANOL)
        assert s.rr == pytest.approx(0.5)

    def test_unparseable_truth_scores_zero(self):
        assert score_molecule("m1", [ETHANOL], "!!bad!!").rr == 0.0

    def test_empty_predictions(self):
        assert score_molecule("m1", [], ETHANOL).rr == 0.0

    def test_precomputed_keys_match_fresh(self):
        keys = [inchikey14(PROPANE), inchikey14(ETHANOL)]
        fresh = score_molecule("m1", [PROPANE, ETHANOL], ETHANOL)
        cached = score_molecule(
            "m1", [], ETHANOL, predicted_keys=keys, true_key=inchikey14(ETHANOL)
        )
        assert cached.rr == fresh.rr

    def test_novelty_class_recorded(self):
        assert score_molecule("m1", [ETHANOL], ETHANOL, novelty_class=2).novelty_class == 2


class TestEvaluatePredictions:
    def test_scores_each_molecule(self):
        scores = evaluate_predictions(
            {"m1": [ETHANOL], "m2": [PROPANE, BENZENE]},
            {"m1": ETHANOL, "m2": BENZENE},
        )
        by_id = {s.molecule_id: s for s in scores}
        assert by_id["m1"].rr == 1.0
        assert by_id["m2"].rr == pytest.approx(0.5)

    def test_missing_prediction_scores_zero(self):
        """A molecule absent from the submission is wrong, not excluded."""
        scores = evaluate_predictions({}, {"m1": ETHANOL})
        assert len(scores) == 1
        assert scores[0].rr == 0.0

    def test_extra_predictions_ignored(self):
        scores = evaluate_predictions({"m1": [ETHANOL], "ghost": [ETHANOL]}, {"m1": ETHANOL})
        assert len(scores) == 1

    def test_novelty_classes_applied(self):
        scores = evaluate_predictions(
            {"m1": [ETHANOL]}, {"m1": ETHANOL}, novelty_classes={"m1": 3}
        )
        assert scores[0].novelty_class == 3


class TestAggregates:
    def _mixed(self):
        return evaluate_predictions(
            {
                "a": [ETHANOL],                      # rank 1 -> 1.0
                "b": [PROPANE, ETHANOL],             # rank 2 -> 0.5
                "c": [PROPANE],                      # miss   -> 0.0
            },
            {"a": ETHANOL, "b": ETHANOL, "c": ETHANOL},
            novelty_classes={"a": 1, "b": 2, "c": 3},
        )

    def test_mrr(self):
        assert mrr_at_k(self._mixed()) == pytest.approx((1.0 + 0.5 + 0.0) / 3)

    def test_recall(self):
        assert recall_at_k(self._mixed()) == pytest.approx(2 / 3)

    def test_top1(self):
        assert top1_accuracy(self._mixed()) == pytest.approx(1 / 3)

    def test_empty_aggregates_are_zero(self):
        assert mrr_at_k([]) == 0.0
        assert recall_at_k([]) == 0.0
        assert top1_accuracy([]) == 0.0
        assert summarise([])["n"] == 0.0

    def test_summarise_by_class_splits_cohorts(self):
        summary = summarise_by_class(self._mixed())
        assert summary["overall"]["n"] == 3
        assert summary["class1"]["mrr"] == pytest.approx(1.0)
        assert summary["class2"]["mrr"] == pytest.approx(0.5)
        assert summary["class3"]["mrr"] == pytest.approx(0.0)

    def test_summarise_by_class_omits_absent_classes(self):
        scores = evaluate_predictions(
            {"a": [ETHANOL]}, {"a": ETHANOL}, novelty_classes={"a": 1}
        )
        summary = summarise_by_class(scores)
        assert "class1" in summary
        assert "class2" not in summary

    def test_format_summary_is_tabular(self):
        text = format_summary(summarise_by_class(self._mixed()))
        assert "MRR@25" in text
        assert "overall" in text
        assert len(text.splitlines()) >= 5
