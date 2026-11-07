"""Independent metric check on ties, failures and exact synthetic membership."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from verify_precise_fixed_metrics import check_metrics, discrimination, outcomes  # noqa: E402


class IndependentMetricTests(unittest.TestCase):
    def test_ties_reject_and_failures_remain_in_denominators(self):
        pairs = [{"pair_id": str(i), "label": label} for i, label in enumerate(("genuine", "genuine", "impostor", "impostor"))]
        saved = {"all_attempts_failure_as_reject": {"genuine_comparisons": 2, "impostor_comparisons": 2,
                 "genuine_accepts": 0, "false_accepts": 0, "tar": 0, "far": 0},
                 "failures": {"genuine": 1, "impostor": 0},
                 "descriptive_valid_score_discrimination": {"auc": .75, "eer_linear_interpolation": 1 / 3}}
        result = check_metrics(pairs, np.array([.5, np.nan, .5, .1]), .5, saved)
        self.assertEqual(result["failed_genuine"], 1)
        self.assertEqual(result["genuine_accepts"], 0)
        saved["all_attempts_failure_as_reject"]["false_accepts"] = 1
        with self.assertRaises(ValueError):
            check_metrics(pairs, np.array([.5, np.nan, .5, .1]), .5, saved)

    def test_roc_uses_all_ties_and_both_classes_without_test_cutoff(self):
        self.assertEqual(discrimination(["genuine", "impostor"], [.5, .5]), {"auc": .5, "eer_linear_interpolation": .5})
        self.assertEqual(discrimination(["genuine", "impostor"], [.9, .1]), {"auc": 1, "eer_linear_interpolation": 0})
        with self.assertRaises(ValueError):
            discrimination(["genuine", "impostor"], [.9, float("inf")])

    def test_exact_saved_membership_and_unimputed_failures_are_required(self):
        pairs = [{"pair_id": "a", "label": "genuine"}, {"pair_id": "b", "label": "impostor"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            path.write_text("pair_id,status,score\na,ok,0.9\nb,error,\n")
            self.assertTrue(np.isnan(outcomes(path, pairs)[1]))
            for text in ("pair_id,status,score\na,ok,0.9\n", "pair_id,status,score\na,ok,0.9\nb,error,0\n"):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    outcomes(path, pairs)
            self.assertNotIn("jpeg", json.dumps(pairs))


if __name__ == "__main__":
    unittest.main()
