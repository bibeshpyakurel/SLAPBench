"""Common-list comparison contracts using artificial metadata and scores only."""

import contextlib
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from precise_matching import __main__ as cli  # noqa: E402
from precise_matching import comparison, data, evaluation  # noqa: E402
from test_precise_matching import fixtures, write_csv  # noqa: E402


class ComparisonTests(unittest.TestCase):
    def test_different_score_scales_and_failures_keep_identical_primary_population(self):
        images, pairs, a = fixtures()
        b = [replace(s, score=s.score * 100) for s in a]
        b[4:8] = [evaluation.Score("test_pair_0", "ok", 10),
                  evaluation.Score("test_pair_1", "ok", 30),
                  evaluation.Score("test_pair_2", "encoder_error", None),
                  evaluation.Score("test_pair_3", "ok", 0)]
        report = comparison.compare_methods(images, pairs, {"frozen_dinov2": a, "adapted_dinov2": b}, "0")
        self.assertEqual(report["methods"]["frozen_dinov2"]["threshold"]["cutoff"], 0.2)
        self.assertEqual(report["methods"]["adapted_dinov2"]["threshold"]["cutoff"], 20)
        for method in report["methods"].values():
            self.assertEqual(method["test"]["pooled"]["all_attempts_failure_as_reject"]["impostor_comparisons"], 2)
        # Ordering is explicit: adapted is A, frozen is B, so positive FAR
        # difference means frozen accepts the impostor that adapted failed on.
        paired = report["paired_comparisons"][0]
        self.assertEqual((paired["method_a"], paired["method_b"]), ("adapted_dinov2", "frozen_dinov2"))
        genuine, impostor = (paired["test"]["pooled"][label] for label in ("genuine", "impostor"))
        self.assertEqual(genuine["all_attempts_failure_as_reject"], {
            "both_accept": 0, "a_only_accept": 1, "b_only_accept": 1, "both_reject": 0,
            "acceptance_rate_difference_b_minus_a": 0})
        self.assertEqual(impostor["coverage_counts"], {
            "both_valid": 1, "a_only_valid": 0, "b_only_valid": 1, "neither_valid": 0})
        self.assertEqual(impostor["all_attempts_failure_as_reject"]["acceptance_rate_difference_b_minus_a"], 0.5)
        self.assertEqual(impostor["supplemental_both_valid_only"]["comparisons"], 1)
        self.assertEqual(impostor["supplemental_both_valid_only"]["acceptance_rate_difference_b_minus_a"], 0)
        content = json.dumps(report, allow_nan=False)
        for private in ("PRIVATE", "pair_", "_capture", images[0].sha256):
            self.assertNotIn(private, content)

    def test_absent_hand_and_all_failed_class_have_explicit_empty_support(self):
        images, pairs, scores = fixtures()
        failed = [replace(s, status="missing_input", score=None) if s.pair_id.startswith("test") else s
                  for s in scores]
        report = comparison.compare_methods(images, pairs, {"verifinger_sum": failed, "verifinger_fusion": failed}, "0")
        groups = report["paired_comparisons"][0]["test"]
        for label in ("genuine", "impostor"):
            self.assertEqual(groups["pooled"][label]["coverage_counts"]["neither_valid"], 2)
            self.assertEqual(groups["pooled"][label]["all_attempts_failure_as_reject"]["both_reject"], 2)
            self.assertIsNone(groups["pooled"][label]["supplemental_both_valid_only"]["acceptance_rate_difference_b_minus_a"])
            self.assertEqual(groups["left"][label]["attempts"], 0)
            self.assertIsNone(groups["left"][label]["all_attempts_failure_as_reject"]["acceptance_rate_difference_b_minus_a"])
        self.assertIsNone(report["methods"]["verifinger_sum"]["test"]["pooled"]["descriptive_valid_score_discrimination"])

    def test_hand_summaries_reuse_pooled_development_cutoff(self):
        images, pairs, scores = fixtures()
        left_images = [replace(im, image_id=im.image_id + "_left", hand="left", capture_id=im.capture_id + "_left",
                               sha256=hashlib.sha256((im.sha256 + "left").encode()).hexdigest(),
                               decoded_sha256=hashlib.sha256((im.decoded_sha256 + "left").encode()).hexdigest())
                       for im in images]
        left_pairs = [replace(p, pair_id=p.pair_id + "_left", image_a=p.image_a + "_left", image_b=p.image_b + "_left")
                      for p in pairs]
        left_scores = [replace(s, pair_id=s.pair_id + "_left", score=s.score / 2) for s in scores]
        # Left-only development would choose 0.1. The pooled cutoff is 0.2,
        # rejecting left test genuine 0.15 while accepting 0.25.
        left_scores[4] = replace(left_scores[4], score=0.15)
        left_scores[5] = replace(left_scores[5], score=0.25)
        all_scores = scores + left_scores
        report = comparison.compare_methods(images + left_images, pairs + left_pairs,
                                            {"adapted_dinov2": all_scores, "frozen_dinov2": all_scores}, "0")
        method = report["methods"]["adapted_dinov2"]
        self.assertEqual(method["threshold"]["cutoff"], 0.2)
        self.assertEqual(method["test"]["left"]["all_attempts_failure_as_reject"]["genuine_accepts"], 1)
        changed = [replace(s, score=1e9) if s.pair_id.startswith("test") else s for s in all_scores]
        revised = comparison.compare_methods(images + left_images, pairs + left_pairs,
                                             {"adapted_dinov2": changed, "frozen_dinov2": all_scores}, "0")
        self.assertEqual(revised["methods"]["adapted_dinov2"]["threshold"], method["threshold"])

    def test_every_arm_must_cover_exact_list_and_pass_metadata_and_score_checks(self):
        images, pairs, scores = fixtures()
        for candidate in (scores[:-1], scores + [scores[0]],
                          scores + [replace(scores[0], pair_id="train_pair_0")],
                          [replace(scores[0], score=float("nan")), *scores[1:]]):
            with self.subTest(rows=len(candidate)), self.assertRaises(data.InputError):
                comparison.compare_methods(images, pairs, {"frozen_dinov2": scores, "adapted_dinov2": candidate}, "0")
        images[0] = replace(images[0], exposure_reviewed="unknown")
        with self.assertRaises(data.InputError):
            comparison.compare_methods(images, pairs, {"frozen_dinov2": scores, "adapted_dinov2": scores}, "0")

    def test_supported_arms_only_and_order_invariance(self):
        images, pairs, scores = fixtures()
        for names in (["frozen_dinov2"], ["frozen_dinov2", "frozen_dinov2"], ["frozen_dinov2", "PRIVATE"]):
            with self.subTest(names=names), self.assertRaises(data.InputError):
                comparison.validate_arms(names)
        methods = {name: scores for name in sorted(comparison.ARMS)}
        report = comparison.compare_methods(images, pairs, methods, "0")
        reordered = {name: list(reversed(rows)) for name, rows in reversed(list(methods.items()))}
        self.assertEqual(comparison.compare_methods(images, list(reversed(pairs)), reordered, "0"), report)
        self.assertEqual(len(report["paired_comparisons"]), 6)


class ComparisonCliTests(unittest.TestCase):
    def inputs(self, root):
        images, pairs, scores = fixtures()
        for name, fields, rows in (("images", data.IMAGE_FIELDS, images), ("pairs", data.PAIR_FIELDS, pairs),
                                  ("a", evaluation.SCORE_FIELDS, scores), ("b", evaluation.SCORE_FIELDS, scores)):
            write_csv(root / (name + ".csv"), fields, [asdict(row) for row in rows])
        return ["compare", "--images", str(root / "images.csv"), "--pairs", str(root / "pairs.csv"),
                "--method", "frozen_dinov2", str(root / "a.csv"),
                "--method", "adapted_dinov2", str(root / "b.csv"), "--target-far", "0"]

    def test_cli_roundtrip_preserves_private_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.inputs(root)
            before = {p.name: p.read_bytes() for p in root.glob("*.csv")}
            output = root / "reports" / "comparison.json"
            with patch.object(cli, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main([*args, "--output", str(output)]), 0)
            content = output.read_text()
            self.assertEqual(len(json.loads(content)["paired_comparisons"]), 1)
            self.assertNotIn(tmp, content)
            self.assertNotIn("PRIVATE", content)
            self.assertEqual({p.name: p.read_bytes() for p in root.glob("*.csv")}, before)
            with patch.object(cli, "ROOT", root), patch.object(cli, "load_protocol") as load, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main([*args, "--output", str(output)]), 2)
            load.assert_not_called()
            self.assertEqual(output.read_text(), content)

    def test_unknown_or_repeated_arm_refused_before_inputs_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.inputs(root)
            for arm in ("PRIVATE", "frozen_dinov2"):
                invalid = [arm if arg == "adapted_dinov2" else arg for arg in args]
                error = io.StringIO()
                with patch.object(cli, "ROOT", root), patch.object(cli, "load_protocol") as load, \
                        contextlib.redirect_stderr(error):
                    self.assertEqual(cli.main([*invalid, "--output", str(root / "reports" / "bad.json")]), 2)
                load.assert_not_called()
                self.assertNotIn("PRIVATE", error.getvalue())

    def test_comparison_preserves_metadata_gate_and_rejects_changed_method_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.inputs(root)
            path = root / "images.csv"
            original = path.read_text()
            path.write_text(original.replace(",true,true", ",unknown,true", 1))
            output = root / "reports" / "blocked.json"
            with patch.object(cli, "ROOT", root), patch.object(cli, "load_scores") as load, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main([*args, "--output", str(output)]), 1)
            load.assert_not_called()
            self.assertNotIn("methods", json.loads(output.read_text()))
            path.write_text(original)
            original_load = evaluation.load_scores

            def mutate(path):
                rows = original_load(path)
                path.write_text(path.read_text() + "\n")
                return rows

            with patch.object(cli, "ROOT", root), patch.object(cli, "load_scores", side_effect=mutate), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main([*args, "--output", str(root / "reports" / "changed.json")]), 2)
            self.assertFalse((root / "reports" / "changed.json").exists())


if __name__ == "__main__":
    unittest.main()
