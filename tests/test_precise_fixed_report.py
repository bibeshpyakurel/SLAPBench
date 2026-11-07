"""Saved-outcome reporting contracts; synthetic names/scores only."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from precise_matching.data import InputError  # noqa: E402
from precise_matching.evaluation import Score, select_cutoff, write_scores  # noqa: E402
from precise_matching.fixed_evaluation import fixed_summary  # noqa: E402
from precise_matching.fixed_pairs import FixedConfig, FixedPair, canonical_digest  # noqa: E402
from precise_matching.training import digest, write_json  # noqa: E402
from report_precise_shared_pairs import (  # noqa: E402
    dependency_counts, descriptive_scores, paired_dino, read_dino, read_historical_qwen, read_sdk,
)


def fixture():
    a, b, c, d = (f"{subject}_synthetic_{capture}_15.jpeg"
                  for subject, capture in (("001", "a"), ("001", "b"), ("1", "c"), ("001", "d")))
    training = [FixedPair("train-g", a, b, "genuine", "thumbs", False),
                FixedPair("train-i", a, c, "impostor", "thumbs", True)]
    testing = [FixedPair("test-g", a, d, "genuine", "thumbs", False),
               FixedPair("test-i", b, c, "impostor", "thumbs", True)]
    return training, testing


class FixedReportTests(unittest.TestCase):
    def test_dependency_counts_are_aggregate_and_do_not_claim_confirmed_identities(self):
        training, testing = fixture()
        report = dependency_counts(training, testing)
        self.assertEqual(report["train_test_overlap"], {"exact_pairs": 0, "images": 3, "numeric_groups": 1})
        self.assertEqual(report["training"]["maximum_pairs_using_one_image"], 2)
        self.assertNotIn("jpeg", json.dumps(report))

    def test_sdk_zero_scores_and_unknown_component_status_are_preserved(self):
        _, testing = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for p in testing:
                (root / f"random_2100_{p.label}_seed2026.txt").write_text(
                    f"{p.image_a} {p.image_b} 0 | 1 2 sum=3 nfing=2\n")
            scores, _ = read_sdk(root, testing)
            self.assertEqual([s.score for s in scores["verifinger_fusion"]], [0, 0])
            self.assertEqual([s.score for s in scores["verifinger_sum"]], [3, 3])
            report = descriptive_scores(testing, scores["verifinger_fusion"])
            self.assertIsNone(report["decision_threshold"])
            self.assertEqual(report["pooled"]["zero_impostor"], 1)
            self.assertEqual(report["label_ambiguity_sensitivity"]["attempted"], 1)
            self.assertNotIn("jpeg", json.dumps(report))
            (root / "random_2100_genuine_seed2026.txt").write_text(
                f"{testing[0].image_a} {testing[0].image_b} 0\n")
            scores, _ = read_sdk(root, testing)
            self.assertIsNone(scores["verifinger_sum"][0].score)

    def test_new_sdk_source_names_require_full_unchanged_membership(self):
        _, testing = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = {label: "new_" + label + ".txt" for label in ("genuine", "impostor")}
            for p in testing:
                (root / names[p.label]).write_text(f"{p.image_a} {p.image_b} 0 | 1 2 sum=3 nfing=2\n")
            before = {name: (root / name).read_bytes() for name in names.values()}
            scores, _ = read_sdk(root, testing, filenames=names)
            self.assertEqual([s.score for s in scores["verifinger_fusion"]], [0, 0])
            self.assertEqual(before, {name: (root / name).read_bytes() for name in names.values()})
            with self.assertRaises(InputError):
                read_sdk(root, [replace(testing[0], image_a="OTHER"), testing[1]], filenames=names)

    def test_historical_qwen_requires_exact_unordered_pairs_and_preserves_parse_failures(self):
        _, testing = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for p, token in zip(testing, ("90", "ERROR"), strict=True):
                (root / f"res_random_2100_{p.label}_seed2026_qwen3_vl-8b.txt").write_text(
                    f"C:\\source\\{p.image_b} C:\\source\\{p.image_a} {token}\n")
            scores, _ = read_historical_qwen(root, testing)
            self.assertEqual(scores, [Score("test-g", "ok", 90), Score("test-i", "error", None)])
            report = descriptive_scores(testing, scores)
            self.assertEqual(report["pooled"]["attempted"], 2)
            self.assertEqual(report["pooled"]["failures"], 1)
            self.assertIsNone(report["pooled"]["discrimination"])
            with self.assertRaises(InputError):
                read_historical_qwen(root, [replace(testing[0], image_a="OTHER"), testing[1]])
            path = root / "res_random_2100_genuine_seed2026_qwen3_vl-8b.txt"
            path.write_text(path.read_text().replace("90", "101"))
            with self.assertRaises(InputError):
                read_historical_qwen(root, testing)

    def test_paired_counts_use_arm_cutoffs_failures_and_unchanged_denominators(self):
        _, testing = fixture()
        a = [Score("test-g", "ok", .9), Score("test-i", "ok", .8)]
        b = [Score("test-g", "decode_error", None), Score("test-i", "ok", .1)]
        report = paired_dino(testing, a, b, .5, .2)
        self.assertEqual(report["pooled"]["genuine"]["frozen_only_accept"], 1)
        self.assertEqual(report["pooled"]["genuine"]["frozen_only_valid"], 1)
        self.assertEqual(report["pooled"]["impostor"]["adapted_minus_frozen_acceptance_rate"], -1)
        self.assertIsNone(report["left"]["genuine"]["adapted_minus_frozen_acceptance_rate"])
        with self.assertRaises(InputError):
            paired_dino(testing, a[:-1], b, .5, .2)

    def test_dino_report_replays_raw_scores_calibration_selection_and_test_claim(self):
        training, testing = fixture()
        config = FixedConfig(epochs=2, warmup_steps=0, device="cpu", precision="fp32")
        provenance = {"model_identity": {"arm": "frozen_dinov2"}, "code_commit": "synthetic",
                      "packages": {}, "hardware": {}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cal, test = root / "cal", root / "test"
            cal.mkdir(); test.mkdir()
            train_scores = [Score("train-g", "ok", .9), Score("train-i", "ok", .2)]
            test_scores = [Score("test-g", "ok", .8), Score("test-i", "ok", .1)]
            write_scores(cal / "scores.csv", train_scores)
            write_json(cal / "run_record.json", {"phase": "training_calibration", "config": asdict(config),
                "pairs_sha256": canonical_digest([asdict(p) for p in training]), "provenance": provenance})
            threshold = select_cutoff([.2], config.target_far)
            threshold["selected_on"] = "training_impostors_resubstitution"
            write_json(cal / "selection.json", {"selection_frozen": True, "test_scores_accessed": False,
                "checkpoint_rule": config.checkpoint_rule, "threshold_rule": config.threshold_rule,
                "model_identity": provenance["model_identity"], "threshold": threshold,
                "scores_sha256": digest(cal / "scores.csv"), "run_record_sha256": digest(cal / "run_record.json")})
            pairs_sha = canonical_digest([asdict(p) for p in testing])
            selection_sha = digest(cal / "selection.json")
            write_json(cal / "test_claim.json", {"output": str(test), "selection_sha256": selection_sha,
                                                "testing_pairs_sha256": pairs_sha})
            write_json(test / "run_record.json", {"phase": "held_out_pair_test", "provenance": provenance,
                "config": asdict(config), "testing_pairs_sha256": pairs_sha, "selection_sha256": selection_sha})
            write_scores(test / "scores.csv", test_scores)
            write_json(test / "aggregate.json", {"results": fixed_summary(testing, test_scores, .2), "threshold": threshold})
            write_json(test / "completion.json", {"complete": True, "scores_sha256": digest(test / "scores.csv"),
                                                  "aggregate_sha256": digest(test / "aggregate.json")})
            report, scores = read_dino(test, cal, training, testing, config, "frozen_dinov2")
            self.assertEqual(scores, test_scores)
            self.assertEqual(report["threshold"]["training_valid_impostors"], 1)
            self.assertFalse(any("development" in key for key in report["threshold"]))
            self.assertEqual(report["results"]["pooled"]["all_attempts_failure_as_reject"]["tar"], 1)
            with self.assertRaises(InputError):
                read_dino(test, cal, training, testing, config, "adapted_dinov2")
            with (test / "scores.csv").open("a") as handle:
                handle.write("unexpected,error,\n")
            with self.assertRaises(InputError):
                read_dino(test, cal, training, testing, config, "frozen_dinov2")


if __name__ == "__main__":
    unittest.main()
