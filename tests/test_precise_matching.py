"""Synthetic metadata and scores only; never construct real subject partitions."""

import contextlib
import csv
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))
from precise_matching import __main__ as cli  # noqa: E402
from precise_matching import data, evaluation  # noqa: E402


def fixtures():
    images, pairs = [], []
    for partition in data.PARTITIONS:
        for subject in ("PRIVATE_A", "PRIVATE_B"):
            for capture in ("one", "two"):
                key = f"{partition}_{subject}_{capture}"
                raw = hashlib.sha256(("raw:" + key).encode()).hexdigest()
                decoded = hashlib.sha256(("decoded:" + key).encode()).hexdigest()
                images.append(data.Image(key, partition + subject, "right", key + "_capture",
                                         partition, raw, decoded, "true", "true"))
        a, b, c, d = images[-4:]
        for index, (left, right, label) in enumerate(((a, b, "genuine"), (c, d, "genuine"),
                                                     (a, c, "impostor"), (b, d, "impostor"))):
            pairs.append(data.Pair(f"{partition}_pair_{index}", left.image_id, right.image_id, label))
    values = {"development": (0.8, 0.9, 0.1, 0.2), "test": (0.7, 0.2, 0.3, 0.0)}
    scores = [evaluation.Score(f"{partition}_pair_{index}", "ok", score)
              for partition, numbers in values.items() for index, score in enumerate(numbers)]
    return images, pairs, scores


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ProtocolTests(unittest.TestCase):
    def test_valid_metadata_is_aggregate_and_does_not_claim_training_readiness(self):
        images, pairs, _ = fixtures()
        report = data.validate_protocol(images, pairs)
        self.assertTrue(report["metadata_checks_passed"])
        self.assertEqual(report["partitions"]["test"], {
            "images": 4, "subjects": 2, "genuine_pairs": 2, "impostor_pairs": 2})
        self.assertNotIn("PRIVATE", json.dumps(report))
        self.assertIn("authorize training", " ".join(report["limits"]))

    def test_subject_separation_includes_unused_images_and_other_hand(self):
        images, pairs, _ = fixtures()
        images.append(replace(images[0], image_id="unused", hand="left", partition="test",
                              capture_id="unused", sha256="a" * 64, decoded_sha256="b" * 64))
        report = data.validate_protocol(images, pairs)
        self.assertEqual(report["blockers"]["subject_partition_conflicts"], 1)

    def test_capture_and_digest_conflicts_across_partitions(self):
        for field, issue in (("capture_id", "capture_metadata_conflicts"),
                             ("sha256", "sha256_partition_conflicts"),
                             ("decoded_sha256", "decoded_sha256_partition_conflicts")):
            with self.subTest(field=field):
                images, pairs, _ = fixtures()
                images[4] = replace(images[4], **{field: getattr(images[0], field)})
                self.assertGreater(data.validate_protocol(images, pairs)["blockers"][issue], 0)

    def test_unknown_capture_exposure_and_hash_evidence_block(self):
        images, pairs, _ = fixtures()
        images[0] = replace(images[0], capture_id="", capture_verified="unknown",
                            exposure_reviewed="false", sha256="", decoded_sha256="")
        report = data.validate_protocol(images, pairs)
        self.assertFalse(report["metadata_checks_passed"])
        for key in ("capture_id_missing_images", "capture_unverified_images",
                    "exposure_unreviewed_images", "sha256_missing_images", "decoded_sha256_missing_images"):
            self.assertEqual(report["blockers"][key], 1)

    def test_same_capture_reencoding_is_not_an_independent_genuine(self):
        images, pairs, _ = fixtures()
        images[1] = replace(images[1], capture_id=images[0].capture_id)
        self.assertEqual(data.validate_protocol(images, pairs)["blockers"]["same_capture_pairs"], 1)

    def test_decoded_duplicates_with_different_bytes_block_genuine(self):
        images, pairs, _ = fixtures()
        images[1] = replace(images[1], decoded_sha256=images[0].decoded_sha256)
        blockers = data.validate_protocol(images, pairs)["blockers"]
        self.assertEqual(blockers["decoded_sha256_identical_pairs"], 1)
        self.assertEqual(blockers["decoded_sha256_capture_conflicts"], 1)

    def test_duplicate_image_and_parent_capture_comparisons_block(self):
        images, pairs, _ = fixtures()
        pair = pairs[0]
        pairs.append(replace(pair, pair_id="reversed", image_a=pair.image_b, image_b=pair.image_a))
        blockers = data.validate_protocol(images, pairs)["blockers"]
        self.assertEqual(blockers["duplicate_unordered_image_pairs"], 1)
        self.assertEqual(blockers["duplicate_unordered_capture_pairs"], 1)

    def test_alternate_images_do_not_hide_reused_capture_pair(self):
        images, pairs, _ = fixtures()
        images.extend([replace(images[0], image_id="alternate_a"), replace(images[1], image_id="alternate_b")])
        pairs.append(data.Pair("alternate_pair", "alternate_a", "alternate_b", "genuine"))
        blockers = data.validate_protocol(images, pairs)["blockers"]
        self.assertEqual(blockers["duplicate_unordered_capture_pairs"], 1)
        self.assertNotIn("duplicate_unordered_image_pairs", blockers)

    def test_pair_integrity_errors_and_missing_classes(self):
        mutations = (
            (replace(fixtures()[1][0], image_b="absent"), "pairs_with_unknown_images"),
            (replace(fixtures()[1][0], image_b=fixtures()[0][4].image_id), "cross_partition_pairs"),
            (replace(fixtures()[1][0], image_b=fixtures()[0][0].image_id), "self_image_pairs"),
            (replace(fixtures()[1][0], label="impostor"), "pair_label_conflicts"),
        )
        for pair, issue in mutations:
            with self.subTest(issue=issue):
                images, pairs, _ = fixtures()
                pairs[0] = pair
                self.assertIn(issue, data.validate_protocol(images, pairs)["blockers"])
        images, pairs, _ = fixtures()
        images[0] = replace(images[0], hand="left")
        self.assertIn("mixed_hand_pairs", data.validate_protocol(images, pairs)["blockers"])
        report = data.validate_protocol(images, [pair for pair in pairs if pair.label == "genuine"])
        self.assertEqual(report["blockers"]["missing_evaluation_classes"], 2)

    def test_csv_schema_rejects_bad_hand_flags_digest_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, pairs, _ = fixtures()
            write_csv(root / "pairs.csv", data.PAIR_FIELDS, [asdict(row) for row in pairs])
            for changes in ({"hand": "thumbs"}, {"sha256": "not-a-hash"},
                            {"exposure_reviewed": "yes"}, {"partition": "validation"},
                            {"image_id": images[1].image_id}):
                with self.subTest(changes=changes):
                    altered = [replace(images[0], **changes), *images[1:]]
                    write_csv(root / "images.csv", data.IMAGE_FIELDS, [asdict(row) for row in altered])
                    with self.assertRaises(data.InputError):
                        data.load_protocol(root / "images.csv", root / "pairs.csv")

    def test_csv_rejects_repeated_headers_ragged_rows_whitespace_and_empty_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            for source in ("a,a\nx,y\n", "a,b\nx\n", "a,b\nx,y,z\n", "a,b\n x,y\n", "a,b\n"):
                with self.subTest(source=source):
                    path.write_text(source)
                    with self.assertRaises(data.InputError):
                        data.read_table(path, ("a", "b"))

    def test_in_memory_calls_enforce_the_same_schema_as_csv_loading(self):
        images, pairs, scores = fixtures()
        for candidate in ([*images, images[0]], [replace(images[0], hand="thumbs"), *images[1:]]):
            with self.subTest(rows=len(candidate)), self.assertRaises(data.InputError):
                data.validate_protocol(candidate, pairs)
            with self.assertRaises(data.InputError):
                evaluation.evaluate(candidate, pairs, scores, "0")


class EvaluationTests(unittest.TestCase):
    def test_cutoff_is_development_only_and_test_far_is_not_retuned(self):
        images, pairs, scores = fixtures()
        report = evaluation.evaluate(images, pairs, scores, "0")
        self.assertEqual(report["threshold"]["cutoff"], 0.2)
        self.assertEqual(report["development"]["valid_scores_only"]["far"], 0)
        self.assertEqual(report["test"]["valid_scores_only"]["far"], 0.5)
        self.assertEqual(report["test"]["valid_scores_only"]["tar"], 0.5)
        changed = [replace(row, score=1e100) if row.pair_id.startswith("test") else row for row in scores]
        self.assertEqual(evaluation.evaluate(images, pairs, changed, "0")["threshold"], report["threshold"])

    def test_decimal_far_budget_ties_and_arbitrary_finite_score_scale(self):
        threshold = evaluation.select_cutoff([2, 2, 1, 0], "0.25")
        self.assertEqual(threshold["allowed_development_false_accepts"], 1)
        self.assertEqual(threshold["development_false_accepts"], 0)
        self.assertEqual(evaluation.select_cutoff([2, 2, 1, 0], "0.5")["cutoff"], 1)
        self.assertEqual(evaluation.select_cutoff([1.0] * 100, "0.29")["allowed_development_false_accepts"], 29)
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        self.assertEqual(evaluation.select_cutoff([maximum, -maximum], "0")["cutoff"], maximum)
        self.assertEqual(evaluation.select_cutoff([-8, -3], "0")["cutoff"], -3)

    def test_invalid_far_and_nonfinite_scores_are_rejected(self):
        for target in ("-0.1", "1", "nan", "inf", "1/0", "garbage", 0.1):
            with self.subTest(target=target), self.assertRaises(data.InputError):
                evaluation.select_cutoff([0.1], target)
        for numbers in ([], [math.nan], [math.inf], [-math.inf]):
            with self.subTest(numbers=numbers), self.assertRaises(data.InputError):
                evaluation.select_cutoff(numbers, "0")

    def test_failures_remain_in_denominators_and_zero_is_a_valid_score(self):
        images, pairs, scores = fixtures()
        scores[4] = replace(scores[4], status="encoder_error", score=None)
        scores[6] = replace(scores[6], status="decode_error", score=None)
        report = evaluation.evaluate(images, pairs, scores, "0")
        test = report["test"]
        self.assertEqual(test["failures"], {"genuine": 1, "impostor": 1})
        self.assertEqual(test["coverage"], {"genuine": 0.5, "impostor": 0.5})
        self.assertEqual(test["valid_scores_only"]["impostor_comparisons"], 1)
        self.assertEqual(test["all_attempts_failure_as_reject"]["impostor_comparisons"], 2)
        self.assertEqual(test["status_counts"]["ok"], 2)

    def test_all_test_failures_are_reported_without_fabricated_curves(self):
        images, pairs, scores = fixtures()
        failed = [replace(row, status="error", score=None) if row.pair_id.startswith("test") else row
                  for row in scores]
        test = evaluation.evaluate(images, pairs, failed, "0")["test"]
        self.assertIsNone(test["descriptive_valid_score_discrimination"])
        self.assertIsNone(test["valid_scores_only"]["far"])
        self.assertEqual(test["all_attempts_failure_as_reject"]["tar"], 0)

    def test_development_failures_do_not_relax_false_accept_budget(self):
        images, pairs, scores = fixtures()
        scores[2] = replace(scores[2], status="error", score=None)
        threshold = evaluation.evaluate(images, pairs, scores, "0.5")["threshold"]
        self.assertEqual(threshold["development_valid_impostors"], 1)
        self.assertEqual(threshold["allowed_development_false_accepts"], 0)
        scores[3] = replace(scores[3], status="error", score=None)
        with self.assertRaises(data.InputError):
            evaluation.evaluate(images, pairs, scores, "0.5")

    def test_missing_duplicate_extra_training_and_nonfinite_score_rows_rejected(self):
        images, pairs, scores = fixtures()
        invalid = (scores[:-1], scores + [scores[0]],
                   scores + [replace(scores[0], pair_id="unknown")],
                   scores + [replace(scores[0], pair_id="train_pair_0")],
                   [replace(scores[0], score=math.nan), *scores[1:]],
                   [replace(scores[0], status="error"), *scores[1:]])
        for rows in invalid:
            with self.subTest(rows=len(rows)), self.assertRaises(data.InputError):
                evaluation.evaluate(images, pairs, rows, "0")
        images[0] = replace(images[0], capture_verified="unknown")
        with self.assertRaises(data.InputError):
            evaluation.evaluate(images, pairs, scores, "0")

    def test_score_csv_rejects_imputation_and_invalid_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            for status, score in (("ok", "nan"), ("ok", "inf"), ("ok", ""), ("ok", "x"),
                                  ("error", "0"), ("unknown", "")):
                with self.subTest(status=status, score=score):
                    write_csv(path, evaluation.SCORE_FIELDS, [{"pair_id": "private", "status": status, "score": score}])
                    with self.assertRaises(data.InputError):
                        evaluation.load_scores(path)

    def test_known_roc_auc_eer_perfect_reversed_and_tied(self):
        for genuine, impostor, auc, eer in (([2, 3], [0, 1], 1, 0),
                                            ([0, 1], [2, 3], 0, 1), ([2], [2], 0.5, 0.5)):
            with self.subTest(genuine=genuine):
                result = evaluation.discrimination(genuine, impostor)
                self.assertEqual(result["auc"], auc)
                self.assertEqual(result["eer_linear_interpolation"], eer)
                self.assertEqual(result["roc_det_rates"][-1], {"far": 1, "tar": 1, "frr": 0})


class CliTests(unittest.TestCase):
    def inputs(self, root):
        images, pairs, scores = fixtures()
        for name, fields, rows in (("images", data.IMAGE_FIELDS, images), ("pairs", data.PAIR_FIELDS, pairs),
                                   ("scores", evaluation.SCORE_FIELDS, scores)):
            write_csv(root / f"{name}.csv", fields, [asdict(row) for row in rows])
        return ["--images", str(root / "images.csv"), "--pairs", str(root / "pairs.csv")]

    def test_end_to_end_preserves_inputs_and_exports_only_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.inputs(root)
            before = {path.name: path.read_bytes() for path in root.glob("*.csv")}
            output = root / "reports" / "synthetic" / "evaluation.json"
            with patch.object(cli, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                result = cli.main(["evaluate", *args, "--scores", str(root / "scores.csv"),
                                   "--target-far", "0", "--output", str(output)])
            self.assertEqual(result, 0)
            content = output.read_text()
            self.assertNotIn("PRIVATE", content)
            self.assertNotIn(tmp, content)
            self.assertNotIn("pair_", content)
            self.assertEqual(json.loads(content)["test"]["valid_scores_only"]["far"], 0.5)
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.glob("*.csv")})

    def test_validation_blockers_export_counts_and_skip_score_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.inputs(root)
            content = (root / "images.csv").read_text().replace(",true,true", ",unknown,true", 1)
            (root / "images.csv").write_text(content)
            output = root / "reports" / "blockers.json"
            with patch.object(cli, "ROOT", root), patch.object(cli, "load_scores") as load, \
                 contextlib.redirect_stdout(io.StringIO()):
                result = cli.main(["evaluate", *args, "--scores", str(root / "scores.csv"),
                                   "--target-far", "0", "--output", str(output)])
            self.assertEqual(result, 1)
            load.assert_not_called()
            report = json.loads(output.read_text())
            self.assertEqual(report["blockers"], {"capture_unverified_images": 1})
            self.assertNotIn("threshold", report)

    def test_output_boundary_overwrite_and_symlink_escape_refused_before_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "reports"
            reports.mkdir()
            existing = reports / "keep.json"
            existing.write_text("original")
            (reports / "escape").symlink_to(root, target_is_directory=True)
            for output in (root / "public.json", existing, reports / "escape" / "outside.json"):
                with self.subTest(output=output), patch.object(cli, "ROOT", root), \
                     patch.object(cli, "load_protocol") as load, contextlib.redirect_stderr(io.StringIO()):
                    result = cli.main(["validate", "--images", "absent", "--pairs", "absent", "--output", str(output)])
                self.assertEqual(result, 2)
                load.assert_not_called()
            self.assertEqual(existing.read_text(), "original")

    def test_errors_do_not_disclose_private_paths_or_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "reports" / "error.json"
            error = io.StringIO()
            with patch.object(cli, "ROOT", root), contextlib.redirect_stderr(error):
                result = cli.main(["validate", "--images", str(root / "PRIVATE.csv"), "--pairs", "absent", "--output", str(output)])
            self.assertEqual(result, 2)
            self.assertNotIn("PRIVATE", error.getvalue())
            self.assertNotIn(tmp, error.getvalue())
            self.assertFalse(output.exists())

    def test_input_mutation_prevents_report_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self.inputs(root)
            original = data.load_protocol

            def mutate(*paths):
                loaded = original(*paths)
                paths[0].write_text(paths[0].read_text() + "\n")
                return loaded

            output = root / "reports" / "mutation.json"
            with patch.object(cli, "ROOT", root), patch.object(cli, "load_protocol", side_effect=mutate), \
                 contextlib.redirect_stderr(io.StringIO()):
                result = cli.main(["validate", *args, "--output", str(output)])
            self.assertEqual(result, 2)
            self.assertFalse(output.exists())

    def test_module_entry_point_help_runs_without_model_dependencies(self):
        result = subprocess.run([sys.executable, "-m", "precise_matching", "--help"],
                                cwd=CODE_DIR, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0)
        self.assertIn("validate,evaluate", result.stdout)


if __name__ == "__main__":
    unittest.main()
