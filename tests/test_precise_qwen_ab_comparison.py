"""Synthetic-only collaborator A/B ingestion and comparison contracts."""

import contextlib
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import compare_precise_qwen_ab as ab  # noqa: E402
from precise_matching.data import InputError  # noqa: E402
from precise_matching.evaluation import Score  # noqa: E402
from precise_matching.fixed_pairs import GROUP_SCOPE, SHARED_SCOPE, FixedConfig, FixedPair, TRAIN_FILE, canonical_digest  # noqa: E402
from precise_matching.training import digest  # noqa: E402


def pairs():
    return [FixedPair("g-right", "001_synthetic_a_13.jpeg", "001_synthetic_b_13.jpeg", "genuine", "right", False),
            FixedPair("i-left", "001_synthetic_a_14.jpeg", "002_synthetic_b_14.jpeg", "impostor", "left", False),
            FixedPair("g-thumb", "001_synthetic_a_15.jpeg", "001_synthetic_b_15.jpeg", "genuine", "thumbs", False),
            FixedPair("i-thumb", "001_synthetic_a_15.jpeg", "1_synthetic_c_15.jpeg", "impostor", "thumbs", True)]


def write_outputs(root, rows, tokens):
    paths = {label: root / (label + ".txt") for label in ("genuine", "impostor")}
    for label, path in paths.items():
        path.write_text("".join(f"C:\\source\\{p.image_b} C:\\source\\{p.image_a} {token}\n"
                                for p, token in zip(rows, tokens, strict=True) if p.label == label))
    return paths


def run_record(prompt, count):
    return {"model": "Qwen/Qwen3-VL-8B-Instruct", "base_revision": "a" * 40,
            "checkpoint_manifest_sha256": "b" * 64, "training_pairs_consumed": count,
            "prompt_sha256": canonical_digest(prompt), "checkpoint_selection_used_test_results": False,
            "environment": {"torch": "synthetic"},
            **{name: "synthetic fixture evidence only" for name in (
                "training_membership_evidence", "checkpoint_selection_evidence", "evaluation_preprocessing",
                "generation_configuration_evidence", "raw_responses_evidence")}}


class QwenABTests(unittest.TestCase):
    def test_supplied_tokens_are_truth_independent_and_source_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = pairs(); paths = write_outputs(Path(tmp), rows, ("B", "a", "A", "b"))
            original = {key: path.read_bytes() for key, path in paths.items()}
            outcomes, tokens = ab.read_ab(paths, rows)
            self.assertEqual(tokens, {"A": 1, "B": 1, "a": 1, "b": 1})
            report = ab.operating_summary(rows, outcomes)
            self.assertEqual(report["all_attempts_failure_as_reject"]["tar"], .5)
            self.assertEqual(report["all_attempts_failure_as_reject"]["far"], .5)
            self.assertEqual(original, {key: path.read_bytes() for key, path in paths.items()})
            self.assertNotIn("jpeg", json.dumps(report))

    def test_missing_duplicate_extra_wrong_class_and_numeric_score_outputs_fail(self):
        for change in ("missing", "duplicate", "extra", "class", "numeric"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                rows = pairs(); paths = write_outputs(Path(tmp), rows, ("A", "B", "A", "B"))
                text = paths["genuine"].read_text()
                if change == "missing": paths["genuine"].write_text(text.splitlines()[0] + "\n")
                elif change == "duplicate": paths["genuine"].write_text(text + text.splitlines()[0] + "\n")
                elif change == "extra": paths["genuine"].write_text(text + "OTHER ANOTHER A\n")
                elif change == "class": paths = {"genuine": paths["impostor"], "impostor": paths["genuine"]}
                elif change == "numeric": paths["genuine"].write_text(text.replace(" A\n", " 90\n"))
                with self.assertRaises(InputError):
                    ab.read_ab(paths, rows)

    def test_failed_attempts_stay_in_primary_denominators_and_binary_roc_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = pairs(); paths = write_outputs(Path(tmp), rows, ("UNKNOWN", "ERROR", "A", "B"))
            outcomes, _ = ab.read_ab(paths, rows)
            summary = ab.operating_summary(rows, outcomes)
            self.assertEqual(summary["failures"], {"genuine": 1, "impostor": 1})
            self.assertEqual(summary["all_attempts_failure_as_reject"]["tar"], .5)
            self.assertEqual(summary["valid_only"]["tar"], 1)
            dino = [Score(p.pair_id, "ok", .9 if p.label == "genuine" else .1) for p in rows]
            comparison = ab.compare_operating_points(rows, outcomes,
                {arm: dino for arm in ab.ARMS}, {arm: .5 for arm in ab.ARMS})
            self.assertIsNone(comparison["binary_qwen_roc_auc_eer"])
            group = comparison["paired_dino_to_qwen"]["frozen_dinov2"]["pooled"]["genuine"]
            self.assertEqual(group["attempts"], 2)
            self.assertEqual(group["coverage_counts"]["a_only_valid"], 1)
            self.assertEqual(group["all_attempts_failure_as_reject"]["acceptance_rate_difference_b_minus_a"], -.5)
            self.assertEqual(comparison["methods"]["qwen3_vl_trained_ab"]["excluding_flagged_test_pairs"]["attempts"]["impostor"], 1)

    def test_paired_cutoffs_remain_arm_specific_and_missing_dino_rows_fail(self):
        rows = pairs()
        qwen = {p.pair_id: ab.Decision(p.pair_id, "ok", True) for p in rows}
        dino = [Score(p.pair_id, "ok", .5) for p in rows]
        report = ab.compare_operating_points(rows, qwen, {arm: dino for arm in ab.ARMS},
                                             {"frozen_dinov2": .1, "adapted_dinov2": .9})
        self.assertEqual(report["methods"]["frozen_dinov2"]["pooled"]["all_attempts_failure_as_reject"]["tar"], 1)
        self.assertEqual(report["methods"]["adapted_dinov2"]["pooled"]["all_attempts_failure_as_reject"]["tar"], 0)
        with self.assertRaises(InputError):
            ab.compare_operating_points(rows, qwen, {arm: dino[:-1] for arm in ab.ARMS}, {arm: .5 for arm in ab.ARMS})

    def test_metadata_does_not_accept_defaults_unknown_identity_or_test_selection(self):
        prompt = {"system_prompt": "synthetic system", "user_prompt": "synthetic A/B prompt"}
        original = run_record(prompt, 4)
        ab.validate_run_record(original, prompt, 4)
        for update in ({"prompt_sha256": "c" * 64}, {"training_pairs_consumed": 3},
                       {"base_revision": "unknown"}, {"checkpoint_selection_used_test_results": True},
                       {"environment": {}}, {"raw_responses_evidence": ""}):
            with self.subTest(update=update), self.assertRaises(InputError):
                ab.validate_run_record({**original, **update}, prompt, 4)

    def test_cli_public_private_boundaries_fail_before_source_access(self):
        args = [value for name in ("bundle", "config", "training-json", "genuine-output", "impostor-output",
                                   "qwen-run-record", "review", "dino-run-root", "private-receipt", "output")
                for value in ("--" + name, "/tmp/PRIVATE_UNAVAILABLE")]
        with patch.object(ab, "digest") as read, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ab.main(args), 1)
        read.assert_not_called()

    def test_cli_synthetic_roundtrip_review_binding_overwrite_and_input_mutation(self):
        for scope in (SHARED_SCOPE, GROUP_SCOPE):
            with self.subTest(scope=scope):
                self.cli_roundtrip(scope)

    def cli_roundtrip(self, scope):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); private = root / "local/precise"; private.mkdir(parents=True)
            rows = pairs()
            if scope == GROUP_SCOPE:
                rows[-1] = replace(rows[-1], image_b="003_synthetic_c_15.jpeg", numeric_identity_disagreement=False)
            paths = write_outputs(private, rows, ("A", "B", "A", "B"))
            sources = {key: private / (key + ".json") for key in ("bundle", "config", "training_json", "qwen_run_record", "review")}
            sources.update({label + "_output": path for label, path in paths.items()})
            prompt = {"system_prompt": "synthetic system", "user_prompt": "synthetic prompt"}
            sources["training_json"].write_text(json.dumps(prompt))
            training = [replace(p, pair_id="train-" + p.pair_id,
                                image_a=p.image_a.replace("_synthetic_", "_training_")) for p in rows]
            if scope == GROUP_SCOPE:
                def unseen(name):
                    prefix, rest = name.split("_", 1)
                    return str(int(prefix) + 100) + "_" + rest
                training = [replace(p, image_a=unseen(p.image_a), image_b=unseen(p.image_b)) for p in training]
            sources["bundle"].write_text(json.dumps({"scope": scope, "labels": "as_supplied",
                "training": [asdict(p) for p in training], "testing": [asdict(p) for p in rows],
                "source_sha256": {ab.TRAIN_JSON if scope == GROUP_SCOPE else TRAIN_FILE: digest(sources["training_json"])}}))
            sources["config"].write_text(json.dumps(asdict(FixedConfig(warmup_steps=0))))
            record = run_record(prompt, len(rows)); record["training_json_sha256"] = digest(sources["training_json"])
            sources["qwen_run_record"].write_text(json.dumps(record))
            review = {"comparison_provenance_reviewed": True, "non_test_selection_reviewed": True,
                      "review_record": "Synthetic test only, not actual authority", "input_sha256": {
                          key: digest(path) for key, path in sources.items() if key != "review"}}
            sources["review"].write_text(json.dumps(review))
            dino = private / "runs"
            for name in ("frozen-test", "adapted-test", "frozen-calibration", "adapted-calibration"):
                d = dino / name; d.mkdir(parents=True)
                if name.endswith("test"):
                    (d / "run_record.json").write_text(json.dumps({"provenance": {**({"benchmark_scope": scope} if scope == GROUP_SCOPE else {}), "input_sha256": {
                        key: digest(sources[key]) for key in ("bundle", "config")}}}))
            output, receipt = root / "reports/ab.json", private / "receipt.json"
            args = [value for key, path in sources.items() for value in ("--" + key.replace("_", "-"), str(path))]
            args += ["--dino-run-root", str(dino), "--output", str(output), "--private-receipt", str(receipt)]
            scores = [Score(p.pair_id, "ok", .8 if p.label == "genuine" else .1) for p in rows]
            with patch.object(ab, "ROOT", root), patch.object(ab, "read_dino", return_value=({"threshold": {"cutoff": .5}}, scores)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ab.main(args), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["results"]["methods"]["qwen3_vl_trained_ab"]["pooled"]["all_attempts_failure_as_reject"]["tar"], 1)
            self.assertNotIn("jpeg", output.read_text())
            self.assertEqual(result["benchmark_scope"], scope)
            if scope == GROUP_SCOPE:
                self.assertEqual(result["dependence"]["train_test_overlap"], {"exact_pairs": 0, "images": 0, "numeric_groups": 0})
                self.assertNotIn("All test images/identity groups occur in training", output.read_text())
            with patch.object(ab, "ROOT", root), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(ab.main(args), 1)
            output2, receipt2 = root / "reports/second.json", private / "second-receipt.json"
            args2 = args[:-4] + ["--output", str(output2), "--private-receipt", str(receipt2)]
            reader = ab.read_ab

            def mutate(*values):
                result = reader(*values)
                paths["genuine"].write_text(paths["genuine"].read_text() + "\n")
                return result

            with patch.object(ab, "ROOT", root), patch.object(ab, "read_dino", return_value=({"threshold": {"cutoff": .5}}, scores)), \
                    patch.object(ab, "read_ab", side_effect=mutate), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(ab.main(args2), 1)
            self.assertFalse(output2.exists())
            self.assertFalse(receipt2.exists())


if __name__ == "__main__":
    unittest.main()
