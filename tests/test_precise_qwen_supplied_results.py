"""New collaborator summary consistency; synthetic pair/output values only."""

import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from audit_precise_qwen_results import summary_check, static_training_arguments, paired_saved_decisions  # noqa: E402
from compare_precise_qwen_ab import Decision, read_ab  # noqa: E402
from precise_matching.data import InputError  # noqa: E402
from precise_matching.fixed_pairs import FixedPair  # noqa: E402


class SuppliedQwenTests(unittest.TestCase):
    def test_summary_must_reproduce_preserved_decisions_and_exact_prompt(self):
        pairs = [FixedPair("g", "101_synthetic_a_15.jpeg", "101_synthetic_b_15.jpeg", "genuine", "thumbs", False),
                 FixedPair("i", "101_synthetic_a_15.jpeg", "102_synthetic_b_15.jpeg", "impostor", "thumbs", False)]
        decisions = {"g": Decision("g", "ok", True), "i": Decision("i", "ok", True)}
        prompt = {"system_prompt": "synthetic system", "user_prompt": "synthetic prompt"}
        summary = {**prompt, "prompt_sha256": hashlib.sha256(json.dumps(list(prompt.values()), ensure_ascii=False).encode()).hexdigest(),
            "genuine_counts": {"attempted": 1, "predicted_a": 1, "predicted_b": 0, "invalid": 0, "errors": 0},
            "impostor_counts": {"attempted": 1, "predicted_a": 1, "predicted_b": 0, "invalid": 0, "errors": 0},
            "confusion_matrix": {"TP": 1, "FN": 0, "FP": 1, "TN": 0},
            "metrics": {"overall_accuracy_all_attempts_percent": 50}}
        result = summary_check(summary, pairs, decisions, prompt)
        self.assertEqual(result["accuracy_all_attempts"], .5)
        for changed in ({"prompt_sha256": "wrong"}, {"user_prompt": "different"},
                        {"confusion_matrix": {"TP": 1, "FN": 0, "FP": 0, "TN": 1}},
                        {"metrics": {"overall_accuracy_all_attempts_percent": 100}}):
            with self.subTest(changed=changed), self.assertRaises(InputError):
                summary_check({**summary, **changed}, pairs, decisions, prompt)

    def test_new_invalid_token_is_an_explicit_failure_with_source_unchanged(self):
        pairs = [FixedPair("g", "101_synthetic_a_15.jpeg", "101_synthetic_b_15.jpeg", "genuine", "thumbs", False),
                 FixedPair("i", "101_synthetic_a_15.jpeg", "102_synthetic_b_15.jpeg", "impostor", "thumbs", False)]
        with tempfile.TemporaryDirectory() as tmp:
            paths = {label: Path(tmp) / (label + ".txt") for label in ("genuine", "impostor")}
            for p in pairs:paths[p.label].write_text(f"{p.image_a} {p.image_b} INVALID\n")
            before = {k: p.read_bytes() for k, p in paths.items()}
            decisions, tokens = read_ab(paths, pairs)
            self.assertEqual(tokens, {"INVALID": 2})
            self.assertTrue(all(r.accept is None and r.status == "invalid_response" for r in decisions.values()))
            self.assertEqual(before, {k: p.read_bytes() for k, p in paths.items()})

    def test_paired_comparison_uses_full_saved_score_list_and_frozen_cutoff(self):
        from precise_matching.evaluation import Score
        pairs = [FixedPair("g", "101_synthetic_a_15.jpeg", "101_synthetic_b_15.jpeg", "genuine", "thumbs", False),
                 FixedPair("i", "101_synthetic_a_15.jpeg", "102_synthetic_b_15.jpeg", "impostor", "thumbs", False)]
        dino = [Score("g", "ok", .4), Score("i", "ok", .1)]
        qwen = {"g": Decision("g", "ok", True), "i": Decision("i", "ok", True)}
        result = paired_saved_decisions(pairs, dino, qwen, .5)
        self.assertEqual(result["genuine"]["all_attempts_failure_as_reject"]["b_only_accept"], 1)
        self.assertEqual(result["impostor"]["all_attempts_failure_as_reject"]["b_only_accept"], 1)
        with self.assertRaises(InputError):paired_saved_decisions(pairs, dino[:-1], qwen, .5)

    def test_training_arguments_are_read_statically_without_pickle_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "training_args.bin"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("archive/data.pkl", pickle.dumps({"num_train_epochs": 3, "learning_rate": .0001,
                    "bf16": True, "load_best_model_at_end": False, "unrelated": "private fixture"}, protocol=2))
            self.assertEqual(static_training_arguments(path), {"num_train_epochs": 3, "learning_rate": .0001,
                             "bf16": True, "load_best_model_at_end": False})


if __name__ == "__main__":unittest.main()
