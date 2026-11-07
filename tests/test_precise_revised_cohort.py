"""Synthetic November-5 intake contracts; no real images or allocations."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import audit_precise_revised_cohort as audit  # noqa: E402


def fixture(root):
    intake, images = root / "intake", root / "images"
    intake.mkdir(); images.mkdir()
    records, phases, inventory, old_train, old_test = [], {"train": [], "test": []}, [], [], []
    for suffix, fingers in audit.HANDS.items():
        a, b, c, d, e, f, g = [f"{subject}_synthetic_{capture}_{suffix}.jpeg" for subject, capture in
                                (("001", "a"), ("001", "b"), ("1", "c"), ("002", "d"),
                                 ("003", "e"), ("003", "f"), ("004", "g"))]
        for name in (a, b, c, d, e, f, g):
            (images / name).write_bytes(b"not a decodable image; inventory-only intake")
            inventory.append({"filename": name})
        for phase, rows in (("train", ((a, b, "genuine", 0), (a, d, "impostor", 2))),
                            ("test", ((e, f, "genuine", 0), (e, g, "impostor", 2)))):
            for left, right, label, fused in rows:
                phases[phase].append((left, right, label, fused, fingers))
                if phase == "train":
                    records.append({"image_paths": [left, right], "pair_class": label,
                        "correct_answer_option": "A" if label == "genuine" else "B", "verifinger_score": fused,
                        "finger_score_sum": fingers, "nfing": fingers, "filename_suffix": suffix})
                target = old_train if phase == "test" else old_test
                target.append({"image_a": left, "image_b": right, "label": label})
    prompt = {"system_prompt": "synthetic system", "user_prompt": "synthetic A/B prompt", "options": {"A": "same", "B": "different"}}
    data = {**prompt, "pairs": records, "counts": {"genuine": 3, "impostor": 3, "total": 6},
        "suffix_counts": {s: {"genuine": 1, "impostor": 1} for s in audit.HANDS},
        "selection": {"train_numeric_groups": 2, "test_numeric_groups": 2, "forced_alias_groups_to_train": 1,
                      "seed": 2026, "test_fraction": .2, "test_pairs_per_class": 3,
                      "split_group_rule": "PRIVATE_SYNTHETIC_POLICY_EXAMPLE"}}
    (intake / audit.TRAIN_JSON).write_text(json.dumps(data))
    for role, name in audit.LISTS.items():
        phase, label = role.split("_")
        (intake / name).write_text("".join(f"{a} {b} {score} | {' '.join(['1'] * fingers)} sum={fingers} nfing={fingers}\n"
            for a, b, kind, score, fingers in phases[phase] if kind == label))
    inventory_path, bundle, record, earlier = [root / name for name in ("inventory.json", "bundle.json", "record.json", "earlier.json")]
    inventory_path.write_text(json.dumps({"records": inventory}))
    bundle.write_text(json.dumps({"scope": "shared_seen_image_fixed_pairs", "labels": "as_supplied",
                                 "training": old_train, "testing": old_test}))
    record.write_text(json.dumps({"provenance": {"input_sha256": {"bundle": audit.sha(bundle)}, "code_commit": "synthetic"}}))
    earlier.write_text(json.dumps(prompt))
    return intake, inventory_path, bundle, record, images, earlier


class RevisedCohortTests(unittest.TestCase):
    def test_disjoint_new_groups_do_not_erase_prior_training_exposure(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(Path(tmp))
            report, private = audit.reconcile(*paths[:5])
            self.assertEqual(report["new_train_test_overlap"], {"pairs": 0, "images": 0, "numeric_groups": 0})
            exposure = report["comparison_with_completed_benchmark"]["new_testing_previously_exposed_to_completed_dino_training"]
            self.assertEqual(exposure, {"pairs": 6, "images": 9, "numeric_groups": 2})
            self.assertTrue(report["training_json_matches_supplied_training_lists_and_scores"])
            self.assertEqual(report["alias_group_placement"]["in_new_testing"], 0)
            self.assertFalse(private["image_pixels_read"])
            self.assertFalse(private["approval_issued"])
            self.assertNotIn("jpeg", json.dumps(report))
            self.assertNotIn("PRIVATE_SYNTHETIC_POLICY_EXAMPLE", json.dumps(report))

    def test_retained_fused_zeros_and_json_list_score_disagreements_remain_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(Path(tmp))
            report, _ = audit.reconcile(*paths[:5])
            self.assertEqual(report["score_status"]["train_genuine"]["fused_zero"], 3)
            self.assertEqual(report["score_status"]["test_genuine"]["nonpositive_component_sum"], 0)
            p = paths[0] / audit.TRAIN_JSON
            data = json.loads(p.read_text());data["pairs"][0]["verifinger_score"] = 5
            p.write_text(json.dumps(data))
            report, _ = audit.reconcile(*paths[:5])
            self.assertFalse(report["training_json_matches_supplied_training_lists_and_scores"])

    def test_invalid_target_prior_bundle_change_and_missing_local_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fixture(Path(tmp))
            image = next(paths[4].iterdir());image.unlink()
            report, _ = audit.reconcile(*paths[:5])
            self.assertEqual(report["missing_local_image_files"], 1)
            original = paths[2].read_text();paths[2].write_text(original + "\n")
            with self.assertRaises(ValueError):audit.reconcile(*paths[:5])
            paths[2].write_text(original)
            p = paths[0] / audit.TRAIN_JSON
            data = json.loads(p.read_text());data["pairs"][0]["correct_answer_option"] = "B"
            p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):audit.reconcile(*paths[:5])

    def test_cli_private_boundary_prompt_check_and_immutable_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);paths = fixture(root)
            output, private = root / "reports/new.json", root / "local/precise/new-review"
            args = [value for name, path in zip(("intake", "inventory", "approved-bundle", "approved-record", "image-root", "earlier-training-json"), paths, strict=True)
                    for value in ("--" + name, str(path))]
            args += ["--output", str(output), "--private-dir", str(private)]
            source_hashes = {p: audit.sha(p) for p in paths[0].iterdir()}
            with patch.object(audit, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(args), 0)
            report = json.loads(output.read_text())
            self.assertTrue(all(report["prompts_equal_to_earlier_json"].values()))
            self.assertEqual(source_hashes, {p: audit.sha(p) for p in paths[0].iterdir()})
            with patch.object(audit, "ROOT", root), patch.object(audit, "reconcile") as read, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit.main(args), 1)
            read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
