"""Synthetic shared-cohort checks, never private fingerprint inputs."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import audit_precise_shared_cohort as audit  # noqa: E402


class SharedCohortTests(unittest.TestCase):
    def fixture(self, root):
        intake, images = root / "intake", root / "images"
        (intake / "context").mkdir(parents=True)
        images.mkdir()
        # Numeric strings are invented test identities; different padding is deliberate.
        names = [f"{subject}_synthetic_{capture}_13.jpeg" for subject, capture in
                 (("001", "a"), ("001", "b"), ("1", "c"), ("001", "d"))]
        for name in names:
            (images / name).write_bytes(b"names only; not a fingerprint")
        a, b, c, d = names
        records = []
        for left, right, label, fused in ((a, b, "genuine", 0), (a, c, "impostor", 2)):
            records.append({"image_paths": [left, right], "pair_class": label,
                "correct_answer_option": {"genuine": "A", "impostor": "B"}[label],
                "finger_score_sum": 4, "verifinger_score": fused, "nfing": 4,
                "filename_suffix": "13", "source_line": 1})
        data = {"pairs": records, "counts": {"genuine": 1, "impostor": 1, "total": 2}}
        (intake / audit.TRAIN).write_text(json.dumps(data))
        (intake / audit.TEST["genuine"]).write_text(f"{a} {d} 0\n")
        (intake / audit.TEST["impostor"]).write_text(f"{b} {c} 0\n")
        (intake / audit.REFERENCE).write_text(
            f"{Path(a).with_suffix('.dat')} {Path(b).with_suffix('.dat')} 0 | 1 1 1 1 sum=4 nfing=4\n")
        return intake, images

    def test_overlap_and_numeric_disagreements_do_not_relabel_or_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            intake, images = self.fixture(Path(tmp))
            before = (intake / audit.TRAIN).read_bytes()
            report, issues = audit.reconcile(intake, images)
            self.assertEqual(report["training"]["numeric_id_label_disagreements"], 1)
            self.assertEqual(report["testing"]["numeric_id_label_disagreements"], 1)
            self.assertEqual(report["train_test_overlap"]["exact_pairs"], 0)
            self.assertEqual(report["train_test_overlap"]["images"], 3)
            self.assertEqual(len(issues), 2)
            self.assertTrue(all(case["provided_class"] == "impostor" for case in issues))
            self.assertNotIn("synthetic", json.dumps(report))
            self.assertEqual(before, (intake / audit.TRAIN).read_bytes())

    def test_positive_sum_retains_fused_zero_and_dat_mapping_has_line_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            intake, images = self.fixture(Path(tmp))
            report, _ = audit.reconcile(intake, images)
            self.assertEqual(report["training"]["retained_fused_zero_by_class"]["genuine"], 1)
            self.assertTrue(report["training"]["declared_class_counts_match"])
            self.assertTrue(report["full_genuine_reference"]["template_dat_names"])
            self.assertTrue(report["full_genuine_reference"]["matches_json_genuine_cohort"])
            self.assertEqual(report["full_genuine_reference"]["json_source_line_and_score_disagreements"], 0)

    def test_inconsistent_answer_and_nonpositive_selected_sum_are_rejected(self):
        for change in ({"correct_answer_option": "B"}, {"finger_score_sum": 0}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                intake, images = self.fixture(Path(tmp))
                data = json.loads((intake / audit.TRAIN).read_text())
                data["pairs"][0].update(change)
                (intake / audit.TRAIN).write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    audit.reconcile(intake, images)

    def test_cli_private_boundary_and_repeat_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intake, images = self.fixture(root)
            output, private = root / "reports/audit.json", root / "local/precise/review"
            args = ["--intake-dir", str(intake), "--images", str(images),
                    "--output", str(output), "--private-dir", str(private)]
            with patch.object(audit, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(args), 0)
            original = output.read_bytes()
            self.assertTrue((private / "label_review_cases.json").exists())
            with patch.object(audit, "ROOT", root), patch.object(audit, "reconcile") as read, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit.main(args), 1)
            read.assert_not_called()
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
