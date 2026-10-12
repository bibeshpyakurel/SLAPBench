"""Synthetic score checks for threshold ties, export integrity and privacy."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))
import audit_verifinger_scores as audit  # noqa: E402


class VeriFingerAuditTests(unittest.TestCase):
    def fixture(self, root):
        dataset = root / "dataset"
        dataset.mkdir()
        genuine = root / "genuine.txt"
        impostor = root / "impostor.txt"
        genuine.write_text(
            r"C:\private\PRIVA_1_1_13.jpeg C:\private\PRIVA_1_2_13.jpeg 9 | 2 2 2 3 sum=9 nfing=4"
            "\n/private/PRIVA_1_1_15.jpeg /private/PRIVA_1_2_15.jpeg 3 | 1 2 sum=3 nfing=2\n")
        impostor.write_text(
            "/private/PRIVA_1_1_13.jpeg /private/PRIVB_1_1_13.jpeg 1 | 0 0 0 1 sum=1 nfing=4\n"
            "/private/PRIVA_1_1_15.jpeg /private/PRIVB_1_1_15.jpeg 0 | 0 0 sum=0 nfing=2\n")
        for subject, capture, position in (("PRIVA", 1, 13), ("PRIVA", 2, 13),
                                           ("PRIVA", 1, 15), ("PRIVA", 2, 15),
                                           ("PRIVB", 1, 13), ("PRIVB", 1, 15)):
            (dataset / f"{subject}_1_{capture}_{position}.jpeg").write_bytes(b"synthetic filename fixture")
        return genuine, impostor, dataset

    def test_parser_and_windows_basename_normalization(self):
        row = audit.parse_line(r"C:\private\PRIVA_1_1_15.jpeg /private/PRIVA_1_2_15.jpeg 7 | 2 5 sum=7 nfing=2", "genuine", 1)
        self.assertEqual(row.names, ("PRIVA_1_1_15.jpeg", "PRIVA_1_2_15.jpeg"))
        self.assertEqual(row.position, "15")
        self.assertEqual(row.fingers, (2, 5))

    def test_rejects_bad_sum_count_duplicate_fields_and_invalid_scores(self):
        for tail in ("1 2 sum=4 nfing=2", "1 2 sum=3 nfing=4",
                     "1 2 sum=3 nfing=2 sum=3", "1 2 sum=3",
                     "1 -2 sum=0 nfing=2", "1 2.5 sum=3 nfing=2"):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                audit.parse_line("PRIVA_1_1_15.jpeg PRIVA_1_2_15.jpeg 3 | " + tail, "genuine", 1)

    def test_operating_point_respects_integer_error_budget_and_ties(self):
        g, i = [3, 5], [2, 2, 1, 0]
        zero = audit.operating_point(g, i, "0.0001")
        self.assertEqual((zero["threshold"], zero["false_accepts"], zero["genuine_accepts"]), (3, 0, 2))
        tied = audit.operating_point(g, i, "0.25")
        self.assertEqual((tied["allowed_false_accepts"], tied["false_accepts"]), (1, 0))
        half = audit.operating_point(g, i, "0.5")
        self.assertEqual((half["threshold"], half["false_accepts"], half["actual_far"]), (2, 2, 0.5))
        with self.assertRaises(ValueError):
            audit.operating_point(g, [], "0.001")

    def test_known_perfect_and_tied_roc_auc_eer(self):
        perfect = audit.discrimination(audit.curve([4, 6], [0, 2]))
        self.assertEqual(perfect, {"auc": 1.0, "eer_linear_interpolation": 0.0})
        tied = audit.discrimination(audit.curve([2], [2]))
        self.assertEqual(tied, {"auc": 0.5, "eer_linear_interpolation": 0.5})

    def test_audit_preserves_sources_and_separates_thumb_four_finger_and_private_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genuine, impostor, dataset = self.fixture(root)
            before = genuine.read_bytes(), impostor.read_bytes()
            report, curves, records = audit.audit(genuine, impostor, dataset)
            self.assertEqual(before, (genuine.read_bytes(), impostor.read_bytes()))
            self.assertEqual(report["source_byte_checks_passed"], 2)
            self.assertEqual(report["combined"]["unique_image_basenames"], 6)
            self.assertEqual(report["classes"]["genuine"]["missing_local_basenames"], 0)
            self.assertEqual(report["slices"]["four_finger_only"]["genuines"], 1)
            self.assertEqual(report["slices"]["thumbs"]["genuines"], 1)
            self.assertNotIn("PRIVA", json.dumps(report))
            self.assertNotIn("private", json.dumps(curves))
            self.assertIn("PRIVA", json.dumps(records))
            public, private = root / "reports", root / "local"
            audit.write_outputs(public, private, report, curves, records)
            self.assertEqual({p.name for p in public.iterdir()}, {"audit.json", "curves.csv"})
            self.assertNotIn("PRIVA", (public / "curves.csv").read_text())
            self.assertNotIn(b"\r", (public / "curves.csv").read_bytes())

    def test_reference_pair_overlap_is_unordered_and_excludes_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            genuine, impostor, dataset = self.fixture(root)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=("img1_path", "img2_path", "label", "category"))
                writer.writeheader()
                writer.writerow({"img1_path": "PRIVA_1_2_13.jpeg", "img2_path": "PRIVA_1_1_13.jpeg", "label": "mated", "category": "primary_genuine"})
                writer.writerow({"img1_path": "PRIVA_1_1_15.jpeg", "img2_path": "PRIVB_1_1_15.jpeg", "label": "non-mated", "category": "diagnostic"})
            report, _, _ = audit.audit(genuine, impostor, dataset, manifest)
            self.assertEqual(report["reference_manifest_overlap"]["genuine"]["shared_unordered_basename_pairs"], 1)
            self.assertEqual(report["reference_manifest_overlap"]["impostor"]["reference_unique_primary_pairs"], 0)

    def test_overwrite_refusal_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private = root / "private"
            private.mkdir()
            sentinel = private / "keep.txt"
            sentinel.write_text("keep")
            with self.assertRaises(ValueError):
                audit.write_outputs(root / "public", private, {}, [], {})
            self.assertEqual(sentinel.read_text(), "keep")
            self.assertFalse((root / "public").exists())

    def test_cli_rejects_identifying_output_outside_local_before_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["audit", "--genuine", "g", "--impostor", "i", "--output-dir", str(root / "reports" / "new"), "--private-dir", str(root / "public")]
            with patch.object(audit, "ROOT", root), patch.object(sys, "argv", args), \
                 patch.object(audit, "audit") as read, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    audit.main()
                self.assertEqual(error.exception.code, 2)
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
