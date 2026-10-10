"""Synthetic-image checks for inspection privacy, decoding and source preservation."""

import json
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))
import inspect_precise_images as inspection  # noqa: E402


class ImageInspectionTests(unittest.TestCase):
    def fixture(self, root):
        for subject in ("PRIVATEA", "PRIVATEB"):
            for position in ("13", "14"):
                for capture in (1, 2):
                    Image.new("L", (80, 60), 100).save(root / f"{subject}_1_{capture}_{position}.jpeg")
        Image.new("L", (60, 40), 150).save(root / "PRIVATEA_1_15.jpeg")
        (root / "PRIVATEC_1_1_13.jpeg").write_bytes(b"broken image")

    def test_headers_sample_and_privacy_with_unchanged_source_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            originals = {p.name: p.read_bytes() for p in root.iterdir()}
            report, records, _ = inspection.inspect(root, 2, 7)
            self.assertEqual(report["headers"]["jpeg_files"], 10)
            self.assertEqual(report["headers"]["unreadable"], 1)
            self.assertEqual(report["sample"]["images"], 9)
            self.assertEqual(report["sample"]["decoded"], 9)
            self.assertEqual(report["sample"]["exact_decoded_grayscale_duplicate_groups"], 1)
            self.assertEqual(report["sample"]["source_byte_checks_passed"], 9)
            self.assertNotIn("PRIVATE", json.dumps(report))
            self.assertNotIn(tmp, json.dumps(report))
            self.assertIn("PRIVATE", json.dumps(records))
            for name, content in originals.items():
                self.assertEqual((root / name).read_bytes(), content)

    def test_deterministic_selection_and_no_subject_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            a, private_a, _ = inspection.inspect(root, 1, 19)
            b, private_b, _ = inspection.inspect(root, 1, 19)
            self.assertEqual(a, b)
            self.assertEqual(private_a, private_b)
            self.assertNotIn("train", private_a[0])
            self.assertNotIn("pair_id", json.dumps(private_a))

    def test_existing_directories_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public, private = root / "public", root / "private"
            private.mkdir()
            original = private / "original.txt"
            original.write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                inspection.write_outputs(public, private, {}, [{}], [])
            self.assertFalse(public.exists())
            self.assertEqual(original.read_text(), "keep")

    def test_generated_image_files_are_only_in_private_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            report, records, previews = inspection.inspect(root, 1, 19)
            public, private = root / "public", root / "private"
            inspection.write_outputs(public, private, report, records, previews)
            self.assertEqual([p.name for p in public.iterdir()], ["inspection.json"])
            self.assertTrue((private / "inspection_records.json").is_file())
            self.assertTrue((private / "contact_sheet_01.png").is_file())
            self.assertNotIn("PRIVATE", (public / "inspection.json").read_text())

    def test_sample_decode_failure_is_counted_and_source_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            broken = root / "PRIVATED_1_15.jpeg"
            broken.write_bytes(b"not a JPEG")
            report, records, previews = inspection.inspect(root, 1, 19)
            self.assertEqual(report["sample"]["unreadable"], 1)
            failures = [r for r in records[0]["sample_records"] if "error" in r]
            self.assertEqual(len(failures), 1)
            self.assertTrue(failures[0]["source_bytes_unchanged"])
            self.assertEqual(broken.read_bytes(), b"not a JPEG")
            self.assertEqual(sum(preview is None for _, preview in previews), 1)
            self.assertNotIn("PRIVATE", json.dumps(report))

    def test_cli_rejects_outputs_outside_public_private_roots_before_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            choices = ((root / "outside", root / "local" / "review"),
                       (root / "reports" / "review", root / "outside"))
            for public, private in choices:
                args = ["inspect_precise_images.py", "--output-dir", str(public),
                        "--private-dir", str(private)]
                with patch.object(inspection, "ROOT", root), patch.object(sys, "argv", args), \
                     patch.object(inspection, "inspect") as read, \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        inspection.main()
                    self.assertEqual(error.exception.code, 2)
                    read.assert_not_called()
                self.assertFalse(public.exists())
                self.assertFalse(private.exists())


if __name__ == "__main__":
    unittest.main()
