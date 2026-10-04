"""Offline inventory contracts: aggregate privacy, byte duplicates and no writes."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))
import audit_precise_dataset as audit  # noqa: E402


class PreciseInventoryTests(unittest.TestCase):
    def test_duplicate_content_and_label_conflicts_without_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Use identifiers without underscores to match the source convention.
            sources = {
                "PRIVATEA_1_1_13.jpeg": b"same bytes",
                "PRIVATEA_2_13.jpeg": b"different bytes",
                "PRIVATEB_1_1_13.jpeg": b"same bytes",
                "PRIVATEA_1_1_14.jpeg": b"same bytes",
                "PRIVATEA_1_1_15.jpeg": b"thumb bytes",
                "INVALID_PRIVATE.jpeg": b"unrecognized",
            }
            for name, content in sources.items():
                (root / name).write_bytes(content)
            (root / "nested").mkdir()
            (root / "nested/PRIVATEC_1_1_13.jpeg").write_bytes(b"not inspected")
            report = audit.inventory(root)
            self.assertEqual(report["jpeg_images"], 6)
            self.assertEqual(report["unrecognized_filename_count"], 1)
            self.assertEqual(report["uninspected_subdirectories"], 1)
            self.assertEqual(report["four_finger_slap_subjects"], 2)
            self.assertEqual(report["slap_subjects_with_both_hands"], 1)
            self.assertEqual(report["positions"]["13"]["groups_with_multiple_distinct_byte_contents"], 1)
            self.assertEqual(report["exact_byte_duplicates"], {
                "unique_contents": 4, "duplicate_content_groups": 1,
                "redundant_files": 2, "content_groups_spanning_subject_labels": 1,
                "content_groups_spanning_position_labels": 1,
            })
            rendered = json.dumps(report)
            self.assertNotIn("PRIVATE", rendered)
            self.assertNotIn(tmp, rendered)
            for name, content in sources.items():
                self.assertEqual((root / name).read_bytes(), content)

    def test_duplicate_files_do_not_count_as_distinct_impressions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in (1, 2):
                (root / f"PRIVATE_{n}_1_13.jpeg").write_bytes(b"identical")
            position = audit.inventory(root)["positions"]["13"]
            self.assertEqual(position["groups_with_multiple_files"], 1)
            self.assertEqual(position["groups_with_multiple_distinct_byte_contents"], 0)

    def test_cli_refuses_to_overwrite_existing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "report.json"
            output.write_text("original", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(CODE_DIR / "audit_precise_dataset.py"),
                 "--root", str(Path(tmp) / "missing"), "--output", str(output)],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "original")
            self.assertIn("Output already exists", result.stderr)

    def test_read_failure_does_not_emit_source_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "PRIVATE_1_1_13.jpeg"
            source.write_bytes(b"bytes")
            stderr = io.StringIO()
            with patch.object(sys, "argv", ["audit", "--root", str(root)]):
                with patch.object(Path, "open", side_effect=OSError(str(source))):
                    with contextlib.redirect_stderr(stderr):
                        self.assertEqual(audit.main(), 1)
            self.assertNotIn("PRIVATE", stderr.getvalue())
            self.assertNotIn(tmp, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
