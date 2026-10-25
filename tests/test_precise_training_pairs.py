"""Synthetic intake checks; never use private research inputs in tests."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import audit_precise_training_pairs as audit  # noqa: E402


class TrainingPairAuditTests(unittest.TestCase):
    def fixture(self, root):
        dataset = root / "dataset"
        dataset.mkdir()
        names = [f"SYNTH{subject}_1_{capture}_{position}.jpeg"
                 for subject, capture, position in
                 (("A", 1, 13), ("A", 2, 13), ("A", 3, 13), ("B", 1, 13),
                  ("B", 2, 13), ("C", 1, 15), ("C", 2, 15))]
        for name in names:
            (dataset / name).write_bytes(b"synthetic filename fixture only")
        a1, a2, a3, b1, b2, c1, c2 = names
        contents = {
            "training_genuine": f"{a1} {a2} 0\n{c1} {c2} 7 | 2 5 sum=7 nfing=2\n",
            "training_impostor": f"{a1} {b1} 1 | 0 0 0 1 sum=1 nfing=4\n",
            "testing_genuine": f"{a1} {a3} 9 | 2 2 2 3 sum=9 nfing=4\n",
            "testing_impostor": f"{a2} {b2} 0 | 0 0 0 0 sum=0 nfing=4\n",
        }
        sources = {}
        for role, content in contents.items():
            sources[role] = root / (role + ".txt")
            sources[role].write_text(content)
        return sources, dataset

    def test_fused_only_zero_retains_genuine_label_and_unknown_components(self):
        row = audit.parse_line("A_1_1_13.jpeg A_1_2_13.jpeg 0", "genuine", 49)
        self.assertEqual((row.label, row.fused, row.line), ("genuine", 0, 49))
        self.assertIsNone(row.fingers)
        self.assertIsNone(row.summed)
        self.assertIsNone(row.nfing)
        with self.assertRaises(ValueError):
            audit.parse_line("A_1_1_13.jpeg A_1_2_13.jpeg 0 |", "genuine", 1)

    def test_invalid_scores_and_components_are_rejected_without_private_text(self):
        for suffix in ("nan", "-1", "0.5", "0 extra", "3 | 1 2 sum=4 nfing=2"):
            with self.subTest(suffix=suffix), self.assertRaises(ValueError) as raised:
                audit.parse_line("SECRET_A SECRET_B " + suffix, "genuine", 7)
            self.assertNotIn("SECRET", str(raised.exception))

    def test_different_pairs_can_share_images_and_subjects(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources, dataset = self.fixture(Path(tmp))
            report, details = audit.audit(sources, dataset)
            overlap = report["training_testing_overlap"]["all_slaps"]
            self.assertEqual(overlap["shared_pairs"], 0)
            self.assertEqual(overlap["shared_images"], 2)
            self.assertEqual(overlap["shared_subjects"], 2)
            self.assertTrue(overlap["exact_pairs_disjoint"])
            self.assertFalse(overlap["filename_subjects_disjoint"])
            self.assertEqual(report["classes"]["training_genuine"]["fused_only_records"], 1)
            self.assertEqual(report["training_testing_overlap"]["four_finger_only"]["training_records"], 2)
            self.assertNotIn("SYNTH", json.dumps(report))
            self.assertIn("SYNTH", json.dumps(details))

    def test_reversed_pairs_and_windows_paths_detect_exact_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources, dataset = self.fixture(Path(tmp))
            sources["testing_genuine"].write_text(
                r"C:\private\SYNTHA_1_2_13.jpeg /private/SYNTHA_1_1_13.jpeg 8" + "\n")
            report, _ = audit.audit(sources, dataset)
            overlap = report["training_testing_overlap"]["all_slaps"]
            self.assertEqual(overlap["shared_pairs"], 1)
            self.assertFalse(overlap["exact_pairs_disjoint"])

    def test_bad_labels_missing_images_and_duplicate_pairs_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources, dataset = self.fixture(Path(tmp))
            line = "SYNTHA_1_1_13.jpeg ABSENT_1_1_13.jpeg 0\n"
            sources["training_genuine"].write_text(line + line)
            report, _ = audit.audit(sources, dataset)
            group = report["classes"]["training_genuine"]
            self.assertEqual(group["filename_label_mismatches"], 2)
            self.assertEqual(group["missing_local_image_basenames"], 1)
            self.assertEqual(group["duplicate_unordered_pair_excess"], 1)

    def test_deterministic_outputs_source_preservation_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources, dataset = self.fixture(root)
            before = {role: path.read_bytes() for role, path in sources.items()}
            first = audit.audit(sources, dataset)
            self.assertEqual(first, audit.audit(sources, dataset))
            self.assertEqual(before, {role: path.read_bytes() for role, path in sources.items()})
            public, private = root / "reports", root / "local"
            audit.write_outputs(public, private, *first)
            saved = (public / "audit.json").read_bytes()
            with self.assertRaises(ValueError):
                audit.write_outputs(public, private, {}, {})
            self.assertEqual(saved, (public / "audit.json").read_bytes())

    def test_changed_source_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources, dataset = self.fixture(Path(tmp))
            original = audit.read_source

            def change(path, label):
                rows, metadata = original(path, label)
                path.write_bytes(path.read_bytes() + b"\n")
                return rows, metadata

            with patch.object(audit, "read_source", side_effect=change), self.assertRaises(ValueError):
                audit.audit(sources, dataset)

    def test_cli_private_output_boundary_and_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources, dataset = self.fixture(root)
            args = ["audit"]
            for role, path in sources.items():
                args.extend(["--" + role.replace("_", "-"), str(path)])
            args.extend(["--dataset-root", str(dataset), "--output-dir", str(root / "reports/new"),
                         "--private-dir", str(root / "wrong")])
            with patch.object(audit, "ROOT", root), patch.object(sys, "argv", args), \
                 patch.object(audit, "audit") as read, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    audit.main()
                read.assert_not_called()
            args[-1] = str(root / "local/new")
            with patch.object(audit, "ROOT", root), patch.object(sys, "argv", args), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(), 0)
            report = json.loads((root / "reports/new/audit.json").read_text())
            self.assertEqual(report["source_byte_checks_passed"], 4)
            self.assertNotIn("SYNTH", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
