"""Synthetic full-decoding and provenance contracts; no private data in CI."""

import contextlib
import csv
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import audit_precise_readiness as readiness  # noqa: E402


class ReadinessTests(unittest.TestCase):
    def fixture(self, base):
        root = base / "images"
        root.mkdir()
        names = ["PRIVATEA_1_1_13.jpeg", "PRIVATEA_1_2_13.jpeg",
                 "PRIVATEB_1_1_13.jpeg", "PRIVATEB_1_2_13.jpeg"]
        for name, color in zip(names, (80, 80, 80, 130), strict=True):
            Image.new("L", (40, 30), color).save(root / name)
        (root / "PRIVATEC_1_15.jpeg").write_bytes(b"invalid image")
        sources = {}
        for role in readiness.SOURCES:
            path = base / (role + ".txt")
            selected = names[:2] if role.endswith("genuine") else names[1:3]
            path.write_text(" ".join(selected) + " 0\n", encoding="utf-8")
            sources[role] = path
        manifest = base / "history.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["img1_path", "img2_path"])
            writer.writeheader()
            writer.writerow({"img1_path": names[0], "img2_path": names[2]})
        return root, sources, manifest

    def test_full_decode_duplicates_privacy_and_private_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, sources, manifest = self.fixture(Path(tmp))
            originals = {p: p.read_bytes() for p in [*root.iterdir(), *sources.values(), manifest]}
            report, private = readiness.audit(root, sources, [manifest])
            self.assertEqual(report["images"], 5)
            self.assertEqual(report["decoded_images"], 4)
            self.assertEqual(report["decode_failures"], 1)
            self.assertEqual(report["source_byte_checks_passed"], 5)
            self.assertEqual(report["grayscale_decoded_duplicates"]["duplicate_groups"], 1)
            self.assertEqual(report["grayscale_decoded_duplicates"]["groups_spanning_subject_labels"], 1)
            self.assertEqual(report["historical_list_exposure"]["local_subjects"], 2)
            self.assertFalse(report["ready_for_split_creation"])
            for row in private["records"]:
                self.assertIsNone(row["capture_id"])
                self.assertEqual(row["exposure_reviewed"], "unknown")
                self.assertEqual(row["actual_training_use"], "unknown")
                self.assertNotIn("partition", row)
            text = json.dumps(report)
            for value in ("PRIVATE", tmp, ".jpeg", "image_a", "image_b"):
                self.assertNotIn(value, text)
            for path, content in originals.items():
                self.assertEqual(path.read_bytes(), content)

    def test_near_screen_matches_exhaustive_radius_and_mae(self):
        rng = random.Random(42)
        hashes = [rng.getrandbits(64) for _ in range(30)]
        hashes += [hashes[0] ^ 1, hashes[0] ^ (1 << 23) ^ (1 << 50)]
        rows = [{"filename": str(i), "subject_id": str(i), "position": "13",
                 "status": "ok", "dhash64": f"{h:016x}", "decoded_sha256": str(i)}
                for i, h in enumerate(hashes)]
        thumbnails = {i: bytes([80]) * 1024 for i in range(len(rows))}
        candidates, _, within = readiness.screen_near_duplicates(rows, thumbnails)
        expected = {(str(i), str(j)) for i in range(len(rows)) for j in range(i + 1, len(rows))
                    if (hashes[i] ^ hashes[j]).bit_count() <= 2}
        self.assertEqual({(r["image_a"], r["image_b"]) for r in candidates}, expected)
        self.assertEqual(within, len(expected))
        thumbnails[len(rows) - 1] = bytes([160]) * 1024
        screened, _, _ = readiness.screen_near_duplicates(rows, thumbnails)
        self.assertFalse(any(r["image_b"] == str(len(rows) - 1) for r in screened))

    def test_decoded_duplicates_ignore_jpeg_metadata_and_hash_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, sources, _ = self.fixture(Path(tmp))
            selected = sorted(root.glob("*.jpeg"))[:2]
            selected[1].write_bytes(selected[0].read_bytes() + b"trailing metadata")
            report, private = readiness.audit(root, sources)
            a, b = private["records"][:2]
            self.assertNotEqual(a["sha256"], b["sha256"])
            self.assertEqual(a["decoded_sha256"], b["decoded_sha256"])
            with Image.open(selected[0]) as image:
                gray = image.convert("L")
                expected = readiness.sha256(str(gray.size).encode() + gray.tobytes())
            self.assertEqual(a["decoded_sha256"], expected)
            self.assertEqual(report["byte_duplicates"]["redundant_files"], 1)
            self.assertEqual(report["grayscale_decoded_duplicates"]["redundant_files"], 2)

    def test_change_detection_and_lenient_decoder_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, sources, _ = self.fixture(Path(tmp))
            with patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", True), self.assertRaises(ValueError):
                readiness.audit(root, sources)
            original = readiness.screen_near_duplicates

            def mutate(records, thumbnails):
                sources["training_genuine"].write_text("changed", encoding="utf-8")
                return original(records, thumbnails)

            with patch.object(readiness, "screen_near_duplicates", side_effect=mutate), \
                    self.assertRaisesRegex(ValueError, "Metadata source changed"):
                readiness.audit(root, sources)

    def test_image_change_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, sources, _ = self.fixture(Path(tmp))
            original = readiness.screen_near_duplicates

            def mutate(records, thumbnails):
                next(root.glob("*.jpeg")).write_bytes(b"changed")
                return original(records, thumbnails)

            with patch.object(readiness, "screen_near_duplicates", side_effect=mutate), \
                    self.assertRaisesRegex(ValueError, "Image source changed"):
                readiness.audit(root, sources)
            (root / "link.jpeg").symlink_to(next(root.glob("*.jpeg")))
            with self.assertRaisesRegex(ValueError, "symlinks"):
                readiness.audit(root, sources)

    def test_ragged_historical_manifest_fails_without_image_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, sources, manifest = self.fixture(Path(tmp))
            manifest.write_text("img1_path,img2_path\nPRIVATEA_1_1_13.jpeg\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Incomplete historical image reference"):
                readiness.audit(root, sources, [manifest])

    def test_outputs_and_cli_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, sources, _ = self.fixture(base)
            report, private = readiness.audit(root, sources)
            public, local = base / "reports" / "new", base / "local" / "new"
            readiness.write_outputs(public, local, report, private)
            self.assertEqual([p.name for p in public.iterdir()], ["readiness.json"])
            self.assertIn("PRIVATE", (local / "readiness_records.json").read_text())
            with self.assertRaises(ValueError):
                readiness.write_outputs(public, local, report, private)
            for output, identifying in ((base / "outside", base / "local" / "valid"),
                                        (base / "reports" / "valid", base / "outside")):
                argv = ["audit", "--output-dir", str(output), "--private-dir", str(identifying)]
                for role, path in sources.items():
                    argv += ["--" + role.replace("_", "-"), str(path)]
                with patch.object(readiness, "ROOT", base), patch.object(sys, "argv", argv), \
                        patch.object(readiness, "audit") as read, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    readiness.main()
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
