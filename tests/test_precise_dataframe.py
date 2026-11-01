"""Annotation intake privacy, identity conflicts and restricted reader contracts."""

import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import audit_precise_dataframe as intake  # noqa: E402


@unittest.skipUnless(importlib.util.find_spec("pandas"), "Verification environment supplies pandas")
class DataframeIntakeTests(unittest.TestCase):
    def frame(self):
        import pandas as pd

        rows = []
        for name, fingers in (("syntheticA_x_y_13.jpg", range(2, 6)),
                              ("syntheticB_x_y_14.jpg", range(7, 11))):
            for finger in fingers:
                row = {field: 0.0 for field in intake.FIELDS}
                row.update(Dataset="Precise", Slap=name, SubjectID=938.0, Finger=float(finger))
                rows.append(row)
        frame = pd.DataFrame(rows, columns=intake.FIELDS).astype({"Dataset": object, "Slap": object})
        frame.columns = pd.Index(list(intake.FIELDS), dtype=object)
        return frame

    def test_subject_collapse_is_reported_without_identifiers_or_relabeling(self):
        frame = self.frame()
        before = frame.copy(deep=True)
        names = list(frame.Slap.unique()) + ["syntheticMissing_x_y_13.jpg"]
        report, private = intake.summarize(frame, names)
        self.assertEqual(report["precise"]["rows"], 8)
        self.assertEqual(report["precise"]["matched_local_images"], 2)
        self.assertEqual(report["precise"]["local_images_without_annotation"], 1)
        self.assertEqual(report["identity_mapping"]["dataframe_ids_spanning_multiple_filename_prefixes"], 1)
        self.assertEqual(report["identity_mapping"]["filename_prefix_groups"], 2)
        self.assertEqual(report["precise"]["finger_set_conflicts"], 0)
        self.assertEqual(len(private), 2)
        public = json.dumps(report)
        self.assertNotIn("synthetic", public)
        self.assertNotIn("938", public)
        self.assertFalse(report["split_ready"])
        self.assertTrue(all(row["identity_review"] == "unreviewed" for row in private))
        self.assertTrue(frame.equals(before))

    def test_reader_roundtrip_and_unknown_constructor_or_schema_refusal(self):
        frame, constructors = intake.restricted_frame(pickle.dumps(self.frame(), protocol=4))
        self.assertEqual(len(frame), 8)
        self.assertTrue(any(name == "DataFrame" for _, name in constructors))

        class Unsafe:
            def __reduce__(self):
                return eval, ("1 / 0",)

        with self.assertRaisesRegex(ValueError, "constructor"):
            intake.restricted_frame(pickle.dumps(Unsafe(), protocol=4))
        with self.assertRaisesRegex(ValueError, "schema"):
            intake.restricted_frame(pickle.dumps(self.frame().drop(columns="Finger"), protocol=4))
        with self.assertRaises(ValueError):
            intake.restricted_frame(pickle.dumps(self.frame(), protocol=4) + b"trailing")

    def test_padding_aliases_are_evidence_without_silent_canonicalization(self):
        frame = self.frame()
        frame["SubjectID"] = 7.0
        frame["Slap"] = frame["Slap"].replace({"syntheticA_x_y_13.jpg": "0007_x_y_13.jpg",
                                               "syntheticB_x_y_14.jpg": "7_x_y_14.jpg"})
        report, private = intake.summarize(frame, list(frame.Slap.unique()))
        self.assertEqual(report["precise"]["numeric_subject_label_matches"], 2)
        self.assertEqual(report["precise"]["exact_subject_label_matches"], 1)
        self.assertEqual(report["identity_mapping"]["filename_prefix_groups"], 2)
        self.assertEqual(report["identity_mapping"]["dataframe_id_groups"], 1)
        self.assertEqual({row["filename_subject"] for row in private}, {"0007", "7"})
        self.assertFalse(report["split_ready"])

    def test_unknown_capture_and_rotation_stay_unknown_and_conflicts_are_counted(self):
        frame = self.frame()
        frame.loc[0, "augmented_angle"] = float("nan")
        frame.loc[0, "Finger"] = 12.0
        frame.loc[0, "SubjectID"] = 939.0
        report, private = intake.summarize(frame, list(frame.Slap.unique()))
        self.assertEqual(report["precise"]["multiple_subjects_per_slap"], 1)
        self.assertEqual(report["precise"]["finger_set_conflicts"], 1)
        self.assertEqual(report["precise"]["missing_rotation_rows"], 1)
        self.assertFalse(report["capture_session_ppi_fields_present"])
        self.assertTrue(all(row["independent_capture"] == "unknown" for row in private))
        frame.loc[0, "SubjectID"] = 939.5
        with self.assertRaises(ValueError):
            intake.summarize(frame, list(frame.Slap.unique()))

    def test_intake_preserves_sources_and_refuses_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, images = root / "annotations.P", root / "images"
            images.mkdir()
            source.write_bytes(pickle.dumps(self.frame(), protocol=4))
            for name in self.frame().Slap.unique():
                (images / name).write_bytes(b"not-an-image; intake reads names only")
            before = source.read_bytes()
            output, private = root / "reports/intake", root / "local/precise/intake"
            with patch.object(intake, "ROOT", root):
                report = intake.audit(source, images, output, private)
                self.assertFalse(report["split_ready"])
                self.assertTrue((private / "per_image_review.csv").exists())
                self.assertNotIn("synthetic", (output / "dataframe_summary.json").read_text())
                self.assertEqual(source.read_bytes(), before)
                with patch.object(intake, "restricted_frame") as read, self.assertRaises(ValueError):
                    intake.audit(source, images, output, private)
                read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
