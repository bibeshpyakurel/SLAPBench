"""Numeric-group fixed-pair contracts; synthetic inputs only."""

import contextlib
from dataclasses import asdict, replace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from precise_matching.data import InputError  # noqa: E402
from precise_matching.fixed_pairs import (  # noqa: E402
    GROUP_SCOPE, SHARED_SCOPE, REVIEW_FLAGS, FixedConfig, source_pair, validate_group_holdout,
    prepare_bundle, load_bundle, reviewed_shared_inputs,
)
from precise_matching.fixed_evaluation import test_fixed  # noqa: E402
from precise_matching.training import digest  # noqa: E402
from precise_matching import hardware  # noqa: E402
import run_precise_shared_pairs as entry  # noqa: E402
import audit_precise_revised_cohort as audit  # noqa: E402
import test_precise_revised_cohort as intake_fixtures  # noqa: E402


def pair_fixture():
    phases = []
    for phase, a, b in (("train", "001", "002"), ("test", "003", "004")):
        rows = []
        for suffix in ("13", "14", "15"):
            for left, right, label in ((f"{a}_synthetic_a_{suffix}.png", f"{a}_synthetic_b_{suffix}.png", "genuine"),
                                       (f"{a}_synthetic_a_{suffix}.png", f"{b}_synthetic_c_{suffix}.png", "impostor")):
                rows.append(source_pair(len(rows) + 1, left, right, label, phase))
        phases.append(rows)
    return phases


class GroupScopeTests(unittest.TestCase):
    def test_group_separation_catches_padding_alias_leakage_and_false_hand_metadata(self):
        training, testing = pair_fixture()
        report = validate_group_holdout(training, testing)
        self.assertEqual(report["shared_numeric_groups"], 0)
        self.assertFalse(report["biological_identity_independence_confirmed"])
        changed = [replace(p, image_a=p.image_a.replace("003_", "1_"),
                           image_b=p.image_b.replace("003_", "1_")) for p in testing]
        with self.assertRaises(InputError):
            validate_group_holdout(training, changed)
        with self.assertRaises(InputError):
            validate_group_holdout(training, [replace(testing[0], hand="thumbs"), *testing[1:]])

    def test_prepare_uses_exact_new_sources_and_load_cannot_bypass_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intake, inventory, old, record, images, _ = intake_fixtures.fixture(root)
            value = json.loads(inventory.read_text())
            for row in value["records"]:
                row.update(status="ok", sha256="a" * 64, decoded_sha256="b" * 64)
            inventory.write_text(json.dumps(value))
            report, _ = audit.reconcile(intake, inventory, old, record, images)
            audit_path = root / "audit.json";audit_path.write_text(json.dumps(report))
            output = root / "new-bundle.json"
            result = prepare_bundle(intake, inventory, audit_path, output, scope=GROUP_SCOPE)
            self.assertFalse(result["image_pixels_read"])
            bundle, training, testing, _ = load_bundle(output, images)
            self.assertEqual(bundle["scope"], GROUP_SCOPE)
            self.assertEqual(len(training), 6);self.assertEqual(len(testing), 6)
            bundle["testing"][0]["image_a"] = bundle["training"][0]["image_a"]
            output.write_text(json.dumps(bundle))
            with self.assertRaises(InputError):load_bundle(output, images)

    def test_review_scope_must_match_bundle_and_group_test_guard_precedes_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, config, review = [root / name for name in ("bundle.json", "config.json", "review.json")]
            bundle.write_text(json.dumps({"scope": GROUP_SCOPE}))
            config.write_text(json.dumps(asdict(FixedConfig())))
            record = {flag: True for flag in REVIEW_FLAGS}
            record.update(scope=GROUP_SCOPE, authorized_phases=["train", "calibrate", "test"], review_record="synthetic review",
                          input_sha256={"bundle": digest(bundle), "config": digest(config)})
            review.write_text(json.dumps(record))
            reviewed_shared_inputs(bundle, config, review, "train")
            record["scope"] = SHARED_SCOPE;review.write_text(json.dumps(record))
            with self.assertRaises(InputError):reviewed_shared_inputs(bundle, config, review, "train")
            training, testing = pair_fixture()
            with patch("precise_matching.fixed_evaluation.score_fixed") as score, self.assertRaises(InputError):
                test_fixed(None, training, testing, None, FixedConfig(), root / "none", root / "test", {}, scope=GROUP_SCOPE)
            score.assert_not_called()


class DriverMetadataTests(unittest.TestCase):
    def test_loaded_module_supplies_metadata_when_management_binary_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "version"
            path.write_text("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.99.02 Release Build\n")
            with patch.object(hardware.shutil, "which", return_value=None):
                self.assertEqual(hardware.cuda_driver_metadata(path), {
                    "driver_version": "595.99.02", "driver_metadata_source": "proc_driver_nvidia"})
                path.write_text("unrecognized version")
                with self.assertRaises(InputError):hardware.cuda_driver_metadata(path)

    def test_management_failure_is_not_hidden_by_another_metadata_source(self):
        with patch.object(hardware.shutil, "which", return_value="/synthetic/nvidia-smi"), \
                patch.object(hardware.subprocess, "check_output", return_value="595.99.02\n"):
            self.assertEqual(hardware.cuda_driver_metadata()["driver_metadata_source"], "nvidia-smi")
        with patch.object(hardware.shutil, "which", return_value="/synthetic/nvidia-smi"), \
                patch.object(hardware.subprocess, "check_output", side_effect=subprocess.CalledProcessError(1, "synthetic")), \
                self.assertRaises(subprocess.CalledProcessError):
            hardware.cuda_driver_metadata()


@unittest.skipUnless(importlib.util.find_spec("torch"), "Runs in matcher CPU CI")
class GroupRuntimeTests(unittest.TestCase):
    def test_cli_fresh_training_calibration_and_both_arm_tests_have_group_scope(self):
        import test_precise_training_fixed_pairs as runtime_fixtures
        from PIL import Image
        helper = runtime_fixtures.FixedRuntimeTests();helper.setUp()
        training, testing = pair_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);private = root / "local/precise";private.mkdir(parents=True)
            images = root / "images";images.mkdir()
            lock = root / "code/precise_matching/environment/requirements-linux-x86_64-py312-cu124.lock"
            lock.parent.mkdir(parents=True);lock.write_text("synthetic lock")
            code = root / "code/synthetic.py";code.write_text("# synthetic source")
            mapping = {}
            keys = sorted({key for p in training + testing for key in (p.image_a, p.image_b)})
            for i, key in enumerate(keys):
                image = Image.new("L", (4, 4), 10 + i * 10)
                path = images / key;image.save(path)
                mapping[key] = {"path": key, "sha256": digest(path),
                                "decoded_sha256": hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()}
            bundle, config, review = [private / name for name in ("bundle.json", "config.json", "review.json")]
            bundle.write_text(json.dumps({"schema_version": 1, "scope": GROUP_SCOPE, "labels": "as_supplied",
                "images": mapping, "training": [asdict(p) for p in training], "testing": [asdict(p) for p in testing]}))
            cfg = FixedConfig(epochs=2, batch_pairs=2, accumulation=2, warmup_steps=0, device="cpu", precision="fp32")
            config.write_text(json.dumps(asdict(cfg)))
            record = {flag: True for flag in REVIEW_FLAGS}
            record.update(scope=GROUP_SCOPE, authorized_phases=["train", "calibrate", "test"], review_record="synthetic user review",
                          input_sha256={"bundle": digest(bundle), "config": digest(config)})
            review.write_text(json.dumps(record))
            common = [value for name, path in (("bundle", bundle), ("config", config), ("review", review),
                      ("source-root", images), ("weights", private / "weights")) for value in ("--" + name, str(path))]
            run = private / "runs/train"
            with patch.object(entry, "ROOT", root), patch.object(entry, "git_output", return_value="synthetic"), \
                    patch.object(entry, "package_versions", return_value={}), \
                    patch.object(entry, "source_record", return_value={"code/synthetic.py": digest(code)}), \
                    patch.object(entry, "load_encoder", side_effect=lambda *a, **kw: helper.model()), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main(["train", *common, "--output", str(run)]), 0)
                for arm in ("frozen_dinov2", "adapted_dinov2"):
                    cal, test = private / ("runs/" + arm + "-cal"), private / ("runs/" + arm + "-test")
                    extra = ["--arm", arm] + (["--training-run", str(run)] if arm == "adapted_dinov2" else [])
                    self.assertEqual(entry.main(["calibrate", *common, *extra, "--output", str(cal)]), 0)
                    self.assertEqual(entry.main(["test", *common, *extra, "--calibration", str(cal), "--output", str(test)]), 0)
                    aggregate = json.loads((test / "aggregate.json").read_text())
                    self.assertEqual(aggregate["scope"], "held_out_numeric_prefix_groups_identity_semantics_unconfirmed")
                    self.assertFalse(any("All testing images" in value for value in aggregate["limits"]))


if __name__ == "__main__":
    unittest.main()
