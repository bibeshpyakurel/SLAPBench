"""Offline integrity and storage contracts for separately authorized model setup."""

import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import local_model_backends  # noqa: E402
import model_registry  # noqa: E402
import run_verification  # noqa: E402
from setup_next_models import main, save_location, verify_snapshot  # noqa: E402


class NextModelSetupTests(unittest.TestCase):
    def test_selected_download_uses_pinned_revision_without_other_models(self):
        calls = []
        hub = SimpleNamespace(HfApi=lambda: None,
                              snapshot_download=lambda **kwargs: calls.append(kwargs))
        with tempfile.TemporaryDirectory() as tmp:
            args = ["setup_next_models.py", "--download", "--models", "qwen35",
                    "--models-dir", tmp]
            with patch.object(sys, "argv", args), patch.dict(sys.modules, {"huggingface_hub": hub}):
                main()
            repo, folder, revision = model_registry.ADDITIONAL_MODELS["qwen35"]
            self.assertEqual(calls, [{"repo_id": repo, "revision": revision,
                                     "local_dir": str(Path(tmp) / folder), "max_workers": 3}])

    def test_existing_verification_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            report.write_text("original")
            with patch.object(sys, "argv", ["setup_next_models.py", "--verify", "--report", str(report)]):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual(report.read_text(), "original")

    def test_all_new_models_have_distinct_registered_runner_paths(self):
        for key, (_, folder, revision) in model_registry.ADDITIONAL_MODELS.items():
            with self.subTest(model=key):
                config = run_verification.MODELS[key]
                self.assertEqual(config["hf_id"], str(model_registry.ADDITIONAL_MODELS_DIR / folder))
                self.assertEqual(config["revision"], revision)
                self.assertNotIn(key, model_registry.LOCAL_MODELS)

    def test_machine_local_location_roundtrip_and_environment_override(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            config = Path(tmp) / "storage.json"
            root = Path(tmp) / "weights"
            save_location(root, config)
            self.assertEqual(model_registry.additional_models_dir(config), root)
            with patch.dict(os.environ, {"SLAPBENCH_MODELS_DIR": tmp}):
                self.assertEqual(model_registry.additional_models_dir(config), Path(tmp))

    def test_verification_detects_same_size_weight_corruption(self):
        content = b"original model fixture"
        item = SimpleNamespace(rfilename="weights.safetensors", size=len(content),
                               lfs=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest()))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / item.rfilename
            path.write_bytes(content)
            self.assertEqual(verify_snapshot(directory, [item])[0]["bytes"], len(content))
            path.write_bytes(b"X" + content[1:])
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_snapshot(directory, [item])

    def test_regular_files_use_git_blob_hash_and_reject_truncation(self):
        content = b'{"model_type":"fixture"}'
        blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content, usedforsecurity=False)
        item = SimpleNamespace(rfilename="config.json", size=len(content),
                               lfs=None, blob_id=blob.hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / item.rfilename
            path.write_bytes(content)
            self.assertEqual(len(verify_snapshot(directory, [item])), 1)
            path.write_bytes(content[:-1])
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                verify_snapshot(directory, [item])

    def test_missing_files_and_empty_snapshots_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            item = SimpleNamespace(rfilename="missing")
            with self.assertRaisesRegex(ValueError, "Missing snapshot"):
                verify_snapshot(Path(tmp), [item])
            with self.assertRaisesRegex(ValueError, "no files"):
                verify_snapshot(Path(tmp), [])

    def test_unavailable_cuda_fails_before_any_weights_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, {"torch": fake_torch, "transformers": SimpleNamespace()}):
                with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                    local_model_backends.load_local_model("qwen35", {"hf_id": tmp})


if __name__ == "__main__":
    unittest.main()
