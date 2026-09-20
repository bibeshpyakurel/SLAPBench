"""Offline checks that setup downloads to the paths inference actually loads."""

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import model_registry  # noqa: E402
import run_verification  # noqa: E402
import setup_models  # noqa: E402


class ModelSetupTests(unittest.TestCase):
    def test_inference_paths_match_download_directory_names(self):
        for key, (_, folder) in model_registry.LOCAL_MODELS.items():
            with self.subTest(model=key):
                self.assertEqual(
                    run_verification.MODELS[key]["hf_id"],
                    str(model_registry.MODELS_DIR / folder),
                )
        self.assertEqual(run_verification.MODELS["pixtral"]["hf_id"],
                         model_registry.PIXTRAL_ID)

    def test_download_passes_the_same_paths_to_hugging_face(self):
        calls = []

        def fake_download(**kwargs):
            calls.append(kwargs)
            return kwargs.get("local_dir", "/cache/pixtral")

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(setup_models, "MODELS_DIR", Path(tmp)):
                with patch.dict(sys.modules, {
                    "huggingface_hub": types.SimpleNamespace(snapshot_download=fake_download)
                }):
                    setup_models.download_models()

            for key, (repo_id, folder) in model_registry.LOCAL_MODELS.items():
                with self.subTest(model=key):
                    self.assertIn({"repo_id": repo_id, "repo_type": "model",
                                   "local_dir": str(Path(tmp) / folder)}, calls)
            self.assertIn({"repo_id": model_registry.PIXTRAL_ID,
                           "repo_type": "model"}, calls)

    def test_dangling_symlink_is_not_replaced_by_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / "models"
            link.symlink_to(Path(tmp) / "unmounted-drive")
            with patch.object(setup_models, "MODELS_DIR", link):
                with patch.dict(sys.modules, {
                    "huggingface_hub": types.SimpleNamespace(snapshot_download=lambda **_: None)
                }):
                    with self.assertRaises(SystemExit):
                        setup_models.download_models()
            self.assertTrue(link.is_symlink())
            self.assertFalse(link.exists())

    def test_failed_download_is_reported_as_failure(self):
        def unavailable(**_):
            raise RuntimeError("repository unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(setup_models, "MODELS_DIR", Path(tmp)):
                with patch.dict(sys.modules, {
                    "huggingface_hub": types.SimpleNamespace(snapshot_download=unavailable)
                }):
                    with self.assertRaisesRegex(SystemExit, "Model downloads failed"):
                        setup_models.download_models()


if __name__ == "__main__":
    unittest.main()
