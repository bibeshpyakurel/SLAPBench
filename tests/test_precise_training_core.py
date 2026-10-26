"""Synthetic matcher contracts; no private dataset or network requests in tests."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import check_precise_training_runtime as runtime  # noqa: E402
from precise_matching.data import InputError  # noqa: E402
from precise_matching.encoders import configure_trainable, cosine_scores, encode, validate_checkpoint  # noqa: E402
from precise_matching.images import image_tensor, prepare_image  # noqa: E402
from precise_matching.losses import verification_loss  # noqa: E402


class PreprocessingTests(unittest.TestCase):
    def image(self, root):
        path = root / "synthetic.png"
        Image.new("L", (1600, 1500), 0).save(path)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_whole_image_retained_with_centered_white_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, digest = self.image(Path(tmp))
            before = path.read_bytes()
            image = prepare_image(path, digest)
            self.assertEqual((image.mode, image.size), ("RGB", (448, 448)))
            self.assertEqual(image.getpixel((200, 13)), (255, 255, 255))
            self.assertEqual(image.getpixel((200, 14)), (0, 0, 0))
            self.assertEqual(image.getpixel((200, 433)), (0, 0, 0))
            self.assertEqual(image.getpixel((200, 434)), (255, 255, 255))
            self.assertEqual(before, path.read_bytes())

    def test_wrong_digest_and_permissive_decoder_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, digest = self.image(Path(tmp))
            for bad in ("", "BAD", "f" * 64):
                with self.assertRaises(InputError):
                    prepare_image(path, bad)
            with patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", True), self.assertRaises(InputError):
                prepare_image(path, digest)

    def test_bad_bytes_report_no_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PRIVATE_SYNTHETIC_NAME"
            path.write_bytes(b"invalid image")
            with self.assertRaises(InputError) as error:
                prepare_image(path, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertNotIn(path.name, str(error.exception))

    def test_decoded_digest_is_checked_before_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, digest = self.image(Path(tmp))
            with Image.open(path) as source:
                gray = source.convert("L")
                decoded = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
            prepare_image(path, digest, decoded)
            with self.assertRaises(InputError):
                prepare_image(path, digest, "0" * 64)


class EncoderBoundaryTests(unittest.TestCase):
    def test_only_selected_layers_become_trainable(self):
        class Parameter:
            requires_grad = True

            def requires_grad_(self, value):
                self.requires_grad = value

            def numel(self):
                return 3

        names = ["embeddings.weight", "encoder.layer.9.weight", "encoder.layer.10.weight",
                 "encoder.layer.11.weight", "layernorm.weight"]
        params = [(name, Parameter()) for name in names]
        model = SimpleNamespace(named_parameters=lambda: params, train=lambda mode: None)
        self.assertEqual(configure_trainable(model, True)["trainable_parameters"], 9)
        self.assertEqual([p.requires_grad for _, p in params], [False, False, True, True, True])
        self.assertEqual(configure_trainable(model, False)["trainable_parameters"], 0)

    def test_missing_groups_and_altered_checkpoint_rejected(self):
        with self.assertRaises(InputError):
            configure_trainable(SimpleNamespace(named_parameters=lambda: []), True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            config = {"model_type": "dinov2", "hidden_size": 384, "num_hidden_layers": 12,
                      "num_attention_heads": 6, "patch_size": 14}
            (path / "config.json").write_text(json.dumps(config))
            (path / "model.safetensors").write_bytes(b"synthetic checkpoint bytes")
            with self.assertRaises(InputError):
                validate_checkpoint(path)
            digest = hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest()
            config_digest = hashlib.sha256((path / "config.json").read_bytes()).hexdigest()
            with patch("precise_matching.encoders.WEIGHTS_SHA256", digest), \
                    patch("precise_matching.encoders.CONFIG_SHA256", config_digest):
                validate_checkpoint(path)
                config["hidden_size"] = 385
                (path / "config.json").write_text(json.dumps(config))
                with self.assertRaises(InputError):
                    validate_checkpoint(path)

    def test_runtime_refuses_output_escape_and_overwrite_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(runtime, "ROOT", root):
                for output, private in [(root / "bad.json", root / "local/run"),
                                        (root / "reports/run.json", root / "public")]:
                    with self.assertRaises(InputError):
                        runtime.output_paths(output, private)
                path = root / "reports/run.json"
                path.parent.mkdir()
                path.write_text("original")
                with self.assertRaises(InputError):
                    runtime.output_paths(path, root / "local/run")
                self.assertEqual(path.read_text(), "original")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch integration runs in the isolated matcher environment")
class TorchContractTests(unittest.TestCase):
    def test_pair_loss_counts_and_expected_separation(self):
        import torch

        embeddings = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
        args = (["a", "a", "b", "b"], ["a1", "a2", "b1", "b2"], ["left"] * 4)
        loss, counts = verification_loss(embeddings, *args)
        self.assertEqual(counts, {"positive_pairs": 2, "negative_pairs": 4})
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_invalid_hand_capture_classes_and_normalization_rejected(self):
        import torch

        embeddings = torch.tensor([[1., 0.]] * 4)
        cases = [(["a"] * 4, ["1", "2", "3", "4"], ["left"] * 4),
                 (["a", "a", "b", "b"], ["1", "1", "3", "4"], ["left"] * 4),
                 (["a", "a", "b", "b"], ["1", "2", "3", "4"], ["left", "right", "left", "left"])]
        for args in cases:
            with self.assertRaises(InputError):
                verification_loss(embeddings, *args)
        with self.assertRaises(InputError):
            verification_loss(embeddings * 2, ["a", "a", "b", "b"], list("1234"), ["left"] * 4)

    def test_sixteen_image_design_counts_and_tensor_scaling(self):
        import torch

        embeddings = torch.nn.functional.normalize(torch.randn(16, 8), dim=1)
        _, counts = verification_loss(embeddings, [str(i // 2) for i in range(16)],
                                      [str(i) for i in range(16)], ["right"] * 16)
        self.assertEqual(counts, {"positive_pairs": 8, "negative_pairs": 112})
        tensor = image_tensor(Image.new("RGB", (448, 448), (0, 0, 0)))
        self.assertEqual(tuple(tensor.shape), (3, 448, 448))
        self.assertAlmostEqual(float(tensor[0, 0, 0]), -0.485 / 0.229, places=5)

    def test_encoder_normalizes_cls_and_rejects_zero_and_nan(self):
        import torch

        class FakeEncoder:
            def __init__(self, vector):
                self.vector = vector

            def __call__(self, **kwargs):
                return SimpleNamespace(last_hidden_state=self.vector)

        pixels = torch.zeros(2, 3, 4, 4)
        values = torch.tensor([[[3., 4.]], [[3., 4.]]])
        torch.testing.assert_close(encode(FakeEncoder(values), pixels),
                                   torch.tensor([[0.6, 0.8], [0.6, 0.8]]))
        for vector in (values * 0, values * float("nan")):
            with self.assertRaises(InputError):
                encode(FakeEncoder(vector), pixels)

    def test_continuous_similarity_and_shape_contract(self):
        import torch

        a = torch.tensor([[1., 0.], [0., 1.]])
        b = torch.tensor([[-1., 0.], [0.6, 0.8]])
        torch.testing.assert_close(cosine_scores(a, b), torch.tensor([-1., 0.8]))
        with self.assertRaises(InputError):
            cosine_scores(a, b * 2)
        with self.assertRaises(InputError):
            encode(SimpleNamespace(__call__=lambda **kw: None), torch.zeros(0, 3, 14, 14))


if __name__ == "__main__":
    unittest.main()
