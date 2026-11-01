"""CUDA preflight gates and synthetic update contracts; CI never loads weights."""

import contextlib
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import check_precise_gpu_runtime as runtime  # noqa: E402
import check_precise_training_runtime as original  # noqa: E402
from precise_matching.data import InputError  # noqa: E402
from precise_matching.training import Config  # noqa: E402


class EarlyGpuProbeGates(unittest.TestCase):
    def test_existing_outputs_untouched_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "local/existing"
            existing.mkdir(parents=True)
            (existing / "keep.txt").write_text("preserve")
            args = ["--weights", "unavailable", "--config", "unavailable",
                    "--output", str(root / "reports/capacity.json"), "--private-dir", str(existing)]
            with patch.object(original, "ROOT", root), patch.object(runtime, "load_encoder") as load, \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(runtime.main(args), 1)
            load.assert_not_called()
            self.assertEqual(list(p.name for p in existing.iterdir()), ["keep.txt"])

    def test_cpu_configuration_is_not_a_gpu_capacity_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(json.dumps(asdict(Config(device="cpu", precision="fp32"))))
            private = root / "local/probe"
            with patch.object(original, "ROOT", root), patch.object(runtime, "run_gpu_probe") as probe, \
                    contextlib.redirect_stderr(io.StringIO()):
                result = runtime.main(["--weights", "unavailable", "--config", str(config),
                                       "--output", str(root / "reports/capacity.json"), "--private-dir", str(private)])
            self.assertEqual(result, 1)
            probe.assert_not_called()
            self.assertFalse(private.exists())


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU matcher CI supplies torch")
class SyntheticGpuProbeContracts(unittest.TestCase):
    def test_cuda_and_bf16_gates_precede_weight_loading(self):
        for available, bf16 in ((False, True), (True, False)):
            with self.subTest(cuda=available), patch("torch.cuda.is_available", return_value=available), \
                    patch("torch.cuda.is_bf16_supported", return_value=bf16), \
                    patch.object(runtime, "load_encoder") as load, self.assertRaises(InputError):
                runtime.run_gpu_probe(Path("unavailable"), Config(), Path("unavailable"))
            load.assert_not_called()

    def model(self):
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = torch.nn.Linear(3, 4).requires_grad_(False)
                self.head = torch.nn.Linear(4, 8)

            def forward(self, pixel_values):
                values = self.head(self.frozen(pixel_values.mean(dim=(2, 3))))
                return SimpleNamespace(last_hidden_state=values.unsqueeze(1))

        torch.manual_seed(23)
        return Tiny()

    def test_accumulated_updates_preserve_frozen_weights_and_both_hand_batches(self):
        config = Config(epochs=1, microbatches=2, subjects_per_batch=2, accumulation=2,
                        warmup_steps=0, device="cpu", precision="fp32")
        model, observe = self.model(), Mock()
        before = original.frozen_digest(model)
        hands = []
        loss = runtime.verification_loss

        def checked_loss(vectors, subjects, captures, hand, margin):
            hands.append(set(hand))
            return loss(vectors, subjects, captures, hand, margin)

        with patch.object(runtime, "verification_loss", side_effect=checked_loss):
            report = runtime.synthetic_updates(model, config, side=14, observe=observe)
        self.assertEqual(report["batch_pairs"], {"positive_pairs": 2, "negative_pairs": 4})
        self.assertEqual(report["optimizer_updates"], 2)
        self.assertEqual(report["microbatches"], 4)
        self.assertEqual(report["optimizer_state_entries"], 2)
        self.assertGreater(report["changed_trainable_tensors"], 0)
        self.assertTrue(report["losses_finite"] and report["gradients_finite"])
        self.assertEqual(original.frozen_digest(model), before)
        self.assertEqual(observe.call_count, 4)
        self.assertEqual(hands, [{"left"}, {"right"}, {"left"}, {"right"}])

    def test_frozen_only_model_and_single_update_are_refused(self):
        config = Config(device="cpu", precision="fp32")
        with self.assertRaises(InputError):
            runtime.synthetic_updates(self.model().requires_grad_(False), config, side=14)
        with self.assertRaises(InputError):
            runtime.synthetic_updates(self.model(), config, side=14, updates=1)


if __name__ == "__main__":
    unittest.main()
