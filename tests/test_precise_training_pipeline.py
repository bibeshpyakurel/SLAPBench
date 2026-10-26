"""Synthetic-only training, resume, provenance and final-test separation checks."""

import csv
import contextlib
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from precise_matching.data import Image, InputError, Pair, IMAGE_FIELDS, PAIR_FIELDS  # noqa: E402
from precise_matching.evaluation import Score, evaluate_development  # noqa: E402
from precise_matching.training import Config, Sampler, digest, reviewed_inputs, train  # noqa: E402
import train_precise_matcher as entry  # noqa: E402


def fixture():
    images = []
    for part in ("train", "development", "test"):
        for subject in ("a", "b"):
            for hand in (("left", "right") if part == "train" else ("left",)):
                for capture in ("1", "2"):
                    key = f"{part}-{subject}-{hand}-{capture}"
                    images.append(Image(key, part + subject, hand, key, part,
                                        hashlib.sha256(key.encode()).hexdigest(),
                                        hashlib.sha256((key + "decoded").encode()).hexdigest(), "true", "true"))
    pairs = []
    for part in ("development", "test"):
        pairs += [Pair(part + "-g", part + "-a-left-1", part + "-a-left-2", "genuine"),
                  Pair(part + "-i", part + "-a-left-1", part + "-b-left-1", "impostor")]
    return images, pairs


class InputGateTests(unittest.TestCase):
    def test_cli_refuses_public_outputs_before_input_or_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = [value for name in ("images", "pairs", "paths", "config", "review", "source-root", "weights")
                    for value in ("--" + name, str(root / "PRIVATE_UNAVAILABLE_INPUT"))]
            args += ["--output", str(root / "reports/forbidden")]
            error = io.StringIO()
            with patch.object(entry, "ROOT", root), patch.object(entry, "load_encoder") as load, \
                    contextlib.redirect_stderr(error):
                self.assertEqual(entry.main(args), 1)
            load.assert_not_called()
            self.assertNotIn("PRIVATE_UNAVAILABLE_INPUT", error.getvalue())
            self.assertFalse((root / "reports").exists())

    def test_development_evaluator_never_accepts_test_or_training_scores(self):
        images, pairs = fixture()
        report = evaluate_development(images, pairs,
                                     [Score("development-g", "ok", 0.9), Score("development-i", "ok", 0.2)], "0.001")
        self.assertEqual(report["selection_source"], "development_only")
        self.assertEqual(report["threshold"]["cutoff"], 0.2)
        for bad in (Score("test-i", "ok", 100), Score("development-g", "decode_error", None)):
            with self.assertRaises(InputError):
                evaluate_development(images, pairs, [Score("development-i", "ok", 0.2), bad], "0.001")

    def test_sampler_uses_training_subjects_and_alternating_same_hand(self):
        images, _ = fixture()
        first, second = Sampler(images, 2, 13), Sampler(images, 2, 13)
        for index in range(8):
            a, b = first.next(), second.next()
            self.assertEqual(a, b)
            self.assertEqual({r.partition for r in a}, {"train"})
            self.assertEqual({r.hand for r in a}, {("left", "right")[index % 2]})
            self.assertEqual(len({r.capture_id for r in a}), 4)
        with self.assertRaises(InputError):
            Sampler([r for r in images if r.image_id != "train-a-right-2"], 2, 13)

    def test_review_is_digest_bound_and_paths_exclude_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, pairs = fixture()
            sources = {key: root / (key + ".csv") for key in ("images", "pairs", "paths")}
            sources["config"] = root / "config.json"
            for key, rows, fields in (("images", [asdict(i) for i in images], IMAGE_FIELDS),
                                      ("pairs", [asdict(p) for p in pairs], PAIR_FIELDS)):
                with sources[key].open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            needed = {r.image_id for r in images if r.partition == "train"}
            needed |= {key for p in pairs if p.pair_id.startswith("development") for key in (p.image_a, p.image_b)}
            with sources["paths"].open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("image_id", "path"))
                writer.writerows((key, key + ".png") for key in sorted(needed))
            sources["config"].write_text(json.dumps(asdict(Config())))
            review = root / "review.json"

            def authorize():
                review.write_text(json.dumps({"protocol_reviewed": True, "split_reviewed": True,
                    "capture_exposure_reviewed": True, "execution_authorized": True,
                    "review_record": "synthetic-fixture-only",
                    "input_sha256": {key: digest(path) for key, path in sources.items()}}))

            authorize()
            args = [sources[k] for k in ("images", "pairs", "paths", "config")] + [review, root]
            result = reviewed_inputs(*args)
            self.assertEqual(set(result[2]), needed)
            sources["config"].write_text("{}")
            with self.assertRaises(InputError):
                result[-1]()
            with self.assertRaises(InputError):
                reviewed_inputs(*args)
            sources["config"].write_text(json.dumps(asdict(Config())))
            with sources["paths"].open("a") as stream:
                stream.write("test-a-left-1,private-test.png\n")
            authorize()  # Even an asserted review cannot make a test path acceptable.
            with self.assertRaises(InputError):
                reviewed_inputs(*args)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU matcher CI job supplies torch")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def model(self):
        import torch

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(48, 8)
                self.dropout = torch.nn.Dropout(0.15)

            def forward(self, pixel_values):
                return SimpleNamespace(last_hidden_state=self.linear(
                    self.dropout(pixel_values.flatten(1))).unsqueeze(1))

        torch.manual_seed(22)
        return TinyEncoder()

    def config(self):
        return Config(epochs=2, microbatches=4, subjects_per_batch=2, accumulation=2,
                      warmup_steps=1, learning_rate=0.001, minimum_learning_rate=0.0001,
                      device="cpu", precision="fp32")

    def loader(self, row):
        import torch
        self.assertNotEqual(row.partition, "test")
        generator = torch.Generator().manual_seed(int(row.sha256[:8], 16))
        return torch.randn(3, 4, 4, generator=generator)

    def test_cli_success_with_reviewed_synthetic_images_and_fake_encoder(self):
        from PIL import Image as PILImage
        import torch

        class WholeImageTiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(48, 8)

            def forward(self, pixel_values):
                values = torch.nn.functional.adaptive_avg_pool2d(pixel_values, (4, 4))
                return SimpleNamespace(last_hidden_state=self.linear(values.flatten(1)).unsqueeze(1))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "code").mkdir()
            (root / "code/train_precise_matcher.py").write_text("# synthetic SCM fixture\n")
            images, pairs = fixture()
            for index, row in enumerate(images):
                if row.partition != "test":
                    path = root / (row.image_id + ".png")
                    gray = PILImage.new("L", (16, 15), index * 13)
                    gray.save(path)
                    images[index] = replace(row, sha256=digest(path), decoded_sha256=hashlib.sha256(
                        str(gray.size).encode() + gray.tobytes()).hexdigest())
            for name, rows, fields in (("images", [asdict(i) for i in images], IMAGE_FIELDS),
                                       ("pairs", [asdict(p) for p in pairs], PAIR_FIELDS)):
                with (root / (name + ".csv")).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            needed = {r.image_id for r in images if r.partition == "train"}
            needed |= {key for p in pairs if p.pair_id.startswith("development") for key in (p.image_a, p.image_b)}
            with (root / "paths.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("image_id", "path"))
                writer.writerows((key, key + ".png") for key in sorted(needed))
            (root / "config.json").write_text(json.dumps(asdict(self.config())))
            (root / "review.json").write_text(json.dumps({"protocol_reviewed": True, "split_reviewed": True,
                "capture_exposure_reviewed": True, "execution_authorized": True, "review_record": "synthetic-only",
                "input_sha256": {name: digest(root / (name + extension)) for name, extension in
                                 (("images", ".csv"), ("pairs", ".csv"), ("paths", ".csv"), ("config", ".json"))}}))
            lock = root / "code/precise_matching/environment/requirements-linux-x86_64-py312-cu124.lock"
            lock.parent.mkdir(parents=True)
            lock.write_text("# synthetic lock fixture\n")
            args = [value for name, extension in (("images", ".csv"), ("pairs", ".csv"), ("paths", ".csv"),
                                                  ("config", ".json"), ("review", ".json"))
                    for value in ("--" + name, str(root / (name + extension)))]
            args += ["--source-root", str(root), "--weights", str(root / "fake-weights"),
                     "--output", str(root / "local/precise/runs/synthetic")]
            # SCM inventory and model loading are external boundaries; real image
            # decoding, input-review checks, optimization and selection run below.
            with patch.object(entry, "ROOT", root), patch.object(entry, "load_encoder", return_value=WholeImageTiny()), \
                    patch.object(entry, "git_output", side_effect=[b"", "synthetic-commit\n"]), \
                    patch.object(entry, "package_versions", return_value={"runtime": "synthetic"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main(args), 0)
            result = json.loads((root / "local/precise/runs/synthetic/attempt-0001/result.json").read_text())
            self.assertTrue(result["training_complete"])
            self.assertFalse(result["test_scores_accessed"])
            self.assertFalse(any(root.glob("test-*.png")))

    def test_exact_resume_matches_uninterrupted_despite_dropout(self):
        import torch

        images, pairs = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provenance = {"source": "synthetic", "torch_runtime": torch.__version__}
            full = train(self.model(), images, pairs, self.loader, self.config(), root / "full", provenance)
            partial = train(self.model(), images, pairs, self.loader, self.config(), root / "split",
                            provenance, stop_after_epochs=1)
            checkpoint = root / "split" / partial["last_checkpoint"]
            (root / "split/attempt-0002").mkdir()  # Preserve a crash before attempt.json was written.
            before = checkpoint.read_bytes()
            resumed = train(self.model(), images, pairs, self.loader, self.config(), root / "split",
                            provenance, resume=checkpoint)
            a = torch.load(root / "full" / full["last_checkpoint"], weights_only=True)
            b = torch.load(root / "split" / resumed["last_checkpoint"], weights_only=True)
            self.assertEqual(a["history"], b["history"])
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
            self.assertEqual(before, checkpoint.read_bytes())
            self.assertFalse(resumed["test_scores_accessed"])
            self.assertTrue(resumed["training_complete"])
            self.assertTrue(resumed["selection_frozen"])
            self.assertTrue((root / "split/attempt-0003/result.json").exists())

    def test_resume_rejects_changed_provenance_config_and_corrupt_bytes(self):
        images, pairs = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            first = train(self.model(), images, pairs, self.loader, self.config(), output,
                          {"source": "synthetic"}, stop_after_epochs=1)
            checkpoint = output / first["last_checkpoint"]
            for config, provenance in ((replace(self.config(), margin=0.1), {"source": "synthetic"}),
                                       (self.config(), {"source": "changed"})):
                with self.assertRaises(InputError):
                    train(self.model(), images, pairs, self.loader, config, output, provenance, resume=checkpoint)
            checkpoint.write_bytes(b"corrupt fixture")
            with self.assertRaises(InputError):
                train(self.model(), images, pairs, self.loader, self.config(), output,
                      {"source": "synthetic"}, resume=checkpoint)

    def test_decode_failure_preserves_attempt_and_refuses_overwrite(self):
        images, pairs = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"

            def fail(row):
                raise InputError("Synthetic decode failure.")

            with self.assertRaises(InputError):
                train(self.model(), images, pairs, fail, self.config(), output, {})
            self.assertTrue((output / "attempt-0001/failure.json").exists())
            with self.assertRaises(InputError):
                train(self.model(), images, pairs, self.loader, self.config(), output, {})

    def test_input_change_before_checkpoint_stops_selection(self):
        images, pairs = fixture()
        calls = 0

        def integrity():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise InputError("Synthetic input changed during the epoch.")

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            with self.assertRaises(InputError):
                train(self.model(), images, pairs, self.loader, self.config(), output, {}, integrity=integrity)
            self.assertFalse(list(output.glob("attempt-*/*.pt")))
            self.assertTrue((output / "attempt-0001/failure.json").exists())


if __name__ == "__main__":
    unittest.main()
