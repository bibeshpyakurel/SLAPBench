"""Final-test selection/threshold/failure contracts with synthetic inputs only."""

import contextlib
import csv
from dataclasses import asdict, replace
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import test_precise_matcher as entry  # noqa: E402
import test_precise_training_pipeline as helpers  # noqa: E402
from precise_matching.data import IMAGE_FIELDS, PAIR_FIELDS, InputError  # noqa: E402
from precise_matching.encoders import MODEL_ID, REVISION, WEIGHTS_SHA256, CONFIG_SHA256  # noqa: E402
from precise_matching.testing import (ImageFailure, final_test, inspect_selection,
                                     reviewed_test_inputs, restore_selection, score_test)  # noqa: E402
from precise_matching.training import Config, digest, train  # noqa: E402


class EarlyGateTests(unittest.TestCase):
    def test_cli_public_output_refused_without_loading_private_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = [value for name in ("images", "pairs", "paths", "config", "review", "source-root",
                                      "weights", "training-run", "selection")
                    for value in ("--" + name, str(root / "PRIVATE_UNAVAILABLE"))]
            args += ["--output", str(root / "reports/not-private")]
            error = io.StringIO()
            with patch.object(entry, "ROOT", root), patch.object(entry, "load_encoder") as load, \
                    contextlib.redirect_stderr(error):
                self.assertEqual(entry.main(args), 1)
            load.assert_not_called()
            self.assertNotIn("PRIVATE_UNAVAILABLE", error.getvalue())
            self.assertFalse((root / "reports").exists())


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU matcher CI job supplies torch")
class FinalTestContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def model(self):
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(48, 8)

            def forward(self, pixel_values):
                values = torch.nn.functional.adaptive_avg_pool2d(pixel_values, (4, 4))
                return SimpleNamespace(last_hidden_state=self.linear(values.flatten(1)).unsqueeze(1))

        torch.manual_seed(22)
        return Tiny()

    def pixels(self, row):
        import torch
        return torch.randn(3, 4, 4, generator=torch.Generator().manual_seed(int(row.sha256[:8], 16)))

    def fixture(self, root):
        from PIL import Image
        import hashlib

        images, pairs = helpers.fixture()
        source = root / "data"
        source.mkdir()
        for index, row in enumerate(images):
            if row.partition == "test":
                p = source / (row.image_id + ".png")
                gray = Image.new("L", (16, 15), index * 13)
                gray.save(p)
                images[index] = replace(row, sha256=digest(p), decoded_sha256=hashlib.sha256(
                    str(gray.size).encode() + gray.tobytes()).hexdigest())
        for name, rows, fields in (("images", images, IMAGE_FIELDS), ("pairs", pairs, PAIR_FIELDS)):
            with (root / (name + ".csv")).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(asdict(row) for row in rows)
        config = Config(epochs=2, microbatches=2, subjects_per_batch=2, accumulation=1,
                        warmup_steps=0, learning_rate=0.001, minimum_learning_rate=0.0001,
                        device="cpu", precision="fp32")
        (root / "config.json").write_text(json.dumps(asdict(config)))
        with (root / "paths.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("image_id", "path"))
            needed = {key for pair in pairs if pair.pair_id.startswith("test-")
                      for key in (pair.image_a, pair.image_b)}
            writer.writerows((key, key + ".png") for key in sorted(needed))
        sources = {name: root / (name + extension) for name, extension in
                   (("images", ".csv"), ("pairs", ".csv"), ("paths", ".csv"), ("config", ".json"))}
        hashes = {key: digest(path) for key, path in sources.items()}
        inventory = {}
        for name in ("data", "images", "encoders", "losses", "evaluation", "training"):
            rel = "code/precise_matching/" + name + ".py"
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# synthetic source inventory boundary\n")
            inventory[rel] = digest(p)
        script = root / "code/train_precise_matcher.py"
        script.write_text("# synthetic trainer source boundary\n")
        inventory[str(script.relative_to(root))] = digest(script)
        lock = root / "code/precise_matching/environment/requirements-linux-x86_64-py312-cu124.lock"
        lock.parent.mkdir()
        lock.write_text("# synthetic lock fixture\n")
        provenance = {"model": MODEL_ID, "model_revision": REVISION, "weights_sha256": WEIGHTS_SHA256,
                      "model_config_sha256": CONFIG_SHA256,
                      "input_sha256": {k: hashes[k] for k in ("images", "pairs", "config")},
                      "source_sha256": inventory, "packages": {"runtime": "synthetic"},
                      "python": entry.platform.python_version(), "environment_lock_sha256": digest(lock)}
        run = root / "local/precise/runs/training"
        train(self.model(), images, pairs, self.pixels, config, run, provenance)
        selection = run / "attempt-0001/result.json"
        info = inspect_selection(run, selection, config, hashes)
        hashes.update(info["hashes"])
        review = root / "review.json"
        review.write_text(json.dumps({**{flag: True for flag in
            ("protocol_reviewed", "split_reviewed", "capture_exposure_reviewed", "execution_authorized",
             "selection_reviewed", "final_test_authorized")},
            "input_sha256": hashes, "review_record": "synthetic-fixture-only"}))
        args = [sources[k] for k in ("images", "pairs", "paths", "config")] + [review, source, run, selection]
        return images, pairs, config, args

    def test_fixed_cutoff_test_only_pixels_claim_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, args = self.fixture(root)
            images, pairs, _, config, info, hashes, integrity = reviewed_test_inputs(*args)
            before = {k: p.read_bytes() for k, p in info["files"].items()}
            seen = []

            def loader(row):
                self.assertEqual(row.partition, "test")
                seen.append(row.image_id)
                return self.pixels(row)

            report = final_test(self.model(), images, pairs, loader, config, info,
                                root / "evaluation", hashes, integrity=integrity)
            self.assertEqual(report["threshold"], info["result"]["selected"]["threshold"])
            self.assertEqual(report["test"]["attempts"], {"genuine": 1, "impostor": 1})
            self.assertEqual(len(seen), len(set(seen)))
            self.assertNotIn("test-a-left", json.dumps(report))
            self.assertEqual(before, {k: p.read_bytes() for k, p in info["files"].items()})
            with self.assertRaises(InputError):
                final_test(self.model(), images, pairs, loader, config, info, root / "retry", hashes)

    def test_incomplete_or_changed_selection_refused_before_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, config, args = self.fixture(root)
            result = json.loads(args[-1].read_text())
            for key in ("training_complete", "selection_frozen"):
                args[-1].write_text(json.dumps({**result, key: False}))
                with self.assertRaises(InputError):
                    reviewed_test_inputs(*args)
            args[-1].write_text(json.dumps(result))
            info = inspect_selection(args[-2], args[-1], config,
                                     {k: digest(root / (k + ext)) for k, ext in
                                      (("images", ".csv"), ("pairs", ".csv"), ("config", ".json"))})
            info["files"]["checkpoint"].write_bytes(b"corrupt synthetic checkpoint")
            with self.assertRaises(InputError):
                reviewed_test_inputs(*args)

    def test_path_population_and_review_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, args = self.fixture(root)
            reviewed = reviewed_test_inputs(*args)
            with (root / "paths.csv").open("a") as stream:
                stream.write("train-a-left-1,unavailable.png\n")
            with self.assertRaises(InputError):
                reviewed[-1]()
            review = json.loads((root / "review.json").read_text())
            review["input_sha256"]["paths"] = digest(root / "paths.csv")
            (root / "review.json").write_text(json.dumps(review))
            with self.assertRaises(InputError):
                reviewed_test_inputs(*args)

    def test_replayed_development_rejects_tampered_cutoff_and_test_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, pairs, config, args = self.fixture(root)
            info = reviewed_test_inputs(*args)[4]
            dev = json.loads(info["files"]["development"].read_text())
            for changed in [dict(dev, scores=dev["scores"] +
                                 [{"pair_id": "test-i", "status": "ok", "score": 1.}]),
                            dict(dev, metrics={**dev["metrics"], "threshold": {"cutoff": 99.}})]:
                info["files"]["development"].write_text(json.dumps(changed))
                # Bind the changed fixture bytes to exercise semantic replay,
                # independently of the ordinary immutable-byte guard.
                info["hashes"] = {k: digest(p) for k, p in info["files"].items()}
                with self.assertRaises(InputError):
                    restore_selection(self.model(), info, images, pairs, config)

    def test_completed_history_and_later_attempt_are_required(self):
        import torch

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, pairs, config, args = self.fixture(root)
            info = reviewed_test_inputs(*args)[4]
            path = info["files"]["last_checkpoint"]
            original = path.read_bytes()
            saved = torch.load(path, weights_only=True)
            saved["epochs_completed"] -= 1
            torch.save(saved, path)
            info["hashes"] = {k: digest(p) for k, p in info["files"].items()}
            with self.assertRaises(InputError):
                restore_selection(self.model(), info, images, pairs, config)
            path.write_bytes(original)
            (info["root"] / "attempt-0002").mkdir()
            with self.assertRaises(InputError):
                reviewed_test_inputs(*args)

    def test_changed_artifacts_stop_before_test_pixels_and_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, args = self.fixture(root)
            images, pairs, _, config, info, hashes, integrity = reviewed_test_inputs(*args)
            info["files"]["development"].write_text("{}")
            with patch.object(entry, "load_encoder") as unrelated_load, \
                    patch("precise_matching.testing.score_test") as score:
                with self.assertRaises(InputError):
                    final_test(self.model(), images, pairs, self.pixels, config, info,
                               root / "evaluation", hashes, integrity=integrity)
            score.assert_not_called()
            unrelated_load.assert_not_called()
            self.assertFalse((info["root"] / "final_test_adapted_claim.json").exists())

    def test_decode_and_encoder_failures_preserve_pair_population(self):
        import torch

        images, pairs = helpers.fixture()

        def decode_failure(row):
            self.assertEqual(row.partition, "test")
            raise ImageFailure("decode_error")

        selected, scores = score_test(self.model(), images, pairs, decode_failure, "cpu")
        self.assertEqual(len(selected), 2)
        self.assertEqual([s.status for s in scores], ["decode_error", "decode_error"])

        class InvalidEncoder(torch.nn.Module):
            def forward(self, **kwargs):
                return SimpleNamespace(last_hidden_state=torch.full((1, 1, 8), float("nan")))

        _, scores = score_test(InvalidEncoder(), images, pairs, self.pixels, "cpu")
        self.assertEqual([s.status for s in scores], ["encoder_error", "encoder_error"])
        self.assertTrue(all(s.score is None for s in scores))

    def test_nonobject_review_and_escaped_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, args = self.fixture(root)
            review_path = root / "review.json"
            original = review_path.read_bytes()
            review_path.write_text("[]")
            with self.assertRaises(InputError):
                reviewed_test_inputs(*args)
            review_path.write_bytes(original)
            rows = (root / "paths.csv").read_text().splitlines()
            rows[1] = rows[1].split(",")[0] + ",/outside/PRIVATE_SYNTHETIC.png"
            (root / "paths.csv").write_text("\n".join(rows) + "\n")
            review = json.loads(original)
            review["input_sha256"]["paths"] = digest(root / "paths.csv")
            review_path.write_text(json.dumps(review))
            with self.assertRaises(InputError):
                reviewed_test_inputs(*args)

    def test_image_failure_denominators_and_fatal_failure_claim_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, args = self.fixture(root)
            images, pairs, _, config, info, hashes, _ = reviewed_test_inputs(*args)

            def missing(row):
                raise ImageFailure("missing_input")

            report = final_test(self.model(), images, pairs, missing, config, info, root / "failed-images", hashes)
            self.assertEqual(report["test"]["status_counts"], {"missing_input": 2})
            self.assertEqual(report["test"]["coverage"], {"genuine": 0., "impostor": 0.})
            self.assertIsNone(report["test"]["valid_scores_only"]["tar"])
            self.assertEqual(report["test"]["all_attempts_failure_as_reject"]["genuine_comparisons"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, args = self.fixture(root)
            images, pairs, _, config, info, hashes, _ = reviewed_test_inputs(*args)
            count = 0

            def fatal(row):
                nonlocal count
                count += 1
                if count == 1:
                    return self.pixels(row)
                raise InputError("Synthetic source-byte drift")

            with self.assertRaises(InputError):
                final_test(self.model(), images, pairs, fatal, config, info, root / "fatal", hashes)
            self.assertTrue((root / "fatal/failure.json").exists())
            events = [json.loads(line) for line in (root / "fatal/events.jsonl").read_text().splitlines()]
            self.assertEqual(events[0]["kind"], "image")
            self.assertEqual(events[0]["status"], "ok")
            self.assertEqual(events[-1]["kind"], "aborted")
            self.assertTrue((info["root"] / "final_test_adapted_claim.json").exists())
            self.assertFalse((root / "fatal/report.json").exists())

    def test_cli_end_to_end_real_decode_fake_model_no_test_tuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, inputs = self.fixture(root)
            args = [value for name, path in zip(
                ("images", "pairs", "paths", "config", "review", "source-root", "training-run", "selection"),
                inputs, strict=True) for value in ("--" + name, str(path))]
            args += ["--weights", str(root / "fake-weights"),
                     "--output", str(root / "local/precise/runs/evaluation"),
                     "--report", str(root / "reports/aggregate.json")]
            (root / "code/test_precise_matcher.py").write_text("# synthetic scoring SCM fixture\n")
            with patch.object(entry, "ROOT", root), patch.object(entry, "load_encoder", return_value=self.model()), \
                    patch.object(entry, "git_output", side_effect=[b"", "synthetic-commit\n"]), \
                    patch.object(entry, "versions", return_value={"runtime": "synthetic"}), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main(args), 0)
            report = json.loads((root / "reports/aggregate.json").read_text())
            self.assertFalse(report["test_used_for_selection"])
            self.assertEqual(report["test"]["status_counts"], {"ok": 2})
            self.assertNotIn("test-a-left", json.dumps(report))
            self.assertFalse(any((root / "data").glob("train-*.png")))

    def test_changed_test_scores_change_rates_without_changing_cutoff(self):
        import torch

        reports = []
        for score in (-1., 1.):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, _, _, args = self.fixture(root)
                images, pairs, _, config, info, hashes, _ = reviewed_test_inputs(*args)
                with patch("precise_matching.testing.cosine_scores", return_value=torch.tensor([score])):
                    reports.append(final_test(self.model(), images, pairs, self.pixels,
                                              config, info, root / "evaluation", hashes))
        self.assertEqual(reports[0]["threshold"], reports[1]["threshold"])
        self.assertEqual(reports[0]["test"]["all_attempts_failure_as_reject"]["tar"], 0.)
        self.assertEqual(reports[1]["test"]["all_attempts_failure_as_reject"]["tar"], 1.)


if __name__ == "__main__":
    unittest.main()
