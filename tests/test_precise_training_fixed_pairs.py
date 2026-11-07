"""Synthetic explicit-pair contracts; no private inputs or model downloads."""

from collections import Counter
import contextlib
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
from precise_matching.data import InputError  # noqa: E402
from precise_matching.evaluation import Score  # noqa: E402
from precise_matching.fixed_pairs import (  # noqa: E402
    FixedConfig, FixedPair, HANDS, REVIEW_FLAGS, TEST_FILES, TRAIN_FILE, load_bundle,
    prepare_bundle, reviewed_shared_inputs, source_pair, validate_pairs,
)
from precise_matching.fixed_training import explicit_pair_loss, shuffled_batches, train_fixed  # noqa: E402
from precise_matching.fixed_evaluation import (  # noqa: E402
    calibrate_fixed, fixed_summary, inspect_calibration, score_fixed, test_fixed,
)
from precise_matching.training import digest  # noqa: E402
import run_precise_shared_pairs as entry  # noqa: E402

TORCH = importlib.util.find_spec("torch") is not None


def pair_fixture():
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "e"),
             ("d", "f"), ("e", "g"), ("f", "h"), ("g", "h")]
    training = [FixedPair(f"train-{i}", a, b, "genuine" if i % 2 else "impostor",
                          ("left", "right", "thumbs")[i % 3], i == 0)
                for i, (a, b) in enumerate(edges)]
    testing = [FixedPair(f"test-{i}", a, b, "genuine" if i % 2 else "impostor",
                         ("left", "right", "thumbs")[i % 3], i == 0)
               for i, (a, b) in enumerate([("a", "d"), ("b", "e"), ("c", "f"), ("d", "g")])]
    return training, testing


class FixedInputTests(unittest.TestCase):
    def test_explicit_alias_target_is_preserved_and_thumbs_accepted(self):
        pair = source_pair(1, "001_synthetic_a_15.jpeg", "1_synthetic_b_15.jpeg", "impostor", "train")
        self.assertEqual(pair.label, "impostor")
        self.assertTrue(pair.numeric_identity_disagreement)
        self.assertEqual(pair.hand, "thumbs")

    def test_schedule_visits_every_pair_once_without_cross_pairs_or_dropped_tail(self):
        training, _ = pair_fixture()
        batches = shuffled_batches(training, 3, 2026, 0)
        self.assertEqual([len(b) for b in batches], [3, 3, 2])
        self.assertCountEqual([p for b in batches for p in b], training)
        self.assertEqual(batches, shuffled_batches(training, 3, 2026, 0))
        self.assertNotEqual(batches, shuffled_batches(training, 3, 2026, 1))
        for bad in ([training[0], replace(training[0], pair_id="different")],
                    [replace(training[0], label="A")], [replace(training[0], hand="unknown")]):
            with self.assertRaises(InputError):
                validate_pairs(bad)

    def test_review_binds_actual_phase_and_inputs_without_capture_assertions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, config, review = (root / name for name in ("bundle.json", "config.json", "review.json"))
            bundle.write_text('{"scope":"shared_seen_image_fixed_pairs"}')
            config.write_text(json.dumps(asdict(FixedConfig())))
            value = {flag: True for flag in REVIEW_FLAGS}
            value.update(scope="shared_seen_image_fixed_pairs", review_record="synthetic human review",
                         authorized_phases=["train", "calibrate", "test"],
                         input_sha256={"bundle": digest(bundle), "config": digest(config)})
            review.write_text(json.dumps(value))
            _, _, integrity = reviewed_shared_inputs(bundle, config, review, "train")
            integrity()
            value["results_reviewed"] = False
            review.write_text(json.dumps(value))
            with self.assertRaises(InputError):
                reviewed_shared_inputs(bundle, config, review, "train")
            with self.assertRaises(InputError):
                integrity()
            value["results_reviewed"] = True
            value["authorized_phases"] = ["train"]
            review.write_text(json.dumps(value))
            with self.assertRaises(InputError):
                reviewed_shared_inputs(bundle, config, review, "test")
            bundle.write_text('{"changed":true}')
            with self.assertRaises(InputError):
                reviewed_shared_inputs(bundle, config, review, "train")

    def test_invalid_configuration_and_public_output_refused_before_model_loading(self):
        for update in ({"seed": True}, {"warmup_steps": -1}, {"learning_rate": float("nan")},
                       {"checkpoint_rule": "best_test"}, {"device": "cpu", "precision": "bf16"}):
            with self.subTest(update=update), self.assertRaises(InputError):
                replace(FixedConfig(), **update).validate(8)
        args = ["train"] + [value for name in ("bundle", "config", "review", "source-root", "weights", "output")
                            for value in ("--" + name, "/tmp/PRIVATE_UNAVAILABLE")]
        with patch.object(entry, "load_encoder") as load, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(entry.main(args), 1)
        load.assert_not_called()

    def test_private_preparation_uses_existing_inventory_without_reading_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            intake, images = root / "intake", root / "images"
            intake.mkdir(); images.mkdir()
            records, testing, names = [], [], set()
            for suffix in HANDS:
                a, b, c, d = [f"{subject}_synthetic_{capture}_{suffix}.jpeg"
                               for subject, capture in (("001", "a"), ("001", "b"), ("1", "c"), ("001", "d"))]
                names.update((a, b, c, d))
                for left, right, label in ((a, b, "genuine"), (a, c, "impostor")):
                    records.append({"image_paths": [left, right], "pair_class": label,
                                    "correct_answer_option": "A" if label == "genuine" else "B"})
                testing.extend(((a, d, "genuine"), (b, c, "impostor")))
            (intake / TRAIN_FILE).write_text(json.dumps({"pairs": records}))
            for label, filename in TEST_FILES.items():
                (intake / filename).write_text("".join(f"{a} {b} 0\n" for a, b, kind in testing if kind == label))
            for name in names:
                (images / name).write_bytes(b"not an image: preparation must never decode this")
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps({"records": [{"filename": name, "status": "ok", "sha256": "a" * 64,
                                                          "decoded_sha256": "b" * 64} for name in names]}))
            audit = root / "audit.json"
            counts = {"records": 6, "class_counts": {"genuine": 3, "impostor": 3},
                      "numeric_id_label_disagreements": 3,
                      "hand_class_counts": {f"{s}_{label}": 1 for s in HANDS for label in TEST_FILES}}
            audit.write_text(json.dumps({"sources": {name: {"sha256": digest(intake / name)}
                                                     for name in (TRAIN_FILE, *TEST_FILES.values())},
                                         "training": counts, "testing": counts}))
            output = root / "bundle.json"
            summary = prepare_bundle(intake, inventory, audit, output)
            self.assertFalse(summary["image_pixels_read"])
            self.assertFalse(summary["approval_issued"])
            self.assertEqual(summary["training_pairs"], 6)
            _, training, test, paths = load_bundle(output, images)
            self.assertEqual(Counter(p.hand for p in training), {"right": 2, "left": 2, "thumbs": 2})
            self.assertEqual(len(test), 6)
            self.assertEqual(len(paths), 12)
            with self.assertRaises(InputError):
                prepare_bundle(intake, inventory, audit, output)
            value = json.loads(output.read_text())
            value["images"][next(iter(names))]["path"] = "../inventory.json"
            output.write_text(json.dumps(value))
            with self.assertRaises(InputError):
                load_bundle(output, images)

    def test_summary_preserves_failures_pooled_cutoff_and_sensitivity_denominators(self):
        _, testing = pair_fixture()
        scores = [Score(p.pair_id, "ok", value) for p, value in zip(testing, (.9, .8, .1, .7), strict=True)]
        scores[3] = Score(testing[3].pair_id, "decode_error", None)
        report = fixed_summary(testing, scores, .5)
        rates = report["pooled"]["all_attempts_failure_as_reject"]
        self.assertEqual((rates["tar"], rates["far"]), (.5, .5))
        self.assertEqual(report["label_ambiguity_sensitivity"]["remaining"]["attempts"]["impostor"], 1)
        self.assertEqual(report["by_slap_type"]["right"]["all_attempts_failure_as_reject"]["tar"], 1)
        self.assertFalse(report["uncertainty"]["independent_pair_intervals"])
        self.assertNotIn("test-", json.dumps(report))
        for bad in (scores[:-1], scores + [scores[0]], [replace(scores[0], score=float("nan")), *scores[1:]]):
            with self.assertRaises(InputError):
                fixed_summary(testing, bad, .5)


@unittest.skipUnless(TORCH, "Torch contracts run in the matcher CPU CI job")
class FixedRuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(1)
        torch.manual_seed(7)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen = torch.nn.Linear(3, 4)
                self.frozen.requires_grad_(False)
                self.projection = torch.nn.Linear(4, 4)
                self.dropout = torch.nn.Dropout(.1)

            def forward(self, pixel_values):
                x = self.frozen(pixel_values.mean(dim=(2, 3)))
                x = self.projection(self.dropout(x))
                return SimpleNamespace(last_hidden_state=x[:, None, :])

        self.factory = Tiny
        self.state = Tiny().state_dict()
        self.training, self.testing = pair_fixture()
        self.pixels = {key: torch.rand(3, 2, 2) for key in "abcdefgh"}
        self.config = FixedConfig(epochs=2, batch_pairs=3, accumulation=2, warmup_steps=0,
                                  device="cpu", precision="fp32", learning_rate=.01, minimum_learning_rate=.001)

    def model(self):
        model = self.factory()
        model.load_state_dict(self.state)
        return model

    def test_explicit_loss_single_class_tail_and_gradient_weights(self):
        torch = self.torch
        a = torch.tensor([[1., 0.], [0., 1.]])
        b = torch.tensor([[1., 0.], [1., 0.]])
        self.assertAlmostEqual(float(explicit_pair_loss(a, b, ["genuine", "impostor"], .2)), 0)
        self.assertAlmostEqual(float(explicit_pair_loss(a, b, ["impostor", "genuine"], .2)), .9, places=6)
        self.assertAlmostEqual(float(explicit_pair_loss(a[:1], b[:1], ["impostor"], .2)), .8, places=6)
        with self.assertRaises(InputError):
            explicit_pair_loss(a, b, ["genuine"], .2)

    def test_resume_matches_uninterrupted_dropout_and_preserves_exact_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            direct_model, resumed_model = self.model(), self.model()
            used = []

            def load(key):
                used.append(key)
                return self.pixels[key]

            direct = train_fixed(direct_model, self.training, load, self.config, root / "direct", {"synthetic": True})
            self.assertEqual(len(used), len(self.training) * 2 * self.config.epochs)
            partial = train_fixed(resumed_model, self.training, load, self.config, root / "resume", {"synthetic": True},
                                  stop_after_epochs=1)
            resumed = train_fixed(resumed_model, self.training, load, self.config, root / "resume", {"synthetic": True},
                                  resume=root / "resume" / partial["checkpoint"])
            self.assertEqual(direct["history"], resumed["history"])
            self.assertTrue(resumed["training_complete"])
            self.assertEqual(direct["epochs_completed"], 2)
            for name, value in direct_model.state_dict().items():
                self.assertTrue(self.torch.equal(value, resumed_model.state_dict()[name]), name)
                if name.startswith("frozen"):
                    self.assertTrue(self.torch.equal(value, self.state[name]))
            self.assertFalse(direct["test_scores_accessed"])
            with self.assertRaises(InputError):
                train_fixed(self.model(), self.training, load, self.config, root / "resume", {"synthetic": True},
                            resume=root / "resume" / partial["checkpoint"])

    def test_failed_training_retains_failure_and_no_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"

            def fail(key):
                raise InputError("Synthetic image failure")

            with self.assertRaises(InputError):
                train_fixed(self.model(), self.training, fail, self.config, output, {})
            self.assertTrue((output / "attempt-0001/failure.json").exists())
            self.assertFalse((output / "selection.json").exists())
            self.assertFalse(json.loads((output / "attempt-0001/failure.json").read_text())["test_scores_accessed"])

    def test_failed_calibration_never_seals_and_test_requires_replay_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provenance = {"model_identity": {"synthetic": True}}

            def fail(key):
                raise InputError("Synthetic image failure")

            with self.assertRaises(InputError):
                calibrate_fixed(self.model(), self.training, fail, self.config, root / "bad", provenance)
            self.assertTrue((root / "bad/scores.csv").exists())
            self.assertFalse((root / "bad/selection.json").exists())
            with patch("precise_matching.fixed_evaluation.score_fixed") as score, self.assertRaises(OSError):
                test_fixed(self.model(), self.training, self.testing, fail, self.config,
                           root / "bad", root / "test", provenance)
            score.assert_not_called()
            self.assertFalse((root / "test").exists())

    def test_training_threshold_replay_test_claim_and_failure_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provenance = {"model_identity": {"synthetic": True}}
            model = self.model()
            selection = calibrate_fixed(model, self.training, self.pixels.__getitem__, self.config, root / "cal", provenance)
            self.assertFalse(selection["test_scores_accessed"])
            report = test_fixed(model, self.training, self.testing, self.pixels.__getitem__, self.config,
                                root / "cal", root / "test", provenance)
            self.assertEqual(report["threshold"], selection["threshold"])
            self.assertEqual(report["results"]["pooled"]["attempts"], {"genuine": 2, "impostor": 2})
            with self.assertRaises(FileExistsError):
                test_fixed(model, self.training, self.testing, self.pixels.__getitem__, self.config,
                           root / "cal", root / "repeat", provenance)
            self.assertFalse((root / "repeat").exists())
            value = json.loads((root / "cal/selection.json").read_text())
            value["threshold"]["cutoff"] = -1
            (root / "cal/selection.json").write_text(json.dumps(value))
            with self.assertRaises(InputError):
                inspect_calibration(root / "cal", self.training, self.config, provenance)

    def test_decode_failure_propagates_to_all_affected_pairs_and_encoder_failure_is_not_retried(self):
        def load(key):
            if key == "a":
                raise InputError("Synthetic failure")
            return self.pixels[key]

        scores = score_fixed(self.model(), self.training, load, "cpu", 2)
        self.assertEqual(sum(s.status == "decode_error" for s in scores), 2)
        model = self.model()
        with patch.object(model, "forward", side_effect=RuntimeError("synthetic encoder error")) as forward:
            scores = score_fixed(model, self.training, self.pixels.__getitem__, "cpu", 4)
        self.assertEqual(forward.call_count, 2)
        self.assertTrue(all(s.status == "encoder_error" and s.score is None for s in scores))

    def test_cli_roundtrip_train_both_arms_calibrate_and_test_with_real_synthetic_images(self):
        from PIL import Image
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private, source = root / "local/precise", root / "images"
            private.mkdir(parents=True); source.mkdir()
            lock = root / "code/precise_matching/environment/requirements-linux-x86_64-py312-cu124.lock"
            lock.parent.mkdir(parents=True)
            lock.write_text("synthetic lock")
            code = root / "code/synthetic.py"
            code.write_text("# synthetic source")
            images = {}
            for index, key in enumerate(self.pixels):
                path = source / (key + ".png")
                image = Image.new("L", (4, 4), 10 + index * 20)
                image.save(path)
                images[key] = {"path": path.name, "sha256": digest(path),
                               "decoded_sha256": hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()}
            bundle, config, review = (private / name for name in ("bundle.json", "config.json", "review.json"))
            bundle.write_text(json.dumps({"schema_version": 1, "scope": "shared_seen_image_fixed_pairs",
                "labels": "as_supplied", "images": images,
                "training": [asdict(p) for p in self.training], "testing": [asdict(p) for p in self.testing]}))
            config.write_text(json.dumps(asdict(self.config)))
            value = {flag: True for flag in REVIEW_FLAGS}
            value.update(scope="shared_seen_image_fixed_pairs", review_record="synthetic human review",
                         authorized_phases=["train", "calibrate", "test"],
                         input_sha256={"bundle": digest(bundle), "config": digest(config)})
            review.write_text(json.dumps(value))
            common = [value for name, path in (("bundle", bundle), ("config", config), ("review", review),
                                               ("source-root", source), ("weights", private / "weights"))
                      for value in ("--" + name, str(path))]
            run = private / "runs/train"
            with patch.object(entry, "ROOT", root), patch.object(entry, "git_output", return_value="synthetic"), \
                    patch.object(entry, "package_versions", return_value={}), \
                    patch.object(entry, "source_record", return_value={"code/synthetic.py": digest(code)}), \
                    patch.object(entry, "load_encoder", side_effect=lambda *a, **kw: self.model()), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(entry.main(["train", *common, "--output", str(run)]), 0)
                for arm in ("frozen_dinov2", "adapted_dinov2"):
                    cal = private / "runs" / (arm + "-cal")
                    test = private / "runs" / (arm + "-test")
                    extra = ["--arm", arm] + (["--training-run", str(run)] if arm == "adapted_dinov2" else [])
                    self.assertEqual(entry.main(["calibrate", *common, *extra, "--output", str(cal)]), 0)
                    self.assertEqual(entry.main(["test", *common, *extra, "--calibration", str(cal),
                                                 "--output", str(test)]), 0)
                    self.assertTrue((test / "aggregate.json").exists())


if __name__ == "__main__":
    unittest.main()
