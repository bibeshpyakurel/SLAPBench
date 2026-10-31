"""Frozen-control phase/review/failure contracts; synthetic images/models only."""

import contextlib
import csv
from dataclasses import asdict, replace
import hashlib
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
import run_precise_frozen as entry  # noqa: E402
from precise_matching.baseline import (ARTIFACTS, MODEL, inspect_baseline, reviewed_baseline_inputs,
                                       run_baseline)  # noqa: E402
from precise_matching.data import IMAGE_FIELDS, PAIR_FIELDS, InputError, load_protocol  # noqa: E402
from precise_matching.evaluation import evaluate_development, load_scores  # noqa: E402
from precise_matching.testing import ImageFailure, score_partition  # noqa: E402
from precise_matching.training import Config, digest  # noqa: E402
from test_precise_training_pipeline import fixture  # noqa: E402


class EarlyBaselineGates(unittest.TestCase):
    def test_public_output_refused_before_input_or_model_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["development", *[arg for name in ("images", "pairs", "paths", "config", "review",
                                                      "source-root", "weights")
                                      for arg in ("--" + name, str(root / "PRIVATE_UNAVAILABLE"))],
                    "--output", str(root / "reports/forbidden")]
            error = io.StringIO()
            with patch.object(entry, "ROOT", root), patch.object(entry, "load_encoder") as load, \
                    contextlib.redirect_stderr(error):
                self.assertEqual(entry.main(args), 1)
            load.assert_not_called()
            self.assertNotIn("PRIVATE_UNAVAILABLE", error.getvalue())
            self.assertFalse((root / "reports").exists())


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU matcher CI supplies torch")
class BaselineContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def model(self, *, frozen=True):
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 8)

            def forward(self, pixel_values):
                values = self.linear(pixel_values.mean(dim=(2, 3)))
                return SimpleNamespace(last_hidden_state=values.unsqueeze(1))

        torch.manual_seed(7)
        return Tiny().requires_grad_(not frozen)

    def pixels(self, row):
        import torch
        return torch.randn(3, 4, 4, generator=torch.Generator().manual_seed(int(row.sha256[:8], 16)))

    def setup_inputs(self, root):
        from PIL import Image

        images, pairs = fixture()
        source = root / "data"
        source.mkdir()
        for index, row in enumerate(images):
            if row.partition != "train":
                path = source / (row.image_id + ".png")
                gray = Image.new("L", (16, 15), index * 13)
                gray.save(path)
                images[index] = replace(row, sha256=digest(path), decoded_sha256=hashlib.sha256(
                    str(gray.size).encode() + gray.tobytes()).hexdigest())
        for name, rows, fields in (("images", images, IMAGE_FIELDS), ("pairs", pairs, PAIR_FIELDS)):
            with (root / (name + ".csv")).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(asdict(row) for row in rows)
        config = Config(epochs=1, microbatches=2, accumulation=1, warmup_steps=0,
                        subjects_per_batch=2, device="cpu", precision="fp32")
        (root / "config.json").write_text(json.dumps(asdict(config)))
        return images, pairs, config, source

    def reviewed(self, root, phase, baseline=None):
        images, pairs = load_protocol(root / "images.csv", root / "pairs.csv")
        config = Config(**json.loads((root / "config.json").read_text()))
        by_id = {i.image_id: i for i in images}
        keys = {key for p in pairs if by_id[p.image_a].partition == phase for key in (p.image_a, p.image_b)}
        paths = root / (phase + "_paths.csv")
        with paths.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("image_id", "path"))
            writer.writerows((key, key + ".png") for key in sorted(keys))
        hashes = {k: digest(p) for k, p in {"images": root / "images.csv", "pairs": root / "pairs.csv",
                                           "config": root / "config.json", "paths": paths}.items()}
        if baseline is not None:
            hashes.update(inspect_baseline(baseline, images, pairs, config, hashes)["hashes"])
        review = root / (phase + "_review.json")
        review.write_text(json.dumps({"phase": phase, "input_sha256": hashes, "review_record": "synthetic-only",
            **{flag: True for flag in ("protocol_reviewed", "split_reviewed", "capture_exposure_reviewed",
                                      "execution_authorized", "frozen_control_reviewed",
                                      "selection_reviewed", "final_test_authorized")}}))
        args = (root / "images.csv", root / "pairs.csv", paths, root / "config.json", review,
                root / "data", phase, baseline)
        return args, reviewed_baseline_inputs(*args)

    def provenance(self):
        return {**MODEL, "source_sha256": {"fixture": "synthetic-source"}, "packages": {"runtime": "synthetic"},
                "python": entry.platform.python_version(), "environment_lock_sha256": "a" * 64,
                "source_root": "synthetic-source"}

    def develop(self, root):
        self.setup_inputs(root)
        _, loaded = self.reviewed(root, "development")
        images, pairs, _, config, _, hashes, integrity = loaded
        run = root / "local/precise/runs/frozen-development"
        run_baseline(self.model(), images, pairs, self.pixels, config, "development", run, hashes,
                     self.provenance(), integrity=integrity)
        return run

    def test_development_only_freezes_cutoff_and_preserves_weights_without_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_inputs(root)
            _, loaded = self.reviewed(root, "development")
            images, pairs, _, config, _, hashes, integrity = loaded
            model = self.model()
            before = {k: v.clone() for k, v in model.state_dict().items()}
            seen = []

            def load(row):
                self.assertEqual(row.partition, "development")
                seen.append(row.image_id)
                return self.pixels(row)

            output = root / "development"
            with patch("torch.optim.AdamW") as optimizer:
                report = run_baseline(model, images, pairs, load, config, "development", output, hashes,
                                      self.provenance(), integrity=integrity)
            optimizer.assert_not_called()
            self.assertEqual(len(seen), len(set(seen)))
            self.assertTrue(all((v == before[k]).all() for k, v in model.state_dict().items()))
            self.assertFalse(report["test_scores_accessed"])
            self.assertNotIn("development-a-left", json.dumps(report))
            self.assertTrue((output / "selection.json").exists())
            self.assertEqual(len(load_scores(output / "scores.csv")), 2)
            self.assertFalse(any((root / "data").glob("train-*.png")))

    def test_final_test_reads_only_test_and_exports_exact_combined_comparison_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.develop(root)
            _, loaded = self.reviewed(root, "test", run)
            images, pairs, _, config, info, hashes, integrity = loaded
            original = {name: (run / name).read_bytes() for name in ARTIFACTS.values()}

            def load(row):
                self.assertEqual(row.partition, "test")
                return self.pixels(row)

            report = run_baseline(self.model(), images, pairs, load, config, "test", root / "test", hashes,
                                  self.provenance(), info=info, integrity=integrity)
            self.assertEqual(report["threshold"], info["development"]["threshold"])
            self.assertFalse(report["test_used_for_selection"])
            self.assertTrue(report["development_selection_replayed"])
            self.assertEqual(report["test"]["attempts"], {"genuine": 1, "impostor": 1})
            self.assertEqual(len(load_scores(root / "test/scores.csv")), 4)
            self.assertEqual(original, {name: (run / name).read_bytes() for name in ARTIFACTS.values()})
            with self.assertRaises(InputError):
                run_baseline(self.model(), images, pairs, load, config, "test", root / "retry", hashes,
                             self.provenance(), info=info, integrity=integrity)

    def test_review_flags_phase_and_digest_changes_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_inputs(root)
            args, loaded = self.reviewed(root, "development")
            review_path = args[4]
            original = review_path.read_text()
            review = json.loads(original)
            for change in ({"frozen_control_reviewed": False}, {"phase": "test"}, {"input_sha256": {}}):
                review_path.write_text(json.dumps({**review, **change}))
                with self.assertRaises(InputError):
                    reviewed_baseline_inputs(*args)
            review_path.write_text(original)
            args[2].write_text(args[2].read_text() + "\n")
            with self.assertRaises(InputError):
                loaded[-1]()

    def test_path_mappings_reject_other_partitions_and_escape_even_with_reviewed_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_inputs(root)
            args, _ = self.reviewed(root, "development")
            original = args[2].read_text()
            for content in (original + "test-a-left-1,unavailable.png\n",
                            original.replace("development-a-left-1.png", "../escape.png")):
                args[2].write_text(content)
                review = json.loads(args[4].read_text())
                review["input_sha256"]["paths"] = digest(args[2])
                args[4].write_text(json.dumps(review))
                with self.assertRaises(InputError):
                    reviewed_baseline_inputs(*args)

    def test_test_phase_requires_separate_selection_and_execution_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = self.develop(root)
            args, _ = self.reviewed(root, "test", run)
            review = json.loads(args[4].read_text())
            for flag in ("selection_reviewed", "final_test_authorized", "capture_exposure_reviewed"):
                args[4].write_text(json.dumps({**review, flag: False}))
                with self.assertRaises(InputError):
                    reviewed_baseline_inputs(*args)
            args[4].write_text(json.dumps(review))
            (run / "final_test_frozen_claim.json").write_text("{}")
            with self.assertRaises(InputError):
                reviewed_baseline_inputs(*args)

    def test_trainable_or_unreviewed_models_and_training_partition_are_rejected_before_pixels(self):
        images, pairs = fixture()
        config = Config(epochs=1, microbatches=2, accumulation=1, warmup_steps=0, device="cpu", precision="fp32")
        with tempfile.TemporaryDirectory() as tmp:
            loader = unittest.mock.Mock()
            with self.assertRaises(InputError):
                run_baseline(self.model(frozen=False), images, pairs, loader, config, "development",
                             Path(tmp) / "out", {}, self.provenance())
            with self.assertRaises(InputError):
                run_baseline(self.model(), images, pairs, loader, config, "development",
                             Path(tmp) / "out", {}, {**self.provenance(), "model_revision": "wrong"})
            with self.assertRaises(InputError):
                score_partition(self.model(), images, pairs, loader, "cpu", partition="train")
            loader.assert_not_called()

    def test_development_failures_preserve_outcomes_without_frozen_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, pairs, config, _ = self.setup_inputs(root)
            _, loaded = self.reviewed(root, "development")

            def missing(row):
                raise ImageFailure("missing_input")

            with self.assertRaises(InputError):
                run_baseline(self.model(), images, pairs, missing, config, "development", root / "out",
                             loaded[5], self.provenance(), integrity=loaded[-1])
            self.assertTrue((root / "out/development_scores.json").exists())
            self.assertTrue((root / "out/failure.json").exists())
            self.assertFalse((root / "out/selection.json").exists())

    def test_final_test_failure_counts_and_fatal_attempt_claim_are_retained(self):
        for fatal in (False, True):
            with self.subTest(fatal=fatal), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = self.develop(root)
                _, loaded = self.reviewed(root, "test", run)
                images, pairs, _, config, info, hashes, integrity = loaded
                count = 0

                def load(row, fatal=fatal):
                    nonlocal count
                    count += 1
                    if fatal and count == 1:
                        return self.pixels(row)
                    if fatal:
                        raise RuntimeError("synthetic resource failure")
                    raise ImageFailure("decode_error")

                if fatal:
                    with self.assertRaises(RuntimeError):
                        run_baseline(self.model(), images, pairs, load, config, "test", root / "out", hashes,
                                     self.provenance(), info=info, integrity=integrity)
                    self.assertTrue((root / "out/failure.json").exists())
                    self.assertFalse((root / "out/report.json").exists())
                    events = (root / "out/events.jsonl").read_text()
                    self.assertIn('"aborted"', events)
                    self.assertIn('"ok"', events)
                else:
                    report = run_baseline(self.model(), images, pairs, load, config, "test", root / "out", hashes,
                                          self.provenance(), info=info, integrity=integrity)
                    self.assertEqual(report["test"]["status_counts"], {"decode_error": 2})
                    self.assertEqual(report["test"]["coverage"], {"genuine": 0., "impostor": 0.})
                    self.assertIsNone(report["test"]["valid_scores_only"]["tar"])
                self.assertTrue((run / "final_test_frozen_claim.json").exists())

    def test_changed_development_selection_report_or_runtime_stops_before_test_pixels(self):
        for change in ("cutoff", "incomplete", "runtime", "bytes"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = self.develop(root)
                _, loaded = self.reviewed(root, "test", run)
                images, pairs, _, config, info, hashes, integrity = loaded
                provenance = self.provenance()
                if change == "cutoff":
                    path = run / "selection.json"
                    selection = json.loads(path.read_text())
                    selection["threshold"]["cutoff"] = 99
                    path.write_text(json.dumps(selection))
                    # Rebinding digests cannot evade semantic development replay.
                    hashes["baseline_selection"] = digest(path)
                elif change == "incomplete":
                    (run / "report.json").unlink()
                elif change == "runtime":
                    provenance["packages"] = {"runtime": "changed"}
                else:
                    (run / "development_scores.json").write_text("[]")
                loader = unittest.mock.Mock()
                with self.assertRaises((InputError, OSError)):
                    run_baseline(self.model(), images, pairs, loader, config, "test", root / "out", hashes,
                                 provenance, info=info)
                loader.assert_not_called()
                self.assertFalse((run / "final_test_frozen_claim.json").exists())

    def test_test_scores_change_rates_without_changing_saved_cutoff(self):
        import torch
        reports = []
        for value in (-1., 1.):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = self.develop(root)
                _, loaded = self.reviewed(root, "test", run)
                images, pairs, _, config, info, hashes, integrity = loaded
                with patch("precise_matching.testing.cosine_scores", return_value=torch.tensor([value])):
                    reports.append(run_baseline(self.model(), images, pairs, self.pixels, config, "test",
                                                root / "out", hashes, self.provenance(), info=info,
                                                integrity=integrity))
        self.assertEqual(reports[0]["threshold"], reports[1]["threshold"])
        self.assertEqual(reports[0]["test"]["all_attempts_failure_as_reject"]["tar"], 0)
        self.assertEqual(reports[1]["test"]["all_attempts_failure_as_reject"]["tar"], 1)

    def test_cli_two_phase_decode_and_compatibility_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.setup_inputs(root)
            for name in ("baseline", "data", "encoders", "images", "testing", "training"):
                path = root / ("code/precise_matching/" + name + ".py")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# synthetic source inventory\n")
            (root / "code/run_precise_frozen.py").write_text("# synthetic entry source\n")
            lock = root / "code/precise_matching/environment/requirements-linux-x86_64-py312-cu124.lock"
            lock.parent.mkdir()
            lock.write_text("# synthetic environment lock\n")
            baseline = root / "local/precise/runs/development"
            output = root / "local/precise/runs/test"
            for phase, run, dest in (("development", None, baseline), ("test", baseline, output)):
                args, _ = self.reviewed(root, phase, run)
                cli_args = [phase, *[value for name, path in zip(
                    ("images", "pairs", "paths", "config", "review", "source-root"), args[:6], strict=True)
                    for value in ("--" + name, str(path))], "--weights", str(root / "fake-weights"),
                    "--output", str(dest)]
                if run is not None:
                    cli_args += ["--baseline-run", str(run), "--report", str(root / "reports/test.json")]
                with patch.object(entry, "ROOT", root), patch.object(entry, "load_encoder", return_value=self.model()), \
                        patch.object(entry, "git_output", side_effect=[b"", "synthetic-commit\n"]), \
                        patch.object(entry, "package_versions", return_value={"runtime": "synthetic"}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(entry.main(cli_args), 0)
                if phase == "development":
                    test_args, _ = self.reviewed(root, "test", baseline)
                    blocked_args = ["test", *[value for name, path in zip(
                        ("images", "pairs", "paths", "config", "review", "source-root"), test_args[:6], strict=True)
                        for value in ("--" + name, str(path))], "--weights", str(root / "fake-weights"),
                        "--baseline-run", str(baseline), "--output", str(root / "local/precise/runs/blocked")]
                    with patch.object(entry, "ROOT", root), patch.object(entry, "load_encoder") as model_load, \
                            patch.object(entry, "git_output", side_effect=[b"", "synthetic-commit\n"]), \
                            patch.object(entry, "package_versions", return_value={"runtime": "incompatible"}), \
                            contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(entry.main(blocked_args), 1)
                    model_load.assert_not_called()
                    self.assertFalse((baseline / "final_test_frozen_claim.json").exists())
            report = json.loads((root / "reports/test.json").read_text())
            self.assertEqual(report["test"]["status_counts"], {"ok": 2})
            self.assertNotIn(str(root), json.dumps(report))
            self.assertFalse(any((root / "data").glob("train-*.png")))
            # The combined CSV is exactly what common-list comparison expects.
            from precise_matching.comparison import compare_methods
            images, pairs = load_protocol(root / "images.csv", root / "pairs.csv")
            scores = load_scores(output / "scores.csv")
            compared = compare_methods(images, pairs, {"frozen_dinov2": scores, "adapted_dinov2": scores}, "0.001")
            self.assertEqual(compared["methods"]["frozen_dinov2"]["threshold"], report["threshold"])
            self.assertEqual(evaluate_development(images, pairs, [s for s in scores if s.pair_id.startswith("development")],
                                                 "0.001")["threshold"], report["threshold"])


if __name__ == "__main__":
    unittest.main()
