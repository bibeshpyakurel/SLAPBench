"""New-cohort offline reporting provenance; all fixtures are synthetic."""

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from precise_matching.data import InputError  # noqa: E402
from precise_matching.encoders import WEIGHTS_SHA256  # noqa: E402
from precise_matching.fixed_pairs import GROUP_SCOPE, FixedConfig  # noqa: E402
from precise_matching.training import digest  # noqa: E402
from report_precise_group_holdout import check_final_training, content_overlap  # noqa: E402
from render_precise_shared_report import summary_rows  # noqa: E402
from test_precise_fixed_report import fixture  # noqa: E402
from precise_matching.fixed_evaluation import fixed_summary  # noqa: E402
from precise_matching.evaluation import Score  # noqa: E402


class NumericGroupReportTests(unittest.TestCase):
    def test_cross_phase_content_duplicates_are_detected_without_pixels(self):
        training, testing = fixture()
        names = {key for rows in (training, testing) for p in rows for key in (p.image_a, p.image_b)}
        bundle = {"images": {key: {"sha256": "raw-" + key, "decoded_sha256": "decoded-" + key} for key in names}}
        self.assertEqual(content_overlap(bundle, training, testing), {"sha256": 3, "decoded_sha256": 3})
        bundle["images"][testing[0].image_b] = bundle["images"][training[0].image_a]
        self.assertEqual(content_overlap(bundle, training, testing), {"sha256": 3, "decoded_sha256": 3})

    def test_fresh_complete_schedule_and_parameter_verification_are_required(self):
        config = FixedConfig(warmup_steps=0)
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); (run / "attempt-0001").mkdir()
            (run / "attempt-0001/attempt.json").write_text(json.dumps({"start_epoch": 0, "resume_sha256": None}))
            checkpoint = run / "final.pt"; checkpoint.write_bytes(b"SYNTHETIC CHECKPOINT, NO MODEL")
            record = {"config": asdict(config), "provenance": {"benchmark_scope": GROUP_SCOPE,
                "weights_sha256": WEIGHTS_SHA256, "input_sha256": {"bundle": "b", "config": "c"}}}
            (run / "run_record.json").write_text(json.dumps(record))
            history = [{"epoch": i, "pairs_seen": 2} for i in (1, 2, 3)]
            selection = {"checkpoint": "final.pt", "training_complete": True, "checkpoint_rule": "fixed_final_epoch",
                         "test_scores_accessed": False, "epochs_completed": 3, "history": history}
            (run / "selection.json").write_text(json.dumps(selection))
            identity = {"checkpoint_sha256": digest(checkpoint), "selection_sha256": digest(run / "selection.json")}
            verification = {"passed": True, "base_weights_sha256": WEIGHTS_SHA256, **identity,
                            "epochs_completed": 3, "optimizer_updates_completed": config.steps(2),
                            "counts": {"changed_frozen_tensors": 0}, "history": history}
            check_final_training(run, verification, config, [None, None], identity, "b", "c")
            for update in ({"optimizer_updates_completed": 0}, {"passed": False},
                           {"base_weights_sha256": "wrong"}, {"counts": {"changed_frozen_tensors": 1}},
                           {"history": [{"pairs_seen": 1}] * 3}):
                with self.subTest(update=update), self.assertRaises(InputError):
                    check_final_training(run, {**verification, **update}, config, [None, None], identity, "b", "c")
            (run / "attempt-0001/attempt.json").write_text(json.dumps({"start_epoch": 1, "resume_sha256": "old"}))
            with self.assertRaises(InputError):
                check_final_training(run, verification, config, [None, None], identity, "b", "c")

    def test_checkpoint_schedule_checks_epoch_tail_updates_and_complete_membership(self):
        from verify_precise_shared_checkpoint import completed_schedule
        config = FixedConfig(warmup_steps=0)
        saved = {"history": [{"epoch": i, "pairs_seen": 33} for i in (1, 2, 3)],
                 "scheduler": {"last_epoch": 6}}
        self.assertEqual(completed_schedule(saved, config), 6)
        for update in ({"scheduler": {"last_epoch": 5}},
                       {"history": [{"epoch": i, "pairs_seen": 33} for i in (1, 2)]},
                       {"history": [{"epoch": i, "pairs_seen": 32 + i} for i in (1, 2, 3)]}):
            with self.subTest(update=update), self.assertRaises(InputError):
                completed_schedule({**saved, **update}, config)

    def test_cli_replays_new_cohort_without_historical_qwen_or_pixels(self):
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO
        from unittest.mock import patch
        from precise_matching.fixed_pairs import FixedPair, canonical_digest
        from precise_matching.evaluation import select_cutoff, write_scores
        from report_precise_shared_pairs import public_training_threshold
        from precise_matching.training import write_json
        import report_precise_group_holdout as report
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); private = root / "local/precise"; private.mkdir(parents=True)
            train = [FixedPair("tr-g", "101_synthetic_a_15.jpeg", "101_synthetic_b_15.jpeg", "genuine", "thumbs", False),
                     FixedPair("tr-i", "101_synthetic_a_15.jpeg", "102_synthetic_b_15.jpeg", "impostor", "thumbs", False)]
            test = [FixedPair("te-g", "201_synthetic_a_15.jpeg", "201_synthetic_b_15.jpeg", "genuine", "thumbs", False),
                    FixedPair("te-i", "201_synthetic_a_15.jpeg", "202_synthetic_b_15.jpeg", "impostor", "thumbs", False)]
            config = FixedConfig(warmup_steps=0)
            inputs = {key: private / (key + ".json") for key in ("bundle", "config", "cohort_audit", "checkpoint_verification", "review")}
            source_sha = {}
            for name in (report.TRAIN_JSON, *report.LISTS.values()):
                path = private / name
                if name == report.TRAIN_JSON:
                    path.write_text("{}")
                else:
                    label = "genuine" if "genuine" in name else "impostor"
                    pairs = train if name.startswith("train") else test
                    path.write_text("".join(f"{p.image_a} {p.image_b} 0 | 1 2 sum=3 nfing=2\n"
                                            for p in pairs if p.label == label))
                source_sha[name] = digest(path)
            audit = {"training_json_matches_supplied_training_lists_and_scores": True,
                     "new_train_test_overlap": {"pairs": 0, "images": 0, "numeric_groups": 0},
                     "sources": {name: {"sha256": value} for name, value in source_sha.items()},
                     "comparison_with_completed_benchmark": {"new_testing_previously_exposed_to_completed_dino_training": {},
                                                             "test_membership_change": {}}}
            write_json(inputs["cohort_audit"], audit)
            names = {key for p in train + test for key in (p.image_a, p.image_b)}
            bundle = {"scope": GROUP_SCOPE, "labels": "as_supplied", "training": [asdict(p) for p in train],
                      "testing": [asdict(p) for p in test], "source_sha256": {**source_sha, "audit": digest(inputs["cohort_audit"])},
                      "images": {key: {"sha256": "raw-" + key, "decoded_sha256": "decoded-" + key} for key in names}}
            write_json(inputs["bundle"], bundle); write_json(inputs["config"], asdict(config))
            hashes = {key: digest(inputs[key]) for key in ("bundle", "config")}
            write_json(inputs["review"], {"scope": GROUP_SCOPE, "input_sha256": hashes})
            hashes["review"] = digest(inputs["review"])
            runs = private / "runs"; training = runs / "adapted-training"; training.mkdir(parents=True)
            (training / "attempt-0001").mkdir()
            write_json(training / "attempt-0001/attempt.json", {"start_epoch": 0, "resume_sha256": None})
            checkpoint = training / "final.pt"; checkpoint.write_bytes(b"SYNTHETIC, NO REAL MODEL")
            history = [{"epoch": i, "pairs_seen": 2} for i in (1, 2, 3)]
            selection = {"checkpoint": "final.pt", "training_complete": True, "checkpoint_rule": "fixed_final_epoch",
                         "test_scores_accessed": False, "epochs_completed": 3, "history": history}
            write_json(training / "selection.json", selection)
            identity = {"checkpoint_sha256": digest(checkpoint), "selection_sha256": digest(training / "selection.json")}
            verification = {"passed": True, "base_weights_sha256": WEIGHTS_SHA256, **identity,
                            "epochs_completed": 3, "optimizer_updates_completed": config.steps(2),
                            "counts": {"changed_frozen_tensors": 0}, "history": history}
            write_json(inputs["checkpoint_verification"], verification)
            for arm in report.ARMS:
                short = arm.split("_")[0]; cal = runs / (short + "-calibration"); outcome = runs / (short + "-test")
                cal.mkdir(); outcome.mkdir()
                model = {"arm": arm, **(identity if short == "adapted" else {})}
                provenance = {"benchmark_scope": GROUP_SCOPE, "model_identity": model, "input_sha256": hashes,
                              "weights_sha256": WEIGHTS_SHA256, "code_commit": "SYNTHETIC", "packages": {}, "hardware": {}}
                if short == "adapted":
                    write_json(training / "run_record.json", {"config": asdict(config), "provenance": provenance})
                write_scores(cal / "scores.csv", [Score(p.pair_id, "ok", .9 if p.label == "genuine" else .2) for p in train])
                write_json(cal / "run_record.json", {"phase": "training_calibration", "config": asdict(config),
                    "pairs_sha256": canonical_digest([asdict(p) for p in train]), "provenance": provenance})
                threshold = select_cutoff([.2], config.target_far); threshold["selected_on"] = "training_impostors_resubstitution"
                self.assertEqual(public_training_threshold(threshold)["training_valid_impostors"], 1)
                write_json(cal / "selection.json", {"selection_frozen": True, "test_scores_accessed": False,
                    "checkpoint_rule": config.checkpoint_rule, "threshold_rule": config.threshold_rule, "model_identity": model,
                    "threshold": threshold, "scores_sha256": digest(cal / "scores.csv"),
                    "run_record_sha256": digest(cal / "run_record.json")})
                test_sha = canonical_digest([asdict(p) for p in test]); selection_sha = digest(cal / "selection.json")
                write_json(cal / "test_claim.json", {"output": str(outcome), "selection_sha256": selection_sha, "testing_pairs_sha256": test_sha})
                write_json(outcome / "run_record.json", {"phase": "held_out_pair_test", "provenance": provenance,
                    "config": asdict(config), "testing_pairs_sha256": test_sha, "selection_sha256": selection_sha})
                scores = [Score(p.pair_id, "ok", .8 if p.label == "genuine" else .1) for p in test]
                write_scores(outcome / "scores.csv", scores)
                write_json(outcome / "aggregate.json", {"scope": "held_out_numeric_prefix_groups_identity_semantics_unconfirmed",
                           "results": fixed_summary(test, scores, .2), "threshold": threshold})
                write_json(outcome / "completion.json", {"complete": True, "scores_sha256": digest(outcome / "scores.csv"),
                                                         "aggregate_sha256": digest(outcome / "aggregate.json")})
            args = [value for key, path in inputs.items() for value in ("--" + key.replace("_", "-"), str(path))]
            output = root / "reports/new.json"; receipt = private / "receipt.json"
            args += ["--run-root", str(runs), "--intake", str(private), "--output", str(output), "--private-receipt", str(receipt)]
            with patch.object(report, "ROOT", root), redirect_stdout(StringIO()):
                self.assertEqual(report.main(args), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result["cross_phase_content_hash_overlap"], {"sha256": 0, "decoded_sha256": 0})
            self.assertEqual(result["benchmark_scope"], GROUP_SCOPE)
            self.assertNotIn("metrics", result["historical_qwen_scores"])
            self.assertEqual(result["sdk_descriptive"]["verifinger_fusion"]["pooled"]["zero_impostor"], 1)
            self.assertNotIn("jpeg", output.read_text())
            with patch.object(report, "ROOT", root), redirect_stderr(StringIO()):
                self.assertEqual(report.main(args), 1)

    def test_zero_ambiguity_is_not_named_as_two_excluded_rows(self):
        _, testing = fixture()
        from dataclasses import replace
        testing = [replace(p, numeric_identity_disagreement=False) for p in testing]
        scores = [Score(p.pair_id, "ok", .9 if p.label == "genuine" else .1) for p in testing]
        results = fixed_summary(testing, scores, .5)
        data = {"dino": {arm: {"results": results, "threshold": {"cutoff": .5}}
                         for arm in ("frozen_dinov2", "adapted_dinov2")}}
        rows = summary_rows(data)
        self.assertNotIn("excluding_two_flagged_test_pairs", {r["population"] for r in rows})
        self.assertEqual([r["tar"] for r in rows if r["population"] == "label_ambiguity_sensitivity"], [1, 1])


if __name__ == "__main__":
    unittest.main()
