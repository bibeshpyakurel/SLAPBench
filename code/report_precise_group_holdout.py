"""Reproduce the reviewed numeric-group-holdout DINO contribution offline.

No inference, filtering, threshold search, identity confirmation or old-checkpoint
reuse. Original shared-cohort scores and report remain separate and unchanged.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from audit_precise_revised_cohort import LISTS, TRAIN_JSON
from precise_matching.data import InputError
from precise_matching.encoders import WEIGHTS_SHA256
from precise_matching.fixed_pairs import GROUP_SCOPE, FixedConfig, FixedPair, validate_group_holdout
from precise_matching.training import digest, write_json
from report_precise_shared_pairs import ARMS, dependency_counts, descriptive_scores, paired_dino, read_dino, read_sdk

ROOT = Path(__file__).resolve().parents[1]


def content_overlap(bundle, training, testing):
    """Use existing reviewed image hashes only; never decode pixels for reporting."""
    groups = [{key for p in rows for key in (p.image_a, p.image_b)} for rows in (training, testing)]
    return {field: len({bundle["images"][key][field] for key in groups[0]}
                       & {bundle["images"][key][field] for key in groups[1]})
            for field in ("sha256", "decoded_sha256")}


def check_final_training(run, verification, config, training, identity, bundle_sha, config_sha, *, expected_base_sha=WEIGHTS_SHA256, expected_profile=None, resume_review=None):
    record = json.loads((run / "run_record.json").read_text())
    selection = json.loads((run / "selection.json").read_text())
    first = json.loads((run / "attempt-0001/attempt.json").read_text())
    checkpoint = (run / selection["checkpoint"]).resolve()
    attempts = sorted(p.name for p in run.iterdir() if p.is_dir() and p.name.startswith("attempt-"))
    valid_attempts = attempts == ["attempt-0001"]
    if resume_review is not None:
        from check_precise_native_resume import check_resume_chain
        if expected_profile != "dinov2_large_native_v1":
            raise InputError("Reviewed resume reporting is limited to the explicit native profile.")
        check_resume_chain(resume_review, run, record, selection)
        valid_attempts = attempts == ["attempt-0001", "attempt-0002"]
    if (not valid_attempts or not checkpoint.is_relative_to(run.resolve())
            or record.get("config") != asdict(config)
            or record["provenance"].get("weights_sha256") != expected_base_sha
            or record["provenance"].get("benchmark_scope") != GROUP_SCOPE
            or expected_profile is not None and record["provenance"].get("model_profile") != expected_profile
            or record["provenance"]["input_sha256"]["bundle"] != bundle_sha
            or record["provenance"]["input_sha256"]["config"] != config_sha
            or first.get("resume_sha256") is not None or first.get("start_epoch") != 0
            or selection.get("training_complete") is not True
            or selection.get("checkpoint_rule") != "fixed_final_epoch"
            or selection.get("test_scores_accessed") is not False
            or selection.get("epochs_completed") != config.epochs
            or digest(checkpoint) != identity["checkpoint_sha256"]
            or digest(run / "selection.json") != identity["selection_sha256"]
            or verification.get("passed") is not True
            or verification.get("base_weights_sha256") != expected_base_sha
            or verification.get("checkpoint_sha256") != identity["checkpoint_sha256"]
            or verification.get("selection_sha256") != identity["selection_sha256"]
            or verification.get("epochs_completed") != config.epochs
            or verification.get("optimizer_updates_completed") != config.steps(len(training))
            or verification.get("counts", {}).get("changed_frozen_tensors") != 0
            or verification.get("history") != selection.get("history")
            or len(verification.get("history", [])) != config.epochs
            or any(h["pairs_seen"] != len(training) for h in verification["history"])):
        raise InputError("Fresh final-checkpoint provenance does not match the reviewed numeric-group run.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "run-root", "intake", "cohort-audit", "checkpoint-verification", "review", "private-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output, receipt = args.output.resolve(), args.private_receipt.resolve()
        if (not output.is_relative_to(ROOT / "reports") or output.exists()
                or not receipt.is_relative_to(ROOT / "local/precise") or receipt.exists()):
            raise InputError("Use new aggregate report and private receipt paths.")
        sources = {name: getattr(args, name) for name in ("bundle", "config", "cohort_audit", "checkpoint_verification", "review")}
        sources.update({name: args.intake / name for name in (TRAIN_JSON, *LISTS.values())})
        for arm in ARMS:
            short = arm.split("_")[0]
            for phase in ("test", "calibration"):
                path = args.run_root / (short + "-" + phase)
                sources.update({str(p.resolve()): p for p in path.iterdir() if p.suffix in {".csv", ".json"}})
        run = args.run_root / "adapted-training"
        for relative in ("run_record.json", "selection.json", "attempt-0001/attempt.json"):
            sources["training_" + relative] = run / relative
        initial = {name: digest(path) for name, path in sources.items()}
        bundle = json.loads(args.bundle.read_text())
        if bundle.get("scope") != GROUP_SCOPE or bundle.get("labels") != "as_supplied":
            raise InputError("Report requires the reviewed numeric-group benchmark.")
        training = [FixedPair(**r) for r in bundle["training"]]
        testing = [FixedPair(**r) for r in bundle["testing"]]
        group_validation = validate_group_holdout(training, testing)
        config = FixedConfig(**json.loads(args.config.read_text()));config.validate(len(training))
        audit = json.loads(args.cohort_audit.read_text())
        if (bundle["source_sha256"]["audit"] != initial["cohort_audit"]
                or audit.get("training_json_matches_supplied_training_lists_and_scores") is not True
                or audit.get("new_train_test_overlap") != {"pairs": 0, "images": 0, "numeric_groups": 0}):
            raise InputError("Source audit/cohort binding is inconsistent.")
        for name in (TRAIN_JSON, *LISTS.values()):
            if initial[name] != bundle["source_sha256"][name] or initial[name] != audit["sources"][name]["sha256"]:
                raise InputError("A supplied source changed since review/preparation.")
        review = json.loads(args.review.read_text())
        if (review.get("scope") != GROUP_SCOPE
                or review.get("input_sha256") != {key: initial[key] for key in ("bundle", "config")}):
            raise InputError("Review does not bind the evaluated benchmark/configuration.")
        dino, scores = {}, {}
        for arm in ARMS:
            short = arm.split("_")[0]
            test, calibration = args.run_root / (short + "-test"), args.run_root / (short + "-calibration")
            dino[arm], scores[arm] = read_dino(test, calibration, training, testing, config, arm)
            raw = json.loads((test / "aggregate.json").read_text())
            record = json.loads((test / "run_record.json").read_text())
            if (raw.get("scope") != "held_out_numeric_prefix_groups_identity_semantics_unconfirmed"
                    or record["provenance"].get("weights_sha256") != WEIGHTS_SHA256
            or record["provenance"].get("benchmark_scope") != GROUP_SCOPE
                    or any(record["provenance"]["input_sha256"][key] != initial[key] for key in ("bundle", "config", "review"))):
                raise InputError("Evaluated arm does not bind the new numeric-group review/scope.")
        verification = json.loads(args.checkpoint_verification.read_text())
        check_final_training(run, verification, config, training, dino["adapted_dinov2"]["model_identity"],
                             initial["bundle"], initial["config"])
        filenames = {label: LISTS["test_" + label] for label in ("genuine", "impostor")}
        sdk, _ = read_sdk(args.intake, testing, filenames=filenames)
        hashes = content_overlap(bundle, training, testing)
        if any(hashes.values()):
            raise InputError("Cross-phase image-content hashes overlap in the group-holdout report.")
        report = {"schema_version": 1, "scope": "numeric_group_holdout_identity_semantics_unconfirmed",
            "benchmark_scope": GROUP_SCOPE, "study_date": "2026-11-07", "review_date": "2026-11-07",
            "configuration": asdict(config), "training_pairs": len(training), "testing_pairs": len(testing),
            "dependence": dependency_counts(training, testing), "numeric_group_validation": group_validation,
            "cross_phase_content_hash_overlap": hashes,
            "prior_adaptation_exposure": audit["comparison_with_completed_benchmark"]["new_testing_previously_exposed_to_completed_dino_training"],
            "old_vs_new_testing_membership": audit["comparison_with_completed_benchmark"]["test_membership_change"],
            "source_audit_sha256": initial["cohort_audit"],
            "input_sha256": {key: initial[key] for key in ("bundle", "config", "review")},
            "source_sha256": {name: initial[name] for name in (TRAIN_JSON, *LISTS.values())}, "training": verification,
            "fresh_initialization": {"base_weights_sha256": WEIGHTS_SHA256, "first_attempt_resumed": False,
                                     "rule": "Fresh pinned base; final epoch selected in advance."},
            "dino": dino, "paired_dino": paired_dino(testing, scores[ARMS[0]], scores[ARMS[1]],
                dino[ARMS[0]]["threshold"]["cutoff"], dino[ARMS[1]]["threshold"]["cutoff"]),
            "sdk_descriptive": {arm: descriptive_scores(testing, outcomes) for arm, outcomes in sdk.items()},
            "historical_qwen_scores": {"status": "not_applicable_changed_testing_pairs"},
            "trained_qwen_comparison": {"status": "pending_matching_new_cohort_outputs_and_run_provenance", "owner": "Konain",
                                        "local_duplicate_finetuning_required": False},
            "limits": ["Numeric-prefix groups, images and exact pairs are disjoint in the supplied cohort; biological identities remain unconfirmed.",
                "The old adapted checkpoint saw every new testing group/image and is not used to initialize this adaptation.",
                "Prior research results on this dataset are known; this is not an untouched-dataset or independent-replication claim.",
                "All padding-alias groups are in training, limiting representativeness of the new group holdout.",
                "Thresholds use training calibration with resubstitution optimism, not independent development selection.",
                "Rows reuse images/groups; no independent-pair confidence interval, significance or population low-FAR guarantee.",
                "SDK zeros remain raw values; component-positive selection and unknown extraction status limit inference.",
                "No historical Qwen export is substituted for the changed test list; actual matching trained-model outputs remain pending.",
                "Descriptive test ROC/EER cannot select checkpoints, cutoffs or hyperparameters."]}
        if any(digest(path) != initial[name] for name, path in sources.items()):
            raise InputError("An input changed during new-cohort report reproduction.")
        receipt.parent.mkdir(parents=True, exist_ok=True);output.parent.mkdir(parents=True, exist_ok=True)
        write_json(receipt, {"input_sha256": initial, "source_paths": {key: str(path.resolve()) for key, path in sources.items()},
            "code_sha256": digest(Path(__file__)), "no_image_pixels_read": True, "no_model_inference": True, "no_threshold_selection": True})
        write_json(output, report)
    except (InputError, OSError, ValueError, TypeError, KeyError, AttributeError):
        print("Numeric-group report did not complete; preserve original runs and inspect private evidence.", file=sys.stderr)
        return 1
    print("Numeric-group report reproduced from sealed outcomes; review before publication.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
