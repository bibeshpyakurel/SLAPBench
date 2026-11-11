"""Reproduce sealed native-large outcomes and exact-pair visual/Qwen comparison.

Offline saved-outcome verification only. No images, model inference, threshold
selection or rewriting of completed small-model/Qwen source reports.
"""

import argparse
from datetime import date, datetime
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from audit_precise_qwen_results import paired_saved_decisions, summary_check
from audit_precise_revised_cohort import LISTS, TRAIN_JSON
from compare_precise_qwen_ab import operating_summary, read_ab
from precise_matching.data import InputError
from precise_matching.fixed_pairs import GROUP_SCOPE, FixedConfig, FixedPair, validate_group_holdout
from precise_matching.native_large import MODEL_ID, REVISION, WEIGHTS_SHA256, ORIGINAL_SIZE, PADDED_SIZE
from precise_matching.training import digest, write_json
from report_precise_group_holdout import check_final_training, content_overlap
from report_precise_shared_pairs import ARMS, dependency_counts, descriptive_scores, paired_dino, read_dino, read_sdk

ROOT = Path(__file__).resolve().parents[1]


def verify_runtime_source(provenance):
    commit = provenance["code_commit"]
    for name, expected in provenance["source_sha256"].items():
        content = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != expected:
            raise InputError("Recorded runtime source does not match its committed bytes.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "review", "run-root", "checkpoint-verification", "sdk-intake", "cohort-audit",
                 "small-aggregate", "qwen-intake", "qwen-aggregate", "output", "private-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--study-date", required=True)
    parser.add_argument("--review-date", default="2026-11-08")
    parser.add_argument("--resume-review", type=Path)
    args = parser.parse_args(argv)
    try:
        if (args.output.exists() or not args.output.resolve().is_relative_to(ROOT / "reports")
                or args.private_receipt.exists() or not args.private_receipt.resolve().is_relative_to(ROOT / "local/precise")):
            raise InputError("Use new public native aggregate and private receipt paths.")
        date.fromisoformat(args.study_date);date.fromisoformat(args.review_date)
        sources = {k: getattr(args, k) for k in ("bundle", "config", "review", "checkpoint_verification", "cohort_audit",
                                                 "small_aggregate", "qwen_aggregate")}
        if args.resume_review is not None:
            sources["resume_review"] = args.resume_review
        for arm in ARMS:
            short = arm.split("_")[0]
            for phase in ("test", "calibration"):
                sources.update({str(p.resolve()): p for p in (args.run_root / (short + "-" + phase)).iterdir()
                                if p.suffix in {".csv", ".json"}})
        training_run = args.run_root / "adapted-training"
        for relative in ("run_record.json", "selection.json", "attempt-0001/attempt.json"):
            sources["training_" + relative] = training_run / relative
        sources.update({"sdk_" + name: args.sdk_intake / name for name in (TRAIN_JSON, *LISTS.values())})
        qwen_reference = json.loads(args.qwen_aggregate.read_text())
        for name, expected in qwen_reference["source_sha256"].items():
            path = args.qwen_intake / name
            if digest(path) != expected:raise InputError("Preserved Qwen evidence changed since source review.")
            sources["qwen_" + name] = path
        initial = {k: digest(p) for k, p in sources.items()}
        bundle = json.loads(args.bundle.read_text());review = json.loads(args.review.read_text())
        if bundle.get("scope") != GROUP_SCOPE or bundle.get("labels") != "as_supplied":
            raise InputError("Native report requires the reviewed numeric-group cohort.")
        training, testing = ([FixedPair(**r) for r in bundle[k]] for k in ("training", "testing"))
        group_validation = validate_group_holdout(training, testing)
        config = FixedConfig(**json.loads(args.config.read_text()));config.validate(len(training))
        from run_precise_native_large import validate_native_policy
        validate_native_policy(bundle, config, review)
        if review["input_sha256"] != {k: initial[k] for k in ("bundle", "config")}:
            raise InputError("Native report review/input binding differs.")
        overlap = content_overlap(bundle, training, testing)
        if any(overlap.values()):raise InputError("Native report has cross-phase image-content overlap.")
        audit = json.loads(args.cohort_audit.read_text())
        if (bundle["source_sha256"]["audit"] != initial["cohort_audit"]
                or audit.get("new_train_test_overlap") != {"pairs": 0, "images": 0, "numeric_groups": 0}):
            raise InputError("Original cohort audit binding changed.")
        for name in (TRAIN_JSON, *LISTS.values()):
            if initial["sdk_" + name] != bundle["source_sha256"][name]:
                raise InputError("SDK/training source changed from reviewed cohort.")
        dino, native_scores = {}, {}
        for arm in ARMS:
            short = arm.split("_")[0];test = args.run_root / (short + "-test")
            dino[arm], native_scores[arm] = read_dino(test, args.run_root / (short + "-calibration"), training, testing, config, arm)
            record = json.loads((test / "run_record.json").read_text());provenance = record["provenance"]
            if (provenance.get("model_profile") != "dinov2_large_native_v1"
                    or provenance.get("model") != MODEL_ID or provenance.get("model_revision") != REVISION
                    or provenance.get("weights_sha256") != WEIGHTS_SHA256
                    or provenance.get("image_preprocessing") != {"original_width_height": list(ORIGINAL_SIZE),
                        "padded_width_height": list(PADDED_SIZE), "resampling": False, "padding": "centered_white"}
                    or provenance.get("attention") != "sdpa_forced_flash_only"
                    or provenance.get("inference_precision") != "bf16"
                    or any(provenance["input_sha256"][k] != initial[k] for k in ("bundle", "config", "review"))):
                raise InputError("Native test outcomes belong to another declared execution profile.")
            verify_runtime_source(provenance)
        verification = json.loads(args.checkpoint_verification.read_text())
        check_final_training(training_run, verification, config, training, dino["adapted_dinov2"]["model_identity"],
            initial["bundle"], initial["config"], expected_base_sha=WEIGHTS_SHA256, expected_profile="dinov2_large_native_v1",
            resume_review=args.resume_review)
        resume = None
        if args.resume_review is not None:
            from check_precise_native_resume import check_resume_chain, inspect_resume_review
            record = json.loads((training_run / "run_record.json").read_text())
            selected = json.loads((training_run / "selection.json").read_text())
            _, _, extra_paths, _ = inspect_resume_review(args.resume_review, training_run, record)
            for key, path in extra_paths.items():
                sources["resume_" + key] = path;initial["resume_" + key] = digest(path)
            for relative in ("attempt-0002/attempt.json", "attempt-0002/result.json"):
                sources["resume_" + relative] = training_run / relative
                initial["resume_" + relative] = digest(training_run / relative)
            resume = check_resume_chain(args.resume_review, training_run, record, selected)
        small = json.loads(args.small_aggregate.read_text())
        if small.get("benchmark_scope") != GROUP_SCOPE or small["input_sha256"]["bundle"] != initial["bundle"]:
            raise InputError("Prior small-model report does not bind the same exact cohort.")
        if (qwen_reference.get("scope") != "source_verified_common_pair_decisions_run_provenance_incomplete"
                or qwen_reference.get("benchmark_scope") != GROUP_SCOPE
                or qwen_reference.get("outcome_membership_verified") is not True):
            raise InputError("Qwen source review has an unexpected scope.")
        prompt_path = args.qwen_intake / "training-source" / TRAIN_JSON
        if digest(prompt_path) != bundle["source_sha256"][TRAIN_JSON]:raise InputError("Qwen uses another training source.")
        prompt = json.loads(prompt_path.read_text());qwen = {}
        for kind in ("base", "finetuned"):
            decisions, tokens = read_ab({label: args.qwen_intake / f"res_test_{label}_2100_nonzero_{kind}_matched.txt"
                                        for label in ("genuine", "impostor")}, testing)
            summary = json.loads((args.qwen_intake / f"summary_{kind}_matched.json").read_text())
            reproduced = summary_check(summary, testing, decisions, prompt)
            if any(qwen_reference["qwen"][kind][k] != v for k, v in reproduced.items()):
                raise InputError("Qwen outcome review does not reproduce on the native cohort.")
            by_hand = {hand: operating_summary([p for p in testing if p.hand == hand], decisions)
                       for hand in ("left", "right", "thumbs")}
            if by_hand != qwen_reference["qwen"][kind]["by_slap_type"]:
                raise InputError("Qwen per-type outcomes do not reproduce from the exact testing attempts.")
            qwen[kind] = {**reproduced, "source_token_counts": tokens,
                "by_slap_type": by_hand,
                "paired_native_dino_to_qwen": {arm: paired_saved_decisions(testing, native_scores[arm], decisions,
                    dino[arm]["threshold"]["cutoff"]) for arm in ARMS}}
        sdk, _ = read_sdk(args.sdk_intake, testing, filenames={label: LISTS["test_" + label] for label in ("genuine", "impostor")})
        attempt = json.loads((training_run / "attempt-0001/attempt.json").read_text())
        selected = json.loads((training_run / "selection.json").read_text())
        elapsed = (datetime.fromisoformat(selected["finished_at_utc"]) - datetime.fromisoformat(attempt["started_at_utc"])).total_seconds()
        report = {"schema_version": 1, "scope": "native_large_numeric_group_holdout_with_source_verified_qwen_limits",
            "benchmark_scope": GROUP_SCOPE, "model_profile": "dinov2_large_native_v1", "study_date": args.study_date,
            "review_date": args.review_date, "configuration": asdict(config), "training_pairs": len(training), "testing_pairs": len(testing),
            "image_preprocessing": {"original_width_height": list(ORIGINAL_SIZE), "padded_width_height": list(PADDED_SIZE),
                                    "resampling": False, "padding": "centered_white"},
            "training_elapsed_seconds": elapsed, "training": verification, "dino": dino,
            "resume_execution": resume,
            "paired_native_dino": paired_dino(testing, native_scores["frozen_dinov2"], native_scores["adapted_dinov2"],
                dino["frozen_dinov2"]["threshold"]["cutoff"], dino["adapted_dinov2"]["threshold"]["cutoff"]),
            "prior_small_same_cohort": small["dino"], "qwen": qwen,
            "qwen_checkpoint_evidence": qwen_reference["checkpoint_evidence"],
            "qwen_source_run_claims": qwen_reference["source_run_claims"],
            "qwen_provenance_limits": qwen_reference["provenance_limits"],
            "sdk_descriptive": {arm: descriptive_scores(testing, scores) for arm, scores in sdk.items()},
            "dependence": dependency_counts(training, testing), "numeric_group_validation": group_validation,
            "cross_phase_content_hash_overlap": overlap,
            "alias_group_placement": audit.get("alias_group_placement", {}),
            "sdk_source_selection_status": audit.get("score_status", {}), "input_sha256": {k: initial[k] for k in ("bundle", "config", "review")},
            "source_reports_sha256": {k: initial[k] for k in ("small_aggregate", "qwen_aggregate", "cohort_audit")},
            "binary_qwen_roc_auc_eer": None,
            "limits": ["Architecture, resolution and numerical execution change jointly; this does not isolate resizing/model-size effects.",
                "Qwen native A/B operating points and DINO nominal-1%-training-FAR cutoffs are different policies, not matched test FAR.",
                "Training calibration is resubstitution, not independent development calibration or a test-FAR guarantee.",
                "Numeric-prefix separation does not establish biological identities or capture independence.",
                "All pairs/labels/thumbs are retained; dependent images/groups preclude independent-pair confidence claims.",
                "This dataset has prior known research/test results; no untouched-dataset or independent-replication claim.",
                "SDK zeros and positive-component-sum selection do not establish extraction success or full-pool performance.",
                "No checkpoint, cutoff or hyperparameter is selected from these final-test summaries."]}
        if any(digest(p) != initial[k] for k, p in sources.items()):raise InputError("An input changed during native report reproduction.")
        args.private_receipt.parent.mkdir(parents=True, exist_ok=True);args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.private_receipt, {"input_sha256": initial, "code_sha256": digest(Path(__file__)),
                   "image_pixels_read": False, "model_inference_performed": False, "new_selection": False})
        write_json(args.output, report)
    except (InputError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        print("Native report did not complete; preserve sealed sources and inspect private artifacts.", file=sys.stderr);return 1
    print("Native-large and exact-pair comparison aggregate reproduced offline.")
    return 0


if __name__ == "__main__":raise SystemExit(main())
