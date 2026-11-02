"""Reproduce shared-benchmark aggregates from immutable private saved outcomes.

No matcher inference, score repair, cutoff search or identity relabeling.
Historical Qwen scores remain explicitly distinct from a trained A/B result.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path, PureWindowsPath
import sys

from audit_precise_training_pairs import read_source
from precise_matching.data import InputError
from precise_matching.encoders import WEIGHTS_SHA256
from precise_matching.evaluation import Score, discrimination, load_scores
from precise_matching.fixed_evaluation import fixed_summary, inspect_calibration, matched_scores
from precise_matching.fixed_pairs import FixedConfig, FixedPair, canonical_digest, validate_pairs
from precise_matching.training import digest, write_json

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("frozen_dinov2", "adapted_dinov2")


def public_training_threshold(threshold):
    """Name the actual calibration population, despite the reused helper's keys."""
    names = {"allowed_development_false_accepts": "allowed_training_false_accepts",
             "development_false_accepts": "training_false_accepts",
             "development_valid_impostors": "training_valid_impostors"}
    if threshold.get("selected_on") != "training_impostors_resubstitution":
        raise InputError("The shared route requires training-only calibration provenance.")
    return {names.get(key, key): value for key, value in threshold.items()}


def dependency_counts(training, testing):
    """Aggregate reuse measurements, without treating ID aliases as verified people."""
    def population(pairs):
        endpoints = Counter(key for p in pairs for key in (p.image_a, p.image_b))
        prefixes = {Path(key).stem.split("_")[0] for key in endpoints}
        if any(not key.isascii() or not key.isdigit() for key in prefixes):
            raise InputError("Unrecognized numeric-group metadata for dependency accounting.")
        groups = {int(key) for key in prefixes}
        return {"pairs": len(pairs), "images": len(endpoints), "raw_prefix_groups": len(prefixes),
                "numeric_groups_unconfirmed_identity_semantics": len(groups),
                "images_in_multiple_pairs": sum(value > 1 for value in endpoints.values()),
                "maximum_pairs_using_one_image": max(endpoints.values(), default=0)}, set(endpoints), groups

    train, train_images, train_groups = population(training)
    test, test_images, test_groups = population(testing)
    train_pairs = {tuple(sorted((p.image_a, p.image_b))) for p in training}
    test_pairs = {tuple(sorted((p.image_a, p.image_b))) for p in testing}
    return {"training": train, "testing": test,
            "train_test_overlap": {"exact_pairs": len(train_pairs & test_pairs),
                                   "images": len(train_images & test_images),
                                   "numeric_groups": len(train_groups & test_groups)},
            "uncertainty_policy": "Descriptive finite-cohort rates and label sensitivity; no independent-pair intervals."}


def read_dino(test_root, calibration, training, testing, config, arm):
    test_root, calibration = Path(test_root).resolve(), Path(calibration).resolve()
    completion = json.loads((test_root / "completion.json").read_text())
    record = json.loads((test_root / "run_record.json").read_text())
    provenance = record["provenance"]
    if (completion.get("complete") is not True
            or completion.get("scores_sha256") != digest(test_root / "scores.csv")
            or completion.get("aggregate_sha256") != digest(test_root / "aggregate.json")
            or record.get("phase") != "held_out_pair_test"
            or record.get("config") != asdict(config)
            or record.get("testing_pairs_sha256") != canonical_digest([asdict(p) for p in testing])
            or provenance.get("model_identity", {}).get("arm") != arm):
        raise InputError("DINO test artifact does not match the completed shared benchmark.")
    selection = inspect_calibration(calibration, training, config, provenance)
    if record["selection_sha256"] != digest(calibration / "selection.json"):
        raise InputError("Test record references a changed calibration selection.")
    claim = json.loads((calibration / "test_claim.json").read_text())
    if (claim.get("output") != str(test_root) or claim.get("selection_sha256") != record["selection_sha256"]
            or claim.get("testing_pairs_sha256") != record["testing_pairs_sha256"]):
        raise InputError("Test claim and saved outcome provenance disagree.")
    scores = load_scores(test_root / "scores.csv")
    results = fixed_summary(testing, scores, selection["threshold"]["cutoff"])
    original = json.loads((test_root / "aggregate.json").read_text())
    if original.get("results") != results or original.get("threshold") != selection["threshold"]:
        raise InputError("Saved DINO aggregate does not reproduce from raw outcomes.")
    return {"threshold": public_training_threshold(selection["threshold"]), "results": results,
            "model_identity": provenance["model_identity"],
            "code_commit": provenance["code_commit"], "packages": provenance["packages"],
            "hardware": provenance["hardware"]}, scores


def read_sdk(intake, testing):
    lookup, paths = {}, []
    for label in ("genuine", "impostor"):
        path = Path(intake) / f"random_2100_{label}_seed2026.txt"
        paths.append(path)
        rows, _ = read_source(path, label)
        for row in rows:
            key = tuple(sorted(row.names))
            if key in lookup:
                raise InputError("Duplicate SDK source pair.")
            lookup[key] = row
    expected = {tuple(sorted((p.image_a, p.image_b))) for p in testing}
    if set(lookup) != expected:
        raise InputError("SDK outcomes do not cover the exact benchmark.")
    outputs = {"verifinger_fusion": [], "verifinger_sum": []}
    for p in testing:
        row = lookup[tuple(sorted((p.image_a, p.image_b)))]
        if row.label != p.label:
            raise InputError("SDK source target disagrees with the preserved benchmark.")
        for arm, value in (("verifinger_fusion", row.fused), ("verifinger_sum", row.summed)):
            outputs[arm].append(Score(p.pair_id, "ok" if value is not None else "missing_input", value))
    return outputs, paths


def read_historical_qwen(context, testing):
    lookup, paths = {}, []
    for label in ("genuine", "impostor"):
        path = Path(context) / f"res_random_2100_{label}_seed2026_qwen3_vl-8b.txt"
        paths.append(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 3:
                raise InputError("Unexpected historical Qwen output schema.")
            a, b, token = parts
            key = tuple(sorted((PureWindowsPath(a).name, PureWindowsPath(b).name)))
            if key in lookup:
                raise InputError("Repeated historical Qwen attempt.")
            if token in {"ERROR", "UNKNOWN", "None"}:
                lookup[key] = label, None
            else:
                try:
                    value = int(token)
                except ValueError:
                    raise InputError("Historical Qwen score is not a preserved integer outcome.") from None
                if not 0 <= value <= 100:
                    raise InputError("Historical Qwen score is outside its documented range.")
                lookup[key] = label, value
    if set(lookup) != {tuple(sorted((p.image_a, p.image_b))) for p in testing}:
        raise InputError("Historical Qwen output pairs do not exactly match testing.")
    scores = []
    for p in testing:
        label, value = lookup[tuple(sorted((p.image_a, p.image_b)))]
        if label != p.label:
            raise InputError("Historical Qwen class file conflicts with benchmark target.")
        scores.append(Score(p.pair_id, "ok" if value is not None else "error", value))
    return scores, paths


def descriptive_scores(testing, scores):
    by_score = matched_scores(testing, scores)

    def subset(pairs):
        genuine = [by_score[p.pair_id].score for p in pairs
                   if p.label == "genuine" and by_score[p.pair_id].score is not None]
        impostor = [by_score[p.pair_id].score for p in pairs
                    if p.label == "impostor" and by_score[p.pair_id].score is not None]
        return {"attempted": len(pairs), "valid": len(genuine) + len(impostor),
                "valid_genuine": len(genuine), "valid_impostor": len(impostor),
                "failures": sum(by_score[p.pair_id].score is None for p in pairs),
                "zero_genuine": sum(v == 0 for v in genuine), "zero_impostor": sum(v == 0 for v in impostor),
                "discrimination": discrimination(genuine, impostor)}

    return {"pooled": subset(testing), "by_slap_type": {
        hand: subset([p for p in testing if p.hand == hand]) for hand in ("right", "left", "thumbs")},
        "label_ambiguity_sensitivity": subset([p for p in testing if not p.numeric_identity_disagreement]),
        "decision_threshold": None, "threshold_reason": "No verified non-test calibration for this supplied arm."}


def paired_dino(testing, a, b, cutoff_a, cutoff_b):
    sa, sb = matched_scores(testing, a), matched_scores(testing, b)
    report = {}
    for hand in ("pooled", "right", "left", "thumbs"):
        groups = {}
        for label in ("genuine", "impostor"):
            pairs = [p for p in testing if p.label == label and (hand == "pooled" or p.hand == hand)]
            counts = {key: 0 for key in ("both_accept", "frozen_only_accept", "adapted_only_accept", "both_reject",
                                        "both_valid", "frozen_only_valid", "adapted_only_valid", "neither_valid")}
            for p in pairs:
                va, vb = sa[p.pair_id].score, sb[p.pair_id].score
                aa, ab = va is not None and va > cutoff_a, vb is not None and vb > cutoff_b
                key = ("both_accept" if aa and ab else "frozen_only_accept" if aa
                       else "adapted_only_accept" if ab else "both_reject")
                counts[key] += 1
                key = ("both_valid" if va is not None and vb is not None else "frozen_only_valid" if va is not None
                       else "adapted_only_valid" if vb is not None else "neither_valid")
                counts[key] += 1
            groups[label] = {"attempts": len(pairs), **counts,
                "adapted_minus_frozen_acceptance_rate": (
                    (counts["adapted_only_accept"] - counts["frozen_only_accept"]) / len(pairs) if pairs else None)}
        report[hand] = groups
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "frozen-test", "frozen-calibration", "adapted-test", "adapted-calibration", "checkpoint-verification",
                 "intake", "private-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output, receipt = args.output.resolve(), args.private_receipt.resolve()
        if (not output.is_relative_to(ROOT / "reports") or output.exists()
                or not receipt.is_relative_to(ROOT / "local/precise") or receipt.exists()):
            raise InputError("Use new aggregate report and private receipt paths.")
        bundle = json.loads(args.bundle.read_text())
        if bundle.get("scope") != "shared_seen_image_fixed_pairs" or bundle.get("labels") != "as_supplied":
            raise InputError("Unexpected benchmark scope/label policy.")
        training = [FixedPair(**row) for row in bundle["training"]]
        testing = [FixedPair(**row) for row in bundle["testing"]]
        validate_pairs(training); validate_pairs(testing)
        config = FixedConfig(**json.loads(args.config.read_text()))
        config.validate(len(training))
        sources = [args.bundle, args.config, args.checkpoint_verification, Path(__file__)]
        dino, outcomes = {}, {}
        for arm in ARMS:
            short = arm.split("_")[0]
            test, calibration = getattr(args, short + "_test"), getattr(args, short + "_calibration")
            for base in (test, calibration):
                sources.extend(path for path in base.iterdir() if path.suffix in {".csv", ".json"})
        sdk_paths = [args.intake / f"random_2100_{label}_seed2026.txt" for label in ("genuine", "impostor")]
        qwen_paths = [args.intake / "context" / f"res_random_2100_{label}_seed2026_qwen3_vl-8b.txt"
                      for label in ("genuine", "impostor")]
        sources.extend(sdk_paths + qwen_paths)
        initial = {str(path.resolve()): digest(path) for path in sources}
        sdk, _ = read_sdk(args.intake, testing)
        historical, _ = read_historical_qwen(args.intake / "context", testing)
        # Confirm SDK/testing source bytes remain the audited source files.
        for path in sdk_paths:
            if digest(path) != bundle["source_sha256"][path.name]:
                raise InputError("SDK testing source changed since bundle preparation.")
        for arm in ARMS:
            short = arm.split("_")[0]
            dino[arm], outcomes[arm] = read_dino(getattr(args, short + "_test"), getattr(args, short + "_calibration"),
                                               training, testing, config, arm)
        # DINO arms must refer to the same reviewed source population/config.
        for arm in ARMS:
            short = arm.split("_")[0]
            record = json.loads((getattr(args, short + "_test") / "run_record.json").read_text())
            if (record["provenance"]["input_sha256"]["bundle"] != digest(args.bundle)
                    or record["provenance"]["input_sha256"]["config"] != digest(args.config)):
                raise InputError("DINO arm references a different benchmark/configuration.")
        verification = json.loads(args.checkpoint_verification.read_text())
        identity = dino["adapted_dinov2"]["model_identity"]
        if (verification.get("passed") is not True or verification.get("base_weights_sha256") != WEIGHTS_SHA256
                or verification.get("checkpoint_sha256") != identity.get("checkpoint_sha256")
                or verification.get("selection_sha256") != identity.get("selection_sha256")
                or verification.get("epochs_completed") != config.epochs
                or verification.get("counts", {}).get("changed_frozen_tensors") != 0
                or len(verification.get("history", [])) != config.epochs
                or any(h["pairs_seen"] != len(training) for h in verification["history"])):
            raise InputError("Final checkpoint integrity/history does not match the evaluated adapted arm.")
        report = {"schema_version": 1, "scope": "shared_held_out_pairs_seen_images_and_identities",
            "configuration": asdict(config), "training_pairs": len(training), "testing_pairs": len(testing),
            "dependence": dependency_counts(training, testing),
            "training": verification,
            "dino": dino, "paired_dino": paired_dino(testing, outcomes[ARMS[0]], outcomes[ARMS[1]],
                dino[ARMS[0]]["threshold"]["cutoff"], dino[ARMS[1]]["threshold"]["cutoff"]),
            "sdk_descriptive": {arm: descriptive_scores(testing, scores) for arm, scores in sdk.items()},
            "historical_qwen_scores": {"status": "historical_score_export_not_matching_trained_AB_result",
                "preprocessing": "Native PIL inputs; model processor still transforms images.",
                "prompt_scope": "Four-finger 0-to-100 confidence prompt; thumbs nevertheless occur in pairs.",
                "model_adapter_and_run_provenance": "Unverified; accompanying script does not load an adapter.",
                "raw_generated_responses": "Not supplied in these parsed-score exports.",
                "metrics": descriptive_scores(testing, historical)},
            "trained_qwen_comparison": {"status": "not_ready_user_confirmed_20261101",
                "required_when_available": "Matching outputs and actual run/checkpoint provenance.",
                "owner": "Konain", "local_duplicate_finetuning_required": False},
            "limits": ["Training and test share all testing images and identity groups.",
                "Targets are supplied benchmark labels; alias semantics remain unconfirmed.",
                "DINO thresholds use training calibration with resubstitution optimism.",
                "SDK raw zeros are preserved; successful template extraction is not established by score presence.",
                "Historical Qwen outputs do not establish trained A/B protocol performance.",
                "ROC/EER are descriptive; no independent-pair interval or population low-FAR claim."]}
        if any(digest(Path(path)) != value for path, value in initial.items()):
            raise InputError("An input changed during aggregate reproduction.")
        output.parent.mkdir(parents=True, exist_ok=True)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        write_json(receipt, {"input_sha256": initial, "outputs": {"aggregate": str(output)},
                            "no_model_inference": True, "no_threshold_selection": True})
        write_json(output, report)
    except (InputError, OSError, ValueError, TypeError, KeyError, AttributeError):
        print("Shared-benchmark report did not complete; no source outputs or labels were changed.", file=sys.stderr)
        return 1
    print("Shared-benchmark aggregate reproduced; review content before publication.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
