"""Compare reviewed collaborator A/B outputs with the sealed shared-pair DINO runs.

Offline only. No model loading, training, cutoff selection or original rewrites.
Run metadata and review need actual source evidence, not generated flags.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path, PureWindowsPath
import re
import sys

from precise_matching.comparison import paired_decisions
from precise_matching.data import InputError
from precise_matching.evaluation import Score
from audit_precise_revised_cohort import TRAIN_JSON
from precise_matching.fixed_pairs import (GROUP_SCOPE, SCOPES, SHARED_SCOPE, FixedConfig, FixedPair, TRAIN_FILE,
                                          canonical_digest, validate_group_holdout, validate_pairs)
from precise_matching.training import digest, write_json
from report_precise_shared_pairs import ARMS, dependency_counts, read_dino

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Decision:
    pair_id: str
    status: str
    accept: bool | None


def read_ab(paths, testing):
    """Preserve source tokens; map A/a and B/b explicitly, independent of truth."""
    validate_pairs(testing)
    expected = {tuple(sorted((p.image_a, p.image_b))): p for p in testing}
    outcomes, tokens = {}, Counter()
    if set(paths) != {"genuine", "impostor"}:
        raise InputError("Both original class output files are required.")
    for source_class, path in paths.items():
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 3:
                raise InputError("Expected two image paths followed by the preserved parsed A/B outcome.")
            a, b, token = fields
            key = tuple(sorted((PureWindowsPath(a).name, PureWindowsPath(b).name)))
            p = expected.get(key)
            if p is None or p.label != source_class or p.pair_id in outcomes:
                raise InputError("Unexpected, repeated or wrong-class A/B pair; no intersection filtering.")
            if token in {"A", "a", "B", "b"}:
                outcome = Decision(p.pair_id, "ok", token in {"A", "a"})
            elif token in {"UNKNOWN", "INVALID", "ERROR"}:
                outcome = Decision(p.pair_id, "execution_error" if token == "ERROR" else ("invalid_response" if token == "INVALID" else "unknown_response"), None)
            else:
                raise InputError("Unknown preserved outcome token; do not guess or repair model replies.")
            tokens[token] += 1
            outcomes[p.pair_id] = outcome
    if set(outcomes) != {p.pair_id for p in testing}:
        raise InputError("A/B outputs must cover all testing attempts, including explicit failures.")
    return outcomes, dict(sorted(tokens.items()))


def operating_summary(pairs, outcomes):
    counts, statuses = Counter(), Counter()
    for p in pairs:
        row = outcomes[p.pair_id]
        if ((row.status == "ok") != (row.accept is not None)
                or row.accept is not None and type(row.accept) is not bool):
            raise InputError("Invalid binary decision status/value.")
        counts[p.label] += 1
        statuses[row.status] += 1
        counts[p.label + "_valid"] += row.accept is not None
        counts[p.label + "_accepts"] += row.accept is True

    def rates(valid_only):
        g = counts["genuine_valid"] if valid_only else counts["genuine"]
        i = counts["impostor_valid"] if valid_only else counts["impostor"]
        tar = counts["genuine_accepts"] / g if g else None
        far = counts["impostor_accepts"] / i if i else None
        return {"genuine_comparisons": g, "impostor_comparisons": i,
                "genuine_accepts": counts["genuine_accepts"], "false_accepts": counts["impostor_accepts"],
                "tar": tar, "far": far,
                "balanced_accuracy": (tar + 1 - far) / 2 if tar is not None and far is not None else None}

    return {"attempts": {label: counts[label] for label in ("genuine", "impostor")},
            "status_counts": dict(sorted(statuses.items())),
            "failures": {label: counts[label] - counts[label + "_valid"] for label in ("genuine", "impostor")},
            "coverage": {label: counts[label + "_valid"] / counts[label] if counts[label] else None
                         for label in ("genuine", "impostor")},
            "all_attempts_failure_as_reject": rates(False), "valid_only": rates(True)}


def validate_run_record(record, training_json, training_count):
    """Check reviewed source claims for this particular A/B protocol, not defaults."""
    prompt = {key: training_json[key] for key in ("system_prompt", "user_prompt")}
    if (record.get("model") != "Qwen/Qwen3-VL-8B-Instruct"
            or not re.fullmatch(r"[0-9a-f]{40}", record.get("base_revision", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", record.get("checkpoint_manifest_sha256", ""))
            or record.get("training_pairs_consumed") != training_count
            or record.get("prompt_sha256") != canonical_digest(prompt)
            or record.get("checkpoint_selection_used_test_results") is not False
            or not isinstance(record.get("environment"), dict) or not record["environment"]):
        raise InputError("Actual reviewed Qwen identity, training, prompt, environment and non-test selection are required.")
    for name in ("training_membership_evidence", "checkpoint_selection_evidence",
                 "evaluation_preprocessing", "generation_configuration_evidence", "raw_responses_evidence"):
        if not isinstance(record.get(name), str) or not record[name].strip():
            raise InputError("Actual run evidence is incomplete; code defaults are not a run receipt.")


def compare_operating_points(testing, qwen, dino_scores, dino_cutoffs):
    expected = {p.pair_id for p in testing}
    if set(qwen) != expected:
        raise InputError("Qwen decisions must cover the whole benchmark.")
    methods = {"qwen3_vl_trained_ab": qwen}
    scores = {"qwen3_vl_trained_ab": {
        key: Score(key, "ok" if row.accept is not None else "error",
                   float(row.accept) if row.accept is not None else None) for key, row in qwen.items()}}
    for arm in ARMS:
        by_id = {s.pair_id: s for s in dino_scores[arm]}
        if len(by_id) != len(dino_scores[arm]) or set(by_id) != expected:
            raise InputError("DINO decisions must cover the same full benchmark.")
        methods[arm] = {key: Decision(key, s.status, s.score > dino_cutoffs[arm] if s.score is not None else None)
                        for key, s in by_id.items()}
        scores[arm] = by_id
    groups = {"pooled": testing, **{hand: [p for p in testing if p.hand == hand]
                                    for hand in ("left", "right", "thumbs")},
              "excluding_flagged_test_pairs": [p for p in testing if not p.numeric_identity_disagreement]}
    return {"methods": {arm: {group: operating_summary(rows, decisions) for group, rows in groups.items()}
                        for arm, decisions in methods.items()},
            "paired_dino_to_qwen": {arm: {group: paired_decisions(
                rows, scores[arm], scores["qwen3_vl_trained_ab"], dino_cutoffs[arm], .5)
                for group, rows in groups.items()} for arm in ARMS},
            "binary_qwen_roc_auc_eer": None,
            "binary_encoding_note": "Internal 1/0 is only an accept/reject indicator for paired counts, not a confidence score."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "config", "training-json", "genuine-output", "impostor-output", "qwen-run-record",
                 "review", "dino-run-root", "private-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output, receipt = args.output.resolve(), args.private_receipt.resolve()
        if (not output.is_relative_to(ROOT / "reports") or output.exists()
                or not receipt.is_relative_to(ROOT / "local/precise") or receipt.exists()):
            raise InputError("Use new aggregate and private receipt paths.")
        sources = {key: getattr(args, key) for key in
                   ("bundle", "config", "training_json", "genuine_output", "impostor_output", "qwen_run_record", "review")}
        hashes = {key: digest(path) for key, path in sources.items()}
        review = json.loads(args.review.read_text())
        if (not isinstance(review, dict) or review.get("comparison_provenance_reviewed") is not True
                or review.get("non_test_selection_reviewed") is not True
                or review.get("input_sha256") != {key: value for key, value in hashes.items() if key != "review"}
                or not isinstance(review.get("review_record"), str) or not review["review_record"].strip()):
            raise InputError("Actual comparison/provenance review must bind all new supplied inputs.")
        bundle = json.loads(args.bundle.read_text())
        scope = bundle.get("scope")
        if scope not in SCOPES or bundle.get("labels") != "as_supplied":
            raise InputError("Unexpected fixed-pair cohort scope.")
        training, testing = ([FixedPair(**row) for row in bundle[phase]] for phase in ("training", "testing"))
        validate_pairs(training); validate_pairs(testing)
        if scope == GROUP_SCOPE:
            validate_group_holdout(training, testing)
        source_name = TRAIN_JSON if scope == GROUP_SCOPE else TRAIN_FILE
        if hashes["training_json"] != bundle["source_sha256"][source_name]:
            raise InputError("Qwen training source is not the supplied shared cohort.")
        record = json.loads(args.qwen_run_record.read_text())
        if record.get("training_json_sha256") != hashes["training_json"]:
            raise InputError("Actual Qwen training-source provenance differs.")
        validate_run_record(record, json.loads(args.training_json.read_text()), len(training))
        config = FixedConfig(**json.loads(args.config.read_text()))
        config.validate(len(training))
        dino, scores, cutoffs = {}, {}, {}
        artifacts = {}
        for arm in ARMS:
            short = arm.split("_")[0]
            test, calibration = args.dino_run_root / (short + "-test"), args.dino_run_root / (short + "-calibration")
            for base in (test, calibration):
                artifacts.update({str(p.resolve()): digest(p) for p in base.iterdir() if p.suffix in {".json", ".csv"}})
            dino[arm], scores[arm] = read_dino(test, calibration, training, testing, config, arm)
            original = json.loads((test / "run_record.json").read_text())
            if (original["provenance"].get("benchmark_scope", SHARED_SCOPE) != scope
                    or any(original["provenance"]["input_sha256"][key] != hashes[key] for key in ("bundle", "config"))):
                raise InputError("DINO arm references another cohort or configuration.")
            cutoffs[arm] = dino[arm]["threshold"]["cutoff"]
        qwen, tokens = read_ab({label: getattr(args, label + "_output") for label in ("genuine", "impostor")}, testing)
        report = {"schema_version": 1, "scope": "reviewed_fixed_pair_dino_vs_trained_qwen_ab", "benchmark_scope": scope,
            "results": compare_operating_points(testing, qwen, scores, cutoffs),
            "qwen_source_token_counts": tokens, "dino_cutoffs_unchanged": cutoffs,
            "qwen_identity": {key: record[key] for key in ("model", "base_revision", "checkpoint_manifest_sha256")},
            "dependence": dependency_counts(training, testing),
            "limits": ["Binary Qwen responses supply one operating point, not continuous-score ROC/AUC/EER.",
                ("Numeric-prefix groups are held out; biological subject/capture independence remains unconfirmed."
                 if scope == GROUP_SCOPE else "All test images/identity groups occur in training; labels remain as supplied."),
                "Review assertions and metadata do not themselves prove training consumption or selection history.",
                "No independent-trial confidence interval, significance or population low-FAR claim.",
                "Failures are retained and rejected for primary counts, not biometric impostor-rejection evidence.",
                "Lowercase parsed labels are mapped explicitly; no raw response is repaired or overwritten."]}
        if (any(digest(path) != hashes[key] for key, path in sources.items())
                or any(digest(Path(path)) != value for path, value in artifacts.items())):
            raise InputError("A reviewed input changed during offline comparison.")
        receipt.parent.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(receipt, {"input_sha256": hashes, "dino_artifact_sha256": artifacts,
            "code_sha256": digest(Path(__file__)), "no_model_inference": True, "no_threshold_selection": True})
        write_json(output, report)
    except (InputError, OSError, ValueError, TypeError, KeyError, AttributeError):
        print("A/B comparison did not complete; preserve sources and inspect actual run/review evidence.", file=sys.stderr)
        return 1
    print("Reviewed A/B operating-point comparison written; no model inference or tuning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
