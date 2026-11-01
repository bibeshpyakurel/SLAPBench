"""Reconcile Konain's explicit JSON cohort and fixed testing lists offline.

Public output is aggregate only. Preserve original labels, references and scores;
numeric-ID disagreements are review issues, not automatic label corrections.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys

from audit_precise_dataset import parse_labels
from audit_precise_training_pairs import read_source

ROOT = Path(__file__).resolve().parents[1]
TRAIN = "slap_verification_binary_ab_13_14_15_balanced.json"
TEST = {label: f"random_2100_{label}_seed2026.txt" for label in ("genuine", "impostor")}
REFERENCE = "context/genuine_filtered_minus2100.txt"


def summarize(rows, available, scope):
    pairs, images, subjects, numeric = [], set(), set(), set()
    counts, issues, by_hand = Counter(), [], Counter()
    for index, a, b, label in rows:
        if label not in {"genuine", "impostor"}:
            raise ValueError("Unexpected class.")
        la, lb = parse_labels(Path(a)), parse_labels(Path(b))
        if la is None or lb is None or not all(s.isascii() and s.isdigit() for s in (la[0], lb[0])):
            raise ValueError("Unrecognized source image labels.")
        pairs.append(tuple(sorted((a, b))))
        images.update((a, b)); subjects.update((la[0], lb[0])); numeric.update((int(la[0]), int(lb[0])))
        counts[label] += 1
        counts["self_pairs"] += a == b
        counts["missing_image_references"] += int(a not in available) + int(b not in available)
        counts["hand_conflicts"] += la[1] != lb[1]
        counts["raw_prefix_label_disagreements"] += (la[0] == lb[0]) != (label == "genuine")
        by_hand[la[1] + "_" + label] += 1
        if (int(la[0]) == int(lb[0])) != (label == "genuine"):
            issues.append({"scope": scope, "source_record": index, "image_a": a, "image_b": b,
                           "provided_class": label, "issue": "numeric_subject_id_disagreement",
                           "action": "Preserve original; review identity/label semantics."})
    return ({"records": len(rows), "class_counts": {key: counts[key] for key in ("genuine", "impostor")},
        "hand_class_counts": dict(sorted(by_hand.items())),
        "unique_pairs": len(set(pairs)), "duplicate_unordered_pairs": len(pairs) - len(set(pairs)),
        "images": len(images), "raw_prefix_groups": len(subjects), "numeric_id_groups": len(numeric),
        "numeric_id_label_disagreements": len(issues),
        **{key: counts[key] for key in ("self_pairs", "missing_image_references", "hand_conflicts", "raw_prefix_label_disagreements")}},
        {"pairs": set(pairs), "images": images, "subjects": subjects, "numeric": numeric}, issues)


def reconcile(intake, image_root, previous=None):
    paths = [intake / TRAIN, *(intake / name for name in TEST.values()), intake / REFERENCE]
    source_bytes = {path: path.read_bytes() for path in paths}
    data = json.loads(source_bytes[intake / TRAIN])
    pairs = data["pairs"]
    records = list(pairs.values()) if isinstance(pairs, dict) else pairs
    names = {p.name for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}}
    stems = {Path(name).stem: name for name in names}
    if not names or len(stems) != len(names):
        raise ValueError("Image names require a unique local stem mapping.")
    training, fused_zero, hand_errors, finger_errors = [], Counter(), 0, 0
    for index, record in enumerate(records, 1):
        paths_pair = record["image_paths"]
        if len(paths_pair) != 2 or not all(isinstance(p, str) for p in paths_pair):
            raise ValueError("Invalid explicit image pair.")
        a, b = (PureWindowsPath(p).name for p in paths_pair)
        label = record["pair_class"]
        if record["correct_answer_option"] != {"genuine": "A", "impostor": "B"}.get(label):
            raise ValueError("Class/answer mismatch.")
        if type(record["finger_score_sum"]) is not int or record["finger_score_sum"] <= 0:
            raise ValueError("Unexpected nonpositive selected component sum.")
        if type(record["verifinger_score"]) is not int or record["verifinger_score"] < 0:
            raise ValueError("Unexpected fused score.")
        suffix = Path(a).stem.split("_")[-1]
        hand_errors += suffix != record["filename_suffix"]
        finger_errors += record["nfing"] != {"13": 4, "14": 4, "15": 2}.get(suffix)
        fused_zero[label] += record["verifinger_score"] == 0
        training.append((index, a, b, label))
    tr, train_sets, issues = summarize(training, names, "training")
    tr.update({"positive_component_sum_records": len(records), "retained_fused_zero_by_class": dict(fused_zero),
               "declared_suffix_disagreements": hand_errors, "finger_count_disagreements": finger_errors,
               "declared_class_counts_match": data["counts"] == {**tr["class_counts"], "total": len(records)}})
    tests, old_test_matches = [], {}
    for label, name in TEST.items():
        rows, _ = read_source(intake / name, label)
        tests.extend((r.line, *r.names, label) for r in rows)
        old = previous / f"testing_{label}.txt" if previous else None
        old_test_matches[label] = old.read_bytes() == source_bytes[intake / name] if old and old.is_file() else None
    te, test_sets, test_issues = summarize(tests, names, "testing")
    issues.extend(test_issues)
    reference, _ = read_source(intake / REFERENCE, "genuine")

    def map_reference(row):
        return tuple(sorted(stems[Path(name).stem] for name in row.names))

    reference_pairs = {map_reference(r) for r in reference if r.summed is not None and r.summed > 0}
    actual_genuine = {tuple(sorted((a, b))) for _, a, b, label in training if label == "genuine"}
    lookup = {r.line: r for r in reference}
    provenance_errors = 0
    for record in records:
        if record["pair_class"] != "genuine":
            continue
        original = lookup.get(record["source_line"])
        pair = tuple(sorted(PureWindowsPath(p).name for p in record["image_paths"]))
        provenance_errors += original is None or map_reference(original) != pair or (
            original.fused, original.summed, original.nfing) != (
            record["verifinger_score"], record["finger_score_sum"], record["nfing"])
    for path, raw in source_bytes.items():
        if path.read_bytes() != raw:
            raise ValueError("Source changed during reconciliation.")
    report = {"schema_version": 1, "scope": "Read-only collaborator fixed-pair cohort reconciliation; no model execution",
        "training": tr, "testing": te, "testing_bytes_identical_to_prior_intake": old_test_matches,
        "train_test_overlap": {"exact_pairs": len(train_sets["pairs"] & test_sets["pairs"]),
            "images": len(train_sets["images"] & test_sets["images"]),
            "raw_prefix_groups": len(train_sets["subjects"] & test_sets["subjects"]),
            "numeric_id_groups": len(train_sets["numeric"] & test_sets["numeric"])},
        "full_genuine_reference": {"records": len(reference),
            "template_dat_names": all(Path(n).suffix.lower() == ".dat" for r in reference for n in r.names),
            "mapping": "Template basenames mapped by unique stem to local JPEG basenames; originals unchanged",
            "positive_sum_pairs": len(reference_pairs), "matches_json_genuine_cohort": reference_pairs == actual_genuine,
            "json_source_line_and_score_disagreements": provenance_errors},
        "sources": {str(path.relative_to(intake)): {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                    for path, raw in source_bytes.items()},
        "limits": ["No capture/session independence is established by capture_type; it describes hand/slap type.",
            "No labels are corrected using numeric equality; disagreements remain private review cases.",
            "The supplied holdout excludes exact pairs, not images or subjects; do not claim subject-disjoint generalization.",
            "The complete 1.6 GB impostor pool is not recounted here; its declared selection metadata is not a run receipt.",
            "No development partition or checkpoint-selection history is supplied by the cohort JSON."]}
    return report, issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intake-dir", "images", "output", "private-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous-intake", type=Path)
    args = parser.parse_args(argv)
    try:
        output, private = args.output.resolve(), args.private_dir.resolve()
        if not output.is_relative_to(ROOT / "reports") or not private.is_relative_to(ROOT / "local/precise"):
            raise ValueError("Use reviewed report/private directories.")
        if output.exists() or private.exists():
            raise ValueError("Existing reports and private review cases are never overwritten.")
        report, issues = reconcile(args.intake_dir, args.images, args.previous_intake)
        private.mkdir(parents=True)
        with (private / "label_review_cases.json").open("x") as handle:
            json.dump(issues, handle, indent=2)
        with (private / "provenance.json").open("x") as handle:
            json.dump({"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "sources": report["sources"], "no_relabeling": True}, handle, indent=2)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        print("Shared-cohort reconciliation failed; no source changes or model execution performed.", file=sys.stderr)
        return 1
    print("Aggregate shared-cohort reconciliation written; review flags are not experiment approvals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
