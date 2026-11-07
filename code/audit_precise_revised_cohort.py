"""Read-only intake of the November 5 revised collaborator pair cohort.

Compare new sources with the completed benchmark without allocating subjects,
changing labels, decoding images, generating pairs or executing a model.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys

from audit_precise_dataset import parse_labels
from audit_precise_shared_cohort import summarize
from audit_precise_training_pairs import read_source

ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSON = "slap_subject_disjoint_train_ab.json"
LISTS = {phase + "_" + label: phase + "_" + label + ("_2100" if phase == "test" else "") + "_nonzero.txt"
         for phase in ("train", "test") for label in ("genuine", "impostor")}
HANDS = {"13": 4, "14": 4, "15": 2}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(a, b):
    return tuple(sorted((a, b)))


def populations(rows):
    return {"pairs": {key(a, b) for _, a, b, _ in rows},
            "images": {name for _, a, b, _ in rows for name in (a, b)},
            "numeric_groups": {int(parse_labels(Path(name))[0]) for _, a, b, _ in rows for name in (a, b)}}


def reconcile(intake, inventory_path, approved_bundle, approved_record, image_root):
    sources = {name: Path(intake) / name for name in (TRAIN_JSON, *LISTS.values())}
    sources.update(inventory=Path(inventory_path), approved_bundle=Path(approved_bundle), approved_record=Path(approved_record))
    hashes = {name: sha(path) for name, path in sources.items()}
    inventory = json.loads(sources["inventory"].read_text())["records"]
    names = {r["filename"] for r in inventory}
    if len(names) != len(inventory):
        raise ValueError("Inventory filenames must be unique.")
    old = json.loads(sources["approved_bundle"].read_text())
    record = json.loads(sources["approved_record"].read_text())
    if (old.get("scope") != "shared_seen_image_fixed_pairs" or old.get("labels") != "as_supplied"
            or record["provenance"]["input_sha256"]["bundle"] != hashes["approved_bundle"]):
        raise ValueError("The prior bundle must match the actual completed training provenance.")
    old_rows = {phase: [(i, p["image_a"], p["image_b"], p["label"]) for i, p in enumerate(old[field], 1)]
                for phase, field in (("train", "training"), ("test", "testing"))}
    data = json.loads(sources[TRAIN_JSON].read_text())
    records = list(data["pairs"].values()) if isinstance(data["pairs"], dict) else data["pairs"]
    json_rows, json_scores = [], []
    hand_errors, finger_errors = 0, 0
    for index, row in enumerate(records, 1):
        label = row["pair_class"]
        if (label not in {"genuine", "impostor"}
                or row["correct_answer_option"] != {"genuine": "A", "impostor": "B"}[label]
                or not isinstance(row["image_paths"], list) or len(row["image_paths"]) != 2):
            raise ValueError("Invalid explicit training target/pair.")
        a, b = (PureWindowsPath(path).name for path in row["image_paths"])
        la, lb = parse_labels(Path(a)), parse_labels(Path(b))
        if la is None or lb is None:
            raise ValueError("Unknown new cohort filename convention.")
        hand_errors += la[1] != lb[1] or row["filename_suffix"] != la[1]
        finger_errors += row["nfing"] != HANDS[la[1]]
        if (type(row["verifinger_score"]) is not int or row["verifinger_score"] < 0
                or type(row["finger_score_sum"]) is not int or row["finger_score_sum"] <= 0):
            raise ValueError("Unexpected selected SDK score representation.")
        json_rows.append((index, a, b, label))
        json_scores.append((key(a, b), label, row["verifinger_score"], row["finger_score_sum"], row["nfing"]))
    phases, list_scores, score_status = {}, [], {}
    for phase in ("train", "test"):
        rows = []
        for label in ("genuine", "impostor"):
            role = phase + "_" + label
            parsed, _ = read_source(sources[LISTS[role]], label)
            for row in parsed:
                a, b = row.names
                rows.append((len(rows) + 1, a, b, label))
                if phase == "train":
                    list_scores.append((key(a, b), label, row.fused, row.summed, row.nfing))
            score_status[role] = {"records": len(parsed), "fused_zero": sum(r.fused == 0 for r in parsed),
                                  "missing_components": sum(r.summed is None for r in parsed),
                                  "nonpositive_component_sum": sum(r.summed is not None and r.summed <= 0 for r in parsed)}
        phases[phase] = rows
    reports, sets, issues = {}, {}, []
    for phase, rows in {"training_json": json_rows, "training_lists": phases["train"], "testing": phases["test"]}.items():
        reports[phase], _, cases = summarize(rows, names, phase)
        sets[phase] = populations(rows)
        issues.extend(cases)
    actual = reports["training_json"]
    selection = data["selection"]
    suffix_counts = Counter(a.rsplit("_", 1)[-1].split(".")[0] + "_" + label for _, a, _, label in json_rows)
    declared_suffix = {suffix + "_" + label: counts[label] for suffix, counts in data["suffix_counts"].items()
                       for label in ("genuine", "impostor")}
    tr, te = sets["training_lists"], sets["testing"]
    old_sets = {phase: populations(rows) for phase, rows in old_rows.items()}
    comparison = {}
    for phase, new_set in (("train", tr), ("test", te)):
        prior = old_sets[phase]
        comparison[phase + "_membership_change"] = {
            "prior_pairs": len(prior["pairs"]), "new_pairs": len(new_set["pairs"]),
            "shared_pairs": len(new_set["pairs"] & prior["pairs"]),
            "added_pairs": len(new_set["pairs"] - prior["pairs"]),
            "removed_pairs": len(prior["pairs"] - new_set["pairs"])}
    comparison["new_testing_previously_exposed_to_completed_dino_training"] = {
        field: len(te[field] & old_sets["train"][field]) for field in ("pairs", "images", "numeric_groups")}
    comparison["new_training_overlap_with_previous_testing"] = {
        field: len(tr[field] & old_sets["test"][field]) for field in ("pairs", "images", "numeric_groups")}
    aliases = defaultdict(set)
    for name in names:
        parsed = parse_labels(Path(name))
        if parsed is None or not parsed[0].isascii() or not parsed[0].isdigit():
            raise ValueError("Existing inventory has unexpected numeric-prefix metadata.")
        aliases[int(parsed[0])].add(parsed[0])
    alias_groups = {group for group, prefixes in aliases.items() if len(prefixes) > 1}
    root = Path(image_root).resolve(strict=True)
    endpoints = tr["images"] | te["images"]
    missing_files = 0
    for name in endpoints:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("New source path escapes authorized image root.")
        missing_files += not path.is_file()
    report = {"schema_version": 1, "scope": "Read-only new collaborator cohort intake; no model execution or approval",
        **reports, "new_train_test_overlap": {field: len(tr[field] & te[field]) for field in ("pairs", "images", "numeric_groups")},
        "training_json_matches_supplied_training_lists_and_scores": Counter(json_scores) == Counter(list_scores),
        "declared_training_class_counts_match": data["counts"] == {**actual["class_counts"], "total": len(json_rows)},
        "declared_suffix_counts_match": suffix_counts == declared_suffix,
        "declared_suffix_errors": hand_errors, "declared_finger_count_errors": finger_errors,
        "score_status": score_status, "missing_local_image_files": missing_files,
        "declared_split_counts_match_observed": selection["train_numeric_groups"] == len(tr["numeric_groups"])
            and selection["test_numeric_groups"] == len(te["numeric_groups"]),
        "alias_group_placement": {"inventory_padding_alias_groups": len(alias_groups),
            "in_new_training": len(alias_groups & tr["numeric_groups"]), "in_new_testing": len(alias_groups & te["numeric_groups"]),
            "declared_forced_training_groups": selection["forced_alias_groups_to_train"]},
        "declared_sampling": {key: selection[key] for key in ("seed", "test_fraction", "test_pairs_per_class")},
        "prompts_equal_to_earlier_json": None,
        "comparison_with_completed_benchmark": comparison,
        "prior_execution_commit": record["provenance"]["code_commit"],
        "sources": {name: {"bytes": path.stat().st_size, "sha256": hashes[name]} for name, path in sources.items()
                    if name not in {"inventory", "approved_bundle", "approved_record"}},
        "limits": ["Observed disjointness concerns numeric filename-prefix groups, not independently verified biological identities.",
            "The dataframe's identity/capture semantics and independence are not established by these pair files.",
            "The existing adapted DINO checkpoint has prior training exposure to new testing groups; do not claim it is unseen-group testing.",
            "This is a different cohort from the approved 39290/4200 benchmark; no replacement, retraining or test authorization is inferred.",
            "Source metadata and selection seeds are declarations, not proof of Konain's actual model consumption or checkpoint selection.",
            "Score equality with the supplied new lists does not establish full upstream-pool provenance or SDK extraction success."]}
    if any(sha(path) != hashes[name] for name, path in sources.items()):
        raise ValueError("Source changed during read-only intake.")
    private = {"source_paths": {name: str(path.resolve()) for name, path in sources.items()}, "input_sha256": hashes,
               "declared_selection_unchanged": selection, "label_review_cases": issues,
               "image_pixels_read": False, "model_execution": False, "approval_issued": False}
    return report, private


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("intake", "inventory", "approved-bundle", "approved-record", "image-root", "earlier-training-json", "output", "private-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output, private = args.output.resolve(), args.private_dir.resolve()
        if (not output.is_relative_to(ROOT / "reports") or output.exists()
                or not private.is_relative_to(ROOT / "local/precise") or private.exists()):
            raise ValueError("Use new aggregate and private receipt paths.")
        earlier_sha = sha(args.earlier_training_json)
        report, receipt = reconcile(args.intake, args.inventory, args.approved_bundle, args.approved_record, args.image_root)
        earlier = json.loads(args.earlier_training_json.read_text())
        new = json.loads((args.intake / TRAIN_JSON).read_text())
        report["prompts_equal_to_earlier_json"] = {key: new[key] == earlier[key]
                                                   for key in ("system_prompt", "user_prompt", "options")}
        if sha(args.earlier_training_json) != earlier_sha:
            raise ValueError("Earlier supplied JSON changed during comparison.")
        receipt["input_sha256"]["earlier_training_json"] = earlier_sha
        receipt["source_paths"]["earlier_training_json"] = str(args.earlier_training_json.resolve())
        receipt["code_sha256"] = sha(Path(__file__))
        if any(sha(Path(receipt["source_paths"][name])) != value
               for name, value in receipt["input_sha256"].items()):
            raise ValueError("An intake source changed before output.")
        private.mkdir(parents=True)
        with (private / "source_review.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True)
            stream.write("\n")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        print("New-cohort intake did not complete; no original sources or model runs were changed.", file=sys.stderr)
        return 1
    print("New-cohort aggregate and private source review written; no model execution or approval issued.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
