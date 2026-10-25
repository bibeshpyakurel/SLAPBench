"""Inspect supplied Precise training/testing lists; never construct or train pairs.

Accept complete VeriFinger exports and image1 image2 fused-score records.
Zero scores remain scores; absent component fields remain unavailable.
Publish aggregate counts only; source provenance and overlap identities stay local.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys

from audit_precise_dataset import parse_labels
from audit_verifinger_scores import parse_line as parse_complete

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("training_genuine", "training_impostor", "testing_genuine", "testing_impostor")


@dataclass(frozen=True)
class PairRecord:
    image1: str
    image2: str
    fused: int
    label: str
    line: int
    fingers: tuple[int, ...] | None = None
    summed: int | None = None
    nfing: int | None = None

    @property
    def names(self):
        return PureWindowsPath(self.image1).name, PureWindowsPath(self.image2).name

    @property
    def pair(self):
        return tuple(sorted(self.names))

    @property
    def labels(self):
        return tuple(parse_labels(Path(name)) for name in self.names)

    @property
    def position(self):
        a, b = self.labels
        return a[1] if a and b and a[1] == b[1] else "unknown_or_mixed"


def parse_line(text, label, number):
    if label not in {"genuine", "impostor"}:
        raise ValueError("Unknown class")
    if "|" in text:
        row = parse_complete(text, label, number)
        return PairRecord(row.image1, row.image2, row.fused, label, number,
                          row.fingers, row.summed, row.nfing)
    try:
        image1, image2, value = text.split()
        fused = int(value)
        if fused < 0:
            raise ValueError("Negative score")
        return PairRecord(image1, image2, fused, label, number)
    except ValueError:
        raise ValueError(f"Invalid score record at line {number}") from None


def read_source(path, label):
    data = path.read_bytes()
    rows = [parse_line(line, label, number) for number, line in
            enumerate(data.decode("utf-8-sig").splitlines(), 1) if line.strip()]
    if not rows:
        raise ValueError("Empty source class")
    return rows, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def population(rows):
    return {
        "pairs": {row.pair for row in rows},
        "images": {name for row in rows for name in row.names},
        "subjects": {label[0] for row in rows for label in row.labels if label},
    }


def class_inventory(rows, dataset_names):
    pop = population(rows)
    mismatched, unknown, bad_finger_counts = 0, 0, 0
    for row in rows:
        a, b = row.labels
        if not a or not b:
            unknown += 1
            continue
        mismatched += (a[0] == b[0]) != (row.label == "genuine")
        if row.nfing is not None and row.position != "unknown_or_mixed":
            bad_finger_counts += row.nfing != (2 if row.position == "15" else 4)
    return {
        "records": len(rows), "unique_unordered_pairs": len(pop["pairs"]),
        "unique_image_basenames": len(pop["images"]), "filename_subjects": len(pop["subjects"]),
        "missing_local_image_basenames": len(pop["images"] - dataset_names),
        "unrecognized_filename_pairs": unknown, "filename_label_mismatches": mismatched,
        "self_image_pairs": sum(row.names[0] == row.names[1] for row in rows),
        "duplicate_unordered_pair_excess": len(rows) - len(pop["pairs"]),
        "slap_type_counts": dict(sorted(Counter(row.position for row in rows).items())),
        "mixed_or_unknown_slap_pairs": sum(row.position == "unknown_or_mixed" for row in rows),
        "complete_component_records": sum(row.fingers is not None for row in rows),
        "fused_only_records": sum(row.fingers is None for row in rows),
        "fused_only_zero_records": sum(row.fingers is None and row.fused == 0 for row in rows),
        "fused_only_by_slap_type": dict(sorted(Counter(
            row.position for row in rows if row.fingers is None).items())),
        "unexpected_supplied_finger_counts": bad_finger_counts,
        "fused_score_range": [min(row.fused for row in rows), max(row.fused for row in rows)],
    }


def compare(training, testing):
    train, test = population(training), population(testing)
    shared = {kind: sorted(train[kind] & test[kind]) for kind in train}
    summary = {"training_records": len(training), "testing_records": len(testing)}
    for kind in train:
        summary[f"training_unique_{kind}"] = len(train[kind])
        summary[f"testing_unique_{kind}"] = len(test[kind])
        summary[f"shared_{kind}"] = len(shared[kind])
    summary.update({"exact_pairs_disjoint": not shared["pairs"],
                    "image_basenames_disjoint": not shared["images"],
                    "filename_subjects_disjoint": not shared["subjects"]})
    return summary, shared


def audit(sources, dataset_root):
    dataset_names = {p.name for p in dataset_root.iterdir()
                     if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg"}}
    if not dataset_names or set(sources) != set(SOURCES):
        raise ValueError("Dataset filenames and all four source roles are required")
    rows, provenance = {}, {}
    for role in SOURCES:
        rows[role], provenance[role] = read_source(sources[role], role.rsplit("_", 1)[1])
        provenance[role]["source_path"] = str(sources[role].resolve())
    training = rows["training_genuine"] + rows["training_impostor"]
    testing = rows["testing_genuine"] + rows["testing_impostor"]
    overlap, details = {}, {}
    for name, positions in (("all_slaps", None), ("four_finger_only", {"13", "14"}),
                            ("right", {"13"}), ("left", {"14"}), ("thumbs", {"15"})):
        tr = [r for r in training if positions is None or r.position in positions]
        te = [r for r in testing if positions is None or r.position in positions]
        overlap[name], details[name] = compare(tr, te)
    cross_class = {}
    for part in ("training", "testing"):
        cross_class[part] = len(population(rows[part + "_genuine"])["pairs"] &
                                population(rows[part + "_impostor"])["pairs"])
    for role, path in sources.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance[role]["sha256"]:
            raise ValueError("Source bytes changed during inspection")
    report = {
        "schema_version": 1,
        "scope": "Offline inventory and overlap of supplied lists; no split creation or model run",
        "pair_identity": "Unordered exact basenames; Windows/POSIX directory prefixes ignored",
        "classes": {role: class_inventory(rows[role], dataset_names) for role in SOURCES},
        "cross_class_pair_overlaps": cross_class, "training_testing_overlap": overlap,
        "source_byte_checks_passed": len(SOURCES),
        "limits": [
            "Filename availability does not verify byte identity with collaborator images.",
            "Pair separation does not imply image or subject separation.",
            "Zero fused scores remain supplied scores, not inferred failures or nonmatch labels.",
            "Absent per-finger fields remain unknown and are not reconstructed.",
            "Lists do not establish capture independence, development allocation, actual training use or binary target encoding.",
        ],
    }
    return report, {"sources": provenance, "overlap_identities": details}


def write_outputs(public, private, report, provenance):
    if public.exists() or private.exists():
        raise ValueError("Choose new output directories; overwrite refused")
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    (public / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (private / "audit_details.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for role in SOURCES:
        parser.add_argument("--" + role.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/Precise")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--private-dir", required=True, type=Path)
    args = parser.parse_args()
    public, private = args.output_dir.resolve(), args.private_dir.resolve()
    if not public.is_relative_to(ROOT / "reports") or not private.is_relative_to(ROOT / "local"):
        parser.error("Aggregate output must be under reports/; identifying output under local/")
    if public.exists() or private.exists():
        parser.error("Choose new output directories; overwrite refused")
    try:
        report, provenance = audit({role: getattr(args, role) for role in SOURCES}, args.dataset_root)
        write_outputs(public, private, report, provenance)
    except (OSError, ValueError, UnicodeError):
        print("Audit failed; sources preserved. Check inputs locally.", file=sys.stderr)
        return 1
    print("Aggregate overlap audit written; identifying records remain local. No training performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
