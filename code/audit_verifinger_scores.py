"""Audit existing Precise VeriFinger score exports without matcher inference.

Source format: image1 image2 fused | finger_scores... sum=INTEGER nfing=INTEGER.
Public outputs contain aggregates and empirical ROC/DET coordinates only.
Original paths, pair records and source hashes stay under ignored local/.
Operating points are selected from the supplied evaluation scores, not development.
"""

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys

from audit_precise_dataset import POSITIONS, parse_labels

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("0.0001", "0.001", "0.01")


@dataclass(frozen=True)
class Record:
    image1: str
    image2: str
    fused: int
    fingers: tuple[int, ...]
    summed: int
    nfing: int
    label: str
    line: int

    @property
    def names(self):
        return PureWindowsPath(self.image1).name, PureWindowsPath(self.image2).name

    @property
    def pair(self):
        return tuple(sorted(self.names))

    @property
    def position(self):
        a, b = (parse_labels(Path(name)) for name in self.names)
        return a[1] if a and b and a[1] == b[1] else "unknown_or_mixed"


def parse_line(text: str, label: str, number: int) -> Record:
    if label not in {"genuine", "impostor"}:
        raise ValueError("Unknown source class")
    try:
        head, tail = text.split("|")
        image1, image2, fused = head.split()
        fields, fingers = {}, []
        for token in tail.split():
            if "=" in token:
                key, value = token.split("=")
                if key in fields:
                    raise ValueError("Repeated field")
                fields[key] = int(value)
            else:
                fingers.append(int(token))
        if set(fields) != {"sum", "nfing"}:
            raise ValueError("Unexpected fields")
        fused = int(fused)
        if min([fused, *fingers, fields["sum"], fields["nfing"]]) < 0:
            raise ValueError("Negative value")
        if sum(fingers) != fields["sum"] or len(fingers) != fields["nfing"]:
            raise ValueError("Inconsistent component fields")
        return Record(image1, image2, fused, tuple(fingers), fields["sum"],
                      fields["nfing"], label, number)
    except ValueError:
        raise ValueError(f"Invalid score record at line {number}") from None


def read_source(path: Path, label: str):
    data = path.read_bytes()
    rows = [parse_line(line, label, i) for i, line in
            enumerate(data.decode("utf-8-sig").splitlines(), 1) if line.strip()]
    if not rows:
        raise ValueError("Empty source class")
    return rows, hashlib.sha256(data).hexdigest()


def operating_point(genuine: list[int], impostor: list[int], target: str):
    far = Fraction(target)
    if not genuine or not impostor or not 0 <= far < 1:
        raise ValueError("Two nonempty classes and a FAR below one required")
    budget = (far.numerator * len(impostor)) // far.denominator
    # Most permissive integer threshold respecting the observed error budget.
    # The whole tied score is rejected when accepting it would exceed the budget.
    threshold = sorted(impostor, reverse=True)[budget] + 1
    false_accepts = sum(score >= threshold for score in impostor)
    genuine_accepts = sum(score >= threshold for score in genuine)
    return {"target_far": float(far), "allowed_false_accepts": budget,
            "threshold": threshold, "false_accepts": false_accepts,
            "impostors": len(impostor), "actual_far": false_accepts / len(impostor),
            "genuine_accepts": genuine_accepts, "genuines": len(genuine),
            "tar": genuine_accepts / len(genuine)}


def curve(genuine: list[int], impostor: list[int]):
    if not genuine or not impostor:
        raise ValueError("Two nonempty score classes required")
    g, i = Counter(genuine), Counter(impostor)
    scores = sorted(g.keys() | i.keys(), reverse=True)
    result = [{"threshold": scores[0] + 1, "false_accepts": 0,
               "genuine_accepts": 0, "far": 0.0, "tar": 0.0, "frr": 1.0}]
    ga, ia = 0, 0
    for score in scores:
        ga += g[score]
        ia += i[score]
        tar, far = ga / len(genuine), ia / len(impostor)
        result.append({"threshold": score, "false_accepts": ia,
                       "genuine_accepts": ga, "far": far, "tar": tar, "frr": 1 - tar})
    return result


def discrimination(points: list[dict]):
    auc = sum((b["far"] - a["far"]) * (a["tar"] + b["tar"]) / 2
              for a, b in zip(points[:-1], points[1:], strict=True))
    eer = None
    for a, b in zip(points[:-1], points[1:], strict=True):
        da, db = a["far"] - a["frr"], b["far"] - b["frr"]
        if da <= 0 <= db:
            alpha = -da / (db - da) if db != da else 0
            eer = a["far"] + alpha * (b["far"] - a["far"])
            break
    return {"auc": auc, "eer_linear_interpolation": eer}


def class_inventory(rows, label, dataset_names):
    images = {name for row in rows for name in row.names}
    subjects, invalid, mismatch, nfing_unexpected = set(), 0, 0, 0
    for row in rows:
        labels = [parse_labels(Path(name)) for name in row.names]
        if not all(labels):
            invalid += 1
            continue
        subjects.update(value[0] for value in labels)
        mismatch += (labels[0][0] == labels[1][0]) != (label == "genuine")
        expected = 2 if row.position == "15" else 4
        nfing_unexpected += row.nfing != expected
    counts = Counter(row.pair for row in rows)
    return {"records": len(rows), "filename_subjects": len(subjects),
            "unique_image_basenames": len(images), "missing_local_basenames": len(images - dataset_names),
            "unrecognized_filename_pairs": invalid, "filename_label_mismatches": mismatch,
            "self_image_pairs": sum(row.names[0] == row.names[1] for row in rows),
            "duplicate_unordered_pair_excess": sum(n - 1 for n in counts.values()),
            "slap_type_counts": dict(sorted(Counter(row.position for row in rows).items())),
            "nfing_counts": dict(sorted(Counter(row.nfing for row in rows).items())),
            "nfing_unexpected_for_slap_type": nfing_unexpected,
            "fused_range": [min(row.fused for row in rows), max(row.fused for row in rows)],
            "sum_range": [min(row.summed for row in rows), max(row.summed for row in rows)]}


def reference_overlap(path: Path, rows):
    pairs = {label: set() for label in ("genuine", "impostor")}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"img1_path", "img2_path", "label", "category"} <= set(reader.fieldnames or []):
            raise ValueError("Reference manifest schema unavailable")
        for record in reader:
            if record["category"] not in {"primary_genuine", "primary_impostor"}:
                continue
            label = {"mated": "genuine", "non-mated": "impostor",
                     "genuine": "genuine", "impostor": "impostor"}.get(record["label"])
            expected = "genuine" if record["category"] == "primary_genuine" else "impostor"
            if label != expected:
                raise ValueError("Unknown or inconsistent reference class")
            pairs[label].add(tuple(sorted(PureWindowsPath(record[key]).name
                                         for key in ("img1_path", "img2_path"))))
    return {label: {"reference_unique_primary_pairs": len(pairs[label]),
                    "shared_unordered_basename_pairs": len(pairs[label] & {r.pair for r in rows if r.label == label})}
            for label in pairs}


def audit(genuine_path: Path, impostor_path: Path, dataset_root: Path, reference: Path | None = None):
    genuine, ghash = read_source(genuine_path, "genuine")
    impostor, ihash = read_source(impostor_path, "impostor")
    rows = genuine + impostor
    names = {p.name for p in dataset_root.iterdir() if p.is_file()}
    subjects = {labels[0] for row in rows for name in row.names
                if (labels := parse_labels(Path(name)))}
    all_images = {name for row in rows for name in row.names}
    report = {"schema_version": 1, "scope": "Offline audit of collaborator-provided evaluation scores",
              "score_rule": "Integer score >= threshold; larger scores indicate a match",
              "threshold_provenance": "Descriptive thresholds selected on these supplied evaluation impostors; not development-fixed",
              "positions_from_filenames": POSITIONS,
              "classes": {"genuine": class_inventory(genuine, "genuine", names),
                          "impostor": class_inventory(impostor, "impostor", names)},
              "combined": {"filename_subjects": len(subjects), "unique_image_basenames": len(all_images),
                           "shared_images_between_classes": len({n for r in genuine for n in r.names} & {n for r in impostor for n in r.names}),
                           "cross_class_unordered_pair_overlap": len({r.pair for r in genuine} & {r.pair for r in impostor})},
              "source_structure": {"all_rows_parsed": True, "all_sums_and_nfing_counts_consistent": True},
              "training_overlap": "Not assessed: training pairs and exposure records not supplied",
              "failure_rates": "Not established: export has scores/nfing but no pre-export failures or SDK status fields",
              "slices": {}}
    coordinates = []
    for slice_name, positions in (("all_slaps", {"13", "14", "15", "unknown_or_mixed"}),
                                  ("four_finger_only", {"13", "14"}),
                                  ("right", {"13"}), ("left", {"14"}), ("thumbs", {"15"})):
        gs = [r for r in genuine if r.position in positions]
        ims = [r for r in impostor if r.position in positions]
        methods = {}
        for method, attribute in (("sdk_fused", "fused"), ("per_finger_sum", "summed")):
            if not gs or not ims:
                methods[method] = {"status": "Unavailable: one score class empty"}
                continue
            gvalues, ivalues = [getattr(r, attribute) for r in gs], [getattr(r, attribute) for r in ims]
            points = curve(gvalues, ivalues)
            methods[method] = {**discrimination(points),
                               "operating_points": [operating_point(gvalues, ivalues, target) for target in TARGETS],
                               "genuine_zero_scores": gvalues.count(0), "impostor_zero_scores": ivalues.count(0)}
            coordinates.extend({"slice": slice_name, "method": method, **point} for point in points)
        report["slices"][slice_name] = {"genuines": len(gs), "impostors": len(ims), "methods": methods}
    if reference:
        report["reference_manifest_overlap"] = reference_overlap(reference, rows)
    # Check source immutability after parsing and summarization.
    if hashlib.sha256(genuine_path.read_bytes()).hexdigest() != ghash or hashlib.sha256(impostor_path.read_bytes()).hexdigest() != ihash:
        raise ValueError("Source bytes changed during audit")
    report["source_byte_checks_passed"] = 2
    private = {"genuine_sha256": ghash, "impostor_sha256": ihash,
               "records": [asdict(row) for row in rows]}
    return report, coordinates, private


def write_outputs(public: Path, private: Path, report, coordinates, records):
    if public.exists() or private.exists():
        raise ValueError("Existing output directory; overwrite refused")
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    (public / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (private / "audit_records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    with (public / "curves.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=("slice", "method", "threshold", "false_accepts", "genuine_accepts", "far", "tar", "frr"))
        writer.writeheader()
        writer.writerows(coordinates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genuine", required=True, type=Path)
    parser.add_argument("--impostor", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/Precise")
    parser.add_argument("--reference-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--private-dir", required=True, type=Path)
    args = parser.parse_args()
    public, private = args.output_dir.resolve(), args.private_dir.resolve()
    if not public.is_relative_to(ROOT / "reports") or not private.is_relative_to(ROOT / "local"):
        parser.error("Public output must be under reports/; identifying output must be under local/")
    if public.exists() or private.exists():
        parser.error("Choose new output directories; overwrite refused")
    try:
        report, coordinates, records = audit(args.genuine, args.impostor, args.dataset_root, args.reference_manifest)
        write_outputs(public, private, report, coordinates, records)
    except (OSError, ValueError, UnicodeError):
        print("Audit failed; sources preserved. Check format/output state locally before retrying.", file=sys.stderr)
        return 1
    print("Aggregate audit and curves written; identifying records stay local")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
