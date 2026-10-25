"""Read-only full decoding, duplicate screening and private exposure inventory.

Never assigns acquisitions, subject partitions, matching lists or exclusions.
All image identifiers, hashes and similarity candidates remain under local/.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

from PIL import Image, ImageFile
import PIL

from audit_precise_dataset import parse_labels
from audit_precise_training_pairs import SOURCES, read_source

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = ((0, 22), (22, 21), (43, 21))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def difference_hash(gray):
    """64 horizontal differences after a documented 9 by 8 LANCZOS resize."""
    small = gray.resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (small[y * 9 + x] > small[y * 9 + x + 1])
    return value


def screen_near_duplicates(records, thumbnails):
    """Exhaustive dHash radius-2 retrieval; MAE is a second descriptive filter.

    At most two changed bits leave at least one of three blocks unchanged.
    This guarantees retrieval for this hash radius, not real near-duplicate recall.
    """
    buckets = defaultdict(list)
    candidates, retrieved, within_radius = [], 0, 0
    for index, row in enumerate(records):
        if row.get("status") != "ok":
            continue
        value = int(row["dhash64"], 16)
        keys = [(i, (value >> offset) & ((1 << width) - 1))
                for i, (offset, width) in enumerate(BLOCKS)]
        neighbors = {other for key in keys for other in buckets[key]}
        for other in sorted(neighbors):
            retrieved += 1
            previous = records[other]
            distance = (value ^ int(previous["dhash64"], 16)).bit_count()
            if distance > 2:
                continue
            within_radius += 1
            if row["decoded_sha256"] == previous["decoded_sha256"]:
                continue
            # Different native sizes can still be derivative candidates.
            absolute_sum = sum(abs(a - b) for a, b in
                               zip(thumbnails[index], thumbnails[other], strict=True))
            if absolute_sum <= 5 * 32 * 32:
                candidates.append({"image_a": previous["filename"],
                                   "image_b": row["filename"],
                                   "dhash_hamming_distance": distance,
                                   "thumbnail_mae": absolute_sum / (32 * 32),
                                   "same_subject_label": row["subject_id"] == previous["subject_id"],
                                   "same_hand_label": row["position"] == previous["position"],
                                   "review_status": "unreviewed"})
        for key in keys:
            buckets[key].append(index)
    return candidates, retrieved, within_radius


def duplicate_summary(records, field):
    groups = defaultdict(list)
    for row in records:
        if row.get(field):
            groups[row[field]].append(row)
    duplicates = [group for group in groups.values() if len(group) > 1]
    return {"unique_contents": len(groups), "duplicate_groups": len(duplicates),
            "redundant_files": sum(len(g) - 1 for g in duplicates),
            "groups_spanning_subject_labels": sum(
                len({r["subject_id"] for r in g}) > 1 for g in duplicates),
            "groups_spanning_position_labels": sum(
                len({r["position"] for r in g}) > 1 for g in duplicates)}


def audit(root, sources, historical_manifests=()):
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise ValueError("Strict decoding required")
    if set(sources) != set(SOURCES) or not root.is_dir():
        raise ValueError("Dataset and four source roles required")
    entries = sorted(root.iterdir())
    if any(p.is_symlink() for p in entries):
        raise ValueError("Dataset symlinks require review")
    files = [p for p in entries if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg"}]
    if not files:
        raise ValueError("No top-level JPEGs")
    memberships, subject_memberships, receipts = {}, {}, {}
    for role, path in sources.items():
        rows, receipt = read_source(path, role.rsplit("_", 1)[1])
        memberships[role] = {name for row in rows for name in row.names}
        subject_memberships[role] = {label[0] for row in rows for label in row.labels if label}
        receipts[str(path.resolve())] = receipt
    historical = set()
    for path in historical_manifests:
        data = path.read_bytes()
        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        if not {"img1_path", "img2_path"}.issubset(reader.fieldnames or []):
            raise ValueError("Historical manifest schema mismatch")
        for row in reader:
            for field in ("img1_path", "img2_path"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError("Incomplete historical image reference")
                historical.add(Path(row[field].replace("\\", "/")).name)
        receipts[str(path.resolve())] = {"bytes": len(data), "sha256": sha256(data)}
    historical_subjects = {labels[0] for name in historical
                           if (labels := parse_labels(Path(name)))}
    records, thumbnails = [], {}
    dimensions, modes, middle_shapes = Counter(), Counter(), Counter()
    for index, path in enumerate(files):
        labels = parse_labels(path)
        if labels is None:
            raise ValueError("Unrecognized label; review locally")
        subject, position = labels
        data = path.read_bytes()
        row = {"filename": path.name, "subject_id": subject, "position": position,
               "filename_tokens": path.stem.split("_"), "sha256": sha256(data),
               "bytes": len(data), "capture_id": None, "capture_verified": "unknown",
               "session_id": None, "acquisition_metadata_status": "not_supplied",
               "actual_training_use": "unknown", "actual_development_use": "unknown",
               "exposure_reviewed": "unknown", "independent_acquisition": "unknown",
               "supplied_list_membership": {role: path.name in names
                                             for role, names in memberships.items()},
               "subject_supplied_list_membership": {role: subject in names
                                                     for role, names in subject_memberships.items()},
               "historical_comparison_list_membership": path.name in historical,
               "subject_historical_comparison_list_membership": subject in historical_subjects}
        middle_shapes[len(path.stem.split("_"))] += 1
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG":
                    raise ValueError("Not JPEG")
                image.load()
                gray = image.convert("L")
                histogram = gray.histogram()
                row.update({"status": "ok", "size": list(image.size), "mode": image.mode,
                            "decoded_sha256": sha256(str(gray.size).encode() + gray.tobytes()),
                            "native_decoded_sha256": sha256(
                                (image.mode + str(image.size)).encode() + image.tobytes()),
                            "dhash64": f"{difference_hash(gray):016x}",
                            "gray_extrema": list(gray.getextrema()),
                            "white_pixel_fraction": histogram[255] / (gray.width * gray.height)})
                thumbnails[index] = gray.resize((32, 32), Image.Resampling.LANCZOS).tobytes()
                dimensions[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
        except (OSError, ValueError, SyntaxError):
            row["status"] = "decode_error"
        records.append(row)
    candidates, retrieved, within_radius = screen_near_duplicates(records, thumbnails)
    # Check the complete inventory and every original byte stream again.
    if entries != sorted(root.iterdir()):
        raise ValueError("Dataset inventory changed")
    for path, row in zip(files, records, strict=True):
        if sha256(path.read_bytes()) != row["sha256"]:
            raise ValueError("Image source changed")
    for path, receipt in receipts.items():
        if sha256(Path(path).read_bytes()) != receipt["sha256"]:
            raise ValueError("Metadata source changed")
    subjects = {r["subject_id"] for r in records}
    public = {
        "schema_version": 1,
        "scope": "Offline full top-level JPEG decode and provenance readiness; no matcher execution",
        "software": {"pillow": PIL.__version__},
        "images": len(records), "subjects": len(subjects),
        "four_finger_images": sum(r["position"] in {"13", "14"} for r in records),
        "decoded_images": sum(r["status"] == "ok" for r in records),
        "decode_failures": sum(r["status"] != "ok" for r in records),
        "source_byte_checks_passed": len(records), "metadata_byte_checks_passed": len(receipts),
        "uninspected_subdirectories": sum(p.is_dir() for p in entries),
        "other_top_level_files": len([p for p in entries if p.is_file()]) - len(files),
        "dimensions_px": dict(sorted(dimensions.items())), "modes": dict(sorted(modes.items())),
        "filename_token_count_histogram": dict(sorted(middle_shapes.items())),
        "constant_grayscale_images": sum(r.get("gray_extrema", [0, 1])[0] ==
                                          r.get("gray_extrema", [0, 1])[1] for r in records),
        "byte_duplicates": duplicate_summary(records, "sha256"),
        "native_decoded_duplicates": duplicate_summary(records, "native_decoded_sha256"),
        "grayscale_decoded_duplicates": duplicate_summary(records, "decoded_sha256"),
        "near_duplicate_screen": {"method": "64-bit horizontal dHash, 9x8 LANCZOS; Hamming <=2; 32x32 LANCZOS grayscale MAE <=5/255",
                                  "retrieved_file_relationships": retrieved,
                                  "relationships_within_hash_radius": within_radius,
                                  "nonidentical_candidates": len(candidates),
                                  "cross_subject_candidates": sum(not c["same_subject_label"] for c in candidates),
                                  "cross_position_candidates": sum(not c["same_hand_label"] for c in candidates),
                                  "confirmed_derivative_relationships": None,
                                  "automatic_exclusions": 0},
        "exposure": {role: {"images_in_supplied_list": sum(r["supplied_list_membership"][role] for r in records),
                             "local_subjects_in_supplied_list": len(subjects & subject_memberships[role])}
                     for role in SOURCES},
        "historical_list_exposure": {"local_images": sum(r["historical_comparison_list_membership"] for r in records),
                                     "local_subjects": len(subjects & historical_subjects)},
        "unknown_capture_images": len(records), "unreviewed_exposure_images": len(records),
        "ready_for_split_creation": False,
        "limitations": [
            "No acquisition/session facts or actual model consumption are inferred from filenames or supplied list membership.",
            "A dHash/thumbnail screen can miss shifted, cropped, rotated or otherwise transformed derivatives and can flag unrelated files.",
            "Candidates require private review; no file is excluded, moved, relabeled or assigned a capture ID.",
            "Decoded grayscale equivalence is distinct from native-mode pixel identity; both conventions are recorded.",
            "A successful decode or nonconstant image is not a biometric quality or independent-acquisition certificate.",
            "Historical manifest membership is not proof that a particular model consumed every listed image.",
        ],
    }
    private = {"schema_version": 1, "dataset_root": str(root.resolve()),
               "sources": receipts, "records": records, "near_duplicate_candidates": candidates,
               "decoded_hash_convention": "SHA-256(str(gray.size).encode() + gray.tobytes()); Pillow L, no EXIF transform",
               "native_hash_convention": "SHA-256((image.mode + str(image.size)).encode() + image.tobytes()); no EXIF transform"}
    return public, private


def write_outputs(public, private, report, details):
    if public.exists() or private.exists():
        raise ValueError("New output directories required")
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    for path, value in ((private / "readiness_records.json", details),
                        (public / "readiness.json", report)):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "datasets/Precise")
    for role in SOURCES:
        parser.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    args = parser.parse_args()
    public, private = args.output_dir.resolve(), args.private_dir.resolve()
    if not public.is_relative_to(ROOT / "reports") or not private.is_relative_to(ROOT / "local"):
        parser.error("Public output must be under reports/; private output under local/")
    if public.exists() or private.exists():
        parser.error("Choose new output directories; overwrite refused")
    try:
        report, details = audit(args.root, {role: getattr(args, role) for role in SOURCES},
                                args.historical_manifest)
        write_outputs(public, private, report, details)
    except (OSError, ValueError, UnicodeError, csv.Error):
        print("Readiness audit failed; inspect inputs locally. No complete report.", file=sys.stderr)
        return 1
    print("Full decoding audit complete. Identifying records remain local; acquisition/exposure review still required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
