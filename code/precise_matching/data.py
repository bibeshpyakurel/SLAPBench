"""Validate private, externally prepared metadata without assigning a split."""

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import io
from pathlib import Path
import re


PARTITIONS = ("train", "development", "test")
IMAGE_FIELDS = (
    "image_id", "subject_id", "hand", "capture_id", "partition", "sha256",
    "decoded_sha256", "capture_verified", "exposure_reviewed",
)
PAIR_FIELDS = ("pair_id", "image_a", "image_b", "label")


class InputError(ValueError):
    """A fixed, identifier-free message suitable for the public CLI."""


def read_table(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    """Reject ambiguous columns, ragged rows, empty tables and whitespace IDs."""
    reader = csv.DictReader(io.StringIO(path.read_bytes().decode("utf-8-sig")), strict=True)
    if reader.fieldnames is None or len(reader.fieldnames) != len(fields):
        raise InputError("Invalid CSV header.")
    if set(reader.fieldnames) != set(fields):
        raise InputError("Invalid CSV header.")
    rows = list(reader)
    if not rows:
        raise InputError("Empty CSV table.")
    for row in rows:
        if None in row or any(value is None or value != value.strip() for value in row.values()):
            raise InputError("Ragged CSV row or surrounding whitespace.")
    return rows


@dataclass(frozen=True)
class Image:
    image_id: str
    subject_id: str
    hand: str
    capture_id: str
    partition: str
    sha256: str
    decoded_sha256: str
    capture_verified: str
    exposure_reviewed: str


@dataclass(frozen=True)
class Pair:
    pair_id: str
    image_a: str
    image_b: str
    label: str


def load_protocol(images_path: Path, pairs_path: Path) -> tuple[list[Image], list[Pair]]:
    images = [Image(**row) for row in read_table(images_path, IMAGE_FIELDS)]
    pairs = [Pair(**row) for row in read_table(pairs_path, PAIR_FIELDS)]
    _check_records(images, pairs)
    return images, pairs


def _check_records(images: list[Image], pairs: list[Pair]) -> None:
    for records, key in ((images, "image_id"), (pairs, "pair_id")):
        ids = [getattr(row, key) for row in records]
        if not all(ids) or len(ids) != len(set(ids)):
            raise InputError("Empty or repeated record identifier.")
    for image in images:
        if not image.subject_id or image.hand not in {"left", "right"}:
            raise InputError("Subject required; hand must be left or right four-finger slap.")
        if image.partition not in PARTITIONS:
            raise InputError("Partition must be train, development or test.")
        for digest in (image.sha256, image.decoded_sha256):
            if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise InputError("Content digest must be empty or lowercase SHA-256.")
        if any(flag not in {"true", "false", "unknown"}
               for flag in (image.capture_verified, image.exposure_reviewed)):
            raise InputError("Review flags must be true, false or unknown.")
    for pair in pairs:
        if not pair.image_a or not pair.image_b or pair.label not in {"genuine", "impostor"}:
            raise InputError("Invalid comparison fields or class.")


def validate_protocol(images: list[Image], pairs: list[Pair]) -> dict:
    """Return aggregate blockers; metadata assertions are not independent proof."""
    _check_records(images, pairs)
    issues = Counter()
    by_id = {row.image_id: row for row in images}
    subjects, captures = defaultdict(list), defaultdict(list)
    for image in images:
        subjects[image.subject_id].append(image)
        if image.capture_id:
            captures[image.capture_id].append(image)
        for field in ("capture_id", "sha256", "decoded_sha256"):
            if not getattr(image, field):
                issues[field + "_missing_images"] += 1
        if image.capture_verified != "true":
            issues["capture_unverified_images"] += 1
        if image.exposure_reviewed != "true":
            issues["exposure_unreviewed_images"] += 1
    issues["subject_partition_conflicts"] = sum(
        len({r.partition for r in group}) > 1 for group in subjects.values())
    issues["capture_metadata_conflicts"] = sum(
        len({(r.subject_id, r.hand, r.partition) for r in group}) > 1
        for group in captures.values())
    for field in ("sha256", "decoded_sha256"):
        groups = defaultdict(list)
        for image in images:
            if getattr(image, field):
                groups[getattr(image, field)].append(image)
        issues[field + "_partition_conflicts"] = sum(
            len({r.partition for r in group}) > 1 for group in groups.values())
        issues[field + "_label_conflicts"] = sum(
            len({(r.subject_id, r.hand) for r in group}) > 1 for group in groups.values())
        issues[field + "_capture_conflicts"] = sum(
            len({r.capture_id for r in group if r.capture_id}) > 1 for group in groups.values())

    pair_counts = {p: Counter() for p in PARTITIONS}
    image_pairs, capture_pairs = set(), set()
    for pair in pairs:
        if pair.image_a not in by_id or pair.image_b not in by_id:
            issues["pairs_with_unknown_images"] += 1
            continue
        a, b = by_id[pair.image_a], by_id[pair.image_b]
        image_key = tuple(sorted((a.image_id, b.image_id)))
        if image_key in image_pairs:
            issues["duplicate_unordered_image_pairs"] += 1
        image_pairs.add(image_key)
        if a.capture_id and b.capture_id:
            capture_key = tuple(sorted((a.capture_id, b.capture_id)))
            if capture_key in capture_pairs:
                issues["duplicate_unordered_capture_pairs"] += 1
            capture_pairs.add(capture_key)
            if a.capture_id == b.capture_id:
                issues["same_capture_pairs"] += 1
        if a.image_id == b.image_id:
            issues["self_image_pairs"] += 1
        if a.partition != b.partition:
            issues["cross_partition_pairs"] += 1
        else:
            pair_counts[a.partition][pair.label] += 1
        if a.hand != b.hand:
            issues["mixed_hand_pairs"] += 1
        if (a.subject_id == b.subject_id) != (pair.label == "genuine"):
            issues["pair_label_conflicts"] += 1
        for field in ("sha256", "decoded_sha256"):
            if getattr(a, field) and getattr(a, field) == getattr(b, field):
                issues[field + "_identical_pairs"] += 1
    partitions = {}
    for partition in PARTITIONS:
        selected = [row for row in images if row.partition == partition]
        partitions[partition] = {
            "images": len(selected), "subjects": len({r.subject_id for r in selected}),
            "genuine_pairs": pair_counts[partition]["genuine"],
            "impostor_pairs": pair_counts[partition]["impostor"],
        }
        if not selected:
            issues["empty_image_partitions"] += 1
        if partition != "train":
            issues["missing_evaluation_classes"] += sum(
                pair_counts[partition][label] == 0 for label in ("genuine", "impostor"))
    blockers = dict(sorted((key, count) for key, count in issues.items() if count))
    return {
        "schema_version": 1, "scope": "Offline supplied-metadata preflight; no split creation",
        "metadata_checks_passed": not blockers, "blockers": blockers,
        "partitions": partitions,
        "limits": [
            "Capture IDs and review flags are supplied assertions, not independently verified acquisition evidence.",
            "Digests are compared as supplied; this tool does not read or hash source images.",
            "Near duplicates, prior exposure claims, model selection and task authorization require separate review.",
            "Passing checks does not certify an untouched cohort or authorize training.",
        ],
    }
