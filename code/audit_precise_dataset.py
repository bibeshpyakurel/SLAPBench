"""Read-only Precise inventory with aggregate output and no model dependencies.

Uses the first and last filename tokens as subject and hand-position labels,
consistent with precise_pairs.py. Intermediate tokens have no assumed meaning.
SHA-256 detects identical file bytes, not independent captures or near-duplicates.
No subject IDs, source filenames, paths or per-image hashes enter the report.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
POSITIONS = {"13": "right_four_finger_slap", "14": "left_four_finger_slap",
             "15": "thumbs"}


def parse_labels(path: Path) -> tuple[str, str] | None:
    parts = path.stem.split("_")
    if len(parts) not in {3, 4} or not all(parts) or parts[-1] not in POSITIONS:
        return None
    return parts[0], parts[-1]


def inventory(root: Path) -> dict:
    if not root.is_dir():
        raise ValueError("Precise directory unavailable")
    entries = sorted(root.iterdir())
    if any(p.is_symlink() for p in entries):
        raise ValueError("Symlink entries require separate review")
    files = [p for p in entries if p.is_file()]
    images = [p for p in files if p.suffix.lower() in {".jpeg", ".jpg"}]
    if not images:
        raise ValueError("No top-level JPEG files found")

    groups = defaultdict(list)
    labels_by_hash = defaultdict(set)
    hash_counts = Counter()
    shapes = Counter()
    unmatched = 0
    total_bytes = 0
    for path in images:
        total_bytes += path.stat().st_size
        shapes[len(path.stem.split("_"))] += 1
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        fingerprint = digest.hexdigest()
        hash_counts[fingerprint] += 1
        labels = parse_labels(path)
        if labels is None:
            unmatched += 1
        else:
            groups[labels].append(fingerprint)
            labels_by_hash[fingerprint].add(labels)

    positions = {}
    for position, name in POSITIONS.items():
        selected = {s: hashes for (s, p), hashes in groups.items() if p == position}
        positions[position] = {
            "meaning_from_existing_pair_builder": name,
            "images": sum(len(v) for v in selected.values()),
            "subjects": len(selected),
            "subject_position_groups": len(selected),
            "groups_with_multiple_files": sum(len(v) >= 2 for v in selected.values()),
            "groups_with_multiple_distinct_byte_contents": sum(
                len(set(v)) >= 2 for v in selected.values()
            ),
            "files_per_group_histogram": dict(sorted(Counter(
                len(v) for v in selected.values()
            ).items())),
        }
    slap_subjects = {s for s, p in groups if p in {"13", "14"}}
    return {
        "schema_version": 1,
        "scope": "Top-level Precise JPEG files; aggregate counts only",
        "methods": {
            "labels": "First/last filename tokens; three or four nonempty tokens",
            "duplicates": "SHA-256 of complete file bytes",
            "image_decoding": "Not performed",
        },
        "jpeg_images": len(images),
        "jpeg_bytes": total_bytes,
        "other_top_level_files": len(files) - len(images),
        "uninspected_subdirectories": sum(p.is_dir() for p in entries),
        "filename_token_count_histogram": dict(sorted(shapes.items())),
        "unrecognized_filename_count": unmatched,
        "subjects_with_recognized_labels": len({s for s, _ in groups}),
        "four_finger_slap_images": sum(positions[p]["images"] for p in ("13", "14")),
        "four_finger_slap_subjects": len(slap_subjects),
        "slap_subjects_with_both_hands": sum(
            (s, "13") in groups and (s, "14") in groups for s in slap_subjects
        ),
        "positions": positions,
        "exact_byte_duplicates": {
            "unique_contents": len(hash_counts),
            "duplicate_content_groups": sum(n >= 2 for n in hash_counts.values()),
            "redundant_files": sum(n - 1 for n in hash_counts.values()),
            "content_groups_spanning_subject_labels": sum(
                len({s for s, _ in labels}) >= 2 for labels in labels_by_hash.values()
            ),
            "content_groups_spanning_position_labels": sum(
                len({p for _, p in labels}) >= 2 for labels in labels_by_hash.values()
            ),
        },
        "limitations": [
            "Different bytes do not prove independent impressions; decoded-pixel and near-duplicate checks remain pending.",
            "Intermediate filename tokens are not interpreted as sessions, sensors or capture IDs.",
            "Hand labels do not establish segmented single-finger labels.",
            "Resolution, image readability/quality, data rights and SDK access are not validated.",
            "No subject split, pair manifest, model evaluation or training is created.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "datasets/Precise")
    parser.add_argument("--output", type=Path,
                        help="New aggregate JSON path; refuses an existing file")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new dated path")
    try:
        report = inventory(args.root)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
            print("Aggregate Precise inventory written")
    except (OSError, ValueError):
        # Filesystem exception text can contain original filenames/subject IDs.
        print("Inventory failed; no complete report was produced", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
