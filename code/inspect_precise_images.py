"""Inspect Precise headers and a small deterministic sample without inference.

Aggregate JSON goes to a new reports/ directory. Original filenames, source
hashes, image-level records and anonymous contact sheets stay under local/.
No matching pairs, subject partitions, segmentation or quality exclusions result.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw, ImageOps, ImageStat
import PIL

from audit_precise_dataset import parse_labels

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(root: Path, sample_subjects: int, seed: int) -> tuple[dict, list, list]:
    if sample_subjects < 1 or not root.is_dir():
        raise ValueError("Positive sample size and available dataset required")
    entries = sorted(root.iterdir())
    if any(p.is_symlink() for p in entries):
        raise ValueError("Source symlinks require separate review")
    files = [p for p in entries if p.is_file() and p.suffix.lower() in {".jpeg", ".jpg"}]
    if not files:
        raise ValueError("No top-level JPEG files")
    groups = defaultdict(list)
    dimensions, modes, dpi, orientations = Counter(), Counter(), Counter(), Counter()
    headers, middle = [], defaultdict(lambda: defaultdict(set))
    for path in files:
        labels = parse_labels(path)
        record = {"filename": path.name, "labels": labels}
        if labels:
            groups[labels].append(path)
        tokens = path.stem.split("_")
        for index, token in enumerate(tokens[1:-1], 1):
            middle[len(tokens)][index].add(token)
        try:
            with Image.open(path) as image:
                size = f"{image.width}x{image.height}"
                dimensions[size] += 1
                modes[image.mode] += 1
                tag = image.info.get("dpi")
                density = "missing" if not tag else "x".join(str(round(float(v), 3)) for v in tag)
                dpi[density] += 1
                orientation = image.getexif().get(274)
                orientations[str(orientation) if orientation in range(1, 9) else "missing_or_unrecognized"] += 1
                record.update({"size": image.size, "mode": image.mode, "dpi_tag": density,
                               "exif_orientation": orientation if orientation in range(1, 9) else None})
        except (OSError, ValueError, SyntaxError):
            record["error"] = "header_unreadable"
        headers.append(record)

    eligible = sorted({s for s, p in groups if p == "13" and len(groups[(s, p)]) >= 2
                       and len(groups.get((s, "14"), [])) >= 2})
    subjects = sorted(random.Random(seed).sample(eligible, min(sample_subjects, len(eligible))))
    selected = []
    for subject in subjects:
        for position in ("13", "14"):
            # Inspection coverage only; these files are not an evaluation pair list.
            selected.extend((groups[(subject, position)][0], groups[(subject, position)][-1]))
    if subjects:
        selected.extend(groups.get((subjects[0], "15"), [])[:2])
    selected.extend([p for p in files if len(p.stem.split("_")) == 3][:9])
    selected = list(dict.fromkeys(selected))

    records, previews, pixel_hashes, sample_positions = [], [], Counter(), Counter()
    for index, path in enumerate(selected, 1):
        alias = f"V{index:03d}"
        labels = parse_labels(path)
        sample_positions[labels[1] if labels else "unrecognized"] += 1
        record = {"alias": alias, "filename": path.name, "labels": labels,
                  "middle_tokens": path.stem.split("_")[1:-1]}
        before = file_hash(path)
        try:
            with Image.open(path) as image:
                image.load()
                gray = image.convert("L")
                stat = ImageStat.Stat(gray)
                histogram = gray.histogram()
                pixels = gray.width * gray.height
                digest = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
                pixel_hashes[digest] += 1
                record.update({"size": gray.size, "mean_gray": round(stat.mean[0], 3),
                               "gray_stddev": round(stat.stddev[0], 3),
                               "white_pixel_fraction": round(histogram[255] / pixels, 6),
                               "black_pixel_fraction": round(histogram[0] / pixels, 6),
                               "decoded_grayscale_sha256": digest})
                previews.append((alias, ImageOps.contain(gray, (520, 340))))
        except (OSError, ValueError, SyntaxError):
            record["error"] = "decode_unreadable"
            previews.append((alias, None))
        after = file_hash(path)
        if before != after:
            raise ValueError("Source changed during inspection")
        record.update({"source_sha256": before, "source_bytes_unchanged": True})
        records.append(record)

    statistics = {}
    for field in ("mean_gray", "gray_stddev", "white_pixel_fraction", "black_pixel_fraction"):
        values = [r[field] for r in records if field in r]
        statistics[field] = {"min": min(values), "max": max(values)} if values else None
    report = {
        "schema_version": 1,
        "scope": "Offline header inventory and small visual-inspection sample; no matching experiment",
        "software": {"pillow": PIL.__version__},
        "headers": {"jpeg_files": len(files), "opened": sum("error" not in r for r in headers),
                    "unreadable": sum("error" in r for r in headers),
                    "dimensions_px": dict(sorted(dimensions.items())), "modes": dict(sorted(modes.items())),
                    "dpi_metadata": dict(sorted(dpi.items())), "exif_orientation_tags": dict(sorted(orientations.items()))},
        "middle_token_cardinalities_by_filename_length": {
            str(length): {str(i): len(values) for i, values in sorted(columns.items())}
            for length, columns in sorted(middle.items())
        },
        "sample": {"seed": seed, "requested_subjects": sample_subjects,
                   "selected_subjects_for_main_coverage": len(subjects), "images": len(records),
                   "selection": "First and last sorted files per hand for seeded subjects with both hands; up to two thumbs and nine three-token variants; not random image-level sampling",
                   "position_counts": dict(sorted(sample_positions.items())),
                   "decoded": sum("error" not in r for r in records),
                   "unreadable": sum("error" in r for r in records),
                   "source_byte_checks_passed": len(records),
                   "exact_decoded_grayscale_duplicate_groups": sum(n >= 2 for n in pixel_hashes.values()),
                   "grayscale_statistics_ranges": statistics},
        "limitations": [
            "Header success does not establish full-image readability outside the decoded sample.",
            "DPI and orientation metadata are not independently verified physical resolution or finger labels.",
            "Intensity summaries and visual review are descriptive, not validated biometric quality scores.",
            "Sample duplicate checks do not rule out duplicates elsewhere, near-duplicates or shared parent captures.",
            "Middle filename tokens are counted but not interpreted as sessions, sensors or capture IDs.",
            "The inspection sample is not a training/development/test split or a matching pair manifest.",
        ],
    }
    return report, [{"header_records": headers, "sample_records": records}], previews


def write_outputs(public_dir: Path, private_dir: Path, report: dict, records: list, previews: list) -> None:
    if public_dir.exists() or private_dir.exists():
        raise ValueError("Choose new output directories; overwrite refused")
    if public_dir.resolve() == private_dir.resolve():
        raise ValueError("Public and private directories must differ")
    public_dir.mkdir(parents=True)
    private_dir.mkdir(parents=True)
    with (private_dir / "inspection_records.json").open("x", encoding="utf-8") as stream:
        json.dump(records[0], stream, indent=2, sort_keys=True)
        stream.write("\n")
    for offset in range(0, len(previews), 12):
        page = Image.new("RGB", (1620, 1520), "#ececec")
        draw = ImageDraw.Draw(page)
        for cell, (alias, preview) in enumerate(previews[offset:offset + 12]):
            x, y = (cell % 3) * 540, (cell // 3) * 380
            draw.text((x + 12, y + 8), alias, fill="black")
            if preview is not None:
                page.paste(preview, (x + (540 - preview.width) // 2, y + 30))
            else:
                draw.text((x + 12, y + 80), "Decode unavailable", fill="black")
        page.save(private_dir / f"contact_sheet_{offset // 12 + 1:02d}.png")
    with (public_dir / "inspection.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "datasets/Precise")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--sample-subjects", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    public, private = args.output_dir.resolve(), args.private_dir.resolve()
    if not public.is_relative_to(ROOT / "reports") or not private.is_relative_to(ROOT / "local"):
        parser.error("Aggregate output must be under reports/ and private output under local/")
    if public.exists() or private.exists():
        parser.error("Choose new output directories; overwrite refused")
    try:
        report, records, previews = inspect(args.root, args.sample_subjects, args.seed)
        write_outputs(public, private, report, records, previews)
    except (OSError, ValueError, SyntaxError):
        print("Inspection failed; inspect local output state before retrying", file=sys.stderr)
        return 1
    print("Aggregate inspection and local-only review files written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
