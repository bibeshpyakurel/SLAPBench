"""
SLAPBench — Archive Non-SLAP Images
=====================================
Moves three categories of files to dataset/archive/, preserving structure.
Generates a full move log at dataset/archive/archive_log.csv.

Groups archived:
  Group 1 — Roll images (devices U, V) + their checksum CSVs   → archive/roll/
  Group 2 — FRGP 15 thumb slap images                          → archive/slap_frgp15/
  Group 3 — Slap-segmented individual finger crops + checksums  → archive/slap_segmented/

Nothing is deleted. Every moved file is logged with original path, archive path,
file size, sha256 (if in checksum CSV), and reason.

Usage:
    python code/archive_non_slap.py          # dry run — prints plan, moves nothing
    python code/archive_non_slap.py --run    # actually moves files
"""

import argparse
import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

DATASET_ROOT  = Path(__file__).parent.parent / "datasets" / "sd302b"
BASELINE_ROOT = DATASET_ROOT / "images" / "baseline"
ARCHIVE_ROOT  = DATASET_ROOT / "archive"
LOG_PATH      = ARCHIVE_ROOT / "archive_log.csv"

FILENAME_RE = re.compile(r"^(\d{8})_([A-Z])_(\d+)_(?:slap|roll)_(\d+)\.png$")

# ── Load all checksum CSVs into a lookup: filename → sha256 ───────────────────
def load_checksums() -> dict[str, str]:
    lookup = {}
    for csv_path in BASELINE_ROOT.rglob("checksum_*.csv"):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                # column is "sha256" in some CSVs, "checksum" in others
                hash_val = row.get("sha256") or row.get("checksum", "")
                lookup[row["filename"]] = hash_val
    return lookup

# ── Classify each file ────────────────────────────────────────────────────────
def classify(path: Path) -> tuple[str | None, str | None]:
    """
    Returns (group_name, reason) or (None, None) if the file should stay.
    group_name maps to the archive subdirectory.
    """
    parts = path.relative_to(BASELINE_ROOT).parts
    if not parts:
        return None, None

    device = parts[0]  # R, S, U, V

    # ── Group 1: Roll images and their checksum CSVs ──────────────────────────
    if device in ("U", "V"):
        if path.suffix == ".png":
            return "roll", f"Device {device} roll image — single-finger sequential capture, not SLAP"
        if path.suffix == ".csv":
            return "roll", f"Checksum CSV for device {device} roll images"
        return None, None

    # ── Group 3: Slap-segmented directory (any device R or S) ─────────────────
    if "slap-segmented" in parts:
        if path.suffix == ".png":
            return "slap_segmented", "Pre-cropped individual finger from slap-segmented directory — deferred to Task 5"
        if path.suffix == ".csv":
            return "slap_segmented", "Checksum CSV for slap-segmented images"
        return None, None

    # ── Group 2: FRGP 15 thumb slap images ───────────────────────────────────
    if path.suffix == ".png":
        m = FILENAME_RE.match(path.name)
        if m:
            frgp = int(m.group(4))
            if frgp == 15:
                return "slap_frgp15", "FRGP 15 — two-thumb simultaneous capture, not 4-finger SLAP"

    return None, None   # keep

# ── Build the move plan ───────────────────────────────────────────────────────
def build_plan(checksums: dict) -> list[dict]:
    plan = []
    for path in sorted(BASELINE_ROOT.rglob("*")):
        if not path.is_file():
            continue
        group, reason = classify(path)
        if group is None:
            continue

        rel = path.relative_to(BASELINE_ROOT)
        dest = ARCHIVE_ROOT / group / rel
        sha = checksums.get(path.name, "")

        plan.append({
            "original_path": str(path.relative_to(DATASET_ROOT)),
            "archive_path":  str(dest.relative_to(DATASET_ROOT)),
            "group":         group,
            "reason":        reason,
            "file_size_bytes": path.stat().st_size,
            "sha256":        sha,
            "filename":      path.name,
        })
    return plan

# ── Execute ───────────────────────────────────────────────────────────────────
def execute(plan: list[dict], dry_run: bool):
    print(f"\n{'DRY RUN — ' if dry_run else ''}Moving {len(plan)} files to archive/\n")

    by_group: dict[str, list] = {}
    for entry in plan:
        by_group.setdefault(entry["group"], []).append(entry)

    for group, entries in sorted(by_group.items()):
        total_mb = sum(e["file_size_bytes"] for e in entries) / 1e6
        print(f"  {group:20s}  {len(entries):5d} files  {total_mb:.1f} MB")
        if dry_run:
            # Show a few examples
            for e in entries[:3]:
                print(f"    {e['original_path']}")
            if len(entries) > 3:
                print(f"    ... and {len(entries)-3} more")

    if dry_run:
        print("\nDry run complete. Run with --run to execute.")
        return

    # ── Actually move ──────────────────────────────────────────────────────────
    moved, failed = 0, 0
    for i, entry in enumerate(plan, 1):
        src  = DATASET_ROOT / entry["original_path"]
        dest = DATASET_ROOT / entry["archive_path"]
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved += 1
        except Exception as exc:
            print(f"  ERROR moving {src.name}: {exc}")
            failed += 1
        if i % 500 == 0:
            print(f"  ... {i}/{len(plan)} moved")

    print(f"\nDone. Moved: {moved}  Failed: {failed}")

    # ── Write log ──────────────────────────────────────────────────────────────
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = ["original_path", "archive_path", "group", "reason",
                  "file_size_bytes", "sha256", "filename", "archived_at"]
    ts = datetime.now().isoformat(timespec="seconds")
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in plan:
            entry["archived_at"] = ts
            writer.writerow(entry)
    print(f"Log written → {LOG_PATH.relative_to(DATASET_ROOT.parent)}")

# ── Summary of what stays ────────────────────────────────────────────────────
def print_what_stays():
    remaining = [p for p in BASELINE_ROOT.rglob("*.png")
                 if classify(p)[0] is None]
    print(f"\n  Files that will stay in baseline/: {len(remaining)}")
    from collections import Counter
    cats = Counter()
    for p in remaining:
        parts = p.relative_to(BASELINE_ROOT).parts
        cats[f"{parts[0]} / {parts[1]} / {parts[2]}"] += 1
    for k, v in sorted(cats.items()):
        print(f"    {k:35s}  {v} images")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true",
                        help="Execute the move (default: dry run)")
    args = parser.parse_args()

    dry_run = not args.run

    print("=" * 60)
    print("SLAPBench Archive Script")
    print("=" * 60)

    checksums = load_checksums()
    print(f"Loaded {len(checksums)} checksum entries.")

    plan = build_plan(checksums)

    total_bytes = sum(e["file_size_bytes"] for e in plan)
    print(f"Plan: {len(plan)} files  ({total_bytes/1e9:.2f} GB) to be archived")

    print_what_stays()
    execute(plan, dry_run=dry_run)
