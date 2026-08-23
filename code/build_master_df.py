"""
SLAPBench — Build Master DataFrame
====================================
One row per (image × finger) — 584 images × 4 fingers = 2,336 rows.

This design supports:
  Task 8 (verification)  — group by image, use image-level columns
  Task 3 (finger ID)     — use per-finger rows directly
  Task 4 (localization)  — use bounding box columns
  Task 6 (rotation)      — use theta_deg column
  Task 5 (NFIQ2)         — fill nfiq2_score column when that task runs

Output: dataset/master_dataframe.csv
"""

import csv
import re
from pathlib import Path
import pandas as pd
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_ROOT  = Path(__file__).parent.parent / "datasets" / "sd302b"
BASELINE_ROOT = DATASET_ROOT / "images" / "baseline"
PARTICIPANTS  = DATASET_ROOT / "participants.csv"
ARCHIVE_LOG   = DATASET_ROOT / "archive" / "archive_log.csv"
OUTPUT        = DATASET_ROOT / "master_dataframe.csv"

FILENAME_RE = re.compile(r"^(\d{8})_([A-Z])_(\d+)_slap_(\d+)\.png$")

FINGER_LABEL = {
    2: "right_index", 3: "right_middle", 4: "right_ring",  5: "right_little",
    7: "left_index",  8: "left_middle",  9: "left_ring",   10: "left_little",
}

ERRATA = {
    "00002302": "thumbs swapped (device L)",
    "00002361": "right ring labeled as right index (device V)",
    "00002420": "slap-segmented finger 1 and 6 swapped (device R)",
    "00002534": "thumbs swapped (device S)",
    "00002561": "thumbs swapped (device S)",
    "00002497": "finger 08 unusable region (device C)",
    "00002354": "fingers 04 and 05 possibly mislabeled (device C)",
}

# ── Load participants ──────────────────────────────────────────────────────────
print("Loading participants...")
participants = pd.read_csv(PARTICIPANTS)
participants["id"] = participants["id"].astype(str).str.zfill(8)
participants = participants.set_index("id")

# ── Load segmentation CSVs ────────────────────────────────────────────────────
print("Loading segmentation CSVs...")
seg_lookup: dict[str, list[dict]] = {}   # image filename → list of finger rows
for csv_path in sorted(BASELINE_ROOT.rglob("segmentation_*.csv")):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            fname = row["filename"]
            seg_lookup.setdefault(fname, []).append({
                "finger_frgp": int(row["frgp"]),
                "tlx": int(row["tlx"]), "tly": int(row["tly"]),
                "trx": int(row["trx"]), "try": int(row["try"]),
                "blx": int(row["blx"]), "bly": int(row["bly"]),
                "brx": int(row["brx"]), "bry": int(row["bry"]),
                "theta_deg": float(row["theta"]),
            })

# ── Load checksum CSVs ────────────────────────────────────────────────────────
print("Loading checksums...")
checksums: dict[str, str] = {}
for csv_path in BASELINE_ROOT.rglob("checksum_*.csv"):
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            hash_val = row.get("sha256") or row.get("checksum", "")
            checksums[row["filename"]] = hash_val

# ── Identify which R subjects have both 500 and 1000 PPI (genuine pair pool) ──
r500_subjects  = set()
r1000_subjects = set()
for p in BASELINE_ROOT.rglob("*.png"):
    m = FILENAME_RE.match(p.name)
    if not m: continue
    subj, dev, res, frgp = m.groups()
    if dev == "R" and int(frgp) in (13, 14):
        if res == "500":  r500_subjects.add(subj)
        if res == "1000": r1000_subjects.add(subj)
genuine_pair_subjects = r500_subjects & r1000_subjects

# ── Build rows ────────────────────────────────────────────────────────────────
print("Scanning images and building rows...")
records = []

for png_path in sorted(BASELINE_ROOT.rglob("*.png")):
    m = FILENAME_RE.match(png_path.name)
    if not m:
        continue

    subj, device, res_str, frgp_str = m.groups()
    frgp     = int(frgp_str)
    res      = int(res_str)
    fname    = png_path.name
    rel_path = str(png_path.relative_to(DATASET_ROOT))

    # Only FRGP 13 (right) and 14 (left) — the 4-finger SLAP images
    if frgp not in (13, 14):
        continue

    which_hand = "right" if frgp == 13 else "left"

    # Image dimensions and file size
    try:
        with Image.open(png_path) as img:
            width_px, height_px = img.size
    except Exception:
        width_px = height_px = None

    file_size_bytes = png_path.stat().st_size
    sha256          = checksums.get(fname, "")

    # Participant metadata
    meta = participants.loc[subj] if subj in participants.index else None
    age            = int(meta["age"])            if meta is not None else None
    yob            = int(meta["yob"])            if meta is not None else None
    gender         = meta["gender"]              if meta is not None else None
    race           = meta["race"]                if meta is not None else None
    work_type      = meta["work_type"]           if meta is not None else None
    collection_day = int(meta["collection_day"]) if meta is not None else None

    # Errata
    has_errata  = subj in ERRATA
    errata_note = ERRATA.get(subj, "")

    # Role
    if has_errata:
        role = "excluded_errata"
    elif device == "R" and res == 1000:
        role = "task8_genuine_pair"
    else:
        role = "primary_eval"

    # Genuine pair availability
    genuine_pair_available = subj in genuine_pair_subjects

    # Per-finger rows from segmentation CSV
    finger_rows = seg_lookup.get(fname, [])
    # Filter to only the 4 main fingers (exclude thumbs FRGP 1/6 that appear in FRGP-15 entries)
    finger_rows = [r for r in finger_rows if r["finger_frgp"] in FINGER_LABEL]

    if not finger_rows:
        # Image has no segmentation data — create one image-level row with nulls
        finger_rows = [{"finger_frgp": None, "tlx": None, "tly": None,
                        "trx": None, "try": None, "blx": None, "bly": None,
                        "brx": None, "bry": None, "theta_deg": None}]

    for fr in finger_rows:
        records.append({
            # ── Identity ──────────────────────────────────────────────────────
            "subject_id":            subj,
            "image_name":            fname,
            "file_path":             rel_path,

            # ── Capture metadata ──────────────────────────────────────────────
            "which_hand":            which_hand,
            "device":                device,
            "resolution_ppi":        res,
            "frgp_slap":             frgp,          # 13 or 14

            # ── Image properties ──────────────────────────────────────────────
            "image_width_px":        width_px,
            "image_height_px":       height_px,
            "file_size_bytes":       file_size_bytes,
            "sha256":                sha256,

            # ── Per-finger segmentation ground truth ──────────────────────────
            "finger_frgp":           fr["finger_frgp"],
            "finger_label":          FINGER_LABEL.get(fr["finger_frgp"], ""),
            "tlx":                   fr["tlx"],
            "tly":                   fr["tly"],
            "trx":                   fr["trx"],
            "try":                   fr["try"],
            "blx":                   fr["blx"],
            "bly":                   fr["bly"],
            "brx":                   fr["brx"],
            "bry":                   fr["bry"],
            "theta_deg":             fr["theta_deg"],

            # ── Participant demographics ───────────────────────────────────────
            "age":                   age,
            "yob":                   yob,
            "gender":                gender,
            "race":                  race,
            "work_type":             work_type,
            "collection_day":        collection_day,

            # ── Quality / flags ────────────────────────────────────────────────
            "has_errata":            has_errata,
            "errata_note":           errata_note,
            "role":                  role,
            "genuine_pair_available": genuine_pair_available,

            # ── Future columns (null until those tasks run) ────────────────────
            "nfiq2_score":           None,   # Task 5 — run NFIQ 2 to populate
        })

# ── Build DataFrame and save ──────────────────────────────────────────────────
df = pd.DataFrame(records)
df.to_csv(OUTPUT, index=False)
print(f"\nSaved {len(df)} rows × {len(df.columns)} columns → {OUTPUT.name}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Master DataFrame Summary ===")
print(f"  Total rows:       {len(df)}")
print(f"  Unique images:    {df['image_name'].nunique()}")
print(f"  Unique subjects:  {df['subject_id'].nunique()}")
print()
print("  Role breakdown:")
for role, cnt in df.drop_duplicates("image_name")["role"].value_counts().items():
    print(f"    {role:25s}  {cnt} images")
print()
print("  Device / resolution / hand breakdown (unique images):")
img_df = df.drop_duplicates("image_name")
print(img_df.groupby(["device","resolution_ppi","which_hand"])["image_name"]
           .count().rename("images").to_string())
print()
print("  Fingers with non-zero theta:")
nonzero = df[df["theta_deg"].notna() & (df["theta_deg"] != 0)]
print(f"    {len(nonzero)} finger rows  ({nonzero['image_name'].nunique()} images)")
if len(nonzero):
    print(f"    Range: {nonzero['theta_deg'].min():.3f}° to {nonzero['theta_deg'].max():.3f}°")
print()
print("  Errata-flagged images:", df.drop_duplicates("image_name")["has_errata"].sum())
print()
print(f"  Columns ({len(df.columns)}):")
for c in df.columns:
    null_n = df[c].isna().sum()
    note = f"  ({null_n} nulls)" if null_n else ""
    print(f"    {c}{note}")
