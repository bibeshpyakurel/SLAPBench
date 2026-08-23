"""
SLAPBench — Build SLAP Images Dataframe
=========================================
One row per SLAP image — exactly the 584 images currently on disk.

No per-finger expansion. No collapsing. Just a clean image-level view
of every SLAP image in the dataset with all image-level attributes.

Output: dataset/slap_images.csv  (584 rows)
"""

from pathlib import Path
import pandas as pd

DATASET_ROOT = Path(__file__).parent.parent / "datasets" / "sd302b"
MASTER_DF    = DATASET_ROOT / "master_dataframe.csv"
OUTPUT       = DATASET_ROOT / "slap_images.csv"

# Image-level columns — one value per image (not per finger)
IMAGE_COLS = [
    "subject_id",
    "image_name",
    "file_path",
    "which_hand",
    "device",
    "resolution_ppi",
    "frgp_slap",
    "image_width_px",
    "image_height_px",
    "file_size_bytes",
    "sha256",
    "age",
    "yob",
    "gender",
    "race",
    "work_type",
    "collection_day",
    "has_errata",
    "errata_note",
    "role",
    "genuine_pair_available",
]

df = pd.read_csv(MASTER_DF)
slap_df = (
    df[IMAGE_COLS]
    .drop_duplicates(subset="image_name")
    .sort_values(["device", "resolution_ppi", "subject_id", "frgp_slap"])
    .reset_index(drop=True)
)

slap_df.to_csv(OUTPUT, index=False)
print(f"Saved {len(slap_df)} rows x {len(slap_df.columns)} columns -> {OUTPUT.name}")
print()
print("=== Breakdown ===")
print(slap_df.groupby(["device", "resolution_ppi", "frgp_slap"]).size()
      .rename("images").to_string())
print()
print("Role breakdown:")
print(slap_df["role"].value_counts().to_string())
print()
print("Columns:", list(slap_df.columns))
