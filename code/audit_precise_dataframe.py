"""Read-only intake of the supplied annotation pickle; aggregate public output only.

Uses a fixed constructor allowlist, never pandas.read_pickle or reference scripts.
Identity/capture conflicts remain unresolved; no images, pairs or splits are made.
"""

import argparse
import builtins
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PureWindowsPath
import pickle
import platform
import sys

from audit_precise_dataset import parse_labels

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("Dataset", "Finger", "Slap", "xMin", "xMax", "yMin", "yMax", "difficult",
          "dirty", "moisture", "overlap", "contrast", "SubjectID", "augmented_angle",
          "BBoxWidth", "BBoxHeight", "BBoxCenterX", "BBoxCenterY",
          "Corner_x1", "Corner_y1", "Corner_x2", "Corner_y2", "Corner_x3", "Corner_y3",
          "Corner_x4", "Corner_y4")
FINGERS = {"13": {2, 3, 4, 5}, "14": {7, 8, 9, 10}, "15": {11, 12}}


def restricted_frame(raw):
    """Accept only reviewed pandas/numpy reconstruction globals, not arbitrary globals.

    This is a format-specific intake boundary, not a general pickle sandbox.
    Unknown constructors are refused; no fallback to unrestricted unpickling.
    """
    import numpy as np
    import pandas as pd
    from numpy._core.multiarray import _reconstruct
    from pandas.core.indexes.base import _new_Index
    from pandas.core.internals.managers import BlockManager
    from pandas._libs.internals import _unpickle_block

    allowed = {
        ("pandas", "DataFrame"): pd.DataFrame, ("pandas", "Index"): pd.Index,
        ("pandas", "RangeIndex"): pd.RangeIndex,
        ("pandas.core.frame", "DataFrame"): pd.DataFrame,
        ("pandas.core.internals.managers", "BlockManager"): BlockManager,
        ("pandas.core.indexes.base", "_new_Index"): _new_Index,
        ("pandas.core.indexes.base", "Index"): pd.Index,
        ("pandas.core.indexes.range", "RangeIndex"): pd.RangeIndex,
        ("pandas._libs.internals", "_unpickle_block"): _unpickle_block,
        ("numpy.core.multiarray", "_reconstruct"): _reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): _reconstruct,
        ("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype,
        ("builtins", "slice"): builtins.slice,
    }
    used = set()

    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in allowed:
                raise ValueError("Unreviewed annotation constructor.")
            used.add((module, name))
            return allowed[(module, name)]

    stream = io.BytesIO(raw)
    frame = Restricted(stream).load()
    if stream.read(1) or type(frame) is not pd.DataFrame:
        raise ValueError("Unexpected annotation container or trailing bytes.")
    if list(frame.columns) != list(FIELDS):
        raise ValueError("Unexpected annotation schema.")
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in FIELDS if c not in {"Dataset", "Slap"}):
        raise ValueError("Unexpected numeric annotation type.")
    for column in ("Dataset", "Slap"):
        if any(type(value) is not str for value in frame[column].dropna()):
            raise ValueError("Unexpected textual annotation type.")
    return frame, sorted([list(item) for item in used])


def summarize(frame, image_names):
    """Reconcile labels as evidence; do not choose an authoritative identity source."""
    import numpy as np

    precise = frame.loc[frame["Dataset"].fillna("").str.strip().str.casefold().eq("precise")].copy()
    if precise.empty or precise[["Slap", "SubjectID", "Finger"]].isna().any().any():
        raise ValueError("Missing required Precise annotations.")
    for column in ("SubjectID", "Finger"):
        values = precise[column].to_numpy()
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Nonintegral Precise identity/finger annotations.")
    names = set(image_names)
    if len(names) != len(image_names):
        raise ValueError("Repeated local image names.")
    precise["_basename"] = precise["Slap"].map(lambda value: PureWindowsPath(value).name)
    private, dataframe_to_prefix, prefix_to_dataframe = [], defaultdict(set), defaultdict(set)
    counters = Counter()
    all_fingers = set().union(*FINGERS.values())
    for name, group in precise.groupby("_basename", sort=True):
        subjects = sorted({str(int(value)) for value in group["SubjectID"]})
        fingers = {int(value) for value in group["Finger"]}
        labels = parse_labels(Path(name))
        counters["multiple_subjects_per_slap"] += len(subjects) != 1
        counters["multiple_source_labels_per_basename"] += group["Slap"].nunique() != 1
        counters["unknown_finger_code_slaps"] += not fingers.issubset(all_fingers)
        counters["unparsed_filename_slaps"] += labels is None
        expected = FINGERS.get(labels[1], set()) if labels else set()
        counters["finger_set_conflicts"] += labels is not None and fingers != expected
        counters["repeated_finger_annotation_slaps"] += len(group) != len(fingers)
        numeric_equal = None
        if labels and len(subjects) == 1:
            prefix, _ = labels
            dataframe_to_prefix[subjects[0]].add(prefix)
            prefix_to_dataframe[prefix].add(subjects[0])
            counters["exact_subject_label_matches"] += prefix == subjects[0]
            numeric_equal = prefix.isdigit() and int(prefix) == int(subjects[0])
            counters["numeric_subject_label_matches"] += numeric_equal
        private.append({"image_name": name, "present_locally": name in names,
            "filename_subject": labels[0] if labels else "", "filename_hand": labels[1] if labels else "",
            "dataframe_subjects": json.dumps(subjects), "finger_codes": json.dumps(sorted(fingers)),
            "annotation_rows": len(group), "expected_fingers": fingers == expected,
            "numeric_identity_match": numeric_equal, "identity_review": "unreviewed",
            "independent_capture": "unknown"})
    annotated = set(precise["_basename"])
    datasets = Counter()
    for value in frame["Dataset"]:
        datasets[value if value in {"Precise", "Children", "LiveDet", "augmented"} else "other_or_missing"] += 1
    rotations = precise["augmented_angle"]
    report = {"schema_version": 1, "scope": "Annotation reconciliation only; no authoritative relabeling or capture inference",
        "dataframe": {"rows": len(frame), "columns": list(FIELDS), "dataset_rows": dict(datasets)},
        "precise": {"rows": len(precise), "distinct_source_slap_labels": int(precise["Slap"].nunique()),
            "distinct_slap_basenames": len(annotated), "distinct_subject_id_values": int(precise["SubjectID"].nunique()),
            "local_images": len(names), "matched_local_images": len(annotated & names),
            "local_images_without_annotation": len(names - annotated),
            "annotation_names_without_local_image": len(annotated - names),
            "finger_rows": {str(int(k)): int(v) for k, v in precise["Finger"].value_counts().items()},
            "nulls": {c: int(precise[c].isna().sum()) for c in FIELDS},
            "zero_rotation_rows": int(rotations.eq(0).sum()), "nonzero_rotation_rows": int((rotations.notna() & rotations.ne(0)).sum()),
            "missing_rotation_rows": int(rotations.isna().sum()), **dict(counters)},
        "identity_mapping": {"filename_prefix_groups": len(prefix_to_dataframe),
            "dataframe_id_groups": len(dataframe_to_prefix),
            "dataframe_ids_spanning_multiple_filename_prefixes": sum(len(v) > 1 for v in dataframe_to_prefix.values()),
            "filename_prefixes_spanning_multiple_dataframe_ids": sum(len(v) > 1 for v in prefix_to_dataframe.values()),
            "filename_prefixes_per_dataframe_id_distribution": dict(sorted(Counter(len(v) for v in dataframe_to_prefix.values()).items()))},
        "capture_session_ppi_fields_present": False, "split_ready": False,
        "limitations": ["The source schema has no explicit capture/session/parent, sensor, physical PPI or exposure fields.",
            "SubjectID and filename prefixes are compared as reported labels; neither is automatically corrected or verified.",
            "Annotation rows and rotations are not independent acquisitions; missing rotation is not imputed as zero.",
            "Rows labeled augmented cannot be assigned to Precise parents from this schema alone.",
            "Only local image names were read; no image bytes, score exports, models or reference scripts were executed.",
            "No exclusion, split allocation, pair generation or real-data model execution is authorized by this report."]}
    return report, private


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(source, images, output, private):
    import numpy as np
    import pandas as pd

    source, images, output, private = (p.resolve() for p in (source, images, output, private))
    if not output.is_relative_to(ROOT / "reports") or not private.is_relative_to(ROOT / "local/precise"):
        raise ValueError("Use reviewed aggregate reports and private Precise metadata paths.")
    if output.exists() or private.exists():
        raise ValueError("Existing intake outputs are never overwritten.")
    if source.stat().st_size > 128 * 1024**2:
        raise ValueError("Annotation source exceeds reviewed intake size.")
    entries = list(images.iterdir())
    if any(p.is_symlink() for p in entries):
        raise ValueError("Image inventory symlinks need review.")
    names = sorted(p.name for p in entries if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"})
    if not names:
        raise ValueError("No local image names available.")
    raw = source.read_bytes()
    frame, constructors = restricted_frame(raw)
    report, records = summarize(frame, names)
    if digest(source.read_bytes()) != digest(raw) or sorted(p.name for p in images.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}) != names:
        raise ValueError("Source or local image-name inventory changed during intake.")
    report["source"] = {"bytes": len(raw), "sha256": digest(raw)}
    private.mkdir(parents=True)
    with (private / "per_image_review.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    provenance = {"received_via": "User-supplied local GitLab folder; upstream revision not verified",
        "source_path": str(source), "source_sha256": digest(raw), "source_bytes": len(raw),
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "pandas": pd.__version__, "numpy": np.__version__, "allowed_constructors_used": constructors,
        "code_sha256": digest(Path(__file__).read_bytes()),
        "image_name_inventory_sha256": digest("\0".join(names).encode()),
        "identity_and_capture_verified": False}
    (private / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    output.mkdir(parents=True)
    (output / "dataframe_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataframe", "images", "output-dir", "private-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        audit(args.dataframe, args.images, args.output_dir, args.private_dir)
    except (OSError, ValueError, TypeError, ImportError, pickle.UnpicklingError, EOFError):
        print("Annotation intake failed; preserve any partial private output. No fallback or relabeling performed.", file=sys.stderr)
        return 1
    print("Aggregate annotation intake complete; identities and captures still require review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
