"""Read-only, aggregate inventory of RidgeBase Task1 single-finger images.

Prints counts only; subject IDs and original filenames never enter the output.
The numeric contactless finger labels are deliberately not mapped to named
fingers, because that mapping has not been validated for this repository.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK1 = ROOT / "datasets/ridgebase/Fingerprint_Train_Test_Split/Task1"


def parse_image(path: Path, modality: str) -> tuple[str, str, str, str, str]:
    parts = path.stem.split("_")
    if modality == "Contactbased":
        if path.suffix.lower() != ".bmp" or len(parts) != 4:
            raise ValueError("Unexpected contact-based filename structure")
        session, subject, hand, finger = parts
        device = "contact"
        if finger.lower() not in {"index", "middle", "ring", "little"}:
            raise ValueError("Unexpected contact-based finger label")
    else:
        if path.suffix.lower() != ".png" or len(parts) not in {8, 9}:
            raise ValueError("Unexpected contactless filename structure")
        session, device, subject, _, hand = parts[:5]
        finger = parts[-1]
        if parts[5] != "image" or not parts[6].startswith("fingerprint"):
            raise ValueError("Unexpected contactless capture structure")
        if finger not in {"0", "1", "2", "3"}:
            raise ValueError("Unexpected contactless finger label")
    if not session.isdigit() or not subject.isdigit() or hand.lower() not in {"left", "right"}:
        raise ValueError("Unexpected session, subject or hand label")
    return session, subject, hand.lower(), finger.lower(), device.lower()


def inventory(task1: Path) -> dict:
    if not task1.is_dir():
        raise FileNotFoundError(f"Task1 directory unavailable: {task1}")
    report = {"scope": "RidgeBase Task1 image files only", "splits": {}}
    split_subjects = {}
    for split in ("Train", "Test"):
        split_subjects[split] = set()
        report["splits"][split] = {}
        for modality, suffix in (("Contactbased", ".bmp"), ("Contactless", ".png")):
            folder = task1 / split / modality
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing Task1 modality directory: {folder}")
            files = sorted(folder.rglob(f"*{suffix}"))
            if not files:
                raise ValueError(f"No {suffix} images in {folder}")
            groups = Counter()
            sessions_by_group = defaultdict(set)
            sessions = Counter()
            fingers = Counter()
            devices = Counter()
            subjects = set()
            for path in files:
                try:
                    session, subject, hand, finger, device = parse_image(path, modality)
                except ValueError as exc:
                    raise ValueError(f"{path}: {exc}") from exc
                key = (subject, hand, finger)
                groups[key] += 1
                sessions_by_group[key].add(session)
                subjects.add(subject)
                sessions[session] += 1
                fingers[finger] += 1
                devices[device] += 1
            split_subjects[split].update(subjects)
            report["splits"][split][modality] = {
                "images": len(files),
                "subjects": len(subjects),
                "subject_hand_finger_groups": len(groups),
                "groups_with_multiple_images": sum(n >= 2 for n in groups.values()),
                "groups_with_two_sessions": sum(len(v) >= 2 for v in sessions_by_group.values()),
                "session_labels": dict(sorted(sessions.items())),
                "finger_labels": dict(sorted(fingers.items())),
                "devices": dict(sorted(devices.items())),
            }
    report["train_test_subject_overlap"] = len(split_subjects["Train"] & split_subjects["Test"])
    report["limitations"] = [
        "Filename/session counts do not prove that captures are independent.",
        "Contactless finger codes 0-3 have not been mapped to named finger positions.",
        "No dataset-license, image-quality, minutiae or matcher validation is implied.",
    ]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_TASK1)
    args = parser.parse_args()
    print(json.dumps(inventory(args.root), indent=2, sort_keys=True))
