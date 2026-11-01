"""Private, explicit-pair shared benchmark; never infer targets from identities.

This route deliberately does not use the subject-disjoint metadata schema.
Preparation consumes existing inventories and source lists, not image pixels.
"""

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import re

from .data import InputError
from .evaluation import target_fraction
from .training import digest, write_json

HANDS = {"13": "right", "14": "left", "15": "thumbs"}
LABELS = {"genuine": "A", "impostor": "B"}
TRAIN_FILE = "slap_verification_binary_ab_13_14_15_balanced.json"
TEST_FILES = {kind: f"random_2100_{kind}_seed2026.txt" for kind in LABELS}
REVIEW_FLAGS = ("protocol_reviewed", "results_reviewed", "cohort_reviewed",
                "labels_as_supplied_reviewed", "execution_authorized")


@dataclass(frozen=True)
class FixedPair:
    pair_id: str
    image_a: str
    image_b: str
    label: str
    hand: str
    numeric_identity_disagreement: bool


@dataclass(frozen=True)
class FixedConfig:
    seed: int = 2026
    epochs: int = 3
    batch_pairs: int = 8
    accumulation: int = 4
    warmup_steps: int = 100
    learning_rate: float = 1e-5
    minimum_learning_rate: float = 1e-6
    weight_decay: float = .01
    margin: float = .2
    target_far: str = "0.01"
    checkpoint_rule: str = "fixed_final_epoch"
    threshold_rule: str = "training_impostors_strict_greater"
    device: str = "cuda"
    precision: str = "bf16"

    def validate(self, pair_count=None):
        integers = (self.seed, self.epochs, self.batch_pairs, self.accumulation, self.warmup_steps)
        if (any(type(v) is not int for v in integers) or not 0 <= self.seed < 2**63
                or min(self.epochs, self.batch_pairs, self.accumulation) < 1 or self.warmup_steps < 0):
            raise InputError("Invalid fixed-pair integer schedule.")
        if pair_count is not None and (pair_count < 1 or self.warmup_steps >= self.steps(pair_count)):
            raise InputError("Warmup must finish before the last scheduled update.")
        numbers = (self.learning_rate, self.minimum_learning_rate, self.weight_decay, self.margin)
        if (any(type(v) not in {int, float} or not math.isfinite(v) for v in numbers)
                or not 0 < self.minimum_learning_rate <= self.learning_rate
                or self.weight_decay < 0 or not -1 <= self.margin <= 1):
            raise InputError("Invalid fixed-pair optimizer settings.")
        if (self.checkpoint_rule != "fixed_final_epoch"
                or self.threshold_rule != "training_impostors_strict_greater"):
            raise InputError("Only the declared fixed selection rules are implemented.")
        if (self.device not in {"cpu", "cuda"} or self.precision not in {"fp32", "bf16"}
                or self.device == "cpu" and self.precision != "fp32"):
            raise InputError("Explicit supported device/precision required; no fallback.")
        target_fraction(self.target_far)

    def steps(self, pair_count):
        return self.epochs * math.ceil(math.ceil(pair_count / self.batch_pairs) / self.accumulation)


def validate_pairs(pairs):
    if not pairs:
        raise InputError("An explicit nonempty pair list is required.")
    ids, unordered = set(), set()
    for p in pairs:
        key = tuple(sorted((p.image_a, p.image_b)))
        if (not p.pair_id or p.pair_id in ids or key in unordered or p.image_a == p.image_b
                or p.label not in LABELS or p.hand not in HANDS.values()
                or type(p.numeric_identity_disagreement) is not bool):
            raise InputError("Invalid, repeated or inconsistent explicit pair.")
        ids.add(p.pair_id)
        unordered.add(key)
    if {p.label for p in pairs} != set(LABELS):
        raise InputError("Both supplied target classes are required.")


def source_pair(index, a, b, label, phase):
    """Parse hand/ambiguity metadata only; the supplied target is unchanged."""
    from audit_precise_dataset import parse_labels

    a, b = (PureWindowsPath(name).name for name in (a, b))
    parts = [parse_labels(Path(name)) for name in (a, b)]
    if (label not in LABELS or any(p is None for p in parts)
            or any(not p[0].isascii() or not p[0].isdigit() for p in parts)
            or parts[0][1] != parts[1][1] or parts[0][1] not in HANDS):
        raise InputError("Unrecognized shared-pair metadata.")
    conflict = (int(parts[0][0]) == int(parts[1][0])) != (label == "genuine")
    return FixedPair(f"{phase}-{index:08d}", a, b, label, HANDS[parts[0][1]], conflict)


def prepare_bundle(intake, inventory_path, audit_path, output):
    """Translate supplied pairs losslessly, using the completed image inventory."""
    from audit_precise_training_pairs import read_source

    intake, output = Path(intake), Path(output)
    if output.exists():
        raise InputError("Preparation requires a new private bundle path.")
    sources = {name: intake / name for name in (TRAIN_FILE, *TEST_FILES.values())}
    sources.update(inventory=Path(inventory_path), audit=Path(audit_path))
    hashes = {name: digest(path) for name, path in sources.items()}
    audit = json.loads(Path(audit_path).read_text())
    if any(hashes[name] != audit["sources"][name]["sha256"]
           for name in (TRAIN_FILE, *TEST_FILES.values())):
        raise InputError("Supplied inputs differ from the completed cohort audit.")
    data = json.loads(sources[TRAIN_FILE].read_text())
    records = data["pairs"]
    records = list(records.values()) if isinstance(records, dict) else records
    training = []
    for index, record in enumerate(records, 1):
        label = record["pair_class"]
        if (label not in LABELS or record["correct_answer_option"] != LABELS[label]
                or len(record["image_paths"]) != 2):
            raise InputError("Explicit A/B target mismatch.")
        training.append(source_pair(index, *record["image_paths"], label, "train"))
    testing = []
    for label, name in TEST_FILES.items():
        rows, _ = read_source(sources[name], label)
        for row in rows:
            testing.append(source_pair(len(testing) + 1, *row.names, label, "test"))
    for phase, pairs in (("training", training), ("testing", testing)):
        validate_pairs(pairs)
        expected = audit[phase]
        if (len(pairs) != expected["records"]
                or Counter(p.label for p in pairs) != expected["class_counts"]
                or sum(p.numeric_identity_disagreement for p in pairs)
                != expected["numeric_id_label_disagreements"]
                or Counter(f"{suffix}_{p.label}" for p in pairs for suffix, hand in HANDS.items()
                           if p.hand == hand) != expected["hand_class_counts"]):
            raise InputError("Pair membership/counts differ from the reviewed audit.")
    if ({tuple(sorted((p.image_a, p.image_b))) for p in training}
            & {tuple(sorted((p.image_a, p.image_b))) for p in testing}):
        raise InputError("Exact training/testing pair overlap is prohibited.")
    inventory = json.loads(Path(inventory_path).read_text())["records"]
    by_name = {row["filename"]: row for row in inventory}
    if len(by_name) != len(inventory):
        raise InputError("Duplicate image-inventory keys.")
    needed = {key for p in training + testing for key in (p.image_a, p.image_b)}
    images = {}
    for key in sorted(needed):
        row = by_name[key]
        if row["status"] != "ok":
            raise InputError("Existing image inventory has unresolved decoding failures.")
        images[key] = {"path": key, "sha256": row["sha256"], "decoded_sha256": row["decoded_sha256"]}
    if any(digest(path) != hashes[name] for name, path in sources.items()):
        raise InputError("Source changed during private preparation.")
    bundle = {"schema_version": 1, "scope": "shared_seen_image_fixed_pairs",
              "labels": "as_supplied", "source_sha256": hashes,
              "images": images, "training": [asdict(p) for p in training],
              "testing": [asdict(p) for p in testing]}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, bundle)
    return {"training_pairs": len(training), "testing_pairs": len(testing),
            "referenced_images": len(images), "bundle_sha256": digest(output),
            "image_pixels_read": False, "approval_issued": False}


def load_bundle(path, source_root):
    bundle = json.loads(Path(path).read_text())
    if (not isinstance(bundle, dict) or bundle.get("schema_version") != 1
            or bundle.get("scope") != "shared_seen_image_fixed_pairs"
            or bundle.get("labels") != "as_supplied"):
        raise InputError("Unexpected private shared-benchmark schema.")
    training = [FixedPair(**row) for row in bundle["training"]]
    testing = [FixedPair(**row) for row in bundle["testing"]]
    validate_pairs(training)
    validate_pairs(testing)
    if ({p.pair_id for p in training} & {p.pair_id for p in testing}
            or {tuple(sorted((p.image_a, p.image_b))) for p in training}
            & {tuple(sorted((p.image_a, p.image_b))) for p in testing}):
        raise InputError("Shared-benchmark phases overlap in pair IDs or exact pairs.")
    needed = {key for p in training + testing for key in (p.image_a, p.image_b)}
    if set(bundle["images"]) != needed:
        raise InputError("Image mapping must cover the exact supplied endpoints.")
    root, paths = Path(source_root).resolve(strict=True), {}
    for key, row in bundle["images"].items():
        path = (root / row["path"]).resolve(strict=True)
        if (not path.is_relative_to(root) or not path.is_file()
                or any(not re.fullmatch(r"[0-9a-f]{64}", row[field])
                       for field in ("sha256", "decoded_sha256"))):
            raise InputError("Invalid reviewed image path/digest mapping.")
        paths[key] = path
    return bundle, training, testing, paths


def reviewed_shared_inputs(bundle_path, config_path, review_path, phase):
    """No model/image loading before actual, digest-bound human review."""
    if phase not in {"train", "calibrate", "test"}:
        raise InputError("Unknown shared-benchmark execution phase.")
    sources = {"bundle": Path(bundle_path), "config": Path(config_path), "review": Path(review_path)}
    hashes = {key: digest(path) for key, path in sources.items()}
    review = json.loads(sources["review"].read_text())
    if (not isinstance(review, dict) or any(review.get(flag) is not True for flag in REVIEW_FLAGS)
            or review.get("scope") != "shared_seen_image_fixed_pairs"
            or review.get("input_sha256") != {key: hashes[key] for key in ("bundle", "config")}
            or not isinstance(review.get("review_record"), str) or not review["review_record"].strip()
            or phase not in review.get("authorized_phases", [])):
        raise InputError("Actual protocol/results/cohort/label review must bind the exact inputs and phase.")
    config = FixedConfig(**json.loads(sources["config"].read_text()))
    config.validate()

    def integrity():
        if any(digest(path) != hashes[key] for key, path in sources.items()):
            raise InputError("Reviewed shared-benchmark inputs changed during execution.")

    integrity()
    return config, hashes, integrity


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
