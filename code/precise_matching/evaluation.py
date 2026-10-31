"""Select a development-only cutoff and apply it unchanged to test scores."""

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from pathlib import Path

from .data import Image, InputError, Pair, read_table, validate_protocol


SCORE_FIELDS = ("pair_id", "status", "score")
STATUSES = {"ok", "decode_error", "encoder_error", "missing_input", "error"}


@dataclass(frozen=True)
class Score:
    pair_id: str
    status: str
    score: float | None


def write_scores(path: Path, scores: list[Score]):
    """Exclusive private CSV export using the same outcome schema as loading."""
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in scores)


def load_scores(path: Path) -> list[Score]:
    scores, seen = [], set()
    for row in read_table(path, SCORE_FIELDS):
        if not row["pair_id"] or row["pair_id"] in seen or row["status"] not in STATUSES:
            raise InputError("Repeated score identifier or unknown score status.")
        seen.add(row["pair_id"])
        value = None
        if row["status"] == "ok":
            try:
                value = float(row["score"])
            except ValueError:
                raise InputError("Successful comparisons require finite numeric scores.") from None
            if not math.isfinite(value):
                raise InputError("Successful comparisons require finite numeric scores.")
        elif row["score"]:
            raise InputError("Failed comparisons require an empty score; do not impute.")
        scores.append(Score(row["pair_id"], row["status"], value))
    return scores


def target_fraction(target_far: str) -> Fraction:
    try:
        if not isinstance(target_far, str):
            raise ValueError
        far = Fraction(target_far)
        if not 0 <= far < 1:
            raise ValueError
        return far
    except (ValueError, ZeroDivisionError):
        raise InputError("Target FAR must be a rate in [0, 1), supplied as text.") from None


def select_cutoff(impostors: list[float], target_far: str) -> dict:
    """Reject the boundary tie; strict > works for any finite score scale."""
    far = target_fraction(target_far)
    if not impostors or not all(math.isfinite(value) for value in impostors):
        raise InputError("Finite development impostor scores are required.")
    budget = far.numerator * len(impostors) // far.denominator
    cutoff = sorted(impostors, reverse=True)[budget]
    return {
        "cutoff": cutoff, "comparison": ">", "target_far": float(far),
        "development_valid_impostors": len(impostors),
        "allowed_development_false_accepts": budget,
        "development_false_accepts": sum(value > cutoff for value in impostors),
        "selected_on": "development_valid_impostors_only",
    }


def discrimination(genuine: list[float], impostor: list[float]) -> dict | None:
    """Discrete ROC/DET rates and descriptive AUC/interpolated EER, no tuning."""
    if not genuine or not impostor:
        return None
    g, i = Counter(genuine), Counter(impostor)
    points = [{"far": 0.0, "tar": 0.0, "frr": 1.0}]
    ga, ia = 0, 0
    for score in sorted(g.keys() | i.keys(), reverse=True):
        ga += g[score]
        ia += i[score]
        far, tar = ia / len(impostor), ga / len(genuine)
        points.append({"far": far, "tar": tar, "frr": 1 - tar})
    auc, eer = 0.0, None
    for a, b in zip(points[:-1], points[1:], strict=True):
        auc += (b["far"] - a["far"]) * (a["tar"] + b["tar"]) / 2
        da, db = a["far"] - a["frr"], b["far"] - b["frr"]
        if eer is None and da <= 0 <= db:
            alpha = -da / (db - da) if db != da else 0
            eer = a["far"] + alpha * (b["far"] - a["far"])
    return {"auc": auc, "eer_linear_interpolation": eer, "roc_det_rates": points}


def summarize(rows: list[tuple[str, Score]], cutoff: float) -> dict:
    valid = {label: [] for label in ("genuine", "impostor")}
    attempts, failures, accepts = Counter(), Counter(), Counter()
    statuses = Counter()
    for label, row in rows:
        attempts[label] += 1
        statuses[row.status] += 1
        if row.score is None:
            failures[label] += 1
        else:
            valid[label].append(row.score)
            accepts[label] += row.score > cutoff

    def rates(denominators):
        return {
            "genuine_accepts": accepts["genuine"], "false_accepts": accepts["impostor"],
            "genuine_comparisons": denominators["genuine"],
            "impostor_comparisons": denominators["impostor"],
            "tar": accepts["genuine"] / denominators["genuine"] if denominators["genuine"] else None,
            "far": accepts["impostor"] / denominators["impostor"] if denominators["impostor"] else None,
        }

    return {
        "attempts": dict(attempts), "status_counts": dict(sorted(statuses.items())),
        "failures": {label: failures[label] for label in valid},
        "coverage": {label: len(values) / attempts[label] if attempts[label] else None
                     for label, values in valid.items()},
        "valid_scores_only": rates({label: len(values) for label, values in valid.items()}),
        "all_attempts_failure_as_reject": rates(attempts),
        "descriptive_valid_score_discrimination": discrimination(valid["genuine"], valid["impostor"]),
    }


def evaluate(images: list[Image], pairs: list[Pair], scores: list[Score], target_far: str) -> dict:
    preflight = validate_protocol(images, pairs)
    if not preflight["metadata_checks_passed"]:
        raise InputError("Protocol metadata has unresolved blockers.")
    by_image = {row.image_id: row for row in images}
    expected = {pair.pair_id: pair for pair in pairs
                if by_image[pair.image_a].partition in {"development", "test"}}
    by_score = {score.pair_id: score for score in scores}
    if len(by_score) != len(scores) or set(by_score) != set(expected):
        raise InputError("Scores must cover every development/test pair exactly once, with no extra rows.")
    grouped = {partition: [] for partition in ("development", "test")}
    for pair_id, pair in expected.items():
        row = by_score[pair_id]
        if row.status not in STATUSES or (row.status == "ok") != (row.score is not None):
            raise InputError("Invalid score status/value combination.")
        if row.score is not None and not math.isfinite(row.score):
            raise InputError("Successful comparisons require finite numeric scores.")
        grouped[by_image[pair.image_a].partition].append((pair.label, row))
    dev = grouped["development"]
    for label in ("genuine", "impostor"):
        if not any(kind == label and row.score is not None for kind, row in dev):
            raise InputError("Development needs valid scores from both classes.")
    threshold = select_cutoff([row.score for label, row in dev
                               if label == "impostor" and row.score is not None], target_far)
    return {
        "schema_version": 1, "scope": "Offline evaluation of supplied scores; no matcher inference",
        "protocol": preflight, "threshold": threshold,
        "development": summarize(dev, threshold["cutoff"]),
        "test": summarize(grouped["test"], threshold["cutoff"]),
        "limits": [
            "Input scores must already come from a frozen configuration; checkpoint provenance requires separate review.",
            "Test scores never select the cutoff; test ROC/DET and EER are descriptive only.",
            "Failures are rejects in the all-attempt view, not evidence of biometric impostor rejection.",
            "No population low-FAR bound or independent-trial confidence interval is claimed.",
        ],
    }


def evaluate_development(images, pairs, scores, target_far):
    """Checkpoint selection without accepting, loading or requiring test scores."""
    if not validate_protocol(images, pairs)["metadata_checks_passed"]:
        raise InputError("Protocol metadata has unresolved blockers.")
    by_image = {row.image_id: row for row in images}
    expected = {p.pair_id: p for p in pairs if by_image[p.image_a].partition == "development"}
    supplied = {s.pair_id: s for s in scores}
    if len(supplied) != len(scores) or set(supplied) != set(expected):
        raise InputError("Checkpoint selection accepts development comparisons only.")
    if any(s.status != "ok" or s.score is None or not math.isfinite(s.score) for s in scores):
        raise InputError("Development failures must be resolved before checkpoint selection.")
    rows = [(expected[s.pair_id].label, s) for s in scores]
    if any(not any(label == kind for label, _ in rows) for kind in ("genuine", "impostor")):
        raise InputError("Development requires both comparison classes.")
    cutoff = select_cutoff([s.score for label, s in rows if label == "impostor"], target_far)
    return {"selection_source": "development_only", "threshold": cutoff,
            "development": summarize(rows, cutoff["cutoff"])}
