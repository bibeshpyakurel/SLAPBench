"""Independently reproduce sealed fixed-pair DINO metrics with NumPy/scikit-learn.

Reads private outcome tables, never images or models. Replays the declared
training cutoff; descriptive test ROC/EER does not select any decision.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]


def discrimination(labels, scores):
    y, values = np.asarray(labels) == "genuine", np.asarray(scores, dtype=float)
    if not np.isfinite(values).all() or not y.any() or y.all():
        raise ValueError("Both classes of finite saved scores are required.")
    far, tar, _ = roc_curve(y, values, drop_intermediate=False)
    difference = far + tar - 1
    right = int(np.flatnonzero(difference >= 0)[0])
    if difference[right] == 0:
        eer = far[right]
    else:
        left = right - 1
        eer = far[left] - difference[left] * (far[right] - far[left]) / (difference[right] - difference[left])
    return {"auc": float(roc_auc_score(y, values)), "eer_linear_interpolation": float(eer)}


def outcomes(path, pairs):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row["pair_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != {p["pair_id"] for p in pairs}:
        raise ValueError("Saved outcomes do not cover the exact fixed membership.")
    values = []
    for p in pairs:
        row = by_id[p["pair_id"]]
        if row["status"] == "ok":
            value = float(row["score"])
            if not np.isfinite(value):
                raise ValueError("Nonfinite saved outcome.")
        else:
            if row["score"]:
                raise ValueError("Failure has an imputed score.")
            value = np.nan
        values.append(value)
    return np.array(values)


def check_metrics(pairs, values, cutoff, saved):
    labels = np.array([p["label"] for p in pairs]); valid = np.isfinite(values)
    accepted = valid & (values > cutoff)
    rates = saved["all_attempts_failure_as_reject"]
    result = {}
    for label, count_key, accepts_key, rate in (("genuine", "genuine_comparisons", "genuine_accepts", "tar"),
                                               ("impostor", "impostor_comparisons", "false_accepts", "far")):
        total = int(np.sum(labels == label)); accepts = int(np.sum(accepted & (labels == label)))
        failures = int(np.sum(~valid & (labels == label)))
        if (rates[count_key] != total or rates[accepts_key] != accepts or saved["failures"][label] != failures
                or not np.isclose(rates[rate], accepts / total, atol=1e-12, rtol=0)):
            raise ValueError("Sealed confusion counts, coverage or rates do not reproduce.")
        result.update({count_key: total, accepts_key: accepts, rate: accepts / total, "failed_" + label: failures})
    curve = discrimination(labels[valid], values[valid])
    if any(not np.isclose(value, saved["descriptive_valid_score_discrimination"][key], atol=1e-12, rtol=0)
           for key, value in curve.items()):
        raise ValueError("Sealed discrimination metrics do not reproduce independently.")
    return {**result, **curve}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "run-root", "aggregate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        if output.exists() or not output.is_relative_to(ROOT / "local/precise"):
            raise ValueError("Use a new private verification receipt.")
        bundle = json.loads(args.bundle.read_text()); aggregate = json.loads(args.aggregate.read_text())
        sources = [args.bundle, args.aggregate]
        for short in ("frozen", "adapted"):
            sources.extend(args.run_root / (short + "-" + phase) / "scores.csv" for phase in ("calibration", "test"))
        hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        results = {}
        for arm in ("frozen_dinov2", "adapted_dinov2"):
            short = arm.split("_")[0]; saved = aggregate["dino"][arm]; cutoff = saved["threshold"]["cutoff"]
            train = outcomes(args.run_root / (short + "-calibration/scores.csv"), bundle["training"])
            impostor = train[np.array([p["label"] == "impostor" for p in bundle["training"]])]
            if not np.isfinite(impostor).all():
                raise ValueError("This verification requires complete training-impostor calibration scores.")
            from fractions import Fraction
            target = Fraction(aggregate["configuration"]["target_far"])
            budget = target.numerator * len(impostor) // target.denominator
            if float(np.sort(impostor)[::-1][budget]) != cutoff:
                raise ValueError("Declared cutoff does not replay from training impostors.")
            test = outcomes(args.run_root / (short + "-test/scores.csv"), bundle["testing"])
            groups = {"pooled": np.ones(len(test), dtype=bool), **{hand: np.array([p["hand"] == hand for p in bundle["testing"]])
                                                                 for hand in ("left", "right", "thumbs")}}
            results[arm] = {"training_impostors": len(impostor), "allowed_training_false_accepts": budget,
                            "training_false_accepts": int(np.sum(impostor > cutoff)), "cutoff": cutoff,
                            "testing": {}}
            for group, mask in groups.items():
                selected = [p for p, keep in zip(bundle["testing"], mask, strict=True) if keep]
                if not selected:
                    continue
                reference = saved["results"]["pooled"] if group == "pooled" else saved["results"]["by_slap_type"][group]
                results[arm]["testing"][group] = check_metrics(selected, test[mask], cutoff, reference)
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != value for p, value in hashes.items()):
            raise ValueError("Input changed during verification.")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump({"passed": True, "implementation": "NumPy confusion counts and scikit-learn ROC/AUC",
                       "new_inference": False, "new_selection": False, "results": results,
                       "source_sha256": hashes}, stream, indent=2)
            stream.write("\n")
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        print("Independent metric verification did not complete; inspect private inputs.", file=sys.stderr)
        return 1
    print("Independent saved-outcome counts, cutoff replay, AUC and EER verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
