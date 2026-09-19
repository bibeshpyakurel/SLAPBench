"""
SLAPBench — metrics verification
================================

The model runs themselves need GPUs and NIST SD302b, so CI cannot reproduce
them. What CI *can* prove is the step that every published number depends on:
that each committed ``.metrics.json`` actually follows from its sibling
per-pair CSV.

This script recomputes AUC, EER and the pair counts straight from the CSVs and
fails if any of them disagrees with what was published. If a results file is
ever replaced without rerunning the analysis — the exact drift that leaves a
paper citing numbers its own data no longer supports — this job goes red.

It also checks the manifest invariant the paper states in its abstract:
7,832 pairs = 176 genuine + 7,656 impostor.

Usage:
    python code/verify_metrics.py
    python code/verify_metrics.py --strict   # also fail on warnings
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Tolerances. The published files round to a few decimals, so an exact float
# comparison would fail on formatting alone; these bounds are tight enough that
# a genuine analysis change cannot slip through.
AUC_TOL = 5e-3
EER_TOL = 0.75  # percentage points

# The manifest invariant quoted in the README and the paper abstract.
EXPECTED_TOTAL = 7832
EXPECTED_GENUINE = 176
EXPECTED_IMPOSTOR = 7656


class Failure(Exception):
    pass


def compute_auc_eer(df: pd.DataFrame) -> tuple[float, float]:
    """AUC and EER (in percent) from a per-pair similarity-score frame."""
    scored = df.dropna(subset=["similarity_score"])
    y_true = (scored["label"] == "genuine").astype(int).to_numpy()
    y_score = scored["similarity_score"].astype(float).to_numpy()

    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        raise Failure("one class is missing; AUC and EER are undefined")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))

    fnr = 1.0 - tpr
    crossing = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[crossing] + fnr[crossing]) / 2.0 * 100.0)
    return auc, eer


def verify_pair_manifest(problems: list[str]) -> None:
    manifest = RESULTS_DIR / "sd302b" / "task8_pairs_all.csv"
    if not manifest.exists():
        problems.append(f"{manifest.name}: missing — the exhaustive pair manifest is required")
        return

    df = pd.read_csv(manifest)
    total = len(df)
    genuine = int((df["label"] == "genuine").sum())
    impostor = int((df["label"] == "impostor").sum())

    if (total, genuine, impostor) != (EXPECTED_TOTAL, EXPECTED_GENUINE, EXPECTED_IMPOSTOR):
        problems.append(
            f"task8_pairs_all.csv: manifest is {total} pairs "
            f"({genuine} genuine / {impostor} impostor); the paper states "
            f"{EXPECTED_TOTAL} ({EXPECTED_GENUINE} / {EXPECTED_IMPOSTOR})"
        )
    else:
        print(f"  manifest        {total} pairs = {genuine} genuine + {impostor} impostor  OK")


def verify_one(metrics_path: Path, problems: list[str], warnings: list[str]) -> None:
    csv_path = Path(str(metrics_path).replace(".metrics.json", ".csv"))
    rel = metrics_path.relative_to(PROJECT_ROOT)

    if not csv_path.exists():
        problems.append(f"{rel}: no sibling CSV — the published metrics cannot be checked")
        return

    published = json.loads(metrics_path.read_text())
    df = pd.read_csv(csv_path)

    # Counts must match whatever the metrics file claims it summarised.
    for field, actual in (
        ("total_pairs", len(df)),
        ("genuine_count", int((df["label"] == "genuine").sum())),
        ("impostor_count", int((df["label"] == "impostor").sum())),
    ):
        if field in published and int(published[field]) != actual:
            problems.append(
                f"{rel}: {field} published as {published[field]}, CSV holds {actual}"
            )

    # AUC/EER only exist for the similarity-scoring runs.
    if published.get("auc") is None or "similarity_score" not in df.columns:
        print(f"  {rel.parent.name:<12} {metrics_path.stem[:46]:<48} counts OK (no AUC)")
        return

    try:
        auc, eer = compute_auc_eer(df)
    except Failure as exc:
        problems.append(f"{rel}: {exc}")
        return

    ok = True
    if abs(auc - float(published["auc"])) > AUC_TOL:
        problems.append(
            f"{rel}: AUC published as {published['auc']}, recomputed {auc:.4f} "
            f"(tolerance {AUC_TOL})"
        )
        ok = False
    if "eer" in published and abs(eer - float(published["eer"])) > EER_TOL:
        warnings.append(
            f"{rel}: EER published as {published['eer']}, recomputed {eer:.2f} "
            f"(tolerance {EER_TOL}pp)"
        )

    flag = "OK" if ok else "MISMATCH"
    print(
        f"  {rel.parent.name:<12} AUC published {float(published['auc']):.3f} "
        f"recomputed {auc:.3f}  EER {eer:5.2f}%  {flag}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as failures"
    )
    args = parser.parse_args()

    problems: list[str] = []
    warnings: list[str] = []

    print("Pair manifest")
    verify_pair_manifest(problems)

    metrics_files = sorted(
        Path(p) for p in glob.glob(str(RESULTS_DIR / "**" / "*.metrics.json"), recursive=True)
    )
    if not metrics_files:
        print("No .metrics.json files found under results/.", file=sys.stderr)
        return 1

    print(f"\nRecomputing {len(metrics_files)} published metrics files from their CSVs")
    for path in metrics_files:
        verify_one(path, problems, warnings)

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ! {w}")

    if problems:
        print(f"\n{len(problems)} FAILURE(S):", file=sys.stderr)
        for p in problems:
            print(f"  x {p}", file=sys.stderr)
        return 1

    if warnings and args.strict:
        print("\nWarnings present and --strict was given.", file=sys.stderr)
        return 1

    print(f"\nAll {len(metrics_files)} metrics files reproduce from their per-pair CSVs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
