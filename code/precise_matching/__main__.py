"""Offline preflight/evaluation CLI. Outputs contain aggregate information only."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from .data import InputError, load_protocol, validate_protocol
from .evaluation import evaluate, load_scores, target_fraction


ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "evaluate"))
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--target-far", help="Required rate for evaluation, e.g. 0.001; no default")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "evaluate" and (args.scores is None or args.target_far is None):
        parser.error("Evaluation requires --scores and --target-far.")
    if args.command == "validate" and (args.scores is not None or args.target_far is not None):
        parser.error("Validation does not accept scores or select a threshold.")
    try:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT / "reports") or output.suffix != ".json":
            raise InputError("Output must be a new JSON file under reports/.")
        if output.exists():
            raise InputError("Output exists; overwrite refused.")
        if args.target_far is not None:
            target_fraction(args.target_far)
        sources = [args.images, args.pairs] + ([args.scores] if args.scores else [])
        before = [hashlib.sha256(path.read_bytes()).digest() for path in sources]
        images, pairs = load_protocol(args.images, args.pairs)
        report = validate_protocol(images, pairs)
        passed = report["metadata_checks_passed"]
        if args.command == "evaluate" and passed:
            report = evaluate(images, pairs, load_scores(args.scores), args.target_far)
        if before != [hashlib.sha256(path.read_bytes()).digest() for path in sources]:
            raise InputError("Input changed during validation; no report written.")
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
        print("Aggregate report written; metadata checks " + ("passed." if passed else "have blockers."))
        return 0 if passed else 1
    except InputError as exc:
        print(str(exc), file=sys.stderr)
    except (OSError, UnicodeError, ValueError, csv.Error):
        print("Input or output could not be processed; inspect the private files locally.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
