"""Offline preflight/evaluation CLI. Outputs contain aggregate information only."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from .data import InputError, load_protocol, validate_protocol
from .comparison import compare_methods, validate_arms
from .evaluation import evaluate, load_scores, target_fraction


ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "evaluate", "compare"))
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--method", action="append", nargs=2, metavar=("ARM", "SCORES"),
                        help="Comparison only: repeat for each supported arm and its private score CSV")
    parser.add_argument("--target-far", help="Required rate for evaluation, e.g. 0.001; no default")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "evaluate" and (args.scores is None or args.target_far is None):
        parser.error("Evaluation requires --scores and --target-far.")
    if args.command == "validate" and (args.scores is not None or args.target_far is not None):
        parser.error("Validation does not accept scores or select a threshold.")
    if args.command != "compare" and args.method:
        parser.error("Only comparison accepts --method.")
    if args.command == "compare" and (args.scores is not None or not args.method or args.target_far is None):
        parser.error("Comparison requires --method arms and --target-far, without --scores.")
    try:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT / "reports") or output.suffix != ".json":
            raise InputError("Output must be a new JSON file under reports/.")
        if output.exists():
            raise InputError("Output exists; overwrite refused.")
        if args.target_far is not None:
            target_fraction(args.target_far)
        methods = args.method or []
        if args.command == "compare":
            validate_arms([name for name, _ in methods])
        sources = [args.images, args.pairs] + ([args.scores] if args.scores else [])
        sources += [Path(path) for _, path in methods]
        before = [hashlib.sha256(path.read_bytes()).digest() for path in sources]
        images, pairs = load_protocol(args.images, args.pairs)
        report = validate_protocol(images, pairs)
        passed = report["metadata_checks_passed"]
        if args.command == "evaluate" and passed:
            report = evaluate(images, pairs, load_scores(args.scores), args.target_far)
        elif args.command == "compare" and passed:
            report = compare_methods(images, pairs, {name: load_scores(Path(path)) for name, path in methods},
                                     args.target_far)
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
