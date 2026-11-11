"""Bind one explicitly reviewed native resume to preserved interruption evidence.

This reads actual review records; it neither creates approval assertions nor
launches training. The fresh-run report gate remains the default.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

from precise_matching.data import InputError
from precise_matching.training import digest

ROOT = Path(__file__).resolve().parents[1]


def interruption_observation_path():
    return ROOT / "reports/precise-native-large-interruption-20261110/integrity.json"


def inspect_resume_review(path, run, record):
    path, run = Path(path).resolve(), Path(run).resolve()
    if not path.is_relative_to(ROOT / "local/precise"):
        raise InputError("Resume review must remain private.")
    review = json.loads(path.read_text())
    if (review.get("scope") != "one_explicit_native_epoch_boundary_resume"
            or review.get("execution_authorized") is not True or review.get("max_new_attempts") != 1
            or review.get("resume_from_epoch") != 2
            or review.get("execution_source_commit") != record["provenance"]["code_commit"]
            or record["provenance"].get("model_profile") != "dinov2_large_native_v1"):
        raise InputError("One explicit unchanged-profile epoch-2 resume review is required.")
    checkpoint = (run / review["resume_checkpoint"]).resolve()
    paths = {"training_record": run / "run_record.json", "resume_checkpoint": checkpoint,
             "interruption_verification": Path(review["interruption_verification"]).resolve(),
             "proposal": Path(review["proposal"]).resolve(),
             "review_record": Path(review["review_record"]).resolve()}
    if (checkpoint != run / "attempt-0001/epoch-0002.pt"
            or any(not paths[k].is_relative_to(ROOT / "local/precise")
                   for k in ("interruption_verification", "review_record"))
            or not paths["proposal"].is_relative_to(ROOT / "docs/notes")):
        raise InputError("Resume evidence paths differ from the concrete reviewed boundary.")
    hashes = {k: digest(p) for k, p in paths.items()}
    if review.get("input_sha256") != {**record["provenance"]["input_sha256"], **hashes}:
        raise InputError("Resume review does not bind all unchanged inputs and recovery evidence.")
    human = json.loads(paths["review_record"].read_text())
    if (not isinstance(human.get("response"), str) or not human["response"].strip()
            or human.get("proposal_sha256") != hashes["proposal"]):
        raise InputError("The recorded actual human response does not bind this proposal.")
    inspected = json.loads(paths["interruption_verification"].read_text())
    if (inspected.get("passed") is not True or inspected.get("epochs_completed") != 2
            or inspected.get("epochs_remaining") != 1 or inspected.get("training_complete") is not False
            or inspected.get("counts", {}).get("changed_frozen_tensors") != 0
            or inspected.get("execution_source_commit") != review["execution_source_commit"]
            or inspected["input_sha256"]["record"] != hashes["training_record"]
            or inspected["input_sha256"]["checkpoint"] != hashes["resume_checkpoint"]
            or inspected["input_sha256"]["console"] != digest(run.parent / "training-console.log")
            or checkpoint.with_suffix(".sha256").read_text().strip() != hashes["resume_checkpoint"]):
        raise InputError("Preserved interruption evidence does not match the approved epoch boundary.")
    return review, inspected, paths, hashes


def check_resume_chain(path, run, record, selection):
    review, inspected, paths, hashes = inspect_resume_review(path, run, record)
    run = Path(run).resolve()
    attempts = sorted(p.name for p in run.iterdir() if p.is_dir() and p.name.startswith("attempt-"))
    second = json.loads((run / "attempt-0002/attempt.json").read_text())
    result = json.loads((run / "attempt-0002/result.json").read_text())
    first = json.loads((run / "attempt-0001/attempt.json").read_text())
    if (attempts != ["attempt-0001", "attempt-0002"] or first.get("start_epoch") != 0
            or first.get("resume_sha256") is not None or second.get("start_epoch") != review["resume_from_epoch"]
            or second.get("resume_sha256") != hashes["resume_checkpoint"]
            or selection.get("history", [])[:2] != inspected["history"]
            or selection.get("checkpoint") != "attempt-0002/epoch-0003.pt"
            or result != selection):
        raise InputError("Actual attempts do not follow the single approved resume chain.")
    first_start = datetime.fromisoformat(first["started_at_utc"])
    resumed_start = datetime.fromisoformat(second["started_at_utc"])
    finished = datetime.fromisoformat(selection["finished_at_utc"])
    observation = interruption_observation_path()
    observed = json.loads(observation.read_text())
    if (observed.get("scope") != inspected["scope"]
            or observed.get("input_sha256") != inspected["input_sha256"]):
        raise InputError("Published interruption timestamp does not bind the inspected original sources.")
    last_log = datetime.fromisoformat(observed["console_last_write_utc"])
    elapsed = (finished - resumed_start).total_seconds()
    if not first_start <= last_log <= resumed_start <= finished:
        raise InputError("Resume attempt timing is inconsistent.")
    logged_first_elapsed = (last_log - first_start).total_seconds()
    retained = sum(h["pairs_seen"] for h in selection["history"])
    discarded = inspected["last_reported_progress"]["pairs_seen"]
    return {"single_explicit_resume_reviewed": True, "review_sha256": digest(Path(path)),
            "interruption_verification_sha256": hashes["interruption_verification"],
            "interruption_observation_sha256": digest(observation),
            "resume_checkpoint_sha256": hashes["resume_checkpoint"], "epochs_restored": 2,
            "attempts": 2, "resume_attempt_elapsed_seconds": elapsed,
            "first_attempt_last_log_write_utc": last_log.isoformat(),
            "first_attempt_logged_elapsed_seconds": logged_first_elapsed,
            "minimum_observed_training_attempt_elapsed_seconds": logged_first_elapsed + elapsed,
            "timing_interpretation": "Pipeline wall intervals, not GPU kernel time; original crash tail is not timed exactly.",
            "retained_trajectory_pair_presentations": retained,
            "minimum_observed_total_pair_presentations": retained + discarded,
            "discarded_partial_epoch_last_reported_progress": inspected["last_reported_progress"],
            "training_wall_interval_includes_interruption_downtime": True,
            "crash_root_cause": inspected["crash_root_cause"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run = args.training_run.resolve()
        if not run.is_relative_to(ROOT / "local/precise/runs"):
            raise InputError("Resume checks use the actual private run.")
        record = json.loads((run / "run_record.json").read_text())
        inspect_resume_review(args.review, run, record)
        attempts = sorted(p.name for p in run.iterdir() if p.is_dir() and p.name.startswith("attempt-"))
        if attempts != ["attempt-0001"] or (run / "selection.json").exists():
            raise InputError("The single reviewed resume has already been used or the run is sealed.")
    except (InputError, OSError, ValueError, KeyError, TypeError):
        print("Resume review preflight did not pass; do not launch a retry.", file=sys.stderr)
        return 1
    print("Explicit resume review/evidence binding checked; no job launched.")
    return 0


if __name__ == "__main__":raise SystemExit(main())
