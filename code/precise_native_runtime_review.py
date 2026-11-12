"""Exact, explicitly reviewed post-training kernel transition for native scoring.

Never changes training records/checkpoints or creates human approval assertions.
"""

import copy
import json
from pathlib import Path

from precise_matching.data import InputError
from precise_matching.fixed_pairs import canonical_digest
from precise_matching.training import digest

ROOT = Path(__file__).resolve().parents[1]
BRIDGE_PATHS = ("code/precise_native_runtime_review.py", "code/score_precise_native_restored.py")
SCOPE = "native_scoring_kernel_42_to_43_same_gpu_driver"


def transition_delta(before, after):
    if set(before) != set(after) or any(before[k] != after[k] for k in before if k != "hardware"):
        raise InputError("Non-hardware training/scoring provenance differs.")
    old, new = before["hardware"], after["hardware"]
    expected = {"platform": ("Linux-7.0.0-42-generic-x86_64-with-glibc2.43",
                              "Linux-7.0.0-43-generic-x86_64-with-glibc2.43"),
                "driver_metadata_source": ("proc_driver_nvidia", "nvidia-smi")}
    if set(old) != set(new) or any(old[k] != new[k] for k in old if k not in expected):
        raise InputError("GPU, driver version, CUDA or another hardware property differs.")
    if (old.get("cuda_device") != "NVIDIA GeForce RTX 4080 SUPER"
            or old.get("driver_version") != "595.99.02"
            or old.get("device") != "cuda"
            or any((old[k], new[k]) != v for k, v in expected.items())):
        raise InputError("Only the exact proposed kernel/discovery transition is eligible.")
    return {k: {"training": old[k], "scoring": new[k]} for k in expected}


def inspect_runtime_review(path, run, current, input_hashes, source_hashes):
    path, run = Path(path).resolve(), Path(run).resolve()
    if (not path.is_relative_to(ROOT / "local/precise")
            or not run.is_relative_to(ROOT / "local/precise/runs")):
        raise InputError("Runtime review and completed training must remain private.")
    record = json.loads((run / "run_record.json").read_text())
    selected = json.loads((run / "selection.json").read_text())
    delta = transition_delta(record["provenance"], current)
    review = json.loads(path.read_text())
    if (review.get("scope") != SCOPE or review.get("execution_authorized") is not True
            or review.get("authorized_phases") != ["calibrate", "test"]
            or review.get("frozen_calibration_retry_authorized") is not True
            or review.get("training_provenance_sha256") != canonical_digest(record["provenance"])
            or review.get("scoring_provenance_sha256") != canonical_digest(current)
            or review.get("source_sha256") != source_hashes):
        raise InputError("Actual review of the precise scoring transition is required.")
    checkpoint = (run / selected["checkpoint"]).resolve()
    files = {"training_record": run / "run_record.json", "selection": run / "selection.json",
             "checkpoint": checkpoint, "verification": Path(review["verification"]).resolve(),
             "proposal": Path(review["proposal"]).resolve(), "human_record": Path(review["human_record"]).resolve()}
    if (not checkpoint.is_relative_to(run)
            or any(not files[k].is_relative_to(ROOT / "local/precise") for k in ("verification", "human_record"))
            or not files["proposal"].is_relative_to(ROOT / "docs/notes")):
        raise InputError("Runtime review evidence has unexpected paths.")
    hashes = {k: digest(p) for k, p in files.items()}
    if review.get("input_sha256") != {**input_hashes, **hashes}:
        raise InputError("Runtime review does not bind the exact preserved inputs/checkpoint.")
    human = json.loads(files["human_record"].read_text())
    verified = json.loads(files["verification"].read_text())
    if (not isinstance(human.get("response"), str) or not human["response"].strip()
            or human.get("proposal_sha256") != hashes["proposal"]
            or verified.get("passed") is not True
            or verified.get("checkpoint_sha256") != hashes["checkpoint"]
            or verified.get("selection_sha256") != hashes["selection"]
            or verified.get("counts", {}).get("changed_frozen_tensors") != 0
            or selected.get("training_complete") is not True or selected.get("epochs_completed") != 3
            or selected.get("checkpoint_rule") != "fixed_final_epoch" or selected.get("test_scores_accessed") is not False):
        raise InputError("Completed checkpoint or actual human review evidence is inconsistent.")
    summary = {"scope": SCOPE, "review_sha256": digest(path), "delta": delta,
               "training_record_sha256": hashes["training_record"],
               "training_execution_source_commit": record["provenance"]["code_commit"],
               "original_checkpoint_sha256": hashes["checkpoint"],
               "interpretation": "Explicit runtime transition, not identical runtime or bitwise-equivalence evidence."}
    return summary, files, hashes


def inspect_scoring_transition(path, run, provenance):
    """Reproduce the actual recorded bridge, never substitute training hardware."""
    training = json.loads((Path(run) / "run_record.json").read_text())["provenance"]
    expected_names = set(training["source_sha256"]) | set(BRIDGE_PATHS)
    if set(provenance["source_sha256"]) != expected_names:
        raise InputError("Unexpected source set in restored-runtime scoring.")
    current = copy.deepcopy(provenance)
    claimed = current.pop("scoring_runtime_transition")
    current["code_commit"] = training["code_commit"]
    current["source_sha256"] = {k: provenance["source_sha256"][k] for k in training["source_sha256"]}
    current["model_identity"] = training["model_identity"]
    result = inspect_runtime_review(path, run, current, provenance["input_sha256"], provenance["source_sha256"])
    if result[0] != claimed:
        raise InputError("Scoring runtime transition does not reproduce from reviewed evidence.")
    return result
