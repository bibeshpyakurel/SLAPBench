"""Offline final-checkpoint integrity for the separately reviewed native-large run."""

import argparse
import json
from pathlib import Path
import sys

from precise_matching.data import InputError
from precise_matching.fixed_pairs import FixedConfig, GROUP_SCOPE
from precise_matching.fixed_training import checked_checkpoint
from precise_matching.native_large import TRAINABLE_PREFIXES, WEIGHTS_SHA256, load_large_encoder, validate_large_checkpoint
from precise_matching.training import digest, write_json
from verify_precise_shared_checkpoint import completed_schedule, parameter_integrity

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-run", "weights", "output"):parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run, output = args.training_run.resolve(), args.output.resolve()
        if (output.exists() or not output.is_relative_to(ROOT / "local/precise")
                or not run.is_relative_to(ROOT / "local/precise/runs")):
            raise InputError("New private native-checkpoint receipt required.")
        record = json.loads((run / "run_record.json").read_text());selection = json.loads((run / "selection.json").read_text())
        checkpoint = (run / selection["checkpoint"]).resolve()
        if (not checkpoint.is_relative_to(run) or selection.get("training_complete") is not True
                or selection.get("test_scores_accessed") is not False or selection.get("checkpoint_rule") != "fixed_final_epoch"
                or selection.get("epochs_completed") != record["config"]["epochs"]
                or record["provenance"].get("model_profile") != "dinov2_large_native_v1"
                or record["provenance"].get("benchmark_scope") != GROUP_SCOPE
                or record["provenance"].get("weights_sha256") != WEIGHTS_SHA256
                or digest(checkpoint) != selection["checkpoint_sha256"]):
            raise InputError("Native final-checkpoint selection/provenance is inconsistent.")
        saved = checked_checkpoint(checkpoint, run, record)
        if saved["history"] != selection["history"] or saved["epochs_completed"] != selection["epochs_completed"]:
            raise InputError("Native checkpoint does not replay its final selection.")
        updates = completed_schedule(saved, FixedConfig(**record["config"]))
        base = load_large_encoder(args.weights, adapted=False, device="cpu").state_dict()
        counts = parameter_integrity(base, saved["model"], trainable_prefixes=TRAINABLE_PREFIXES)
        validate_large_checkpoint(args.weights)
        if digest(checkpoint) != selection["checkpoint_sha256"]:raise InputError("Checkpoint changed during verification.")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, {"passed": True, "scope": "offline_native_large_checkpoint_integrity",
            "verifier_sha256": digest(Path(__file__)), "base_weights_sha256": WEIGHTS_SHA256,
            "checkpoint_sha256": selection["checkpoint_sha256"], "selection_sha256": digest(run / "selection.json"),
            "epochs_completed": selection["epochs_completed"], "history": saved["history"],
            "optimizer_updates_completed": updates, "counts": counts, "image_pixels_read": False,
            "model_inference_performed": False})
    except (InputError, OSError, KeyError, TypeError, ValueError, RuntimeError):
        print("Native checkpoint verification did not complete; inspect private artifacts.", file=sys.stderr);return 1
    print("Native-large final checkpoint verified offline.")
    return 0


if __name__ == "__main__":raise SystemExit(main())
