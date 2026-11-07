"""Verify the completed adapted checkpoint against the pinned base offline.

No images, inference, checkpoint selection or model update is performed.
"""

import argparse
import json
from pathlib import Path
import sys

from precise_matching.data import InputError
from precise_matching.encoders import TRAINABLE_PREFIXES, WEIGHTS_SHA256, load_encoder, validate_checkpoint
from precise_matching.fixed_training import checked_checkpoint
from precise_matching.fixed_pairs import FixedConfig
from precise_matching.training import digest, write_json

ROOT = Path(__file__).resolve().parents[1]


def parameter_integrity(base, adapted):
    import torch

    if (not base or set(base) != set(adapted)
            or any(not any(name.startswith(prefix) for name in base) for prefix in TRAINABLE_PREFIXES)):
        raise InputError("Base/adapted state dictionaries or expected trainable groups disagree.")
    counts = {"frozen_tensors": 0, "frozen_parameters": 0, "changed_frozen_tensors": 0,
              "trainable_tensors": 0, "trainable_parameters": 0, "changed_trainable_tensors": 0}
    for name, original in base.items():
        trained = adapted[name]
        if (original.shape != trained.shape or original.dtype != trained.dtype
                or not torch.isfinite(original).all() or not torch.isfinite(trained).all()):
            raise InputError("Nonfinite or incompatible checkpoint tensor.")
        group = "trainable" if name.startswith(TRAINABLE_PREFIXES) else "frozen"
        counts[group + "_tensors"] += 1
        counts[group + "_parameters"] += original.numel()
        counts["changed_" + group + "_tensors"] += not torch.equal(original, trained)
    if counts["changed_frozen_tensors"]:
        raise InputError("Adaptation changed frozen pretrained tensors.")
    return counts


def completed_schedule(saved, config):
    history = saved["history"]
    if (len(history) != config.epochs
            or [h["epoch"] for h in history] != list(range(1, config.epochs + 1))):
        raise InputError("Checkpoint history does not contain every declared epoch.")
    pair_count = history[0]["pairs_seen"]
    config.validate(pair_count)
    expected = config.steps(pair_count)
    if (saved["scheduler"]["last_epoch"] != expected
            or any(h["pairs_seen"] != pair_count for h in history)):
        raise InputError("Saved scheduler/history does not match the complete fixed-pair schedule.")
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-run", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output, run = args.output.resolve(), args.training_run.resolve()
        if (not output.is_relative_to(ROOT / "local/precise") or output.exists()
                or not run.is_relative_to(ROOT / "local/precise/runs")):
            raise InputError("Use new private verification outputs and the actual training run.")
        record = json.loads((run / "run_record.json").read_text())
        selection = json.loads((run / "selection.json").read_text())
        checkpoint = (run / selection["checkpoint"]).resolve()
        if (not checkpoint.is_relative_to(run) or selection.get("training_complete") is not True
                or selection.get("test_scores_accessed") is not False
                or selection.get("checkpoint_rule") != "fixed_final_epoch"
                or selection.get("epochs_completed") != record["config"]["epochs"]
                or digest(checkpoint) != selection["checkpoint_sha256"]):
            raise InputError("Training final selection is incomplete or inconsistent.")
        saved = checked_checkpoint(checkpoint, run, record)
        if (saved["history"] != selection["history"] or saved["epochs_completed"] != selection["epochs_completed"]):
            raise InputError("Checkpoint does not replay the sealed final epoch/history.")
        config = FixedConfig(**record["config"])
        expected_updates = completed_schedule(saved, config)
        validate_checkpoint(args.weights)
        # Transformers performs version-specific architectural key conversion
        # (for example legacy attention.query -> q_proj) when loading these
        # pinned bytes. Compare the identical loader's canonical state, rather
        # than incorrectly treating raw serialization names as model state.
        base = load_encoder(args.weights, adapted=False, device="cpu").state_dict()
        counts = parameter_integrity(base, saved["model"])
        validate_checkpoint(args.weights)
        if digest(checkpoint) != selection["checkpoint_sha256"]:
            raise InputError("Checkpoint changed during verification.")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, {"scope": "offline_final_checkpoint_parameter_integrity", "passed": True,
            "verifier_sha256": digest(Path(__file__)),
            "checkpoint_sha256": selection["checkpoint_sha256"], "base_weights_sha256": WEIGHTS_SHA256,
            "selection_sha256": digest(run / "selection.json"), "counts": counts,
            "base_state_loading": "Same pinned local encoder loader as training; canonical runtime state keys.",
            "epochs_completed": selection["epochs_completed"], "history": saved["history"],
            "optimizer_updates_completed": expected_updates,
            "image_pixels_read": False, "model_inference_performed": False})
    except (InputError, OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError):
        print("Offline checkpoint verification did not complete; inspect private artifacts.", file=sys.stderr)
        return 1
    print("Final checkpoint parameter integrity verified; no image access or inference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
