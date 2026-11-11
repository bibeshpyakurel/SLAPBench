"""Inspect an interrupted native run's saved epoch boundary, without inference.

This does not authorize a retry or resume, create a final selection, inspect
image pixels, change any checkpoint, or choose a checkpoint using test scores.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys

from precise_matching.data import InputError
from precise_matching.fixed_pairs import FixedConfig, canonical_digest
from precise_matching.fixed_training import checked_checkpoint
from precise_matching.native_large import TRAINABLE_PREFIXES, WEIGHTS_SHA256, load_large_encoder
from precise_matching.training import digest, write_json
from verify_precise_shared_checkpoint import parameter_integrity

ROOT = Path(__file__).resolve().parents[1]


def interrupted_schedule(saved, config, pair_count):
    """Validate a resumable epoch boundary without declaring training complete."""
    import numpy as np
    import torch

    count, history = saved["epochs_completed"], saved["history"]
    config.validate(pair_count)
    expected = config.steps(pair_count) // config.epochs * count
    if (not 0 < count < config.epochs or len(history) != count
            or [h["epoch"] for h in history] != list(range(1, count + 1))
            or any(h["pairs_seen"] != pair_count for h in history)
            or saved["scheduler"]["last_epoch"] != expected):
        raise InputError("Checkpoint is not a complete unfinished epoch boundary.")
    states = list(saved["optimizer"]["state"].values())
    if not states:
        raise InputError("Checkpoint has no optimizer state.")
    for state in states:
        if (int(state["step"]) != expected
                or not all(torch.isfinite(state[k]).all().item() for k in ("exp_avg", "exp_avg_sq"))):
            raise InputError("Optimizer state is nonfinite or does not match completed updates.")
    random.Random().setstate(saved["python_rng"])
    np.random.RandomState().set_state(saved["numpy_rng"])
    torch.Generator(device="cpu").set_state(saved["torch_rng"])
    if config.device == "cuda" and (not saved["cuda_rng"] or any(
            state.dtype != torch.uint8 or state.ndim != 1 or state.numel() == 0 for state in saved["cuda_rng"])):
        raise InputError("Saved CUDA RNG metadata is missing or malformed.")
    return {"epochs_completed": count, "epochs_remaining": config.epochs - count,
            "optimizer_updates_saved": expected, "optimizer_state_entries": len(states),
            "scheduler_and_optimizer_steps_consistent": True, "saved_rng_metadata_valid": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-root", "checkpoint", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root, output = args.run_root.resolve(), args.output.resolve()
        run, checkpoint = root / "adapted-training", args.checkpoint.resolve()
        if (not root.is_relative_to(ROOT / "local/precise/runs")
                or not output.is_relative_to(ROOT / "local/precise") or output.exists()
                or not checkpoint.is_relative_to(run) or (run / "selection.json").exists()):
            raise InputError("Use an unfinished private run, its checkpoint and a new private receipt.")
        record_path = run / "run_record.json"
        record = json.loads(record_path.read_text());provenance = record["provenance"]
        if (provenance.get("model_profile") != "dinov2_large_native_v1"
                or provenance.get("weights_sha256") != WEIGHTS_SHA256):
            raise InputError("Unexpected interrupted execution profile.")
        files = {"record": record_path, "checkpoint": checkpoint,
                 "console": root / "training-console.log"}
        before = {k: digest(p) for k, p in files.items()}
        for name, expected in provenance["source_sha256"].items():
            committed = subprocess.check_output(["git", "show", f"{provenance['code_commit']}:{name}"], cwd=ROOT)
            import hashlib
            if digest(ROOT / name) != expected or hashlib.sha256(committed).hexdigest() != expected:
                raise InputError("Protected execution source differs from committed run provenance.")
        saved = checked_checkpoint(checkpoint, run, record)
        cfg = FixedConfig(**record["config"])
        if asdict(cfg) != record["config"]:
            raise InputError("Interrupted configuration differs from the fixed schedule.")
        last = None
        for line in files["console"].read_text().splitlines():
            if line.startswith('{"progress":'):
                last = json.loads(line)["progress"]
        if last is None:
            raise InputError("No saved progress event is available.")
        pair_count = last["pairs_total"]
        state = interrupted_schedule(saved, cfg, pair_count)
        base = load_large_encoder(args.weights, adapted=False, device="cpu").state_dict()
        counts = parameter_integrity(base, saved["model"], trainable_prefixes=TRAINABLE_PREFIXES)
        if canonical_digest(saved["record"]) != canonical_digest(record):
            raise InputError("Checkpoint record changed during verification.")
        if any(digest(p) != before[k] for k, p in files.items()):
            raise InputError("A preserved input changed during interruption verification.")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, {"scope": "offline_interrupted_native_epoch_boundary_integrity_only",
            "passed": True, **state, "history": saved["history"], "counts": counts,
            "last_reported_progress": last, "input_sha256": before,
            "base_weights_sha256": WEIGHTS_SHA256, "execution_source_commit": provenance["code_commit"],
            "verifier_sha256": digest(Path(__file__)), "image_pixels_read": False,
            "model_inference_performed": False, "retry_authorized": False,
            "final_checkpoint_selected": False, "training_complete": False,
            "crash_root_cause": "not_established_by_checkpoint_integrity"})
    except (InputError, OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError):
        print("Interrupted-run inspection did not complete; preserve the original run.", file=sys.stderr)
        return 1
    print("Saved native epoch boundary verified offline; no retry or final selection authorized.")
    return 0


if __name__ == "__main__":raise SystemExit(main())
