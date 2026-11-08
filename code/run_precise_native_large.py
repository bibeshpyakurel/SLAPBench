"""Execute the separately reviewed native-resolution DINOv2-Large benchmark.

All identifying inputs, checkpoints, image caches and raw outcomes stay local.
No approval receipt is generated here; actual human review is required.
"""

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from precise_matching.data import InputError
from precise_matching.native_large import (CONFIG_SHA256, MODEL_ID, REVISION, WEIGHTS_SHA256,
    ORIGINAL_SIZE, PADDED_SIZE, load_large_encoder as load_encoder,
    native_tensor as image_tensor, prepare_native_image as prepare_image)
from precise_matching.fixed_evaluation import calibrate_fixed, test_fixed
from precise_matching.fixed_pairs import GROUP_SCOPE, load_bundle, reviewed_shared_inputs
from precise_matching.hardware import cuda_driver_metadata
from precise_matching.fixed_training import checked_checkpoint, train_fixed
from precise_matching.training import digest, write_json
from train_precise_matcher import git_output, package_versions

ROOT = Path(__file__).resolve().parents[1]


def private_path(path, subtree="local/precise"):
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT / subtree):
        raise InputError("Shared-benchmark artifacts must remain in the designated private tree.")
    return resolved


def source_record():
    code = sorted((ROOT / "code/precise_matching").glob("*.py"))
    code += [ROOT / "code/run_precise_native_large.py", ROOT / "code/run_precise_shared_pairs.py", ROOT / "code/train_precise_matcher.py"]
    names = [str(path.relative_to(ROOT)) for path in code]
    if git_output("status", "--porcelain", "--untracked-files=all", "--", *names).strip():
        raise InputError("Review and commit matcher source before real execution.")
    return {name: digest(ROOT / name) for name in names}


def progress_printer():
    last = [0.0]

    def event(row):
        now = time.monotonic()
        if now - last[0] >= 30 or row.get("pairs_seen") == row.get("pairs_total") and "pairs_seen" in row:
            print(json.dumps({"progress": row}), flush=True)
            last[0] = now
    return event


def validate_native_policy(bundle, config, review):
    from dataclasses import asdict
    expected = {"accumulation": 32, "batch_pairs": 1, "checkpoint_rule": "fixed_final_epoch", "device": "cuda",
        "epochs": 3, "learning_rate": 1e-5, "margin": .2, "minimum_learning_rate": 1e-6, "precision": "bf16",
        "seed": 2026, "target_far": "0.01", "threshold_rule": "training_impostors_strict_greater",
        "warmup_steps": 100, "weight_decay": .01}
    if (bundle.get("scope") != GROUP_SCOPE or asdict(config) != expected
            or review.get("model_profile") != "dinov2_large_native_v1"
            or review.get("native_pixels_reviewed") is not True
            or review.get("weights_sha256") != WEIGHTS_SHA256):
        raise InputError("Native large execution requires the exact separate model/pixel/schedule review.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    for phase in ("train", "calibrate", "test"):
        command = commands.add_parser(phase)
        for name in ("bundle", "config", "review", "source-root", "weights", "output"):
            command.add_argument("--" + name, type=Path, required=True)
        if phase == "train":
            command.add_argument("--resume", type=Path)
        else:
            command.add_argument("--arm", choices=("frozen_dinov2", "adapted_dinov2"), required=True)
            command.add_argument("--training-run", type=Path)
        if phase == "test":
            command.add_argument("--calibration", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = private_path(args.output, "local/precise/runs")
        private_path(args.bundle)
        private_path(args.review)
        config, hashes, input_integrity = reviewed_shared_inputs(args.bundle, args.config, args.review, args.phase)
        bundle, training, testing, paths = load_bundle(args.bundle, args.source_root)
        config.validate(len(training))
        review = json.loads(args.review.read_text())
        validate_native_policy(bundle, config, review)
        source_hashes = source_record()

        def integrity():
            input_integrity()
            if any(digest(ROOT / name) != value for name, value in source_hashes.items()):
                raise InputError("Reviewed source changed during execution.")

        integrity()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        compatibility_path = (Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
                              / "site-packages/preserved_packages.pth")
        # Identify the committed runtime source, so an unrelated documentation
        # commit cannot invalidate a long-running experiment's next phase.
        runtime_commit = git_output("log", "-1", "--format=%H", "--", *source_hashes, text=True).strip()
        provenance = {"code_commit": runtime_commit,
            "source_sha256": source_hashes, "input_sha256": hashes,
            "source_root": str(args.source_root.resolve()), "weights_path": str(args.weights.resolve()),
            "model": MODEL_ID, "model_revision": REVISION, "weights_sha256": WEIGHTS_SHA256,
            "model_config_sha256": CONFIG_SHA256,
            "environment_lock_sha256": digest(ROOT / "code/precise_matching/environment/requirements-linux-x86_64-py312-cu124.lock"),
            "python": platform.python_version(), "packages": package_versions(), "benchmark_scope": bundle["scope"],
            "interpreter_path": str(Path(sys.executable).resolve()),
            "interpreter_sha256": digest(Path(sys.executable).resolve()),
            "compatibility_binding_sha256": digest(compatibility_path) if compatibility_path.is_file() else None,
            "model_profile": "dinov2_large_native_v1",
            "image_preprocessing": {"original_width_height": list(ORIGINAL_SIZE), "padded_width_height": list(PADDED_SIZE),
                                    "resampling": False, "padding": "centered_white"},
            "attention": "sdpa_forced_flash_only", "inference_precision": "bf16",
            "gradient_checkpointing": "nonreentrant"}
        adapted = args.phase == "train" or args.arm == "adapted_dinov2"
        model = load_encoder(args.weights, adapted=adapted, device=config.device)
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        torch.use_deterministic_algorithms(True)
        provenance["hardware"] = {"device": config.device, "platform": platform.platform(),
            "torch_runtime": str(torch.__version__), "torch_compiled_cuda": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name() if config.device == "cuda" else None,
            **(cuda_driver_metadata() if config.device == "cuda" else {"driver_version": None, "driver_metadata_source": None})}
        provenance["model_identity"] = {"arm": "adapted_dinov2" if adapted else "frozen_dinov2",
                                        "base_weights_sha256": WEIGHTS_SHA256, "revision": REVISION}
        if args.phase != "train" and adapted:
            if args.training_run is None:
                raise InputError("Adapted scoring requires the sealed training run.")
            run = private_path(args.training_run, "local/precise/runs")
            selected = json.loads((run / "selection.json").read_text())
            record = json.loads((run / "run_record.json").read_text())
            if (selected.get("training_complete") is not True or selected.get("epochs_completed") != config.epochs
                    or selected.get("checkpoint_rule") != "fixed_final_epoch"
                    or selected.get("test_scores_accessed") is not False
                    or record.get("config") != json.loads(args.config.read_text())
                    or record.get("provenance") != provenance):
                raise InputError("Adapted run selection/provenance mismatch.")
            checkpoint = (run / selected["checkpoint"]).resolve()
            saved = checked_checkpoint(checkpoint, run, record)
            if (digest(checkpoint) != selected["checkpoint_sha256"]
                    or saved["epochs_completed"] != config.epochs
                    or saved["history"] != selected["history"]):
                raise InputError("Selected final checkpoint does not replay.")
            model.load_state_dict(saved["model"], strict=True)
            provenance["model_identity"]["checkpoint_sha256"] = digest(checkpoint)
            provenance["model_identity"]["selection_sha256"] = digest(run / "selection.json")
        elif args.phase != "train" and args.training_run is not None:
            raise InputError("A frozen arm cannot consume an adapted checkpoint.")

        # Numeric-group separation is checked before model access. This loader
        # additionally limits pixels to the current phase.
        selected_pairs = testing if args.phase == "test" else training
        allowed = {key for pair in selected_pairs for key in (pair.image_a, pair.image_b)}

        @lru_cache(maxsize=8)
        def load(key):
            if key not in allowed:
                raise InputError("An endpoint outside the current phase was requested.")
            row = bundle["images"][key]
            return image_tensor(prepare_image(paths[key], row["sha256"], row["decoded_sha256"]))

        event = progress_printer()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            if args.phase == "train":
                train_fixed(model, training, load, config, output, provenance,
                            integrity=integrity, resume=args.resume, event=event)
            else:
                # Flash attention requires BF16/FP16. This is a declared new
                # scoring protocol; the completed small-model FP32 scores stay intact.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if args.phase == "calibrate":
                        calibrate_fixed(model, training, load, config, output, provenance, integrity=integrity, event=event)
                    else:
                        calibration = private_path(args.calibration, "local/precise/runs")
                        test_fixed(model, training, testing, load, config, calibration, output, provenance,
                                   integrity=integrity, event=event, scope=bundle["scope"])
    except (InputError, OSError, ValueError, TypeError, KeyError, RuntimeError,
            ImportError, subprocess.SubprocessError):
        # The full diagnostic can contain identifying paths, so retain it only
        # in a new ignored sidecar, never in the public console or existing files.
        try:
            failed_output = private_path(args.output)
            diagnostic = failed_output.with_name(failed_output.name + "-failure.json")
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            write_json(diagnostic, {"phase": args.phase, "traceback": traceback.format_exc()})
        except (InputError, OSError):
            pass
        print("Native-large phase did not complete; preserve private records and investigate. No automatic retry.",
              file=sys.stderr)
        return 1
    print("Reviewed native-large phase completed; private artifacts preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
