"""Score the completed native model under one explicitly reviewed runtime bridge.

No training, source/record rewriting, numerical fallback or automatic retry.
Uses the unchanged native encoder, preprocessing, calibration and one-time test.
"""

import argparse
import copy
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import run_precise_native_large as native
from precise_matching.data import InputError
from precise_matching.fixed_pairs import load_bundle, reviewed_shared_inputs
from precise_matching.fixed_evaluation import calibrate_fixed, test_fixed, inspect_calibration
from precise_matching.fixed_training import checked_checkpoint
from precise_matching.training import digest, write_json
from precise_native_runtime_review import BRIDGE_PATHS, inspect_runtime_review

ROOT = Path(__file__).resolve().parents[1]


def checked_output(review, phase, arm, output):
    output = Path(output).resolve()
    expected = review.get("outputs", {}).get(phase + "_" + arm)
    if (not isinstance(expected, str) or Path(expected).resolve() != output
            or output.exists() or output.with_name(output.name + "-failure.json").exists()):
        raise InputError("Use the single unused output path bound to this reviewed scoring phase.")
    return output


def current_core(args, hashes, original_sources):
    import torch
    pth = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages/preserved_packages.pth"
    return {"code_commit": native.git_output("log", "-1", "--format=%H", "--", *original_sources, text=True).strip(),
        "source_sha256": original_sources, "input_sha256": hashes,
        "source_root": str(args.source_root.resolve()), "weights_path": str(args.weights.resolve()),
        "model": native.MODEL_ID, "model_revision": native.REVISION, "weights_sha256": native.WEIGHTS_SHA256,
        "model_config_sha256": native.CONFIG_SHA256,
        "environment_lock_sha256": digest(ROOT / "code/precise_matching/environment/requirements-linux-x86_64-py312-cu124.lock"),
        "python": platform.python_version(), "packages": native.package_versions(),
        "benchmark_scope": "numeric_group_disjoint_fixed_pairs", "interpreter_path": str(Path(sys.executable).resolve()),
        "interpreter_sha256": digest(Path(sys.executable).resolve()), "compatibility_binding_sha256": digest(pth) if pth.is_file() else None,
        "model_profile": "dinov2_large_native_v1", "image_preprocessing": {"original_width_height": list(native.ORIGINAL_SIZE),
            "padded_width_height": list(native.PADDED_SIZE), "resampling": False, "padding": "centered_white"},
        "attention": "sdpa_forced_flash_only", "inference_precision": "bf16", "gradient_checkpointing": "nonreentrant",
        "hardware": {"device": "cuda", "platform": platform.platform(), "torch_runtime": str(torch.__version__),
            "torch_compiled_cuda": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(), **native.cuda_driver_metadata()},
        "model_identity": {"arm": "adapted_dinov2", "base_weights_sha256": native.WEIGHTS_SHA256, "revision": native.REVISION}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibrate", "test"))
    for name in ("bundle", "config", "review", "runtime-review", "completed-training-run", "source-root", "weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("frozen_dinov2", "adapted_dinov2"), required=True)
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args(argv)
    try:
        output = native.private_path(args.output, "local/precise/runs")
        run = native.private_path(args.completed_training_run, "local/precise/runs")
        native.private_path(args.bundle);native.private_path(args.review);native.private_path(args.runtime_review)
        cfg, hashes, input_integrity = reviewed_shared_inputs(args.bundle, args.config, args.review, args.phase)
        bundle, training, testing, paths = load_bundle(args.bundle, args.source_root)
        cfg.validate(len(training));native.validate_native_policy(bundle, cfg, json.loads(args.review.read_text()))
        original_sources = native.source_record()
        if native.git_output("status", "--porcelain", "--", *BRIDGE_PATHS).strip():
            raise InputError("Review and commit the separate scoring bridge before execution.")
        sources = {**original_sources, **{n: digest(ROOT / n) for n in BRIDGE_PATHS}}
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false")
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        if not torch.cuda.is_available():raise InputError("Reviewed CUDA runtime is unavailable; no fallback.")
        core = current_core(args, hashes, original_sources)
        transition, evidence, evidence_hashes = inspect_runtime_review(args.runtime_review, run, core, hashes, sources)
        runtime_review = json.loads(args.runtime_review.read_text())
        checked_output(runtime_review, args.phase, args.arm, output)
        review_hash = digest(args.runtime_review)

        def integrity():
            input_integrity()
            if (digest(args.runtime_review) != review_hash or any(digest(ROOT / n) != v for n, v in sources.items())
                    or any(digest(p) != evidence_hashes[k] for k, p in evidence.items())):
                raise InputError("Reviewed runtime/evidence/source changed during scoring.")

        integrity()
        provenance = copy.deepcopy(core)
        provenance.update(code_commit=native.git_output("log", "-1", "--format=%H", "--", *sources, text=True).strip(),
                          source_sha256=sources, scoring_runtime_transition=transition)
        provenance["model_identity"]["arm"] = args.arm
        adapted = args.arm == "adapted_dinov2"
        model = native.load_encoder(args.weights, adapted=adapted, device=cfg.device)
        torch.use_deterministic_algorithms(True)
        if adapted:
            record = json.loads((run / "run_record.json").read_text());selection = json.loads((run / "selection.json").read_text())
            checkpoint = (run / selection["checkpoint"]).resolve()
            saved = checked_checkpoint(checkpoint, run, record)
            if saved["epochs_completed"] != cfg.epochs or saved["history"] != selection["history"]:
                raise InputError("Original final checkpoint does not replay its sealed selection.")
            model.load_state_dict(saved["model"], strict=True)
            provenance["model_identity"].update(checkpoint_sha256=digest(checkpoint), selection_sha256=digest(run / "selection.json"))
        selected_pairs = testing if args.phase == "test" else training
        allowed = {key for pair in selected_pairs for key in (pair.image_a, pair.image_b)}

        @lru_cache(maxsize=8)
        def load(key):
            if key not in allowed:raise InputError("Image endpoint outside the declared scoring phase.")
            row = bundle["images"][key]
            return native.image_tensor(native.prepare_image(paths[key], row["sha256"], row["decoded_sha256"]))

        with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.autocast("cuda", dtype=torch.bfloat16):
            if args.phase == "calibrate":
                calibrate_fixed(model, training, load, cfg, output, provenance, integrity=integrity, event=native.progress_printer())
            else:
                if args.calibration is None:raise InputError("Sealed calibration required before testing.")
                calibration = native.private_path(args.calibration, "local/precise/runs")
                # Both selections must already be sealed. Each replay checks the
                # complete immutable training score table; no test-score tuning.
                for short in ("frozen", "adapted"):
                    folder = run.parent / (short + "-calibration")
                    recorded = json.loads((folder / "run_record.json").read_text())["provenance"]
                    if (recorded.get("scoring_runtime_transition") != transition
                            or recorded.get("hardware") != provenance["hardware"]):
                        raise InputError("Both calibrations must bind the same reviewed scoring runtime.")
                    inspect_calibration(folder, training, cfg, recorded)
                test_fixed(model, training, testing, load, cfg, calibration, output, provenance,
                           integrity=integrity, event=native.progress_printer(), scope=bundle["scope"])
    except (InputError, OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError, subprocess.SubprocessError):
        try:
            diagnostic = native.private_path(args.output).with_name(args.output.name + "-failure.json")
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            write_json(diagnostic, {"phase": args.phase, "traceback": traceback.format_exc()})
        except (InputError, OSError):pass
        print("Reviewed restored-runtime scoring did not complete; preserve records. No automatic retry.", file=sys.stderr)
        return 1
    print("Explicitly reviewed restored-runtime scoring completed; original training preserved.")
    return 0


if __name__ == "__main__":raise SystemExit(main())
