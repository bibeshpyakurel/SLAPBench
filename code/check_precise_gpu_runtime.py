"""Full-resolution synthetic CUDA capacity check; no fingerprint inputs or scores.

Use the selected local encoder, proposed batch/accumulation/precision and loss.
This checks short-run capacity, not full training readiness or biometric accuracy.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from check_precise_training_runtime import frozen_digest, output_paths
from precise_matching.data import InputError
from precise_matching.encoders import (CONFIG_SHA256, MODEL_ID, REVISION, WEIGHTS_SHA256,
                                        configure_trainable, encode, load_encoder)
from precise_matching.images import MEAN, SIDE, STD
from precise_matching.losses import verification_loss
from precise_matching.training import Config, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
RESERVE_BYTES = 2 * 1024**3


def synthetic_updates(model, config, *, side=SIDE, updates=2, observe=lambda: None):
    """Match training's forward/backward/accumulation and AdamW allocation.

    No scheduler or checkpoint selection is exercised. Nominal learning rate
    tests update invariants; losses and timing do not measure biometric quality.
    Tests inject a tiny CPU model; the CLI requires CUDA and uses SIDE=448.
    """
    import torch

    config.validate()
    if type(updates) is not int or updates < 2:
        raise InputError("At least two updates are needed to exercise allocated AdamW state.")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise InputError("The adapted synthetic capacity check requires trainable parameters.")
    initial = [p.detach().cpu().clone() for p in parameters]
    frozen_before = frozen_digest(model)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in parameters if p.ndim > 1], "weight_decay": config.weight_decay},
        {"params": [p for p in parameters if p.ndim <= 1], "weight_decay": 0.0}],
        lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8)
    batch = config.subjects_per_batch * 2
    subjects = ["synthetic-subject-" + str(i // 2) for i in range(batch)]
    captures = ["synthetic-capture-" + str(i) for i in range(batch)]
    losses, pair_counts = [], None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for micro in range(updates * config.accumulation):
        # Match grayscale-to-RGB scaling without reading/creating image files.
        gray = torch.rand(batch, 1, side, side)
        pixels = ((gray.expand(-1, 3, -1, -1) - torch.tensor(MEAN)[None, :, None, None]) /
                  torch.tensor(STD)[None, :, None, None]).to(config.device)
        with torch.autocast(config.device, dtype=torch.bfloat16, enabled=config.precision == "bf16"):
            embeddings = encode(model, pixels)
        loss, pair_counts = verification_loss(embeddings, subjects, captures,
                                              [("left", "right")[micro % 2]] * batch, config.margin)
        if not torch.isfinite(loss):
            raise InputError("Nonfinite full-resolution synthetic loss.")
        losses.append(float(loss.detach()))
        (loss / config.accumulation).backward()
        if (micro + 1) % config.accumulation == 0:
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise InputError("Missing or nonfinite full-resolution synthetic gradients.")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        observe()
    changed = sum(not torch.equal(a, p.detach().cpu()) for a, p in zip(initial, parameters, strict=True))
    if not changed or frozen_before != frozen_digest(model):
        raise InputError("Synthetic updates must change trainable parameters and preserve frozen parameters.")
    return {"optimizer_updates": updates, "microbatches": len(losses), "images_per_microbatch": batch,
            "accumulation": config.accumulation, "precision": config.precision,
            "nominal_learning_rate": config.learning_rate, "input_side": side,
            "batch_pairs": pair_counts, "losses_finite": True, "gradients_finite": True,
            "changed_trainable_tensors": changed, "optimizer_state_entries": len(optimizer.state),
            "frozen_parameters_unchanged": True}


def run_gpu_probe(weights, config, private):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    config.validate()
    if config.device != "cuda" or not torch.cuda.is_available():
        raise InputError("The full-resolution GPU check requires reviewed CUDA settings; no CPU fallback.")
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise InputError("The configured BF16 mode is unavailable; no precision fallback.")
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                     text=True).splitlines()[0].strip()
    props = torch.cuda.get_device_properties(0)
    free_start, total = torch.cuda.mem_get_info()
    reserved_start = torch.cuda.memory_reserved()
    if free_start < RESERVE_BYTES:
        raise InputError("Insufficient currently free device memory for the 2 GiB external reserve.")
    report = {"schema_version": 1, "scope": "Synthetic CUDA capacity only; no Precise data or biometric evaluation",
        "passed": False, "model": MODEL_ID, "revision": REVISION, "weights_sha256": WEIGHTS_SHA256,
        "model_config_sha256": CONFIG_SHA256, "input_side": SIDE, "config": asdict(config),
        "driver_version": driver, "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
        "device": props.name, "device_total_bytes": total, "device_free_before_bytes": free_start,
        "device_capability": list(torch.cuda.get_device_capability()),
        "bf16_supported": torch.cuda.is_bf16_supported(), "external_reserve_bytes": RESERVE_BYTES,
        "workspace_free_bytes": shutil.disk_usage(ROOT).free, "phases": {}}
    minimum_free = free_start

    def observe():
        nonlocal minimum_free
        torch.cuda.synchronize()
        minimum_free = min(minimum_free, torch.cuda.mem_get_info()[0])
        if minimum_free < RESERVE_BYTES:
            raise InputError("Observed free GPU memory fell below the external 2 GiB reserve.")

    def phase(name, operation):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        write_json(private / (name + "_started.json"), {"phase": name})
        value = operation()
        observe()
        report["phases"][name] = {"seconds": round(time.monotonic() - started, 4),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "free_device_bytes_after": torch.cuda.mem_get_info()[0], **value}
        write_json(private / (name + "_result.json"), report["phases"][name])

    model = load_encoder(weights, adapted=False, device="cuda")
    counts = configure_trainable(model, False)

    def inference():
        model.eval()
        with torch.inference_mode():
            output = encode(model, torch.rand(1, 3, SIDE, SIDE, device="cuda"))
        if list(output.shape) != [1, 384]:
            raise InputError("Unexpected full-resolution DINOv2 embedding shape.")
        return {"embedding_shape": list(output.shape), "precision": "fp32", "gradients_enabled": False}

    phase("frozen_fp32_inference", inference)
    counts = configure_trainable(model, True)
    phase("adapted_training", lambda: synthetic_updates(model, config, observe=observe))
    phase("adapted_fp32_inference", inference)
    peak = max(p["peak_reserved_bytes"] for p in report["phases"].values())
    estimate = free_start + reserved_start - peak
    if estimate < RESERVE_BYTES:
        raise InputError("Peak allocator reservation leaves less than the planned 2 GiB external reserve.")
    report.update({**counts, "minimum_sampled_free_device_bytes": minimum_free,
        "estimated_headroom_at_initial_external_usage_bytes": estimate,
        "external_reserve_met": True, "passed": True, "seconds": round(time.monotonic() - start, 3),
        "limits": ["No source images, real subjects, comparison lists or research scores were accessed.",
            "Random grayscale microbatches exercise capacity; their losses are not biometric evidence.",
            "The nominal LR is used without a scheduler; full checkpoint/resume was checked separately.",
            "Peak allocated/reserved bytes describe this process; sampled driver-free memory and headroom depend on other processes.",
            "Allocator caches are retained between phases; later reserved-memory peaks can include earlier cached allocations.",
            "Inference is single-image FP32, matching the existing development/final-test loaders.",
            "Short-run fit does not establish long-run stability, throughput, protocol approval or actual training readiness.",
            "No microbatch, input size, precision or device is reduced automatically on failure."]})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "config", "output", "private-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    private = None
    created_private = False
    try:
        output, private = output_paths(args.output, args.private_dir)
        before = digest(args.config)
        config = Config(**json.loads(args.config.read_text()))
        config.validate()
        if config.device != "cuda":
            raise InputError("Full-resolution GPU verification requires an explicit CUDA configuration.")
        private.mkdir(parents=True)
        created_private = True
        write_json(private / "execution.json", {"started_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": before, "weights_path": str(args.weights.resolve()),
            "python": platform.python_version(),
            "packages": {name: metadata.version(name) for name in
                         ("torch", "transformers", "numpy", "Pillow", "safetensors")},
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                [Path(__file__).resolve(), *sorted((ROOT / "code/precise_matching").glob("*.py"))]}})
        report = run_gpu_probe(args.weights, config, private)
        if digest(args.config) != before:
            raise InputError("Capacity-check configuration changed during execution.")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
    except (InputError, OSError, ValueError, TypeError, RuntimeError, AssertionError,
            ImportError, subprocess.SubprocessError) as error:
        if created_private and not (private / "failure.json").exists():
            write_json(private / "failure.json", {"status": "failed", "exception_type": type(error).__name__,
                       "no_fingerprint_inputs": True, "automatic_retry_allowed": False})
        print("Synthetic GPU capacity check failed; preserve private phase/failure records. No fallback performed.",
              file=sys.stderr)
        return 1
    print("Synthetic full-resolution CUDA check passed. No fingerprint training or evaluation performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
