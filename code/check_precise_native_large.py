"""Two-update native-resolution DINOv2-Large capacity check with synthetic pixels.

No fingerprint inputs, pairs, scores, model selection or real-data training.
The declared kernel/shape/precision never fall back on resource failure.
"""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import time

from check_precise_training_runtime import frozen_digest
from precise_matching.data import InputError
from precise_matching.encoders import encode
from precise_matching.hardware import cuda_driver_metadata
from precise_matching.native_large import (MODEL_ID, REVISION, WEIGHTS_SHA256, ORIGINAL_SIZE,
    PADDED_SIZE, configure_large_trainable, file_digest, load_large_encoder, validate_large_checkpoint)
from precise_matching.training import write_json

ROOT = Path(__file__).resolve().parents[1]


def probe(weights, output, accumulation=32):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise InputError("Native large probe requires CUDA with BF16 support.")
    if output.exists() or not output.resolve().is_relative_to(ROOT / "local/precise"):
        raise InputError("A new private synthetic-check directory is required.")
    output.mkdir(parents=True)
    config = {"model": MODEL_ID, "revision": REVISION, "weights_sha256": WEIGHTS_SHA256,
        "original_width_height": ORIGINAL_SIZE, "padded_width_height": PADDED_SIZE,
        "batch_pairs": 1, "images_per_microbatch": 2, "accumulation": accumulation,
        "updates": 2, "precision": "bf16", "attention": "sdpa_forced_flash_only",
        "gradient_checkpointing": "nonreentrant", "seed": 2026, "deterministic_algorithms": True,
        "source_sha256": {str(p.relative_to(ROOT)): file_digest(p) for p in
            (Path(__file__), ROOT / "code/precise_matching/native_large.py")}}
    write_json(output / "configuration.json", config)
    try:
        torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
        torch.use_deterministic_algorithms(True)
        free, total = torch.cuda.mem_get_info()
        model = load_large_encoder(weights, adapted=True)
        counts = configure_large_trainable(model, True)
        parameters = [p for p in model.parameters() if p.requires_grad]
        initial = [p.detach().cpu().clone() for p in parameters]
        frozen_before = frozen_digest(model)
        optimizer = torch.optim.AdamW([
            {"params": [p for p in parameters if p.ndim > 1], "weight_decay": .01},
            {"params": [p for p in parameters if p.ndim <= 1], "weight_decay": 0}], lr=1e-5)
        # Synthetic grayscale expanded to RGB; no source image is opened.
        height, width = PADDED_SIZE[1], PADDED_SIZE[0]
        gray = torch.rand(2, 1, height, width, device="cuda")
        pixels = gray.expand(-1, 3, -1, -1).contiguous()
        mean = torch.tensor((.485, .456, .406), device="cuda")[None, :, None, None]
        std = torch.tensor((.229, .224, .225), device="cuda")[None, :, None, None]
        pixels = (pixels - mean) / std
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        started = time.monotonic(); minimum_free = free; losses = []
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            model.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                before_vectors = encode(model, pixels)
            if not torch.isfinite(before_vectors).all():raise InputError("Frozen synthetic forward failed.")
            model.train()
            for update in range(2):
                optimizer.zero_grad(set_to_none=True)
                for micro in range(accumulation):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        vectors = encode(model, pixels)
                        cosine = (vectors[0] * vectors[1]).sum()
                        loss = (1 - cosine) if micro % 2 else torch.relu(cosine - .2)
                    if not torch.isfinite(loss):
                        raise InputError("Nonfinite synthetic pair loss.")
                    (loss / accumulation).backward()
                    losses.append(float(loss.detach()))
                    minimum_free = min(minimum_free, torch.cuda.mem_get_info()[0])
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise InputError("Missing/nonfinite selected gradients.")
                torch.nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
                optimizer.step(); torch.cuda.synchronize()
                minimum_free = min(minimum_free, torch.cuda.mem_get_info()[0])
                print(json.dumps({"synthetic_update": update + 1, "microbatches_completed": len(losses)}), flush=True)
            model.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                after_vectors = encode(model, pixels)
            if not torch.isfinite(after_vectors).all():raise InputError("Adapted synthetic forward failed.")
        elapsed = time.monotonic() - started
        validate_large_checkpoint(weights)
        changed = sum(not torch.equal(a, p.detach().cpu()) for a, p in zip(initial, parameters, strict=True))
        if (not changed or frozen_before != frozen_digest(model) or minimum_free < 2 * 1024**3
                or any(file_digest(ROOT / name) != value for name, value in config["source_sha256"].items())):
            raise InputError("Native synthetic update integrity/headroom check failed.")
        report = {"passed": True, "scope": "synthetic_native_large_capacity_only", "config": config,
            "hardware": {"name": torch.cuda.get_device_name(), "total_bytes": total,
                         "free_before_model_bytes": free, **cuda_driver_metadata()},
            "packages": {name: metadata.version(name) for name in ("torch", "transformers", "numpy", "Pillow")},
            "parameters": counts, "changed_trainable_tensors": changed, "frozen_parameters_unchanged": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "minimum_free_bytes": minimum_free,
            "elapsed_seconds": elapsed, "synthetic_pairs_per_second": len(losses) / elapsed,
            "finite_losses": True, "finite_gradients": True,
            "frozen_and_adapted_bf16_inference_passed": True, "embedding_dtype": str(after_vectors.dtype), "optimizer_state_entries": len(optimizer.state),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "limits": ["Synthetic throughput excludes fingerprint decoding, disk I/O and run-integrity checks.",
                       "Short-run capacity does not prove full-run stability, biometric accuracy or protocol approval.",
                       "No original image pixels, real subject pairs or testing scores were accessed."]}
        write_json(output / "capacity.json", report)
        return report
    except BaseException as exc:
        write_json(output / "failure.json", {"passed": False, "exception_type": type(exc).__name__,
                                            "no_automatic_retry": True})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        report = probe(args.weights, args.output)
    except (InputError, OSError, RuntimeError, ValueError):
        print("Synthetic native large check did not complete; preserve its private failure record.", file=sys.stderr)
        return 1
    print(json.dumps({"passed": report["passed"], "peak_allocated_bytes": report["peak_allocated_bytes"],
                      "elapsed_seconds": report["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
