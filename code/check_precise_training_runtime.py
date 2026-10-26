"""Synthetic pretrained-encoder update/checkpoint probe, never fingerprint training.

CPU probe uses four random 56px images and two optimizer updates; this is not
the proposed 16-image/448px training memory check or a biometric evaluation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from precise_matching.data import InputError
from precise_matching.encoders import MODEL_ID, REVISION, configure_trainable, encode, load_encoder
from precise_matching.losses import verification_loss

ROOT = Path(__file__).resolve().parents[1]


def output_paths(output, private):
    output, private = output.resolve(), private.resolve()
    if (not output.is_relative_to(ROOT / "reports") or output.suffix != ".json"
            or not private.is_relative_to(ROOT / "local")):
        raise InputError("Use a new report JSON under reports/ and private directory under local/.")
    if output.exists() or private.exists():
        raise InputError("Existing outputs are never overwritten.")
    return output, private


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_probe(weights, device, private):
    # Applied before torch import/CUDA initialization; no model-hub requests.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    model = load_encoder(weights, adapted=True, device=device)
    counts = configure_trainable(model, True)
    frozen_before = frozen_digest(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    initial = [p.detach().clone() for p in trainable]
    decay = [p for p in trainable if p.ndim > 1]
    no_decay = [p for p in trainable if p.ndim <= 1]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=1e-5)
    pixels = torch.randn(4, 3, 56, 56, device=device)
    subjects = ("synthetic-A", "synthetic-A", "synthetic-B", "synthetic-B")
    captures = ("capture-A1", "capture-A2", "capture-B1", "capture-B2")
    hands = ("left",) * 4
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, pair_counts = verification_loss(encode(model, pixels), subjects, captures, hands)
        if not torch.isfinite(loss):
            raise InputError("Nonfinite synthetic loss.")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in trainable):
            raise InputError("Missing or nonfinite synthetic gradient.")
        torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    changed = sum(not torch.equal(a, b.detach()) for a, b in zip(initial, trainable, strict=True))
    if not changed or frozen_before != frozen_digest(model):
        raise InputError("Trainable/frozen weight update invariant failed.")
    model.eval()
    with torch.no_grad():
        expected = encode(model, pixels)
    checkpoint = private / "synthetic_checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": 2, "rng": torch.get_rng_state(), "revision": REVISION}, checkpoint)
    restored = torch.load(checkpoint, map_location=device, weights_only=True)
    if restored["step"] != 2 or restored["revision"] != REVISION:
        raise InputError("Synthetic checkpoint metadata mismatch.")
    resumed = load_encoder(weights, adapted=True, device=device)
    resumed.load_state_dict(restored["model"], strict=True)
    resumed_optimizer = torch.optim.AdamW(
        [{"params": [p for p in resumed.parameters() if p.requires_grad and p.ndim > 1], "weight_decay": 0.01},
         {"params": [p for p in resumed.parameters() if p.requires_grad and p.ndim <= 1], "weight_decay": 0.0}], lr=1e-5)
    resumed_optimizer.load_state_dict(restored["optimizer"])
    torch.set_rng_state(restored["rng"].cpu())
    resumed.eval()
    with torch.no_grad():
        actual = encode(resumed, pixels)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return {
        "scope": "Synthetic runtime check only; no fingerprint inputs or research performance",
        "model": MODEL_ID, "revision": REVISION, "device": device,
        "synthetic_images": 4, "input_side": 56, "optimizer_updates": 2,
        "embedding_shape": list(actual.shape), "synthetic_losses": losses,
        "batch_pairs": pair_counts, **counts,
        "changed_trainable_tensors": changed, "frozen_parameters_unchanged": True,
        "checkpoint_embeddings_identical": True,
        "optimizer_state_entries_restored": len(resumed_optimizer.state),
        "seconds": round(time.monotonic() - start, 3),
        "limits": ["Not a fingerprint training run or a test-set evaluation.",
                   "Does not establish full-batch 448px memory fit or usable CUDA when run on CPU.",
                   "Synthetic checkpoint is not a valid research-model checkpoint."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output, private = output_paths(args.output, args.private_dir)
        private.mkdir(parents=True)
        report = run_probe(args.weights, args.device, private)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except (InputError, OSError, ValueError, RuntimeError, AssertionError, ImportError):
        print("Synthetic runtime check failed; inspect the local environment and private outputs.", file=sys.stderr)
        return 1
    print("Synthetic encoder update/checkpoint check passed. No fingerprint training or evaluation performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
