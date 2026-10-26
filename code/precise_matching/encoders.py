"""Pinned local DINOv2 shared encoder; loading never downloads checkpoints."""

import hashlib
import json
from pathlib import Path

from .data import InputError

MODEL_ID = "facebook/dinov2-small"
REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
WEIGHTS_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
CONFIG_SHA256 = "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d"
TRAINABLE_PREFIXES = ("encoder.layer.10.", "encoder.layer.11.", "layernorm.")


def validate_checkpoint(directory: Path):
    """Verify immutable weights and architectural config before loading code."""
    try:
        config_bytes = (directory / "config.json").read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != CONFIG_SHA256:
            raise InputError("Checkpoint configuration differs from the pinned revision.")
        config = json.loads(config_bytes)
        expected = {"model_type": "dinov2", "hidden_size": 384,
                    "num_hidden_layers": 12, "num_attention_heads": 6, "patch_size": 14}
        if any(config.get(key) != value for key, value in expected.items()):
            raise InputError("Checkpoint architecture differs from the selected encoder.")
        digest = hashlib.sha256()
        with (directory / "model.safetensors").open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != WEIGHTS_SHA256:
            raise InputError("Checkpoint weights differ from the pinned revision.")
    except (OSError, ValueError, TypeError):
        raise InputError("Pinned local checkpoint could not be verified.") from None


def configure_trainable(model, adapted: bool):
    """Freeze everything except the selected final blocks/norm in adapted mode."""
    parameters = list(model.named_parameters())
    if not parameters or any(not any(name.startswith(prefix) for name, _ in parameters)
                             for prefix in TRAINABLE_PREFIXES):
        raise InputError("Expected encoder parameter groups are missing.")
    for name, parameter in parameters:
        parameter.requires_grad_(adapted and name.startswith(TRAINABLE_PREFIXES))
    model.train(adapted)
    return {"total_parameters": sum(p.numel() for _, p in parameters),
            "trainable_parameters": sum(p.numel() for _, p in parameters if p.requires_grad)}


def load_encoder(directory: Path, *, adapted=False, device="cpu"):
    import torch
    from transformers import Dinov2Model

    if device not in {"cpu", "cuda"}:
        raise InputError("Choose cpu or cuda explicitly.")
    if device == "cuda" and not torch.cuda.is_available():
        raise InputError("CUDA is unavailable; no automatic CPU fallback.")
    validate_checkpoint(directory)
    model = Dinov2Model.from_pretrained(
        directory, local_files_only=True, use_safetensors=True,
        trust_remote_code=False, attn_implementation="eager",
    ).to(device)
    validate_checkpoint(directory)
    configure_trainable(model, adapted)
    return model


def encode(model, pixels):
    import torch
    import torch.nn.functional as functional

    if (pixels.ndim != 4 or pixels.shape[0] < 1 or pixels.shape[1] != 3
            or min(pixels.shape[2:]) < 1 or not torch.isfinite(pixels).all()):
        raise InputError("Encoder pixels must be finite NCHW RGB tensors.")
    config = getattr(model, "config", None)
    patch = getattr(config, "patch_size", None)
    if patch and any(side % patch for side in pixels.shape[2:]):
        raise InputError("Encoder dimensions must retain complete patches.")
    hidden = model(pixel_values=pixels).last_hidden_state
    if (hidden.ndim != 3 or hidden.shape[0] != pixels.shape[0] or hidden.shape[1] < 1
            or hidden.shape[2] < 1 or hidden.shape[2] != getattr(config, "hidden_size", hidden.shape[2])):
        raise InputError("Encoder returned an incompatible embedding shape.")
    vectors = hidden[:, 0, :].float()
    norms = torch.linalg.vector_norm(vectors, dim=1)
    if not torch.isfinite(vectors).all() or not torch.isfinite(norms).all() or (norms <= 1e-12).any():
        raise InputError("Encoder returned invalid embeddings.")
    return functional.normalize(vectors, dim=1, eps=1e-12)


def cosine_scores(a, b):
    """One continuous score per aligned pair of unit embeddings."""
    import torch

    if a.ndim != 2 or a.shape != b.shape or not a.shape[0] or not a.shape[1]:
        raise InputError("Aligned nonempty embedding matrices are required.")
    for vectors in (a, b):
        norms = torch.linalg.vector_norm(vectors.float(), dim=1)
        if not torch.isfinite(vectors).all() or not torch.allclose(
                norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
            raise InputError("Similarity requires finite unit embeddings.")
    return (a.float() * b.float()).sum(dim=1).clamp(-1, 1)
