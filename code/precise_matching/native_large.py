"""Pinned larger whole-slap encoder and pixel-preserving native preparation.

The completed 448-pixel small-model route remains unchanged. No resizing,
cropping, extra pairs or model downloads occur through this module.
"""

import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageFile

from .data import InputError
from .images import MEAN, STD

MODEL_ID = "facebook/dinov2-large"
REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"
WEIGHTS_SHA256 = "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23"
CONFIG_SHA256 = "12df51c069a2dc1305e34ba71ef58bc2407ea553b75f4722a1715c1bce3bbed0"
TRAINABLE_PREFIXES = ("encoder.layer.22.", "encoder.layer.23.", "layernorm.")
ORIGINAL_SIZE = (1600, 1500)
PADDED_SIZE = (1610, 1512)


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def validate_large_checkpoint(directory):
    directory = Path(directory)
    if (file_digest(directory / "config.json") != CONFIG_SHA256
            or file_digest(directory / "model.safetensors") != WEIGHTS_SHA256):
        raise InputError("Large-model bytes differ from the pinned reviewed source.")
    config = json.loads((directory / "config.json").read_text())
    expected = {"model_type": "dinov2", "hidden_size": 1024, "num_hidden_layers": 24,
                "num_attention_heads": 16, "patch_size": 14}
    if any(config.get(k) != v for k, v in expected.items()):
        raise InputError("Unexpected large-model architecture.")


def configure_large_trainable(model, adapted):
    named = list(model.named_parameters())
    if any(not any(name.startswith(prefix) for name, _ in named) for prefix in TRAINABLE_PREFIXES):
        raise InputError("Large-model selected parameter groups are missing.")
    for name, parameter in named:
        parameter.requires_grad_(adapted and name.startswith(TRAINABLE_PREFIXES))
    model.train(adapted)
    return {"total_parameters": sum(p.numel() for _, p in named),
            "trainable_parameters": sum(p.numel() for _, p in named if p.requires_grad)}


def load_large_encoder(directory, *, adapted=False, device="cuda"):
    import torch
    from transformers import Dinov2Model

    if device not in {"cpu", "cuda"} or device == "cuda" and not torch.cuda.is_available():
        raise InputError("Explicit available CUDA/CPU device required; no fallback.")
    validate_large_checkpoint(directory)
    model = Dinov2Model.from_pretrained(directory, local_files_only=True,
        use_safetensors=True, trust_remote_code=False, attn_implementation="sdpa").to(device)
    if model.config._attn_implementation != "sdpa":
        raise InputError("The declared SDPA attention implementation was not selected.")
    configure_large_trainable(model, adapted)
    if adapted:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def pad_native_image(gray):
    if gray.mode != "L" or gray.size != ORIGINAL_SIZE:
        raise InputError("Native protocol requires the inventoried 1600 x 1500 grayscale image.")
    canvas = Image.new("L", PADDED_SIZE, 255)
    # Center the original raster; never resample it. Exact inverse crop exists.
    canvas.paste(gray, ((PADDED_SIZE[0] - gray.width) // 2, (PADDED_SIZE[1] - gray.height) // 2))
    return canvas.convert("RGB")


def prepare_native_image(path, expected_sha256, expected_decoded_sha256):
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise InputError("Permissive truncated-image decoding is not allowed.")
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise InputError("Native source bytes differ from reviewed input.")
    with Image.open(io.BytesIO(data)) as original:
        original.load(); gray = original.convert("L")
    decoded = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
    if decoded != expected_decoded_sha256:
        raise InputError("Native grayscale content differs from reviewed input.")
    return pad_native_image(gray)


def native_tensor(image):
    import numpy as np
    import torch

    if image.mode != "RGB" or image.size != PADDED_SIZE:
        raise InputError("Expected the declared padded native RGB canvas.")
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
    return (tensor - tensor.new_tensor(MEAN)[:, None, None]) / tensor.new_tensor(STD)[:, None, None]
