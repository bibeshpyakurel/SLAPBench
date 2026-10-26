"""Whole-slap preprocessing; no segmentation, augmentation or population changes."""

import hashlib
import io
from pathlib import Path
import re

from PIL import Image, ImageFile, UnidentifiedImageError

from .data import InputError

SIDE = 448
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def prepare_image(path: Path, expected_sha256: str, expected_decoded_sha256=None):
    """Decode the exact reviewed bytes and retain the entire image on white."""
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise InputError("A reviewed source digest is required.")
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise InputError("Permissive truncated-image decoding is not allowed.")
    if expected_decoded_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_decoded_sha256):
        raise InputError("A reviewed decoded digest is required.")
    try:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise InputError("Image bytes differ from the reviewed input.")
        with Image.open(io.BytesIO(data)) as original:
            original.load()
            gray = original.convert("L")
        decoded = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
        if expected_decoded_sha256 is not None and decoded != expected_decoded_sha256:
            raise InputError("Decoded content differs from the reviewed input.")
        width, height = gray.size
        scale = SIDE / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = gray.resize(size, Image.Resampling.LANCZOS)
        canvas = Image.new("L", (SIDE, SIDE), 255)
        canvas.paste(resized, ((SIDE - size[0]) // 2, (SIDE - size[1]) // 2))
        return canvas.convert("RGB")
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        raise InputError("Image could not be decoded safely.") from None


def image_tensor(image):
    """Convert prepared RGB pixels using the same scaling in both arms."""
    import numpy as np
    import torch

    if image.mode != "RGB" or image.size != (SIDE, SIDE):
        raise InputError("Expected a prepared whole-slap RGB image.")
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return (tensor - tensor.new_tensor(MEAN)[:, None, None]) / tensor.new_tensor(STD)[:, None, None]
