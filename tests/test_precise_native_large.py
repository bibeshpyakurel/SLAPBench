"""Pixel-preserving native preparation contracts; synthetic raster only."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from precise_matching.data import InputError  # noqa: E402
from precise_matching.native_large import ORIGINAL_SIZE, PADDED_SIZE, pad_native_image, prepare_native_image  # noqa: E402


class NativeLargePreparationTests(unittest.TestCase):
    def test_padding_preserves_every_original_pixel_without_resampling(self):
        gray = Image.new("L", ORIGINAL_SIZE)
        gray.putdata(bytes(range(256)) * (ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1] // 256))
        prepared = pad_native_image(gray)
        self.assertEqual(prepared.size, PADDED_SIZE)
        self.assertEqual(prepared.mode, "RGB")
        restored = prepared.crop((5, 6, 1605, 1506)).convert("L")
        self.assertEqual(restored.tobytes(), gray.tobytes())
        self.assertEqual(prepared.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(tuple(side % 14 for side in prepared.size), (0, 0))

    def test_uninventoried_size_or_mode_cannot_change_protocol_implicitly(self):
        for image in (Image.new("L", (448, 448)), Image.new("RGB", ORIGINAL_SIZE)):
            with self.assertRaises(InputError):pad_native_image(image)

    def test_reviewed_bytes_decoded_hash_and_strict_decoding_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.png";gray = Image.new("L", ORIGINAL_SIZE, 120);gray.save(path)
            before = path.read_bytes();raw = hashlib.sha256(before).hexdigest()
            decoded = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
            self.assertEqual(prepare_native_image(path, raw, decoded).size, PADDED_SIZE)
            self.assertEqual(before, path.read_bytes())
            for a, b in (("0" * 64, decoded), (raw, "0" * 64)):
                with self.assertRaises(InputError):prepare_native_image(path, a, b)
            with patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", True), self.assertRaises(InputError):
                prepare_native_image(path, raw, decoded)


if __name__ == "__main__":unittest.main()
