"""CPU tests for image identity, alpha handling and the bounded decode policy."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

import image_io
from cache_features import prepare_pixels


class ImageIOTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="sana_image_io_test_")
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_rgba_composites_on_white_and_cache_uses_same_pixels(self):
        path = self.root / "alpha.png"
        source = Image.new("RGBA", (3, 1))
        source.putdata([(0, 0, 0, 0), (255, 0, 0, 255), (0, 0, 255, 128)])
        source.save(path)
        before = path.read_bytes()
        with image_io.load_rgb(path) as image:
            self.assertEqual(list(image.getdata()), [(255, 255, 255), (255, 0, 0), (127, 127, 255)])
            expected_hash = image_io.rgb_pixel_sha256(image)
        result = image_io.inspect_image(path)
        self.assertNotIn("error", result)
        self.assertTrue(result["alpha_composited"])
        self.assertEqual(result["pixel_sha256"], expected_hash)
        pixels = prepare_pixels(path, 3, expected_sha256=hashlib.sha256(before).hexdigest())
        restored = ((pixels[0] + 1) * 127.5).round().to(dtype=__import__("torch").uint8)
        self.assertEqual(restored[:, 1, 0].tolist(), [255, 255, 255])
        self.assertEqual(restored[:, 1, 1].tolist(), [255, 0, 0])
        self.assertEqual(restored[:, 1, 2].tolist(), [127, 127, 255])
        self.assertEqual(path.read_bytes(), before)

    def test_palette_transparency_is_composited_instead_of_discarded(self):
        path = self.root / "palette.png"
        source = Image.new("P", (2, 1))
        source.putpalette([255, 0, 0, 0, 0, 255] + [0] * (768 - 6))
        source.putdata([0, 1])
        source.save(path, transparency=0)
        with image_io.load_rgb(path) as image:
            self.assertEqual(list(image.getdata()), [(255, 255, 255), (0, 0, 255)])

    def test_exif_rotation_changes_dimensions_without_source_write(self):
        path = self.root / "oriented.jpg"
        source = Image.new("RGB", (7, 11), "red")
        exif = Image.Exif()
        exif[274] = 6
        source.save(path, exif=exif)
        before = path.read_bytes()
        result = image_io.inspect_image(path)
        self.assertNotIn("error", result)
        self.assertEqual((result["original_width"], result["original_height"]), (7, 11))
        self.assertEqual((result["width"], result["height"]), (11, 7))
        self.assertTrue(result["exif_transposed"])
        self.assertEqual(path.read_bytes(), before)

    def test_explicit_pixel_ceiling_rejects_and_restores_pillow_guard(self):
        path = self.root / "bounded.png"
        Image.new("RGB", (10, 10)).save(path)
        original_guard = Image.MAX_IMAGE_PIXELS
        result = image_io.inspect_image(path, max_pixels=64)
        self.assertIn("error", result)
        self.assertEqual(Image.MAX_IMAGE_PIXELS, original_guard)
        with self.assertRaises(ValueError):
            image_io.load_rgb(path, max_pixels=image_io.MAX_PIXELS + 1)
        self.assertEqual(Image.MAX_IMAGE_PIXELS, original_guard)

    def test_known_large_policy_can_be_exercised_without_a_large_allocation(self):
        path = self.root / "simulated_large.png"
        Image.new("RGB", (10, 10), "blue").save(path)
        original_guard = Image.MAX_IMAGE_PIXELS
        with mock.patch.object(image_io, "STANDARD_MAX_IMAGE_PIXELS", 32):
            result = image_io.inspect_image(path, max_pixels=128)
        self.assertNotIn("error", result)
        self.assertTrue(result["large_image_override"])
        self.assertEqual(result["max_pixels"], 128)
        self.assertEqual(Image.MAX_IMAGE_PIXELS, original_guard)

    def test_pixel_hash_preserves_the_previous_rgb_format_across_stripes(self):
        image = Image.new("RGB", (17, 131))
        image.putdata([((i * 17) % 256, (i * 31) % 256, (i * 53) % 256) for i in range(17 * 131)])
        expected = hashlib.sha256(b"RGB:17:131:" + image.tobytes()).hexdigest()
        self.assertEqual(image_io.rgb_pixel_sha256(image), expected)
        self.assertEqual(image_io.rgb_pixel_sha256(image, stripe_rows=1), expected)

    def test_changed_image_hash_is_rejected_before_preprocessing(self):
        path = self.root / "changed.png"
        Image.new("RGB", (5, 5)).save(path)
        with self.assertRaisesRegex(ValueError, "changed since preparation"):
            prepare_pixels(path, 32, expected_sha256="0" * 64)

    def test_broken_image_returns_error_without_mutating_source(self):
        path = self.root / "broken.jpg"
        path.write_bytes(b"not a JPEG")
        result = image_io.inspect_image(path)
        self.assertIn("error", result)
        self.assertEqual(path.read_bytes(), b"not a JPEG")


if __name__ == "__main__":
    unittest.main()
