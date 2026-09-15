"""Read trusted local training images with bounded size and white alpha backing.

No source file is changed. Every decode has an explicit pixel ceiling. Images
above Pillow's normal guard are decoded sequentially; the guard is only adjusted
under a lock while opening the header, then immediately restored. Callers should
use a small worker count because fully decoded images still occupy CPU memory.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
import threading
import warnings

from PIL import Image, ImageOps


MAX_PIXELS = 300_000_000
STANDARD_MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS or 89_478_485
_OPEN_LOCK = threading.RLock()
_LARGE_DECODE_LOCK = threading.RLock()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_bounded(path, max_pixels):
    if not isinstance(max_pixels, int) or not 0 < max_pixels <= MAX_PIXELS:
        raise ValueError(f"max_pixels must be an integer in [1, {MAX_PIXELS}]")
    # Pillow's guard is process-global; all opens in this module use this lock.
    with _OPEN_LOCK:
        previous = Image.MAX_IMAGE_PIXELS
        try:
            Image.MAX_IMAGE_PIXELS = max_pixels
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                image = Image.open(path)
            if image.width < 1 or image.height < 1 or image.width * image.height > max_pixels:
                image.close()
                raise ValueError(f"Image dimensions exceed the explicit {max_pixels}-pixel ceiling")
            return image
        finally:
            Image.MAX_IMAGE_PIXELS = previous


def _header(path, max_pixels):
    with _open_bounded(path, max_pixels) as original:
        size = original.size
        mode = original.mode
        format_name = original.format
        # PNG getexif() may load pixels and invalidate verify(); verify first,
        # then inspect EXIF during the separate guarded decode below.
        original.verify()
    return {
        "original_width": size[0], "original_height": size[1],
        "original_mode": mode, "format": format_name,
        "large_image_override": size[0] * size[1] > STANDARD_MAX_IMAGE_PIXELS,
        "max_pixels": max_pixels,
    }


def _decode_rgb(path, max_pixels):
    with _open_bounded(path, max_pixels) as original:
        orientation = original.getexif().get(274, 1)
        oriented = ImageOps.exif_transpose(original)
        try:
            oriented.load()
            has_alpha = oriented.mode in ("RGBA", "LA") or "transparency" in oriented.info
            if has_alpha:
                rgba = oriented.convert("RGBA")
                try:
                    output = Image.new("RGB", rgba.size, "white")
                    output.paste(rgba, mask=rgba.getchannel("A"))
                finally:
                    rgba.close()
            else:
                output = oriented.convert("RGB")
        finally:
            oriented.close()
    return output, has_alpha, orientation


def load_rgb(path, *, expected_sha256=None, max_pixels=MAX_PIXELS):
    """Return a detached, EXIF-corrected RGB image; caller owns and closes it."""
    path = Path(path)
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError(f"Image changed since preparation: {path}")
    metadata = _header(path, max_pixels)
    guard = _LARGE_DECODE_LOCK if metadata["large_image_override"] else nullcontext()
    with guard:
        rgb, _, _ = _decode_rgb(path, max_pixels)
    return rgb


def rgb_pixel_sha256(image, stripe_rows=64):
    """Canonical white-backed RGB identity without a full-image bytes copy."""
    if image.mode != "RGB":
        raise ValueError("Pixel hashing requires the same RGB image used for training")
    if stripe_rows < 1:
        raise ValueError("stripe_rows must be positive")
    digest = hashlib.sha256(f"RGB:{image.width}:{image.height}:".encode("ascii"))
    for top in range(0, image.height, stripe_rows):
        with image.crop((0, top, image.width, min(image.height, top + stripe_rows))) as stripe:
            digest.update(stripe.tobytes())
    return digest.hexdigest()


def inspect_image(path, *, max_pixels=MAX_PIXELS):
    """Return v1-compatible audit metadata or {'error': ...}; no writes."""
    path = Path(path)
    try:
        encoded_sha = file_sha256(path)
        metadata = _header(path, max_pixels)
        guard = _LARGE_DECODE_LOCK if metadata["large_image_override"] else nullcontext()
        with guard:
            image, alpha_composited, orientation = _decode_rgb(path, max_pixels)
            try:
                result = {**metadata, "image_sha256": encoded_sha,
                          "pixel_sha256": rgb_pixel_sha256(image),
                          "width": image.width, "height": image.height,
                          "exif_transposed": orientation not in (None, 1),
                          "alpha_composited": alpha_composited}
            finally:
                image.close()
        if file_sha256(path) != encoded_sha:
            raise ValueError("Image bytes changed while inspecting")
        return result
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
