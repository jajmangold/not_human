"""Portrait decoding and EXIF-safe normalization without web dependencies."""

from __future__ import annotations

import io

from PIL import Image, ImageOps


class PortraitDimensionsError(ValueError):
    """The decoded portrait exceeds the configured dimension limit."""


def normalize_portrait(payload: bytes, maximum_dimension: int = 4096) -> bytes:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        image = ImageOps.exif_transpose(image)
        if image.width > maximum_dimension or image.height > maximum_dimension:
            raise PortraitDimensionsError(f"portrait dimensions exceed {maximum_dimension}px")
        output = io.BytesIO()
        image.convert("RGB").save(output, format="PNG", optimize=True)
        return output.getvalue()
