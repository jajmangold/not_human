import io
import unittest

from PIL import Image

from web.image_normalization import normalize_portrait


class PortraitOrientationTests(unittest.TestCase):
    def test_normalizer_applies_exif_orientation_before_png_conversion(self):
        source = Image.new("RGB", (40, 20), "navy")
        exif = source.getexif()
        exif[274] = 6  # rotate 90 degrees clockwise for display
        encoded = io.BytesIO()
        source.save(encoded, format="JPEG", exif=exif)

        normalized = normalize_portrait(encoded.getvalue())
        with Image.open(io.BytesIO(normalized)) as result:
            self.assertEqual(result.size, (20, 40))
            self.assertEqual(result.format, "PNG")


if __name__ == "__main__":
    unittest.main()
