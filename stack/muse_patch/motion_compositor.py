"""Region ownership helpers shared by live portrait render paths."""

import numpy as np


def _upper_region_only(
    base_frame,
    animated_frame,
    bbox,
    strength,
    fade_start_ratio,
    fade_end_ratio,
):
    if strength <= 0:
        return base_frame
    height, width = base_frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    face_width, face_height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)

    left, right = x1 - 0.12 * face_width, x2 + 0.12 * face_width
    feather_x = max(2.0, 0.10 * face_width)
    horizontal = np.minimum(
        np.clip((xs - left) / feather_x, 0.0, 1.0),
        np.clip((right - xs) / feather_x, 0.0, 1.0),
    )
    top, top_full = y1 - 0.12 * face_height, y1 - 0.02 * face_height
    fade_start = y1 + fade_start_ratio * face_height
    fade_end = y1 + fade_end_ratio * face_height
    vertical = np.minimum(
        np.clip((ys - top) / max(2.0, top_full - top), 0.0, 1.0),
        np.clip((fade_end - ys) / max(2.0, fade_end - fade_start), 0.0, 1.0),
    )
    alpha = np.outer(vertical, horizontal)[..., None] * min(1.0, float(strength))
    blended = base_frame.astype(np.float32) * (1.0 - alpha) + animated_frame.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def upper_face_only(base_frame, animated_frame, bbox, strength=1.0):
    """Keep blink pixels above the cheeks so MuseTalk remains mouth authority."""
    return _upper_region_only(base_frame, animated_frame, bbox, strength, 0.43, 0.58)


def gaze_region_only(base_frame, animated_frame, bbox, strength=1.0):
    """Composite gaze into the orbital band with a hard-protected lower face.

    ALP gaze anchors can contain small changes outside the pupils.  Ending the
    feather at half the detected face height makes every pixel in the mouth and
    jaw region exactly equal to MuseTalk's output rather than merely low-alpha.
    """
    return _upper_region_only(base_frame, animated_frame, bbox, strength, 0.36, 0.50)
