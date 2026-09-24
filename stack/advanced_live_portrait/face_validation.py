"""Small, dependency-free face-box validation shared with unit tests."""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite


class NoUsableFaceError(ValueError):
    """The portrait does not contain a face large enough for animation."""


def require_usable_face(boxes: Iterable[Iterable[float]], minimum_size: float = 30.0) -> None:
    """Require one finite face box meeting the upstream animator's size gate."""
    for box in boxes:
        values = list(box)
        if len(values) < 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in values[:4])
        if all(isfinite(value) for value in (x1, y1, x2, y2)):
            if x2 - x1 >= minimum_size and y2 - y1 >= minimum_size:
                return
    raise NoUsableFaceError(
        "no usable face detected; upload a clear, front-facing portrait with one visible face"
    )
