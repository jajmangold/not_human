"""Minimal client for the resident AdvancedLivePortrait ``/edit`` endpoint.

Factored out of scripts/calibrate_liveportrait.py so scripts/
measure_controllability.py (#30) does not duplicate it. Uses only the
standard library's HTTP client so it works in the same lightweight
mediapipe-only image the calibration/measurement scripts already run in --
no httpx/requests dependency to add there.
"""

from __future__ import annotations

import urllib.request
import uuid


def render_alp(url: str, portrait: bytes, controls: dict[str, float], timeout: float = 120.0) -> bytes:
    """POST one portrait + control vector to ALP's ``/edit`` and return the PNG."""
    boundary = f"----musetalk-alp-client-{uuid.uuid4().hex}".encode()
    fields: list[bytes] = []
    for name, value in controls.items():
        fields.extend([
            b"--" + boundary + b"\r\n",
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            str(value).encode() + b"\r\n",
        ])
    fields.extend([
        b"--" + boundary + b"\r\n",
        b'Content-Disposition: form-data; name="portrait"; filename="portrait.png"\r\n',
        b"Content-Type: image/png\r\n\r\n",
        portrait,
        b"\r\n--" + boundary + b"--\r\n",
    ])
    request = urllib.request.Request(
        url.rstrip("/") + "/edit", data=b"".join(fields), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError("ALP returned an empty frame")
    return payload
