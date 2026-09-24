"""Fail-soft parsing for optional live-avatar sidecars."""

from __future__ import annotations

from collections.abc import Mapping


NEUTRAL_REACTION = {
    "reaction": "neutral",
    "strength": 0.0,
    "attack_ms": 200,
    "hold_ms": 300,
    "decay_ms": 600,
    "cooldown_ms": 800,
}


def optional_json(result: object, fallback: Mapping[str, object] | None = None) -> dict[str, object]:
    """Return a JSON object from a successful response, otherwise a safe copy."""

    safe = dict(fallback or {})
    if isinstance(result, BaseException) or not bool(getattr(result, "is_success", False)):
        return safe
    try:
        payload = result.json()  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        return safe
    return dict(payload) if isinstance(payload, Mapping) else safe
