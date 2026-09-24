"""Deterministic authority resolution: writers, priority, masks, residual gating."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "AuthorityError",
    "AuthorityConfig",
    "AuthorityWriter",
    "AuthorityResult",
    "resolve_authority",
]

_MAX_AUTHORITY = 1.0


class AuthorityError(ValueError):
    """Raised for structurally invalid authority inputs."""


@dataclass(frozen=True)
class AuthorityConfig:
    """Static policy: canonical keys, default priority, residual gate."""

    keys: tuple[str, ...] = ("face", "root")
    default_priority: int = 0
    residual_gate: float = 0.0

    def __post_init__(self) -> None:
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise AuthorityError("keys must be non-empty and unique")
        if isinstance(self.default_priority, bool) or not isinstance(self.default_priority, int):
            raise AuthorityError("default_priority must be an integer")
        if (
            isinstance(self.residual_gate, bool)
            or not isinstance(self.residual_gate, (int, float))
            or not math.isfinite(float(self.residual_gate))
            or not 0.0 <= self.residual_gate <= _MAX_AUTHORITY
        ):
            raise AuthorityError("residual_gate must be a finite number between 0 and 1")
        object.__setattr__(self, "residual_gate", float(self.residual_gate))


@dataclass(frozen=True)
class AuthorityWriter:
    """One writer's contribution: values, mode, priority, seq, per-key mask."""

    name: str
    values: Mapping[str, float] = field(default_factory=dict)
    mode: str = "absolute"
    priority: int = 0
    seq: int = 0
    mask: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AuthorityError("writer name must be a non-empty string")
        if self.mode not in ("absolute", "additive"):
            raise AuthorityError("mode must be 'absolute' or 'additive'")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise AuthorityError("priority must be an integer")
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 0:
            raise AuthorityError("seq must be a non-negative integer")
        object.__setattr__(
            self, "values", {str(k): _unit(v, f"values[{k!r}]") for k, v in self.values.items()}
        )
        object.__setattr__(
            self, "mask", {str(k): _unit(v, f"mask[{k!r}]") for k, v in self.mask.items()}
        )


@dataclass(frozen=True)
class AuthorityResult:
    """Resolved authority state for one tick."""

    values: Mapping[str, float]
    winner: str | None
    residual: bool
    dropped: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "values": dict(sorted(self.values.items())),
            "winner": self.winner,
            "residual": self.residual,
            "dropped": dict(sorted(self.dropped.items())),
        }


def _unit(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(
        float(value),
    ):
        raise AuthorityError(f"{name} must be a finite number")
    number = float(value)
    if not 0.0 <= number <= _MAX_AUTHORITY:
        raise AuthorityError(f"{name} must be between 0 and 1")
    return number


def resolve_authority(
    config: AuthorityConfig,
    writers: Mapping[str, AuthorityWriter],
    previous: Mapping[str, float] | None = None,
) -> AuthorityResult:
    """Resolve one tick of authority deterministically.

    Rules (no wall-clock or hash-order dependence):

    1. A writer is dropped with reason ``"stale_seq"`` when its ``seq`` is not
       strictly greater than its own last admitted seq.
    2. Per key: additive writers apply first in (priority, seq, name) order,
       clamping the running value to [0, 1].
    3. Absolute writers then replace the running value; the highest
       (priority, seq, name) wins per key.
    4. Masks scale the contribution per key; a mask of 0 disables the writer
       for that key (residual value is used).
    5. Residual gating: if the winning absolute value is below
       ``config.residual_gate`` and a previous value exists, the previous
       value is retained and ``residual`` is set to True.
    6. Keys with no admitted writer fall through to the previous value when
       available, else 0.0.
    """
    known = set(config.keys)
    if not known:
        raise AuthorityError("config keys must not be empty")
    for writer in writers.values():
        unknown = (set(writer.values) | set(writer.mask)) - known
        if unknown:
            raise AuthorityError(
                f"writer {writer.name!r} references unknown keys: {sorted(unknown)}"
            )
    previous_map: dict[str, float] = {}
    if previous is not None:
        unknown = set(previous) - known
        if unknown:
            raise AuthorityError(f"previous references unknown keys: {sorted(unknown)}")
        previous_map = {k: _unit(previous[k], f"previous[{k!r}]") for k in previous}

    dropped: dict[str, str] = {}
    admitted: list[AuthorityWriter] = []
    last_seq: dict[str, int] = {}
    for name in sorted(writers):
        writer = writers[name]
        last = last_seq.get(name)
        if last is not None and writer.seq <= last:
            dropped[writer.name] = "stale_seq"
            continue
        last_seq[name] = writer.seq
        admitted.append(writer)

    out: dict[str, float] = {}
    residual = False
    winner: str | None = None
    for key in config.keys:
        base = previous_map.get(key, 0.0)
        additive = [
            w
            for w in admitted
            if w.mode == "additive" and key in w.values and w.mask.get(key, 1.0) > 0.0
        ]
        absolute = [
            w
            for w in admitted
            if w.mode == "absolute" and key in w.values and w.mask.get(key, 1.0) > 0.0
        ]
        if additive:
            running = base
            for writer in sorted(additive, key=lambda w: (w.priority, w.seq, w.name)):
                gate = writer.mask.get(key, 1.0)
                running = min(_MAX_AUTHORITY, max(0.0, running + writer.values[key] * gate))
        if absolute:
            top = max(absolute, key=lambda w: (w.priority, w.seq, w.name))
            value = min(_MAX_AUTHORITY, max(0.0, top.values[key] * top.mask.get(key, 1.0)))
            if value < config.residual_gate and key in previous_map:
                out[key] = previous_map[key]
                residual = True
            else:
                out[key] = value
            winner = top.name
        elif additive:
            out[key] = running
        elif key in previous_map:
            out[key] = previous_map[key]
        else:
            out[key] = 0.0
    return AuthorityResult(values=out, winner=winner, residual=residual, dropped=dropped)
