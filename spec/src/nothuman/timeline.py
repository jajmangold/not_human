"""Deterministic 50 Hz timestamped timeline buffer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TimelineConfig:
    """Policy for a fixed-capacity timeline."""

    capacity: int
    max_late: int
    max_reorder: int
    interpolate: bool = True

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be >= 1")
        if self.max_late < 0:
            raise ValueError("max_late must be >= 0")
        if self.max_reorder < 0:
            raise ValueError("max_reorder must be >= 0")
        if self.max_reorder > self.max_late:
            raise ValueError("max_reorder must be <= max_late")


@dataclass(frozen=True)
class TimelineMetrics:
    accepted: int = 0
    late_drops: int = 0
    reordered_drops: int = 0
    future_drops: int = 0
    overwritten: int = 0


@dataclass(frozen=True)
class TimelineFrame:
    timestamp: int
    seq: int
    value: Any


@dataclass
class TimelineBuffer:
    """Ring buffer with bounded late, future, and reordered packet policy."""

    config: TimelineConfig
    frames: list[TimelineFrame | None] = field(default_factory=list)
    _last_timestamp: int | None = None
    _metrics: TimelineMetrics = field(default_factory=TimelineMetrics)

    def __post_init__(self) -> None:
        self.frames = [None] * self.config.capacity
        self._last_timestamp = None
        self._metrics = TimelineMetrics()

    def _inc(self, **kwargs: int) -> None:
        m = self._metrics
        self._metrics = TimelineMetrics(
            accepted=m.accepted + kwargs.get("accepted", 0),
            late_drops=m.late_drops + kwargs.get("late_drops", 0),
            reordered_drops=m.reordered_drops + kwargs.get("reordered_drops", 0),
            future_drops=m.future_drops + kwargs.get("future_drops", 0),
            overwritten=m.overwritten + kwargs.get("overwritten", 0),
        )

    def push(self, timestamp: int, value: Any) -> bool:
        """Store a packet when it falls within the configured timing policy."""
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError("timestamp must be a non-negative integer")
        last = self._last_timestamp
        if last is not None:
            gap = last - timestamp
            if gap == 0:
                self._inc(late_drops=1)
                return False
            if gap > 0:
                if gap > self.config.max_late:
                    self._inc(late_drops=1)
                    return False
                if gap > self.config.max_reorder:
                    self._inc(reordered_drops=1)
                    return False
            elif -gap > self.config.max_late and not (
                self.config.capacity == 1 and timestamp == last + 1
            ):
                self._inc(future_drops=1)
                return False
        self._store(timestamp, value)
        self._inc(accepted=1)
        if last is None or timestamp > last:
            self._last_timestamp = timestamp
        return True

    def _store(self, timestamp: int, value: Any) -> None:
        slot = timestamp % self.config.capacity
        if self.frames[slot] is not None:
            self._inc(overwritten=1)
        self.frames[slot] = TimelineFrame(timestamp, timestamp, value)

    def sample(self, timestamp: int) -> Any:
        """Sample with linear interpolation for numeric payloads."""
        if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
            raise ValueError("timestamp must be a non-negative integer")
        retained = sorted((f for f in self.frames if f is not None), key=lambda f: f.timestamp)
        if not retained:
            raise LookupError("timeline is empty")
        if timestamp <= retained[0].timestamp:
            return retained[0].value
        if timestamp >= retained[-1].timestamp:
            return retained[-1].value
        before = max((f for f in retained if f.timestamp <= timestamp), key=lambda f: f.timestamp)
        after = min((f for f in retained if f.timestamp > timestamp), key=lambda f: f.timestamp)
        if not self.config.interpolate:
            return before.value
        if isinstance(before.value, (int, float)) and isinstance(after.value, (int, float)):
            fraction = (timestamp - before.timestamp) / (after.timestamp - before.timestamp)
            return before.value + (after.value - before.value) * fraction
        return before.value

    @property
    def metrics(self) -> TimelineMetrics:
        return self._metrics

    def reset(self) -> None:
        self.frames = [None] * self.config.capacity
        self._last_timestamp = None
        self._metrics = TimelineMetrics()

    def __len__(self) -> int:
        return sum(frame is not None for frame in self.frames)
