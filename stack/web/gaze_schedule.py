"""Phrase-spanning fixation timing for a live conversational portrait."""

from __future__ import annotations

import random


class GazeScheduler:
    """Schedule sparse gaze aversions on one continuous speech clock."""

    def __init__(self, seed: int):
        self._rng = random.Random(seed)
        self._cursor = 0.0
        self._next_shift = self._rng.uniform(1.4, 3.2)

    def _interval(self) -> float:
        return self._rng.uniform(1.8, 4.6)

    def schedule(self, duration: float) -> list[dict[str, int | float]]:
        duration = max(0.0, float(duration))
        start, end = self._cursor, self._cursor + duration
        events: list[dict[str, int | float]] = []
        while self._next_shift < end:
            local = max(0.18, self._next_shift - start)
            move_ms = self._rng.randint(75, 125)
            hold_ms = self._rng.randint(380, 920)
            return_ms = self._rng.randint(130, 220)
            total = (move_ms + hold_ms + return_ms) / 1000.0
            if local + total <= duration - 0.08:
                events.append({
                    "at_ms": round(local * 1000),
                    "move_ms": move_ms,
                    "hold_ms": hold_ms,
                    "return_ms": return_ms,
                    "direction": -1 if self._rng.random() < 0.5 else 1,
                })
                self._next_shift = start + local + total + self._interval()
            else:
                self._next_shift = end + self._rng.uniform(0.35, 1.0)
                break
        self._cursor = end + 0.06
        return events
