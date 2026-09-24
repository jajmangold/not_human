"""Phrase-spanning physiological blink timing for the live portrait."""

from __future__ import annotations

import math
import random


class BlinkScheduler:
    """Place varied blinks on one continuous speech clock, not per phrase."""

    def __init__(self, seed: int, fps: int = 16):
        self._rng = random.Random(seed)
        self._frame_ms = 1000.0 / max(1, fps)
        self._cursor = 0.0
        self._next_blink = self._rng.uniform(2.4, 4.8)

    def _interval(self) -> float:
        # A bounded log-normal renewal process is less clock-like than a
        # uniform timer while retaining a hard refractory period.
        return min(8.2, max(2.6, self._rng.lognormvariate(1.48, 0.28)))

    def schedule(self, duration: float) -> list[dict[str, int]]:
        duration = max(0.0, float(duration))
        start, end = self._cursor, self._cursor + duration
        events: list[dict[str, int]] = []
        while self._next_blink < end:
            close_ms = self._rng.randint(55, 78)
            hold_ms = self._rng.randint(10, 24)
            open_ms = self._rng.randint(115, 155)
            local = max(0.16, self._next_blink - start)
            peak_ms = local * 1000.0 + close_ms
            aligned_peak_ms = round(peak_ms / self._frame_ms) * self._frame_ms
            minimum_peak_ms = math.ceil((160 + close_ms) / self._frame_ms) * self._frame_ms
            aligned_peak_ms = max(aligned_peak_ms, minimum_peak_ms)
            local = (aligned_peak_ms - close_ms) / 1000.0
            event_seconds = (close_ms + hold_ms + open_ms) / 1000.0
            if local + event_seconds <= duration - 0.06:
                events.append(
                    {
                        "at_ms": round(local * 1000),
                        "close_ms": close_ms,
                        "hold_ms": hold_ms,
                        "open_ms": open_ms,
                    }
                )
                self._next_blink = start + local + self._interval()
            else:
                # Carry a blink that landed on a phrase boundary into the next
                # phrase instead of either dropping it or pinning it to a cut.
                self._next_blink = end + 0.18
                break
        self._cursor = end + 0.06  # browser's intentional phrase start lead
        return events
