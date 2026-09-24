"""Coupled conversational blink, gaze, and low-frequency motion scheduling.

Ported verbatim from musetalk-volta's `/ws/live/` demo (web/natural_motion.py,
internal issue #45's "why is it still crappy" follow-up). This is what
actually drives the avatar's blinking and gaze/head micro-motion during
speech -- muse_server.py's resident-side compositor (motion_compositor.py's
gaze_region_only/upper_face_only, reaction_schedule.py's blink_envelope/
gaze_envelope/head_pose_envelope) already reads `blinks`/`gaze_events`/
`head_events` off any job's `reaction` payload; nothing in this agent ever
populated them before this, so the avatar never blinked or moved its gaze
regardless of how well the facial-expression reaction itself rendered.

One instance must live for a whole session (its `_cursor` clock advances
across `schedule()` calls to keep blink/gaze timing continuous and
non-repetitive across phrases) -- not one instance per phrase, which would
restart the random clock every phrase and cluster events near each phrase's
start instead of spreading naturally across the whole conversation.
"""

from __future__ import annotations

import random


STATES = {
    "attend": {"dwell": (2.4, 5.8), "next": ("consider", "settle"), "gain": 0.55},
    "consider": {"dwell": (1.6, 3.8), "next": ("attend", "settle"), "gain": 0.72},
    "settle": {"dwell": (1.2, 2.8), "next": ("attend",), "gain": 0.34},
}


class NaturalMotionScheduler:
    """One persistent clock for events that must not be sampled independently.

    The HSMM-like state controls dwell and amplitude.  Gaze events are placed
    first, then blinks are either kept away from the pre-saccadic interval or
    probabilistically coupled to the slower head-follow portion of the event.
    """

    def __init__(self, seed: int, fps: int = 16):
        self._rng = random.Random(seed)
        self._frame_ms = 1000.0 / max(1, fps)
        self._cursor = 0.0
        self._state = "attend"
        self._state_until = self._rng.uniform(*STATES[self._state]["dwell"])
        self._next_blink = self._rng.uniform(2.8, 5.6)
        self._next_gaze = self._rng.uniform(1.8, 4.2)

    def _blink_interval(self, pause_ratio: float) -> float:
        # Pauses modestly increase opportunities without turning punctuation
        # into a compulsory blink.
        base = self._rng.lognormvariate(1.55, 0.32)
        return min(9.0, max(2.8, base * (1.0 - 0.12 * pause_ratio)))

    def _gaze_interval(self, context: dict[str, float]) -> float:
        focus = max(0.0, min(1.0, float(context.get("gaze_focus", 0.0))))
        return self._rng.uniform(2.5, 6.2) * (1.0 + 0.45 * focus)

    def _advance_state(self, absolute_time: float, context: dict[str, float]) -> None:
        while absolute_time >= self._state_until:
            choices = list(STATES[self._state]["next"])
            if float(context.get("gaze_focus", 0.0)) > 0.12:
                choices.extend(("attend", "attend"))
            if abs(float(context.get("gaze_x", 0.0))) > 0.08:
                choices.append("consider")
            self._state = self._rng.choice(choices)
            low, high = STATES[self._state]["dwell"]
            self._state_until += self._rng.uniform(low, high)

    def schedule(
        self,
        duration: float,
        prosody: dict[str, float] | None = None,
        context: dict[str, float] | None = None,
    ) -> dict[str, object]:
        duration = max(0.0, float(duration))
        prosody, context = prosody or {}, context or {}
        pause_ratio = max(0.0, min(1.0, float(prosody.get("pause_ratio", 0.0))))
        start, end = self._cursor, self._cursor + duration
        self._advance_state(start, context)

        gaze_events: list[dict[str, int | float]] = []
        while self._next_gaze < end:
            local = max(0.20, self._next_gaze - start)
            move_ms = self._rng.randint(70, 105)
            hold_ms = self._rng.randint(420, 980)
            return_ms = self._rng.randint(150, 250)
            total = (move_ms + hold_ms + return_ms) / 1000.0
            if local + total <= duration - 0.10:
                gaze_events.append({
                    "at_ms": round(local * 1000),
                    "move_ms": move_ms,
                    "hold_ms": hold_ms,
                    "return_ms": return_ms,
                    "direction": -1 if self._rng.random() < 0.5 else 1,
                    "head_follow_ms": self._rng.randint(170, 310),
                    "amplitude": round(self._rng.uniform(0.46, 0.74), 3),
                })
                self._next_gaze = start + local + total + self._gaze_interval(context)
            else:
                self._next_gaze = end + self._rng.uniform(0.5, 1.4)
                break

        # Real head pose follows selected gaze shifts after the eyes acquire
        # their target. It is sparse and phrase-bounded; there is no oscillator
        # and no event is allowed to snap unfinished across a phrase boundary.
        head_events: list[dict[str, int | float | str]] = []
        nod = max(0.0, min(0.45, float(context.get("nod", 0.0))))
        if nod >= 0.05 and duration >= 0.82:
            event = {
                "axis": "pitch", "direction": 1, "at_ms": 100,
                "move_ms": 180, "hold_ms": 160, "return_ms": 280,
                "amplitude": round(min(0.58, 0.30 + nod * 0.72), 3),
                "source": "semantic-nod",
            }
            if sum(float(event[key]) for key in ("at_ms", "move_ms", "hold_ms", "return_ms")) <= duration * 1000 - 50:
                head_events.append(event)
        if not head_events:
            for gaze in gaze_events:
                if self._rng.random() >= 0.68:
                    continue
                at_ms = int(gaze["at_ms"]) + int(gaze["head_follow_ms"])
                move_ms = self._rng.randint(210, 290)
                hold_ms = self._rng.randint(180, 380)
                return_ms = self._rng.randint(320, 480)
                if at_ms + move_ms + hold_ms + return_ms > duration * 1000 - 50:
                    continue
                head_events.append({
                    "axis": "yaw", "direction": int(gaze["direction"]),
                    "at_ms": at_ms, "move_ms": move_ms, "hold_ms": hold_ms,
                    "return_ms": return_ms,
                    "amplitude": round(self._rng.uniform(0.38, 0.58), 3),
                    "source": "gaze-follow",
                })

        blinks: list[dict[str, int | bool]] = []
        while self._next_blink < end:
            local = max(0.18, self._next_blink - start)
            coupled = False
            close_ms = self._rng.randint(52, 72)
            hold_ms = self._rng.randint(6, 18)
            open_ms = self._rng.randint(105, 145)
            peak_ms = local * 1000.0 + close_ms
            peak_ms = round(peak_ms / self._frame_ms) * self._frame_ms
            local = max(0.18, (peak_ms - close_ms) / 1000.0)
            # Re-check after frame alignment and after every displacement: a
            # phrase can contain multiple gaze events, and moving past one can
            # otherwise place the blink just before the next one.
            for _ in range(len(gaze_events) + 1):
                conflict = next((
                    gaze for gaze in gaze_events
                    if float(gaze["at_ms"]) / 1000.0 - 0.50
                    <= local < float(gaze["at_ms"]) / 1000.0
                ), None)
                if conflict is None:
                    break
                gaze_at = float(conflict["at_ms"]) / 1000.0
                if self._rng.random() < 0.58:
                    local = gaze_at + (float(conflict["move_ms"]) + 45.0) / 1000.0
                    coupled = True
                else:
                    local = gaze_at + (
                        float(conflict["move_ms"]) + float(conflict["hold_ms"]) + 120.0
                    ) / 1000.0
                peak_ms = round((local * 1000.0 + close_ms) / self._frame_ms) * self._frame_ms
                local = max(0.18, (peak_ms - close_ms) / 1000.0)
            event_seconds = (close_ms + hold_ms + open_ms) / 1000.0
            if local + event_seconds <= duration - 0.07:
                blinks.append({
                    "at_ms": round(local * 1000), "close_ms": close_ms,
                    "hold_ms": hold_ms, "open_ms": open_ms, "coupled": coupled,
                    "peak": round(self._rng.uniform(0.86, 0.97), 3),
                })
                self._next_blink = start + local + self._blink_interval(pause_ratio)
            else:
                self._next_blink = end + 0.22
                break

        # Two correlated latent factors map to all residual axes in the client;
        # these are session-stable, low-amplitude biases rather than gestures.
        orient = max(-1.0, min(1.0, self._rng.gauss(0.0, 0.32)))
        engagement = max(-1.0, min(1.0, self._rng.gauss(0.0, 0.24)))
        asymmetry = max(-1.0, min(1.0, 0.55 * orient + self._rng.gauss(0.0, 0.18)))
        gain = STATES[self._state]["gain"] * (0.82 + 0.18 * float(prosody.get("energy", 0.0)))
        result = {
            "state": self._state,
            "state_remaining_ms": max(0, round((self._state_until - start) * 1000)),
            "gain": round(gain, 4),
            "latent": {
                "orient": round(orient, 4),
                "engagement": round(engagement, 4),
                "asymmetry": round(asymmetry, 4),
            },
            "blinks": blinks,
            "gaze_events": gaze_events,
            "head_events": head_events,
        }
        self._cursor = end + 0.06
        return result
