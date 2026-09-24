"""Pure timing functions for the AdvancedLivePortrait source-bank controller."""

REACTION_SLOTS = {
    "amusement": (4, 5),
    "surprise": (6, 7),
    "skepticism": (8, 9),
    "concern": (10, 11),
    "agreement": (12, 12),
    "disagreement": (13, 13),
    "interest": (14, 14),
    "thinking": (15, 15),
}
QUIET_SLOTS = (0, 1, 3)
BLINK_SLOT = 2
HEAD_POSE_SLOTS = {
    # ALP's pupil_x and Euler yaw use opposite image-space signs on the
    # calibrated portrait: pupil_x=-4.5 moves the iris image-left while
    # yaw=-4 moves the nose image-right. Pair each eye aversion with the
    # rendered head endpoint that follows it visually.
    ("yaw", -1): 17,
    ("yaw", 1): 16,
    ("pitch", -1): 18,
    ("pitch", 1): 19,
}


def smoothstep(value):
    value = min(1.0, max(0.0, float(value)))
    return value * value * (3.0 - 2.0 * value)


def reaction_envelope(frame_index, video_num, fps, plan):
    """Continuous attack/hold/decay around a subtle cross-phrase expression floor."""
    if not plan or plan.get("reaction") not in REACTION_SLOTS:
        return 0.0
    duration = max(1.0 / fps, video_num / fps)
    attack = max(0.08, float(plan.get("attack_ms", 200)) / 1000.0)
    hold = max(0.0, float(plan.get("hold_ms", 320)) / 1000.0)
    decay = max(0.12, float(plan.get("decay_ms", 700)) / 1000.0)
    available = max(0.2, duration - 0.08)
    total = attack + hold + decay
    if total > available:
        scale = available / total
        attack, hold, decay = attack * scale, hold * scale, decay * scale
    time_s = frame_index / fps
    start_ratio = min(0.32, max(0.0, float(plan.get("start_ratio", 0.0))))
    end_ratio = min(0.32, max(0.0, float(plan.get("end_ratio", 0.18))))
    if time_s < attack:
        envelope = start_ratio + (1.0 - start_ratio) * smoothstep(time_s / attack)
    elif time_s < attack + hold:
        envelope = 1.0
    elif time_s < attack + hold + decay:
        envelope = 1.0 - (1.0 - end_ratio) * smoothstep((time_s - attack - hold) / decay)
    else:
        envelope = end_ratio
    strength = min(0.68, max(0.0, float(plan.get("strength", 0.0))))
    return strength * envelope


def blink_envelope(frame_index, fps, plan):
    """Independent asymmetric close/hold/open blink envelope."""
    time_ms = frame_index * 1000.0 / fps
    amount = 0.0
    for event in (plan or {}).get("blinks", []):
        elapsed = time_ms - float(event.get("at_ms", 0))
        close = max(35.0, float(event.get("close_ms", 55)))
        hold = max(0.0, float(event.get("hold_ms", 25)))
        opening = max(60.0, float(event.get("open_ms", 90)))
        if elapsed < 0 or elapsed >= close + hold + opening:
            continue
        if elapsed < close:
            value = smoothstep(elapsed / close)
        elif elapsed < close + hold:
            value = 1.0
        else:
            value = 1.0 - smoothstep((elapsed - close - hold) / opening)
        peak = min(1.0, max(0.65, float(event.get("peak", 1.0))))
        amount = max(amount, value * peak)
    return min(1.0, amount)


def gaze_envelope(frame_index, fps, plan):
    """Return the strongest smooth gaze-aversion event as (direction, amount)."""
    time_ms = frame_index * 1000.0 / fps
    strongest = (0, 0.0)
    for event in (plan or {}).get("gaze_events", []):
        elapsed = time_ms - float(event.get("at_ms", 0))
        move = max(60.0, float(event.get("move_ms", 95)))
        hold = max(120.0, float(event.get("hold_ms", 520)))
        returning = max(90.0, float(event.get("return_ms", 170)))
        if elapsed < 0 or elapsed >= move + hold + returning:
            continue
        if elapsed < move:
            amount = smoothstep(elapsed / move)
        elif elapsed < move + hold:
            amount = 1.0
        else:
            amount = 1.0 - smoothstep((elapsed - move - hold) / returning)
        amount *= min(1.0, max(0.20, float(event.get("amplitude", 1.0))))
        if amount > strongest[1]:
            direction = -1 if float(event.get("direction", -1)) < 0 else 1
            strongest = (direction, amount)
    return strongest


def head_pose_envelope(frame_index, fps, plan):
    """Return one sparse real-Euler head event as (axis, direction, amount)."""
    time_ms = frame_index * 1000.0 / fps
    strongest = (None, 0, 0.0)
    for event in (plan or {}).get("head_events", []):
        axis = str(event.get("axis", "yaw"))
        if axis not in {"yaw", "pitch"}:
            continue
        elapsed = time_ms - float(event.get("at_ms", 0))
        move = max(160.0, float(event.get("move_ms", 240)))
        hold = max(160.0, float(event.get("hold_ms", 420)))
        returning = max(240.0, float(event.get("return_ms", 420)))
        if elapsed < 0 or elapsed >= move + hold + returning:
            continue
        if elapsed < move:
            amount = smoothstep(elapsed / move)
        elif elapsed < move + hold:
            amount = 1.0
        else:
            amount = 1.0 - smoothstep((elapsed - move - hold) / returning)
        amount *= min(0.72, max(0.20, float(event.get("amplitude", 0.48))))
        if amount > strongest[2]:
            direction = -1 if float(event.get("direction", -1)) < 0 else 1
            strongest = (axis, direction, amount)
    return strongest
