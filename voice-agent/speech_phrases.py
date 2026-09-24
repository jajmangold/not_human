"""Incremental phrase-boundary detection for streaming LLM text into TTS.

Ported from musetalk-volta's web/speech_contract.py (its `/ws/live/` demo,
the "previous demo site" confirmed to feel near-instant next to this
LiveKit agent -- internal issue #45). The regexes are that demo's own
tuning, copied verbatim. The length thresholds (`min_phrase_chars`/
`max_phrase_chars`) are parameterized rather than hardcoded, defaulting to
the old demo's own values (50/220) -- but this agent overrides them larger
(see agent.py), a deliberate divergence: live testing of a real multi-turn
conversation measured MuseTalk's ~1.3-2s roughly-fixed per-job overhead
eating ~27% of total resident time (20s of 75s across one 15-phrase call),
and smaller/more-numerous phrases pay that fixed tax more often for the
same amount of speech. Larger phrases trade a slightly later first-audio
moment for real throughput headroom against backlog during a sustained
conversation.
"""

from __future__ import annotations

import re

_SENTENCE_STREAM = re.compile(r'^(.+?[.!?…](?:["\'”’»])?)\s+', re.S)
_SENTENCE_FINAL = re.compile(r'^(.+?[.!?…](?:["\'”’»])?)(?:\s+|$)', re.S)

DEFAULT_MIN_PHRASE_CHARS = 50
DEFAULT_MAX_PHRASE_CHARS = 220


def release_speech_phrases(
    buffer: str,
    final: bool,
    *,
    min_phrase_chars: int = DEFAULT_MIN_PHRASE_CHARS,
    max_phrase_chars: int = DEFAULT_MAX_PHRASE_CHARS,
) -> tuple[list[str], str]:
    """Release breath-sized prose while retaining short sentences for context.

    Kokoro voices are weakest on very short utterances. Complete sentences
    remain the preferred boundary, but a short sentence waits for a
    following sentence unless this is the final flush. Long run-ons split
    at a clause or word boundary.
    """
    released: list[str] = []
    pending = ""
    while True:
        sentence_pattern = _SENTENCE_FINAL if final else _SENTENCE_STREAM
        match = sentence_pattern.search(buffer)
        if not match:
            break
        sentence = match.group(1).strip()
        candidate = f"{pending} {sentence}".strip()
        remainder = buffer[match.end():]
        buffer = remainder
        pending = candidate
        if len(pending) >= min_phrase_chars:
            released.append(pending)
            pending = ""

    if pending:
        # `_SENTENCE` consumes the separator after a held sentence. Reinsert it
        # even when no following delta has arrived yet, or the next token joins
        # directly to the sentence-ending punctuation.
        separator = "" if buffer and buffer[0].isspace() else " "
        buffer = pending + separator + buffer

    cut_floor = max(1, round(max_phrase_chars * 90 / 220))
    while len(buffer) >= max_phrase_chars:
        cut = max(
            buffer.rfind(", ", cut_floor, max_phrase_chars),
            buffer.rfind("; ", cut_floor, max_phrase_chars),
        )
        if cut < cut_floor:
            cut = buffer.rfind(" ", round(max_phrase_chars * 120 / 220), max_phrase_chars)
        if cut < cut_floor:
            released.append(buffer[:max_phrase_chars].strip())
            buffer = buffer[max_phrase_chars:].lstrip()
            continue
        released.append(buffer[:cut + (1 if buffer[cut] in ",;" else 0)].strip())
        buffer = buffer[cut + 1:].lstrip()

    if final and buffer.strip():
        released.append(buffer.strip())
        buffer = ""

    return released, buffer
