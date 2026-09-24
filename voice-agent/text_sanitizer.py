"""Strip leaked tool-call syntax from LLM output text before it reaches
the transcript or TTS.

gemma4-26b has no native function-calling tokens, so the serving layer has
to *parse* its delimiter-based output into structured tool_calls -- and
that parsing is documented to be unreliable (matches llama-cpp-python#2227:
"Gemma tool calls returned as raw native tokens in content instead of
tool_calls"). Confirmed live on internal issue #45, three times now,
in three different leaked shapes -- the model doesn't pick one syntax and
stick to it:
- `set_reaction{reaction: 'thinking', strength: 0.5} Not much...`
- `set_reaction: thinking, 0.5\n\nIt's hard to say for sure...` (a real
  conversation: Kokoro spoke "set reaction thinking zero point five" out
  loud before the actual reply -- the original bracket-only regex didn't
  match this colon/no-brackets variant at all).
- Just the bare strength value, alone on its own line, with no
  "set_reaction" text and no reaction name anywhere nearby --
  `.5`/`.6`/`.3`/etc., each its own isolated paragraph, on nearly every
  turn of a real long conversation once reactions were flowing steadily.
  The reaction name apparently gets consumed correctly as a real tool
  call in this case; only the trailing strength argument spills through
  as plain text. A bare `-` shows up the same way on some turns, likely
  the same underlying spillover in a different partial shape.

This is a safety net, not the fix (the real fix is a corrected chat
template on the model-serving side); it's model-agnostic so it costs
nothing if the upstream issue is ever fixed.

The colon-style branch is deliberately anchored to the known reaction
vocabulary (system_prompt.py's fixed list) plus a numeric strength,
rather than matching up to the next sentence boundary -- an open-ended
"everything up to a period" match would also eat real spoken text on
the (observed-as-plausible) case where the model emits no separator at
all before continuing straight into the real reply on the same line.

The bare-fragment branch (just a leaked number or dash) is deliberately
scoped to a token that stands *entirely alone* on its own line -- anchored
by newlines (or start/end of stream) on both sides, matched via
MULTILINE. This is what keeps it from ever touching a real decimal
embedded in a real sentence ("about 0.5 miles from here"): that decimal
has real text sharing its line, so the anchors don't match. A real reply
consisting of nothing but a bare number or a lone dash is not a shape
this system prompt's rules allow (no bullets/symbols, plain spoken
sentences), so there's no legitimate content this could misfire on.

Buffers a bounded trailing window of already-cleaned text before emitting
so a leak split across multiple streamed chunks (e.g. "set_reaction{reac"
then "tion: 'thinking'...") still gets caught by the regex once both
pieces have arrived, rather than only ever seeing (and missing) a partial
match -- this also protects the bare-fragment branch from a false match
on a number that's merely mid-stream and hasn't seen its own line-ending
yet (the regex only resolves "isolated" once the actual trailing newline,
or true end of stream via flush(), has arrived in the buffer).
"""

from __future__ import annotations

import re

_REACTIONS = (
    "amusement|surprise|skepticism|concern|agreement|disagreement|interest|thinking|neutral"
)
LEAKED_TOOL_CALL_RE = re.compile(
    rf"\bset_reaction\s*(?:"
    rf"[\(\{{][^)}}]*[\)\}}]"  # set_reaction(...) / set_reaction{{...}}
    rf"|:?\s*(?:reaction\s*[:=]\s*)?['\"]?(?:{_REACTIONS})['\"]?"
    rf"\s*,\s*(?:strength\s*[:=]\s*)?[01](?:\.\d+)?"  # set_reaction: thinking, 0.5
    rf")\s*"
    # A bare strength number, or a lone dash, standing entirely alone on
    # its own line -- the spillover-only leak shape (see docstring).
    rf"|(?:(?<=\n)|(?<=\A))[ \t]*(?:[01]?\.\d+|-)[ \t]*(?=\n|\Z)",
    re.IGNORECASE | re.MULTILINE,
)
# Comfortably longer than the pattern itself; costs at most ~1 LLM chunk of
# extra latency before speech starts.
HOLDBACK_CHARS = 80


class LeakedToolCallSanitizer:
    """Stateful, streaming-safe stripper. Feed it text deltas as they
    arrive; it returns only the portion that's safe to emit right now.
    Call flush() once the stream ends to get whatever's left."""

    def __init__(self, holdback_chars: int = HOLDBACK_CHARS) -> None:
        self._holdback_chars = holdback_chars
        self._buffer = ""

    def feed(self, text: str) -> str:
        self._buffer += text
        cleaned = LEAKED_TOOL_CALL_RE.sub("", self._buffer)
        if len(cleaned) > self._holdback_chars:
            to_emit, self._buffer = cleaned[:-self._holdback_chars], cleaned[-self._holdback_chars:]
            return to_emit
        self._buffer = cleaned
        return ""

    def flush(self) -> str:
        remaining, self._buffer = self._buffer, ""
        return remaining
