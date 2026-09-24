"""Unit tests for LeakedToolCallSanitizer.

Extracted from agent.py's llm_node so this logic is independently
testable -- previously it could only be exercised by hand-running
python3 -c snippets during live debugging (internal issue #45).
"""

from __future__ import annotations

from text_sanitizer import LeakedToolCallSanitizer


def _feed_all(chunks: list[str]) -> str:
    sanitizer = LeakedToolCallSanitizer()
    out = "".join(sanitizer.feed(chunk) for chunk in chunks)
    out += sanitizer.flush()
    return out


def test_exact_reported_leak_is_stripped():
    text = (
        "set_reaction{reaction: 'thinking', strength: 0.5} Not much. "
        "I'm just here and ready to chat. Is there something on your mind?"
    )
    assert _feed_all([text]) == (
        "Not much. I'm just here and ready to chat. Is there something on your mind?"
    )


def test_leak_split_across_many_small_chunks_is_still_caught():
    text = "set_reaction{reaction: 'thinking', strength: 0.5} Not much. I'm just here."
    chunks = [text[i:i + 3] for i in range(0, len(text), 3)]
    assert _feed_all(chunks) == "Not much. I'm just here."


def test_clean_reply_passes_through_untouched():
    text = "That's a classic sentence. It's got every letter in the alphabet."
    assert _feed_all([text]) == text


def test_leak_in_the_middle_of_a_reply_is_stripped():
    text = "Sure thing. set_reaction{reaction: 'amusement', strength: 0.7} Here's a joke for you."
    assert _feed_all([text]) == "Sure thing. Here's a joke for you."


def test_multiple_leaks_in_one_reply_are_all_stripped():
    text = (
        "set_reaction{reaction: 'interest', strength: 0.4} First. "
        "set_reaction{reaction: 'amusement', strength: 0.6} Second."
    )
    assert _feed_all([text]) == "First. Second."


def test_empty_feed_returns_empty():
    sanitizer = LeakedToolCallSanitizer()
    assert sanitizer.feed("") == ""
    assert sanitizer.flush() == ""


def test_short_text_is_held_back_until_flush():
    sanitizer = LeakedToolCallSanitizer(holdback_chars=80)
    # Shorter than the holdback window -- nothing should emit yet, in case
    # a leak is still forming across the next chunk.
    assert sanitizer.feed("Hi there") == ""
    assert sanitizer.flush() == "Hi there"


def test_colon_style_leak_is_stripped():
    # The second real live leak (internal issue #45): no brackets at
    # all, just "set_reaction: <reaction>, <strength>" -- the original
    # bracket-only regex missed this entirely and Kokoro spoke it aloud.
    text = (
        "set_reaction: thinking, 0.5\n\n"
        "It's hard to say for sure. They've got a lot of new faces on the roster."
    )
    assert _feed_all([text]) == (
        "It's hard to say for sure. They've got a lot of new faces on the roster."
    )


def test_colon_style_leak_with_no_separator_before_real_text():
    # The riskiest case for a vocabulary-anchored (vs. sentence-boundary)
    # match: no newline or punctuation between the leak and the real
    # reply at all. Must stop right after the strength number, not eat
    # into the real sentence that follows on the same line.
    text = "set_reaction: thinking, 0.5 It's hard to say for sure."
    assert _feed_all([text]) == "It's hard to say for sure."


def test_colon_style_leak_without_labels_or_quotes():
    text = "set_reaction: amusement, 0.7 That's a good one."
    assert _feed_all([text]) == "That's a good one."


def test_unrelated_colon_and_number_is_not_touched():
    # Must not false-positive on ordinary text that happens to contain a
    # colon followed by a number -- the vocabulary anchor is what keeps
    # this safe, confirm it actually holds.
    text = "The score was tied: 2, 2 heading into the fourth quarter."
    assert _feed_all([text]) == text


def test_bare_strength_fragment_alone_on_its_own_line_is_stripped():
    # The third real live leak shape (internal issue #45): no
    # "set_reaction" text and no reaction name anywhere -- just the bare
    # strength value, alone on its own line, appearing on nearly every
    # turn of a real long conversation. Kokoro spoke "point five" out
    # loud, standalone, before the real reply.
    text = "What's the most fun way for you to talk?\n\n.5\n\nMe think hard."
    out = _feed_all([text])
    assert ".5" not in out
    assert "Me think hard." in out
    assert "What's the most fun way for you to talk?" in out


def test_bare_dash_fragment_alone_on_its_own_line_is_stripped():
    # A lone '-' shows up the same way on some turns -- likely the same
    # spillover in a different partial shape.
    text = "Tell me a story. I agree.\n\n-\n\n.5\n\nI require more input."
    out = _feed_all([text])
    assert "\n-\n" not in out
    assert ".5" not in out
    assert "I require more input." in out


def test_bare_fragment_at_start_of_stream_is_stripped():
    text = ".3\n\nMe confused brain. Not quite caught."
    out = _feed_all([text])
    assert not out.lstrip().startswith(".3")
    assert "Me confused brain." in out


def test_embedded_decimal_mid_sentence_is_not_touched():
    # The bare-fragment branch must only fire when the number stands
    # entirely alone on its own line -- a real decimal sharing a line
    # with real text must never be touched.
    text = "The answer is about 0.5 miles from here."
    assert _feed_all([text]) == text


def test_embedded_dash_mid_sentence_is_not_touched():
    text = "Some things - like this - use dashes mid-sentence."
    assert _feed_all([text]) == text
