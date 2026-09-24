"""Unit tests for speech_phrases.release_speech_phrases.

Formalizes what was checked ad hoc during development (internal issue #45
latency investigation) into a real regression suite -- this and
test_reaction_plan.py are the two modules ported verbatim from
musetalk-volta's proven `/ws/live/` demo, and a change here that silently
diverges from that demo's tuned behavior is exactly the kind of bug that
would otherwise only surface as "it feels different" during a live demo.
"""

from __future__ import annotations

from speech_phrases import release_speech_phrases


def _drain_stream(chunks: list[str]) -> list[str]:
    """Simulate an LLM token stream arriving incrementally, the same way
    llm_node receives it, and flush whatever's left at the end."""
    buffer = ""
    released: list[str] = []
    for chunk in chunks:
        buffer += chunk
        new, buffer = release_speech_phrases(buffer, False)
        released.extend(new)
    final, buffer = release_speech_phrases(buffer, True)
    released.extend(final)
    assert buffer == ""
    return released


def test_short_sentence_waits_for_the_next_one():
    # A single short sentence under the 50-char threshold should NOT be
    # released alone mid-stream -- it should merge with what follows.
    released, buffer = release_speech_phrases("Hi there.", final=False)
    assert released == []
    assert "Hi there." in buffer


def test_two_short_sentences_merge_into_one_release():
    released = _drain_stream(["Hi there.", " How are you doing today?"])
    assert released == ["Hi there. How are you doing today?"]


def test_long_sentence_releases_alone():
    text = "That's a genuinely long and complete sentence about nothing in particular."
    released = _drain_stream([text])
    assert released == [text]


def test_multiple_long_sentences_release_separately_in_order():
    s1 = "That's a classic sentence, used to show off every letter in the alphabet."
    s2 = "Are you practicing your typing, or just curious about pangrams?"
    released = _drain_stream([s1, " ", s2])
    assert released == [s1, s2]


def test_streaming_token_by_token_matches_whole_text_release():
    text = "That's a classic sentence. It's got every letter of the alphabet in it."
    token_chunks = [text[i:i + 3] for i in range(0, len(text), 3)]
    streamed = _drain_stream(token_chunks)
    whole, _ = release_speech_phrases(text, final=True)
    assert streamed == whole


def test_final_flush_releases_a_short_trailing_fragment():
    # No terminal punctuation at all (e.g. truncated by max_tokens) --
    # final=True must still flush it rather than losing it silently.
    released, buffer = release_speech_phrases("and then he said", final=True)
    assert released == ["and then he said"]
    assert buffer == ""


def test_long_run_on_without_punctuation_splits_at_a_word_boundary():
    long_run_on = " ".join(["word"] * 80)  # 399 chars, no sentence punctuation
    released, buffer = release_speech_phrases(long_run_on, final=False)
    assert released, "a run-on well past 220 chars must release something"
    for phrase in released:
        assert len(phrase) <= 220
    # Nothing should be silently dropped: released + remaining buffer must
    # reconstruct back to (approximately) the original word sequence.
    reconstructed = " ".join(released + [buffer]).split()
    assert reconstructed == long_run_on.split()


def test_empty_input_releases_nothing():
    released, buffer = release_speech_phrases("", final=False)
    assert released == []
    assert buffer == ""
    released, buffer = release_speech_phrases("", final=True)
    assert released == []
    assert buffer == ""


def test_custom_thresholds_merge_into_fewer_larger_phrases():
    # internal issue #45's throughput follow-up: agent.py overrides the
    # defaults larger (200/420) to reduce how many separate MuseTalk jobs a
    # long reply produces -- each job pays a roughly fixed per-job overhead
    # (measured live: ~27% of total resident time across one real
    # conversation), so fewer/larger phrases meaningfully cut total load.
    # A naive threshold bump (e.g. 120) barely helps once individual
    # sentences already exceed it alone (each one releases immediately
    # without ever trying to merge with its neighbor) -- 200/420 was chosen
    # because it actually halves the phrase count on realistic prose, not
    # just on paper.
    text = (
        "That is a really interesting question, and honestly there are a few different ways to think about it. "
        "One way is to consider the historical context, since a lot of what we see today traces back to decisions "
        "made decades ago. Another angle is the practical side of things, because in real-world use these "
        "considerations often outweigh the theoretical ones. I think the most helpful way to frame it, though, "
        "is to focus on what actually changes the outcome for someone using this every day."
    )
    default_released, _ = release_speech_phrases(text, final=True)
    larger_released, _ = release_speech_phrases(
        text, final=True, min_phrase_chars=200, max_phrase_chars=420,
    )
    assert len(larger_released) <= len(default_released) // 2, (
        f"expected roughly half as many phrases, got {len(default_released)} -> {len(larger_released)}"
    )
    for phrase in larger_released:
        assert len(phrase) <= 420
    # Nothing lost -- reconstructs back to the same words either way.
    assert " ".join(larger_released).split() == text.split()


def test_custom_max_phrase_chars_still_splits_a_run_on_correctly():
    long_run_on = " ".join(["word"] * 150)  # far past any reasonable threshold
    released, buffer = release_speech_phrases(
        long_run_on, final=False, min_phrase_chars=200, max_phrase_chars=420,
    )
    assert released, "a run-on well past max_phrase_chars must release something"
    for phrase in released:
        assert len(phrase) <= 420
    reconstructed = " ".join(released + [buffer]).split()
    assert reconstructed == long_run_on.split()
