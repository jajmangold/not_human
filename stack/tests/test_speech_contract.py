import unittest

from web.speech_contract import QWEN38_SYSTEM_PROMPT, qwen38_request, release_speech_phrases


class SpeechContractTests(unittest.TestCase):
    def test_exact_qwen38_nonthinking_profile(self):
        request = qwen38_request("qwen27b", "hello", 123)
        self.assertEqual(request["temperature"], 0.7)
        self.assertEqual(request["top_p"], 0.8)
        self.assertEqual(request["top_k"], 20)
        self.assertEqual(request["min_p"], 0.0)
        self.assertEqual(request["presence_penalty"], 1.5)
        self.assertEqual(request["repeat_penalty"], 1.0)
        self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": False, "preserve_thinking": False})
        self.assertIn("Kokoro", QWEN38_SYSTEM_PROMPT)
        self.assertIn("Never emit formatting characters", QWEN38_SYSTEM_PROMPT)
        self.assertIn('do not say "asterisk"', QWEN38_SYSTEM_PROMPT)

    def test_short_sentence_waits_for_breath_sized_context(self):
        phrases, remainder = release_speech_phrases("Yes, absolutely. Here is why it works so well in practice. ", False)
        self.assertEqual(phrases, ["Yes, absolutely. Here is why it works so well in practice."])
        self.assertEqual(remainder, "")

    def test_short_final_answer_is_not_lost(self):
        phrases, remainder = release_speech_phrases("That sounds great.", True)
        self.assertEqual(phrases, ["That sounds great."])
        self.assertEqual(remainder, "")

    def test_long_runon_releases_at_clause_boundary(self):
        text = "This is a natural opening clause, " + "with enough ordinary spoken words " * 9
        phrases, remainder = release_speech_phrases(text, False)
        self.assertTrue(phrases)
        self.assertLessEqual(len(phrases[0]), 220)
        self.assertTrue(remainder)

    def test_fragmented_stream_preserves_word_boundaries(self):
        buffer = ""
        released = []
        for fragment in (
            "You're set for September 5, 2026 at 10:03 AM. ",
            "Your confirmation number is ",
            "312-555-0198, and the price is $12.50.",
        ):
            buffer += fragment
            phrases, buffer = release_speech_phrases(buffer, False)
            released.extend(phrases)
        phrases, buffer = release_speech_phrases(buffer, True)
        released.extend(phrases)
        self.assertEqual(" ".join(released), (
            "You're set for September 5, 2026 at 10:03 AM. "
            "Your confirmation number is 312-555-0198, and the price is $12.50."
        ))

    def test_every_fragment_size_preserves_decimal_punctuation(self):
        text = "It costs $12.50. That includes everything you need for the whole day."
        for fragment_size in range(1, 20):
            buffer, released = "", []
            for start in range(0, len(text), fragment_size):
                buffer += text[start:start + fragment_size]
                phrases, buffer = release_speech_phrases(buffer, False)
                released.extend(phrases)
            phrases, buffer = release_speech_phrases(buffer, True)
            released.extend(phrases)
            self.assertEqual(" ".join(released), text, fragment_size)


if __name__ == "__main__":
    unittest.main()
