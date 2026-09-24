import unittest

try:
    from kokoro.text_normalization import normalize_for_speech
except ModuleNotFoundError:  # num2words is installed in the Kokoro image, not the host.
    normalize_for_speech = None


@unittest.skipUnless(normalize_for_speech, "Kokoro runtime dependencies are not installed")
class KokoroNormalizationTests(unittest.TestCase):
    def test_phone_date_time_currency_percent_and_units(self):
        text = "Call (312) 555-0198 on 2026-09-05 at 10:03 AM. It costs $12.50 for 16GB at 99.5%."
        spoken = normalize_for_speech(text)
        self.assertIn("three one two, five five five, zero one nine eight", spoken)
        self.assertIn("September fifth, twenty twenty-six", spoken)
        self.assertIn("ten oh three A M", spoken)
        self.assertIn("twelve dollars and fifty cents", spoken)
        self.assertIn("sixteen gigabytes", spoken)
        self.assertIn("ninety-nine point five percent", spoken)

    def test_markdown_is_not_read_aloud(self):
        spoken = normalize_for_speech("Try [Kokoro](https://example.test) and `voice`.")
        self.assertEqual(spoken, "Try Kokoro and voice.")

    def test_markdown_emphasis_and_bullets_never_become_asterisks(self):
        spoken = normalize_for_speech("**Ready.**\n\n* First point.\n* Second point.")
        self.assertEqual(spoken, "Ready. First point. Second point.")

    def test_simple_multiplication_is_spoken_without_symbols(self):
        self.assertEqual(normalize_for_speech("Two * three equals six."), "Two times three equals six.")

    def test_audio_tags_and_ssml_are_not_spoken(self):
        spoken = normalize_for_speech("[laugh] Hello <break time='1s'/> (sigh) friend.")
        self.assertEqual(spoken, "Hello friend.")

    def test_ambiguous_or_invalid_formats_are_not_invented(self):
        spoken = normalize_for_speech("Keep 03/04/05, 2026-02-30, and 19:45 PM literal.")
        self.assertIn("03/04/05", spoken)
        self.assertIn("2026-02-30", spoken)
        self.assertIn("19:45 PM", spoken)

    def test_phone_extension_and_whole_currency(self):
        spoken = normalize_for_speech("Call +1 (312) 555-0198 ext. 42. Pay $1.00, not -$5.00.")
        self.assertIn("plus one, three one two, five five five, zero one nine eight, extension four two", spoken)
        self.assertIn("one dollar", spoken)
        self.assertIn("negative five dollars", spoken)

    def test_empty_and_pathological_numbers_are_safe(self):
        self.assertEqual(normalize_for_speech(""), "")
        huge = "9" * 100
        self.assertEqual(normalize_for_speech(f"Value {huge}GB."), f"Value {huge}GB.")


if __name__ == "__main__":
    unittest.main()
