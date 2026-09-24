import unittest
from pathlib import Path

from web.optional_services import NEUTRAL_REACTION, optional_json


class FakeResponse:
    def __init__(self, payload=None, success=True, error=None):
        self.payload = payload
        self.is_success = success
        self.error = error

    def json(self):
        if self.error:
            raise self.error
        return self.payload


class OptionalServiceTests(unittest.TestCase):
    def test_successful_mapping_is_copied(self):
        payload = {"reaction": "interest", "strength": 0.4}
        self.assertEqual(optional_json(FakeResponse(payload)), payload)

    def test_exception_http_error_invalid_json_and_non_object_fail_soft(self):
        fallback = NEUTRAL_REACTION
        cases = [
            ConnectionError("offline"),
            FakeResponse({}, success=False),
            FakeResponse(error=ValueError("bad json")),
            FakeResponse(["not", "an", "object"]),
        ]
        for result in cases:
            with self.subTest(result=result):
                parsed = optional_json(result, fallback)
                self.assertEqual(parsed, fallback)
                self.assertIsNot(parsed, fallback)

    def test_optional_sidecars_have_bounded_latency_in_the_live_path(self):
        source = (Path(__file__).resolve().parents[1] / "web/app.py").read_text()
        self.assertGreaterEqual(source.count("timeout=1.5"), 2)
        self.assertIn("return_exceptions=True", source)


if __name__ == "__main__":
    unittest.main()
