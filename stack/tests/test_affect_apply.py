"""Contract test for the affect service's direct-apply endpoint (#45).

Calls the FastAPI route functions directly rather than through a TestClient
(no httpx dependency in the affect image) -- a route decorated with
@app.post is still a plain callable function.
"""

import unittest

try:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "affect"))
    from app import CONTROL_VECTORS, REACTIONS, ApplyRequest, apply, reactions
    from fastapi import HTTPException
except ModuleNotFoundError:
    CONTROL_VECTORS = None


@unittest.skipIf(CONTROL_VECTORS is None, "requires onnxruntime/tokenizers/fastapi (affect image)")
class ApplyEndpointTests(unittest.TestCase):
    def test_reactions_lists_every_control_vector_plus_neutral(self):
        listed = set(reactions()["reactions"])
        self.assertEqual(listed, set(CONTROL_VECTORS) | {"neutral"})

    def test_apply_reuses_the_same_control_vectors_as_classify(self):
        result = apply(ApplyRequest(reaction="surprise", strength=0.5))
        self.assertEqual(result["reaction"], "surprise")
        self.assertEqual(result["strength"], 0.5)
        # CONTROL_VECTORS["surprise"] = {brow_raise: 0.70, eye_open: 0.45, recoil: 0.12}
        # scaled by min(0.65, 0.5) = 0.5 (none of these hit the +/-0.45 clip).
        self.assertAlmostEqual(result["controls"]["brow_raise"], 0.35, places=5)
        self.assertAlmostEqual(result["controls"]["eye_open"], 0.225, places=5)
        self.assertAlmostEqual(result["controls"]["recoil"], 0.06, places=5)

    def test_apply_clamps_strength_the_same_way_regardless_of_input(self):
        # strength=1.0 should clamp the same as classify's min(0.65, strength)
        # -- both go through the shared _controls() helper.
        high = apply(ApplyRequest(reaction="amusement", strength=1.0))
        capped = apply(ApplyRequest(reaction="amusement", strength=0.65))
        self.assertEqual(high["controls"], capped["controls"])

    def test_apply_neutral_returns_no_controls_and_zero_strength(self):
        result = apply(ApplyRequest(reaction="neutral", strength=0.9))
        self.assertEqual(result["strength"], 0.0)
        self.assertEqual(result["controls"], {})

    def test_apply_rejects_unknown_reaction(self):
        with self.assertRaises(HTTPException) as ctx:
            apply(ApplyRequest(reaction="euphoria", strength=0.5))
        self.assertEqual(ctx.exception.status_code, 422)

    def test_apply_request_rejects_out_of_range_strength(self):
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ApplyRequest(reaction="amusement", strength=1.5)


if __name__ == "__main__":
    unittest.main()
