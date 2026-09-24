import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LiveRuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.web = (ROOT / "web/app.py").read_text()
        cls.muse = (ROOT / "muse_patch/muse_server.py").read_text()
        cls.alp = (ROOT / "advanced_live_portrait/app.py").read_text()
        cls.compose = (ROOT / "docker-compose.yml").read_text()
        cls.env = (ROOT / ".env.example").read_text()

    def test_health_and_network_guards_are_truthful(self):
        self.assertIn('JSONResponse(status_code=200 if payload["ok"] else 503', self.web)
        self.assertIn("await _guard_request(request, \"start\", START_RATE_LIMIT)", self.web)
        self.assertIn("await _guard_request(request, \"feedback\", FEEDBACK_RATE_LIMIT)", self.web)
        self.assertIn("_schedule_session_expiry(session_id)", self.web)
        self.assertIn("MUSE_DEMO_AUTH_TOKEN", self.compose)
        self.assertIn("MUSE_DEMO_MAX_RATE_KEYS", self.compose)

    def test_completed_jobs_are_drained_and_disconnects_are_cancelled(self):
        self.assertIn("async def emit_pending_frames()", self.web)
        self.assertIn("for _ in range(2):", self.web)
        self.assertIn('"cancel_path": str(JOBS_ROOT / f"{job_id}.cancel")', self.web)
        self.assertIn("READY_GESTURE_CACHE.setdefault(digest", self.web)
        self.assertIn('for job_id in session_job_ids:', self.web)
        self.assertIn('b"cancel\\n"', self.web)
        self.assertIn("async def watch_disconnect()", self.web)
        self.assertIn("async def run_with_disconnect(coroutine)", self.web)
        self.assertIn("raise ClientDisconnected()", self.web)
        self.assertIn("if cancel_path and os.path.exists(cancel_path)", self.muse)
        self.assertIn("class JobCancelled", self.muse)

    def test_disabled_alp_does_not_advertise_canvas_head_events(self):
        self.assertIn('if not ALP_ENABLED:', self.web)
        self.assertIn('reaction["head_events"] = []', self.web)

    def test_caches_are_bounded(self):
        self.assertIn("MAX_POSE_FLOW_CACHE", self.muse)
        self.assertIn("while len(self._pose_flow_cache) > MAX_POSE_FLOW_CACHE", self.muse)
        self.assertIn("while len(avatar_cache) > MAX_CACHED_AVATARS", self.muse)

    def test_source_bank_encoding_is_lossless_and_versioned(self):
        self.assertIn('MOTION_ENCODER_REVISION = "lossless-rgb-v1"', self.alp)
        self.assertIn('codec="libx264rgb"', self.alp)
        self.assertIn('"-crf", "0"', self.alp)
        self.assertIn('format="PNG"', self.alp)
        self.assertIn("ADVANCED_LIVE_PORTRAIT_FRAMES=28", self.env)
        self.assertIn("ADVANCED_LIVE_PORTRAIT_PROFILE=semantic-bank-v13-blink-brow-fix", self.env)

    def test_motion_controls_are_explicit_and_reportable(self):
        self.assertIn("ALP_EXPRESSION_CONTROLS = {", self.web)
        self.assertIn('@app.get("/api/motion/controls")', self.web)
        self.assertIn('"rotate_pitch"', self.web)
        self.assertIn('"pupil_x"', self.web)
        self.assertIn("minilm_controls:", (ROOT / "web/static/app.js").read_text())

    def test_live_expression_editor_is_wired_to_the_resident_alp(self):
        self.assertIn('@app.post("/api/motion/edit")', self.web)
        self.assertIn('f"{ALP_URL}/edit"', self.web)
        self.assertIn('@app.post("/edit")', self.alp)
        self.assertIn('"X-ALP-Edit": "single-expression-v1"', self.alp)
        self.assertIn("MUSE_DEMO_MOTION_EDIT_RATE_LIMIT", self.compose)
        self.assertIn("MUSE_DEMO_MOTION_EDIT_RATE_LIMIT=240", self.env)


if __name__ == "__main__":
    unittest.main()
