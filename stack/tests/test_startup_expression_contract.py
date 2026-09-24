import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StartupExpressionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.web = (ROOT / "web/app.py").read_text()
        cls.muse = (ROOT / "muse_patch/muse_server.py").read_text()
        cls.affect = (ROOT / "affect/app.py").read_text()
        cls.feedback = (ROOT / "feedback/app.py").read_text()
        cls.qa = (ROOT / "web/startup_qa.py").read_text()
        cls.compose = (ROOT / "docker-compose.yml").read_text()

    def test_ready_gesture_is_rendered_by_muse_before_browser_delivery(self):
        self.assertIn("startup_frames = IDLE_CYCLE_FRAMES +", self.web)
        self.assertIn('"reaction": "startup" if ALP_ENABLED else "idle"', self.web)
        self.assertIn("ready_frames = startup_frames[:ALP_READY_FRAMES]", self.web)
        self.assertNotIn('data={"output": "frames"', self.web)
        self.assertIn('reaction.get("reaction") == "startup"', self.muse)
        self.assertIn('self._render_sources(video_num - ready_count, fps, {"reaction": "idle"})', self.muse)
        self.assertIn("startup audio/source is shorter than ready_frames", self.muse)
        self.assertIn("ready_frames exceeds the prepared source cycle", self.muse)

    def test_settled_expression_bank_excludes_the_ready_prefix(self):
        self.assertGreaterEqual(self.web.count('"bank_start_frame": settle_from'), 2)
        self.assertIn("offset = min(self.bank_start_frame", self.muse)
        self.assertIn("bank_start_frame removes every settled source frame", self.muse)

    def test_minilm_outputs_continuous_compound_controls(self):
        self.assertIn("CONTROL_VECTORS = {", self.affect)
        self.assertIn('"smirk":', self.affect)
        self.assertIn('"controls": _controls(strengths)', self.affect)
        self.assertIn('plan["primitive_mix"] = mixtures[:2]', self.web)

    def test_semantic_bank_contains_sparse_real_head_pose_targets(self):
        alp = (ROOT / "advanced_live_portrait/app.py").read_text()
        self.assertIn("pitch=0.55", alp)
        self.assertIn("yaw=0.65", alp)
        self.assertIn("roll=0.0", alp)
        self.assertIn("pupil_x=-4.5", alp)
        self.assertIn("pupil_x=3.8", alp)
        self.assertIn("yaw=-4.0", alp)
        self.assertIn("yaw=4.0", alp)
        self.assertIn("pitch=-3.0", alp)
        self.assertIn("pitch=3.0", alp)
        self.assertIn('"semantic-bank-v13-blink-brow-fix"', alp)
        self.assertIn("REACTION_BANK_SIZE = 20", alp)

    def test_coherence_profile_removes_coupled_live_actuators(self):
        alp = (ROOT / "advanced_live_portrait/app.py").read_text()
        bank = alp.split("bank = [", 1)[1].split("    ]\n    return gestures", 1)[0]
        self.assertNotIn("pupil_y=", bank)
        self.assertNotIn("roll=0.38", bank)
        self.assertNotIn("roll=-0.32", bank)
        # The closed-eye slot uses full closure (-20) with no eyebrow
        # counter-term: measured against 3 identities (evidence/issue-30/
        # blink-brow-fix/), the previous partial closure (-14) plus a -32
        # "compensation" produced a worst-case brow max_abs_delta of 0.858 --
        # every negative eyebrow value tested made the brow-down leak larger,
        # monotonically, rather than canceling it. Full closure with no
        # counter-term measured lowest at 0.117.
        self.assertIn("blink=-20.0, smile=0.03", bank)
        self.assertNotIn("eyebrow=-32.0", bank)
        self.assertIn("MOTION_PROFILE_REVISION = os.environ.get(\"ALP_MOTION_PROFILE\", \"semantic-bank-v13-blink-brow-fix\")", alp)

    def test_feedback_uses_derolled_corner_line_gaze(self):
        self.assertIn("def derol_landmarks", self.feedback)
        self.assertIn("corner_line_y", self.feedback)
        self.assertIn("points = derol_landmarks", self.feedback)
        self.assertNotIn("(159, 145))", self.feedback)

    def test_startup_qa_uses_fixed_default_and_final_musetalk_frames(self):
        self.assertIn("portrait-d08aa48606ebf5e43a2a.png", self.qa)
        self.assertIn('"bank_start_frame": 8', self.qa)
        self.assertIn('"stream_format": "jpg"', self.qa)
        self.assertIn('report["alp_identity"] = alp_identity', self.qa)
        self.assertIn("identity_tag", self.qa)
        self.assertIn("lower_mouth_delta", self.qa)
        self.assertIn("gaze_mouth_stable", self.qa)
        self.assertIn('"frames": "28"', self.qa)
        self.assertIn("head_pose_mouth_stable", self.qa)
        self.assertIn('"head_pose_route"] = "advanced-live-portrait"', self.web)

    def test_startup_qa_is_non_blocking_and_retains_qwen_contact_sheet(self):
        web_block = self.compose.split("  web:", 1)[1].split("  startup-qa:", 1)[0]
        self.assertNotIn("startup-qa", web_block)
        self.assertIn('restart: "no"', self.compose)
        self.assertIn("contact-sheet.jpg", self.qa)
        self.assertIn("gaze-axis.jpg", self.qa)
        self.assertIn("head-axis.jpg", self.qa)
        self.assertIn("head-yaw-axis.jpg", self.qa)
        self.assertIn("head-pitch-axis.jpg", self.qa)
        self.assertIn("qwen_focused_verdict", self.qa)
        self.assertIn("feature_crop", self.qa)
        self.assertIn('feature_crop(samples[label], face_bbox, "eyes")', self.qa)
        self.assertIn("data:image/jpeg;base64", self.qa)
        self.assertIn('"enable_thinking": False', self.qa)
        self.assertIn('"warn" if visual_blockers or visual_misses else "pass"', self.qa)
        self.assertIn("Qwen visual QA unavailable", self.qa)


if __name__ == "__main__":
    unittest.main()
