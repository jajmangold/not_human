import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class IOSCanvasContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "web/static/index.html").read_text()
        cls.css = (ROOT / "web/static/style.css").read_text()
        cls.javascript = (ROOT / "web/static/app.js").read_text()
        cls.dockerfile = (ROOT / "web/Dockerfile").read_text()

    def test_canvas_is_the_canonical_visible_surface(self):
        self.assertIn('id="live-canvas" role="img"', self.html)
        self.assertNotIn('id="live-video"', self.html)
        self.assertIn("#live-canvas { display: block;", self.css)
        self.assertIn("max-width: 100%; max-height: 62vh", self.css)
        self.assertNotIn("captureStream", self.javascript)

    def test_upload_and_stream_frames_share_the_compatibility_decoder(self):
        self.assertIn("app.js?v=upperbody13", self.html)
        self.assertIn("basePortrait = await decodeImage(file);", self.javascript)
        self.assertGreaterEqual(self.javascript.count("await decodeImage(new Blob"), 2)
        self.assertIn("new Image()", self.javascript)
        self.assertIn("URL.revokeObjectURL(image[objectUrlForImage])", self.javascript)
        self.assertIn("window.AudioContext || window.webkitAudioContext", self.javascript)

    def test_output_feedback_uses_the_backend_mediapipe_path(self):
        self.assertIn("stream === 'output' && source === canvas", self.javascript)
        self.assertIn("'/api/feedback'", self.javascript)
        self.assertIn("Camera frames are", self.javascript)
        self.assertIn("startMocapWorker('webcam')", self.javascript)
        self.assertIn("optional webcam graph remains a lazy local worker", self.javascript)
        self.assertIn("expressionFeedback.blinkSamples", self.javascript)
        self.assertIn("data.blinkId > 0 && data.blinkId === activeBlinkFeedbackId", self.javascript)
        self.assertIn("blinkFeedbackUntil", self.javascript)
        self.assertIn("finishBlinkFeedback(completedBlinkId)", self.javascript)
        self.assertIn("window.portraitTelemetry", self.javascript)
        self.assertIn("mocap-worker.js?v=feedback3", self.javascript)
        self.assertIn("image?.close?.();", self.javascript)

    def test_autonomic_motion_is_a_bounded_semi_markov_process(self):
        self.assertIn("const behaviorStates = {", self.javascript)
        self.assertIn("dwell: [", self.javascript)
        self.assertIn("enterBehaviorState(now", self.javascript)
        self.assertIn("const gain = prior?.gain ?? 0.46", self.javascript)
        self.assertNotIn("expressionFeedback.motionScale", self.javascript)
        self.assertIn("Critically damped second-order response", self.javascript)

    def test_minilm_head_controls_have_bounded_audio_clock_actuators(self):
        self.assertIn("function semanticMotionTarget(now)", self.javascript)
        self.assertIn("controls.lean", self.javascript)
        self.assertIn("controls.recoil", self.javascript)
        semantic_target = self.javascript.split("function semanticMotionTarget(now)", 1)[1].split(
            "function updateSemanticMotion(now)", 1
        )[0]
        self.assertNotIn("controls.gaze_x", semantic_target)
        self.assertNotIn("controls.gaze_y", semantic_target)
        self.assertIn("ALP owns gaze inside an eye-only mask", semantic_target)
        self.assertIn("Gross pitch/roll (including nod and tilt) belongs exclusively", self.javascript)
        self.assertIn("plan?.pose?.nod_impulse", self.javascript)
        self.assertIn("Math.sin(Math.PI * nodPhase) ** 2", self.javascript)
        self.assertIn("const semanticGain = webcamFaceTracked ? 0.35 : 1", self.javascript)
        self.assertIn("const attentionHead = 0", self.javascript)
        self.assertNotIn("hasRealHeadPose", self.javascript)
        self.assertIn("const browserRoll = upperBody.rig", self.javascript)
        self.assertIn("semanticMotion", self.javascript)

    def test_idle_bank_reclassifies_after_each_async_frame_decode(self):
        self.assertIn("function refreshIdleIndices()", self.javascript)
        self.assertIn("if (kind === IDLE_FRAME_PACKET) refreshIdleIndices();", self.javascript)
        self.assertIn("idleGestureStart = data.idle_gesture_start;", self.javascript)

    def test_attention_uses_fixations_not_a_continuous_robot_stare(self):
        self.assertIn("const attention = {", self.javascript)
        self.assertIn("function chooseAttentionFixation(now)", self.javascript)
        self.assertIn("attention.fixation !== 'camera'", self.javascript)
        self.assertIn("const omega = 24", self.javascript)
        self.assertIn("attention.current * Math.max", self.javascript)
        self.assertNotIn("autonomic.rest * Math.max", self.javascript)

    def test_correlated_natural_motion_is_packaged_and_observable(self):
        self.assertIn("COPY natural_motion.py .", self.dockerfile)
        self.assertIn("motion_prior", (ROOT / "web/app.py").read_text())
        self.assertIn("headCurrent", self.javascript)
        self.assertIn("nearNeutralFraction", self.javascript)
        self.assertIn("gazeHeadCorrelation", self.javascript)

    def test_upper_body_uses_pose_fitted_piecewise_mesh_and_audio_clock(self):
        self.assertIn("function upperBodyWarp(image)", self.javascript)
        self.assertIn("drawMeshTriangle", self.javascript)
        self.assertIn("enterRespiratoryState", self.javascript)
        self.assertIn("reaction?.upper_body?.respiration", self.javascript)
        self.assertIn("bodyMetrics: result.body_metrics", self.javascript)
        self.assertIn("breathShoulderCorrelation", self.javascript)
        self.assertIn("normalizedBreathResponse", self.javascript)
        self.assertIn("metrics.shoulderVisibility", self.javascript)
        self.assertIn("upperBody.feedbackGain * correction", self.javascript)
        self.assertIn("0.88, 1.08", self.javascript)
        self.assertIn("upperBody.pose.headRoll", self.javascript)
        self.assertIn("upperBody.pose.headPitch", self.javascript)
        self.assertIn("function upperBodyGeometry(rig)", self.javascript)
        self.assertIn("motion calibration requires ?motion-calibration=1", self.javascript)
        self.assertIn("window.setPortraitMotionCalibration = setMotionCalibration", self.javascript)
        self.assertIn("maxDisplacementPx", self.javascript)
        self.assertIn("motionCalibration.saved", self.javascript)
        self.assertIn("Object.assign(upperBody.pose, saved.upperBodyPose)", self.javascript)
        self.assertIn("const clipTarget = target.map", self.javascript)
        self.assertIn("upperBodyContext.drawImage(image, 0, 0, canvas.width, canvas.height)", self.javascript)
        self.assertNotIn("upperBodySway", self.javascript)

    def test_face_motion_variance_is_telemetry_not_a_noise_chasing_loop(self):
        self.assertIn("variance is telemetry", self.javascript)
        self.assertNotIn("deviation > 0.006 ? 0.97", self.javascript)

    def test_feedback_retries_and_blink_feedback_avoids_duration_aliasing(self):
        self.assertIn("outputFeedbackRetryAt", self.javascript)
        self.assertIn("Math.min(30, 0.5 * (2 **", self.javascript)
        self.assertNotIn("outputFeedbackDisabled", self.javascript)
        self.assertIn("cannot estimate a ~200 ms event's", self.javascript)
        self.assertNotIn("active.at(-1).time - active[0].time + cadence", self.javascript)
        self.assertIn("beginSpeechRespiration(startedAt)", self.javascript)
        self.assertIn("/api/feedback?body=false", self.javascript)
        self.assertIn("feedbackFaceLayer.width = 256", self.javascript)
        self.assertIn("function postBlinkFeedbackFrame(now)", self.javascript)
        self.assertIn("postBlinkFeedbackFrame(now);", self.javascript)
        self.assertIn("activeBlinkFeedbackId === 0", self.javascript)

    def test_speech_frames_use_webcam_style_sample_and_hold(self):
        self.assertIn("drawPair(activeChunk.carryFrame, null, 0, true)", self.javascript)
        self.assertIn("drawPair(lower, null, 0)", self.javascript)
        self.assertIn("sample-and-hold semantics", self.javascript)
        self.assertNotIn("drawPair(activeChunk.carryFrame, lower, mix, true)", self.javascript)

    def test_motion_values_are_copyable_without_importing_the_upstream_webui(self):
        self.assertIn('id="motion-inspector"', self.html)
        self.assertIn('id="copy-motion-values"', self.html)
        self.assertIn('id="download-motion-values"', self.html)
        self.assertIn("window.copyPortraitMotionValues", self.javascript)
        self.assertIn("minilm_controls", self.javascript)
        self.assertIn("face_summary", self.javascript)
        self.assertIn("/api/motion/controls", self.javascript)

    def test_live_expression_editor_has_real_sliders_and_image_endpoint(self):
        editor = (ROOT / "web/static/motion-editor.html").read_text()
        editor_js = (ROOT / "web/static/motion-editor.js").read_text()
        self.assertIn("/api/motion/edit", editor_js)
        self.assertIn("type = 'range'", editor_js)
        self.assertIn("renderLatest", editor_js)
        self.assertIn('id="editor-portrait"', editor)
        self.assertIn('id="editor-output"', editor)
        self.assertIn('id="editor-copy"', editor)

    def test_live_expression_editor_contains_large_portraits_like_the_live_canvas(self):
        editor = (ROOT / "web/static/motion-editor.html").read_text()
        self.assertIn('class="editor-image-frame"', editor)
        self.assertIn("style.css?v=ios2", editor)
        self.assertIn(".editor-image-frame", self.css)
        self.assertIn("max-width: 100%; max-height: 100%;", self.css)
        self.assertIn("height: min(62vh, 720px);", self.css)
        self.assertIn("overflow: hidden", self.css)


if __name__ == "__main__":
    unittest.main()
