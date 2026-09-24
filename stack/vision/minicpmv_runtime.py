"""MiniCPM-V-4.6 (`transformers`), text-only STT-artifact classification.

Image-based calls (ambient caption, on-demand targeted visual query) moved
to the `lfm2vl` service (LiquidAI/LFM2.5-VL-450M via a fixed llama.cpp
build -- see lfm2vl/vendor/README.md and docs/VISION.md) on 2026-09-10,
after real measurement: ~150-185ms/request vs. this model's ~939ms-3.1s,
same-or-better accuracy on captions and count-style questions. MiniCPM-V
stays for classify_transcript specifically, because the swap candidate
tested unreliable at THAT task -- with the exact proven classify prompt,
it answered ARTIFACT for every test transcript, including obviously real
text ("What time is it?"), so it's not a like-for-like replacement across
the board. Two different small models, each kept for what it actually
measured well at.

**Classifier accuracy caveat, found validating the original version of
this**: given a bare transcript with no conversation context, MiniCPM-V
correctly flags context-free fragments ("mmm-hmm", ".") as ARTIFACT, but
calls standalone "Thank you."/"bye" REAL -- reasonably, since either
really can be genuine speech in isolation. Those specific hallucination
shapes are already handled upstream by agent.py's VAD tuning
(min_silence_duration/min_speech_duration -- see agent.py's AgentSession
comments); this classifier is a complementary catch for the other
category (word-salad fragments, coughs transcribed as real-looking
words) it measurably does catch, not a replacement for that fix.

**Real-world accuracy update (2026-09-10)**: a longer live conversation
found this classifier drops real, substantive content at a real, non-
trivial rate ("Talk like a futuristic cyborg robot from now on.", "No
toothbrush, just the pliers." -- both real, both dropped). Enforcement is
OFF by default in agent.py (VISION_STT_FILTER_ENFORCE) as a result --
this classifier's verdicts are logged for future tuning but not currently
acted on. Don't re-enable without a real before/after test against a live
session.

Scene-narrative consolidation (turning the raw per-tick vision log into a
coherent storyline) was prototyped here and then moved to `lfm2vl-narrate`
instead (a separate, larger LFM2.5-VL-3B instance) -- it can look at the
actual current frame while narrating, rather than only synthesizing text
from what a smaller model already (possibly wrongly) said. See
docs/VISION.md.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = os.environ.get("MINICPMV_MODEL_ID", "openbmb/MiniCPM-V-4.6")
# Short on purpose: this only needs to be long enough to catch requests
# that are *already* concurrent (multiple sessions calling in at once),
# not to manufacture a batch by waiting around -- a single active
# conversation should never feel this as added latency worth mentioning.
BATCH_WINDOW_S = float(os.environ.get("MINICPMV_BATCH_WINDOW_S", "0.05"))
MAX_BATCH = int(os.environ.get("MINICPMV_MAX_BATCH", "8"))

# v2 (2026-09-10): the first live test caught a real false positive on
# "What do I look like?  Let's get you a change.  Thank you." -- a real
# question stitched to a probably-hallucinated tail got judged ARTIFACT as
# a whole, and the real question never reached gemma4-26b. v1's prompt
# asked "is this [whole transcript] a genuine utterance," which invites
# exactly that failure: one hallucinated clause anywhere in a multi-clause
# transcript can tip the whole judgment. v2 asks instead whether the
# transcript CONTAINS a genuine, actionable clause anywhere in it --
# dropping should require the transcript to be noise/filler *throughout*,
# not merely "not perfectly coherent." A wrongly dropped real question is
# worse than an occasional stray fragment getting through (the VAD tuning
# already suppresses most of those), so this deliberately biases toward
# REAL on anything ambiguous or mixed. Still not accurate enough for
# enforcement -- see the docstring's real-world accuracy update above.
CLASSIFY_PROMPT_TEMPLATE = (
    "You are filtering speech-to-text output for stray noise before it reaches a "
    "conversational assistant. A transcript can be several clauses long and STT "
    "hallucination often APPENDS a filler clause (\"thank you\", \"the future scene\", "
    "\"close\") onto an otherwise real one -- that does not make the whole thing noise.\n\n"
    "Answer ARTIFACT only if NO part of the transcript carries real, actionable "
    "meaning (pure filler/noise/fragments throughout, e.g. \"mmm-hmm\", a single "
    "stray letter, a cough). Answer REAL if ANY clause is a genuine question, "
    "statement, or request -- even if other clauses in the same transcript look "
    "like noise or don't quite fit. When genuinely unsure, answer REAL.\n\n"
    "Answer with exactly one word: REAL or ARTIFACT.\n\n"
    'Transcript: "{text}"\nAnswer:'
)


class _MicroBatcher:
    """Collects requests for BATCH_WINDOW_S (or until MAX_BATCH is
    reached, whichever comes first), then runs one batched GPU call."""

    def __init__(self, gpu_pool: ThreadPoolExecutor, run_batch_sync) -> None:
        self._gpu_pool = gpu_pool
        self._run_batch_sync = run_batch_sync
        self._queue: list[tuple[object, asyncio.Future]] = []
        self._flush_handle: asyncio.TimerHandle | None = None

    async def submit(self, item: object):
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._queue.append((item, future))
        if len(self._queue) >= MAX_BATCH:
            self._flush_now(loop)
        elif self._flush_handle is None:
            self._flush_handle = loop.call_later(BATCH_WINDOW_S, self._flush_now, loop)
        return await future

    def _flush_now(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._flush_handle is not None:
            self._flush_handle.cancel()
            self._flush_handle = None
        asyncio.ensure_future(self._flush())

    async def _flush(self) -> None:
        batch, self._queue = self._queue, []
        if not batch:
            return
        items = [item for item, _ in batch]
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(self._gpu_pool, self._run_batch_sync, items)
        except Exception as exc:
            for _, future in batch:
                if not future.done():
                    future.set_exception(exc)
            return
        for (_, future), result in zip(batch, results):
            if not future.done():
                future.set_result(result)


class MiniCPMVRuntime:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        # Same sm_70/cuDNN9 conv-engine gap as YOLO/SmolVLM2 -- see the
        # spike's gotcha #3. Still relevant even though this only does
        # text now: MiniCPM-V is one unified checkpoint, its vision tower
        # loads regardless of whether caption()/answer_visual_question()
        # are ever called.
        torch.backends.cudnn.enabled = False
        self._proc = AutoProcessor.from_pretrained(model_id)
        self._proc.tokenizer.padding_side = "left"
        if self._proc.tokenizer.pad_token is None:
            self._proc.tokenizer.pad_token = self._proc.tokenizer.eos_token
        self._model = (
            AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.float16, attn_implementation="sdpa")
            .to("cuda")
            .eval()
        )
        self._classify_gpu_pool = ThreadPoolExecutor(max_workers=1)
        self._classify_batcher = _MicroBatcher(self._classify_gpu_pool, self._classify_batch_sync)

    async def classify_transcript(self, text: str) -> bool:
        """Returns True if `text` looks like real, meaningful speech; False
        if it looks like an STT artifact (filler/fragment/noise) that
        should be dropped before reaching the LLM."""
        return await self._classify_batcher.submit(text)

    def _classify_batch_sync(self, texts: list[str]) -> list[bool]:
        prompts = [
            self._proc.tokenizer.apply_chat_template(
                [{"role": "user", "content": CLASSIFY_PROMPT_TEMPLATE.format(text=text)}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for text in texts
        ]
        inputs = self._proc.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=5, do_sample=False)
        decoded = self._proc.tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return ["ARTIFACT" not in text.upper() for text in decoded]
