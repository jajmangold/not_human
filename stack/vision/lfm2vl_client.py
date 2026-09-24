"""Thin client for the `lfm2vl`/`lfm2vl-narrate` services (LiquidAI's
LFM2.5-VL family, 450M and 3B respectively, via a fixed llama.cpp build --
see lfm2vl/vendor/README.md). Used by app.py for the ambient caption
(/perceive), on-demand targeted visual questions (/query, both against the
450M model), and background scene-narrative consolidation (/narrate,
against the 3B model) -- see docs/VISION.md for the real measurements
behind each choice: 450M for the fast/frequent calls (~150-185ms), 3B for
the infrequent (~20s cadence) narration where richer, more accurate detail
matters more than shaving another ~100ms off a call that isn't on any
conversational hot path anyway.

No client-side batching here, unlike minicpmv_runtime.py's classifier --
llama-server has its own slot-based request handling (n_slots=4 by
default, confirmed in its own startup log), so concurrent requests are
already its job, not ours.
"""

from __future__ import annotations

import base64
import logging
import os

import aiohttp

logger = logging.getLogger("lfm2vl_client")

LFM2VL_URL = os.environ.get("LFM2VL_URL", "http://lfm2vl:8080")
LFM2VL_NARRATE_URL = os.environ.get("LFM2VL_NARRATE_URL", "http://lfm2vl-narrate:8080")
CAPTION_PROMPT = "Describe what is happening in one short sentence."
# Single-request latency measured at ~150-185ms; this leaves real margin
# for slot contention under concurrent load before calling it a failure.
TIMEOUT_S = float(os.environ.get("LFM2VL_TIMEOUT_S", "5.0"))
# Not on any conversational hot path (runs from agent.py's slow
# background narrator, not gated on a turn), so this can afford to be
# generous -- a detailed 4-sentence description measured ~1.4s.
NARRATE_TIMEOUT_S = float(os.environ.get("LFM2VL_NARRATE_TIMEOUT_S", "8.0"))

# See NARRATE_PROMPT's docstring-equivalent comment in agent.py's
# SceneNarrator -- unlike minicpmv_runtime.py's earlier prototype of this
# (text-only synthesis of a log written by the 450M model), this version
# gets the actual CURRENT frame and can verify/correct against it directly
# rather than only trusting someone else's possibly-wrong prior
# description.
NARRATE_PROMPT_TEMPLATE = (
    "You can see the person you're talking to. Look at this image and write ONE "
    "clear, natural sentence describing what's happening right now, in the "
    "context of what's been observed recently below (a less reliable, noisier "
    "log from a smaller, faster model -- use it for context and continuity, but "
    "trust what you can actually see in THIS image over anything in the log that "
    "doesn't match it).\n\n"
    "Recent observation log (less reliable, oldest first):\n"
    "{raw_log}\n\n"
    "Object-presence timeline (from a dedicated detector, more reliable for "
    "what's actually present, though it can't describe actions/attributes):\n"
    "{object_timeline}\n\n"
    "One-sentence narrative, grounded in what you see right now:"
)


async def _ask(http: aiohttp.ClientSession, url: str, jpeg: bytes, prompt: str, *, max_tokens: int, timeout_s: float) -> str | None:
    b64 = base64.b64encode(jpeg).decode()
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    try:
        async with http.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout_s),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.warning("lfm2vl request to %s failed or timed out", url, exc_info=True)
        return None


async def caption(http: aiohttp.ClientSession, jpeg: bytes) -> str | None:
    return await _ask(http, LFM2VL_URL, jpeg, CAPTION_PROMPT, max_tokens=60, timeout_s=TIMEOUT_S)


async def answer_visual_question(http: aiohttp.ClientSession, jpeg: bytes, question: str) -> str | None:
    return await _ask(http, LFM2VL_URL, jpeg, question, max_tokens=60, timeout_s=TIMEOUT_S)


async def narrate(http: aiohttp.ClientSession, jpeg: bytes, raw_log: str, object_timeline: str) -> str | None:
    prompt = NARRATE_PROMPT_TEMPLATE.format(raw_log=raw_log or "(none yet)", object_timeline=object_timeline or "(none available)")
    return await _ask(http, LFM2VL_NARRATE_URL, jpeg, prompt, max_tokens=100, timeout_s=NARRATE_TIMEOUT_S)
