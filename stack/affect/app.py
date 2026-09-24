"""Bounded MiniLM semantic reaction sampler for live portrait phrases."""

from __future__ import annotations

import hashlib
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

MODEL_ROOT = Path(os.environ.get("MINILM_MODEL_DIR", "/models/minilm"))
MODEL_REVISION = os.environ.get("MINILM_REVISION", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
MODEL_SHA256 = os.environ.get("MINILM_MODEL_SHA256", "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452")
TOKENIZER_SHA256 = os.environ.get("MINILM_TOKENIZER_SHA256", "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037")
SNAPSHOT = MODEL_ROOT / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / MODEL_REVISION
MODEL_PATH, TOKENIZER_PATH = SNAPSHOT / "onnx" / "model.onnx", SNAPSHOT / "tokenizer.json"

ANCHORS = {
    "amusement": ["That is funny and delightful.", "I am amused and smiling."],
    "surprise": ["That is unexpected and surprising.", "I am startled by this revelation."],
    "skepticism": ["I doubt that claim and remain skeptical.", "That does not sound convincing."],
    "concern": ["This is worrying and deserves concern.", "I feel sympathy about this problem."],
    "agreement": ["I agree completely and approve.", "Yes, that is correct."],
    "disagreement": ["I disagree and reject that conclusion.", "No, that is incorrect."],
    "interest": ["This is fascinating and I want to know more.", "I am attentive and interested."],
    "thinking": ["I need to consider and reason about this.", "Let me think carefully."],
    "neutral": ["Here is a calm factual explanation.", "This is ordinary neutral information."],
}

LEXICAL_CUES = {
    "amusement": re.compile(r"\b(fun(?:ny)?|hilarious|laugh|joke|delight(?:ful|ed)?)\b", re.I),
    "surprise": re.compile(r"(?:\bcannot believe\b|\b(?:surpris(?:e|ed|ing)|unexpected|unbelievable|astonish(?:ed|ing)?)\b)", re.I),
    "skepticism": re.compile(r"\b(doubt|skeptic(?:al|ism)?|questionable|unconvincing)\b", re.I),
    "concern": re.compile(r"\b(worr(?:y|ied|ying)|danger(?:ous)?|concern(?:ed|ing)?|risk|harm)\b", re.I),
    "agreement": re.compile(r"\b(yes|agree|agreed|exactly|correct|right)\b", re.I),
    "disagreement": re.compile(r"\b(no|not|never|disagree|incorrect|wrong|false)\b", re.I),
    "interest": re.compile(r"\b(fascinat(?:e|ed|ing)|interesting|curious|tell me more)\b", re.I),
    "thinking": re.compile(r"\b(think|consider|reason|perhaps|maybe|wonder)\b", re.I),
}

CONTROL_VECTORS = {
    "amusement": {"smile": 0.45, "smirk": 0.18, "brow_raise": 0.08},
    "surprise": {"brow_raise": 0.70, "eye_open": 0.45, "recoil": 0.12},
    "skepticism": {"smirk": 0.34, "brow_furrow": 0.24, "gaze_x": 0.18, "tilt": 0.12},
    "concern": {"brow_furrow": 0.42, "brow_raise": 0.12, "smile": -0.12},
    "agreement": {"smile": 0.18, "nod": 0.26, "brow_raise": 0.08},
    "disagreement": {"brow_furrow": 0.24, "tilt": -0.14, "smile": -0.10},
    "interest": {"brow_raise": 0.16, "gaze_focus": 0.35, "lean": 0.10},
    "thinking": {"gaze_x": 0.16, "gaze_y": -0.12, "brow_furrow": 0.14, "tilt": 0.08},
}


def _controls(strengths: dict[str, float]) -> dict[str, float]:
    values: dict[str, float] = {}
    for reaction, strength in strengths.items():
        for control, coefficient in CONTROL_VECTORS.get(reaction, {}).items():
            values[control] = values.get(control, 0.0) + coefficient * min(0.65, strength)
    return {name: float(np.clip(value, -0.45, 0.45)) for name, value in values.items()}

_session: ort.InferenceSession | None = None
_tokenizer: Tokenizer | None = None
_prototypes: dict[str, np.ndarray] = {}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _embed(texts: list[str]) -> np.ndarray:
    assert _session is not None and _tokenizer is not None
    encodings = _tokenizer.encode_batch(texts)
    length = min(256, max(len(item.ids) for item in encodings))
    ids = np.zeros((len(encodings), length), dtype=np.int64)
    mask = np.zeros_like(ids)
    types = np.zeros_like(ids)
    for row, item in enumerate(encodings):
        size = min(length, len(item.ids))
        ids[row, :size], mask[row, :size] = item.ids[:size], item.attention_mask[:size]
        types[row, :size] = item.type_ids[:size]
    hidden = _session.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": types})[0]
    pooled = (hidden * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
    return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-8)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _session, _tokenizer, _prototypes
    for path, expected in ((MODEL_PATH, MODEL_SHA256), (TOKENIZER_PATH, TOKENIZER_SHA256)):
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"MiniLM asset missing or checksum mismatch: {path}")
    options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
    _session = ort.InferenceSession(str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"])
    _tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    _tokenizer.enable_truncation(max_length=256)
    names, texts = [], []
    for name, anchors in ANCHORS.items():
        for anchor in anchors: names.append(name); texts.append(anchor)
    vectors = _embed(texts)
    grouped: dict[str, list[np.ndarray]] = {}
    for name, vector in zip(names, vectors): grouped.setdefault(name, []).append(vector)
    _prototypes = {name: np.mean(group, axis=0) for name, group in grouped.items()}
    _prototypes = {name: vector / np.linalg.norm(vector) for name, vector in _prototypes.items()}
    yield
    _session = None; _tokenizer = None; _prototypes = {}


class ReactionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


REACTIONS = tuple(CONTROL_VECTORS) + ("neutral",)


class ApplyRequest(BaseModel):
    """Directly set a named reaction, bypassing MiniLM text classification.

    For a caller that already knows which reaction it wants (e.g. an LLM
    tool call) rather than needing one inferred from spoken text -- reuses
    the same CONTROL_VECTORS/_controls() the classifier already applies, so
    the result is bounded exactly like an automatically classified reaction.
    """

    reaction: str = Field(min_length=1, max_length=32)
    strength: float = Field(default=0.6, ge=0.0, le=1.0)


app = FastAPI(title="MiniLM reaction sampler", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {"ok": _session is not None, "model_revision": MODEL_REVISION, "dimensions": 384}


@app.get("/reactions")
def reactions() -> dict[str, object]:
    """The exact reaction names /apply and /classify can return -- single
    source of truth for a tool-call schema rather than a hardcoded copy."""
    return {"reactions": sorted(REACTIONS)}


@app.post("/apply")
def apply(request: ApplyRequest) -> dict[str, object]:
    if request.reaction not in REACTIONS:
        raise HTTPException(status_code=422, detail=f"unknown reaction: {request.reaction!r}; see /reactions")
    strength = 0.0 if request.reaction == "neutral" else request.strength
    return {
        "reaction": request.reaction,
        "strength": strength,
        "controls": _controls({request.reaction: strength}) if strength else {},
        "attack_ms": 160 if request.reaction in {"surprise", "agreement", "disagreement"} else 260,
        "hold_ms": 500 if request.reaction in {"thinking", "skepticism", "interest"} else 320,
        "decay_ms": 700,
        "cooldown_ms": 900,
    }


@app.post("/classify")
def classify(request: ReactionRequest) -> dict[str, object]:
    if _session is None:
        raise HTTPException(status_code=503, detail="reaction sampler is loading")
    vector = _embed([request.text])[0]
    similarities = {name: float(vector @ prototype) for name, prototype in _prototypes.items()}
    cue_hits = [name for name, pattern in LEXICAL_CUES.items() if pattern.search(request.text)]
    for name in cue_hits:
        similarities[name] += 0.16
    if "disagreement" in cue_hits:
        similarities["agreement"] -= 0.12
    neutral = similarities["neutral"]
    strengths = {
        name: float(np.clip((score - neutral) / 0.45, 0.0, 1.0))
        for name, score in similarities.items() if name != "neutral"
    }
    top = max(strengths, key=strengths.get)
    if strengths[top] < 0.12:
        top = "neutral"
    ordered = sorted(
        (score for name, score in similarities.items() if name not in {"neutral", top}),
        reverse=True,
    )
    margin = max(0.0, similarities[top] - ordered[0]) if top != "neutral" else 0.0
    strength = min(0.78, strengths.get(top, 0.0) * 0.72 + margin * 0.8)
    return {
        "reaction": top,
        "strength": 0.0 if top == "neutral" else strength,
        "primitives": strengths,
        "controls": _controls(strengths),
        "lexical_cues": cue_hits,
        "attack_ms": 160 if top in {"surprise", "agreement", "disagreement"} else 260,
        "hold_ms": 500 if top in {"thinking", "skepticism", "interest"} else 320,
        "decay_ms": 700,
        "cooldown_ms": 900,
    }
