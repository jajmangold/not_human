"""Crosstalk recovery: speech separation upstream of CrispASR.

Genuine simultaneous crosstalk (two people talking at once for most of an
utterance, not just brief turn-taking overlap) is a real failure mode for
single-channel Whisper-style ASR -- verified in the 2026-09-09 hearing-stack
evaluation (Scenario C): at 80-100% overlap, plain transcription recovers
ONE speaker's content and completely LOSES the other's, not a garbled
blend of both.

This adds a speech-separation pass (SepFormer, WSJ0-2mix) upstream of the
CrispASR server: split the mixed audio into 2 estimated single-speaker
streams, transcribe each separately, return both. Verified to recover
BOTH speakers at every overlap level tested (0/10/30/50/80/100%, two
different speaker pairs) where the no-separation baseline lost one
speaker entirely above ~50% overlap. Cheap to run: ~30x realtime on its
own (GPU), negligible against the throughput margins already established
for this stack (7-38x realtime depending on ASR model) -- so running it
unconditionally for a "shared mic, multiple people" context is viable
without hurting the real-time story.

Usage:
    from crisp_separate_and_transcribe import SeparatingTranscriber
    st = SeparatingTranscriber(crispasr_server="http://127.0.0.1:8097")
    texts = st.transcribe(wav_path)  # -> list of 2 strings, one per estimated speaker

When to call this vs. plain single-pass transcription: this is NOT meant
to run on every utterance unconditionally in a normal turn-taking
conversation (see the caveat below) -- gate it behind a cheap signal that
overlap is actually happening. Two practical options, in order of
preference:
  1. If diarization is already running (--diarize-speakers), check
     whether the diarized segment count looks implausibly low relative to
     the audio duration and known number of active speakers (the exact
     symptom this whole investigation found: n_speakers_found collapses
     to 1 under heavy overlap, See Scenario C's ground truth).
  2. A simple double-talk energy heuristic (e.g. two independent VAD
     channels or a multi-band energy-variance check) -- not implemented
     here, this module only does the separation+transcribe step once
     you've decided to call it.

Caveat found during testing: at LOW overlap (0-30%, closer to sequential
turn-taking than true crosstalk), separation still recovers both speakers
but with MORE transcription noise than the plain single-pass baseline,
which already handles that regime well (SepFormer's WSJ0-2mix training
data is synchronized near-full-overlap mixtures, not partial/sequential
ones -- out of its training distribution at low overlap). Don't use this
as a blanket replacement for normal transcription; use it specifically
when overlap is suspected.

Scope caveat, not yet tested: this checkpoint is fixed at 2 sources
(WSJ0-2mix). A 3rd genuinely-simultaneous speaker will not be separated
out correctly -- only tested against 2-speaker overlap, matching the
2-speaker Scenario C corpus this whole investigation's overlap testing
used. speechbrain also ships WSJ0-3mix-trained checkpoints for the
3-speaker case; swap `source=` if that's needed, but verify it the same
way this file's results were verified before trusting it.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

import numpy as np
import scipy.signal as sps
import soundfile as sf
import torch
from speechbrain.inference.separation import SepformerSeparation

SEPFORMER_SR = 8000  # speechbrain/sepformer-wsj02mix's native rate


@dataclass
class SeparatedResult:
    texts: list[str]  # one transcript per estimated source
    n_sources: int


class SeparatingTranscriber:
    def __init__(
        self,
        crispasr_server: str = "http://127.0.0.1:8097",
        device: str | None = None,
        savedir: str = "sepformer-wsj02mix-cache",
    ):
        self.server = crispasr_server.rstrip("/")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # run_opts={"device": device} needs an index-qualified string
        # ("cuda:0") on this speechbrain/torch version -- bare "cuda"
        # raises inside speechbrain and silently falls back to device 0
        # anyway, but pass it correctly rather than rely on the fallback.
        if device == "cuda" and ":" not in device:
            device = "cuda:0"
        self.separator = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj02mix",
            savedir=savedir,
            run_opts={"device": device},
        )

    def _transcribe_wav(self, wav_path: str) -> str:
        boundary = "----crispasrsep"
        with open(wav_path, "rb") as f:
            data = f.read()
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(
            f"{self.server}/inference",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            r = json.loads(resp.read())
        return " ".join(seg.get("text", "") for seg in r.get("segments", []))

    def transcribe(self, wav_path: str, target_sr: int = 16000) -> SeparatedResult:
        """Separate wav_path into estimated per-speaker streams and
        transcribe each. Returns one text per estimated source (SepFormer
        WSJ0-2mix is fixed at 2 sources)."""
        est_sources = self.separator.separate_file(path=wav_path)
        est_sources = est_sources.detach().cpu().numpy()  # (time, n_src)

        texts = []
        for src_idx in range(est_sources.shape[-1]):
            track = est_sources[..., src_idx].squeeze()
            peak = np.max(np.abs(track)) + 1e-9
            track = (track / peak * 0.95).astype(np.float32)
            if target_sr != SEPFORMER_SR:
                n = int(len(track) * target_sr / SEPFORMER_SR)
                track = sps.resample(track, n).astype(np.float32)
            tmp_path = wav_path.rsplit(".", 1)[0] + f".sep_src{src_idx}.wav"
            sf.write(tmp_path, track, target_sr, subtype="PCM_16")
            texts.append(self._transcribe_wav(tmp_path))

        return SeparatedResult(texts=texts, n_sources=est_sources.shape[-1])


if __name__ == "__main__":
    import sys

    wav = sys.argv[1] if len(sys.argv) > 1 else "scenario_c/overlap_100pct.wav"
    server = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:18292"
    st = SeparatingTranscriber(crispasr_server=server)
    result = st.transcribe(wav)
    for i, t in enumerate(result.texts):
        print(f"source {i}: {t!r}")
