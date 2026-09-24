# 06 · The voice loop, and the video bridge that isn't on

*The half that works, the half that doesn't, and how we know which is which.*

Code: [`voice-agent/`](../../voice-agent/) (147 tests, run from the lockfile) ·
Services: [`stack/`](../../stack/)

## What runs end to end

```
mic ─▶ LiveKit room ─▶ STT (CrispASR, HTTP) ─▶ LLM (any OpenAI-compatible endpoint)
                                                   │  tool call: set_reaction(name, strength)
                                                   ▼
                       Kokoro TTS ◀─ phrase splitter ─┴─▶ affect service ─▶ (optional) renderer
```

Verified against real running services, no mocks, in a browser driven by Playwright: a Kokoro →
Whisper round trip transcribed a test sentence back word for word; the custom TTS plugin emitted
19 frames / 145,408 PCM bytes at 24 kHz for a test sentence; a real tool schema made the LLM call
`set_reaction(reaction="surprise", strength=1)` for "I just won the lottery!"; and a real
`<audio>` element for the agent's track appeared in the browser and played.

This is voice. The lip-synced video is discussed below and is **off**.

## Bugs that were worth writing down

**1. The default interruption mode quietly phones home.** LiveKit's "adaptive" interruption
handling called out to LiveKit Cloud's hosted inference. With no credentials configured, this was
visible only as three `WSServerHandshakeError: 401` retries before it fell back. A self-hosted
deployment was making an outbound call nobody had configured it to make. Fix: set the mode to
`vad` explicitly.

**2. Kokoro TTS is not streaming** (one WAV per call), which does not fit the OpenAI SSE speech
contract that the stock plugin drives. Rather than fake SSE on top of a backend that can't stream,
the plugin implements the framework's real chunked-stream / audio-emitter interface directly.
STT and LLM reuse the official plugin classes pointed at self-hosted endpoints, which is lower
risk than hand-rolling those too.

**3. `NEXT_PUBLIC_*` is baked in at build time.** The client URL was left at `ws://127.0.0.1`
(fine for the agent on the same host, wrong for a remote browser). Caught before commit.

**4. A test-rig trap:** Chrome loops a fake-microphone file continuously with no silence, which
starves VAD of an end-of-turn cue and looks exactly like a multi-minute hang. Pad the test audio
with trailing silence.

**5. The second `rtc.Room` connection is descriptor-hungry.** The systemd default of 1024 open
files was exhausted mid-session; raised to 65,536.

**6. `DataStreamAudioReceiver` unconditionally RPCs a playback-started message back to the
sender.** Without `wait_playback_start=True` on the output side, nothing was registered to
receive it.

After 5 and 6: clean operation across many consecutive turns with exact frame-count matches
(5.02 s of speech → 125 frames at 25 fps), which is the evidence that the renderer was lip-syncing
real agent speech and not dummy input.

## The bug that is still open

**The browser never rendered the avatar's audio or video tracks**, even though the server-side
evidence (negotiated `m=video` SDP sections, clean playback-finished cycles) says both were
published, and the token granted `canSubscribe: true`. Repeated Playwright runs with a
`MutationObserver` and full SDK logging saw no subscription event and no element for the avatar's
identity. Hypotheses: something specific to a track published by a participant that joins *after*
the local one, or two `rtc.Room` connections sharing one process and event loop. Not isolated. We
stopped debugging rather than keep guessing. The suggested next step, never done, is to inspect
the browser's *received* SDP offers to see whether the SFU offered the tracks at all.

Separately, the video generator had an unbounded-queue memory leak
([05](05-dispatch-bound.md)). That fix has not been confirmed live either, which is why
`MUSE_AVATAR_ENABLED` defaults to `0` in this repo.

## Speech synthesis, briefly

- **Kokoro** (82M, ONNX, and a GPU FastAPI build) is the fast default voice. Named voice blends and
  a "voice lab" that interpolates style vectors exist in the code; the compose file and the web
  app currently disagree about who serves the lab route (see
  [KNOWN_ISSUES](../KNOWN_ISSUES.md)).
- **Qwen3-TTS voice cloning** ran through a small patch to koboldcpp that exposes per-request
  reference audio and transcript. The project README reports transcript in-context cloning
  raised speaker similarity from ~0.75 to ~0.89. A hand-written int8 port of it was rejected
  ([05](05-dispatch-bound.md)).

## Reactions

The agent doesn't have the model emit a facial expression per frame. An LLM tool call names a
reaction and a strength; the affect service caps strength (0.45 in the reviewed configuration),
schedules blink, gaze and head events, and the renderer owns the rest. That split is why
[02](02-expression-control.md) is about leakage between controls: the vocabulary is small on
purpose, and each word has to move only what it names.
