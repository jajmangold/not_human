# spec/ — contracts for a controllable talking avatar

The Python package is named `nothuman` (kept as-is; the repo is `not_human`). This is the
**contract layer only**: deterministic, tested data types for the things an embodied
conversational avatar exchanges between components, with no model in the hot path.

| contract | what it pins |
|---|---|
| `ControlFrameV1` (protobuf, fields 1–31) | one tick of avatar control, with canonical deterministic JSON |
| `FaceControlV1`, `BodyControlV1` | 52 MediaPipe blendshape categories; 33 pose landmarks |
| `authority.resolve_authority` | who wins when several writers command the same channel: additive writers first, then absolute writers by `(priority, seq, name)`, per-key masks, residual gating, stale-sequence drop, no wall-clock or hash-order dependence |
| `timeline`, `rotation`, `resampler` | timestamped streams, rotation conversions/interpolation, resampling |
| `TranscriptV1`, `KokoroV1` | cancellable ASR segments and TTS PCM chunks with provenance |
| `ExecutionTargetV1` + ADR-0001 | the pinned hardware target (sm_70, dp4a / CUDA-core, tensor cores off) |

**What is not here:** the adapters that would put MuseTalk and LivePortrait behind these
contracts, and the compositor that resolves authority across renderers, were designed but never
built. The running system in `stack/` predates the contracts and does not use them. Treat this
directory as a specification that has tests, not as the architecture of the running code.

Run: `PYTHONPATH=src python -m pytest tests` (190 tests, ~0.4 s, CPU only).

The evidence tables below are the original per-work-item log, kept verbatim.

---

# nothuman

Real-time embodied conversational avatar: explicit controls, deterministic rendering, and governed fleet delivery.

## Bootstrap evidence

P0-001 established the package boundary and development checks without selecting a runtime model or renderer.

| Evidence | Revision or fixture | Result |
| --- | --- | --- |
| Package revision | `0d70cee4d5443c59342ecf2739c33f04fba93af3` | Bootstrap merge |
| ControlFrame revision | `2b93fb8109977ab7f5eeaf80e7953a36c7f7f54f` | PR head before spec repair |
| Schema source | `proto/control_frame_v1.proto` | ControlFrameV1 fields 1–31 |
| Generated binding | `src/nothuman/control_frame_v1_pb2.py` | Checked-in field metadata and version |
| Contract fixture | `tests/fixtures/control_frame_v1.json` | Canonical deterministic JSON |
| Contract tests | `tests/test_control_frame_v1.py` | 11 passed locally: round-trip, bounds, invalid data, compatibility, replay |
| Metrics | CPU contract test runtime: `0.04s` for 11 tests | No model hot path or benchmark claimed in P0-002 |
| Model/data provenance | No model, dataset, weight, or media artifact is introduced | Runtime asset selection deferred to later work items |
| Licensing | No third-party runtime asset or model is introduced | Review when dependencies are selected |

FlashVSR remains deferred. Runtime schemas, timing, authority, coordinates, deployment, and model selection belong to later ADR-backed work items.

## P0-004 timeline evidence

| Evidence | Revision or fixture | Result |
| --- | --- | --- |
| Timeline implementation | `agent/7-pi-timeline-recovery` | Current-main recovery branch |
| Contract fixture | `tests/test_timeline.py` deterministic packet streams | 19 focused tests |
| Metrics | CPU focused suite recorded in PR evidence | No model hot path or GPU benchmark claimed |
| Model/data provenance | Synthetic timestamp/value streams only; no model or dataset | Runtime assets deferred |
| Licensing | No third-party dependency or model added | No new license obligation |

## P0-003 rotation evidence

| Evidence | Revision or fixture | Result |
| --- | --- | --- |
| Rotation implementation | `agent/6-pi-rotation-current` | Current-main synchronized branch |
| Contract tests | `tests/test_rotation.py` | 37 focused tests: conversions, bounds, invalid data, round-trips, interpolation, replay |
| Metrics | CPU focused suite: `0.13s` for 37 tests | No model hot path or GPU benchmark claimed |
| Model/data provenance | Analytic synthetic rotations only | No model or dataset introduced |
| Licensing | No third-party runtime asset or dependency added | No new license obligation |

## P1-001 FaceControlV1 evidence

| Evidence | Revision or fixture | Result |
| --- | --- | --- |
| FaceControlV1 implementation | `src/nothuman/face_control_v1.py` | Pinned 52 categories, confidence, transforms, validity, timestamps |
| Contract fixture | `tests/fixtures/face_control_v1.json` | Canonical deterministic JSON |
| Contract tests | `tests/test_face_control_v1.py` | 16 focused tests: normal, bounds, missing data, replay, forward compatibility |
| Metrics | CPU focused suite: `0.03s` for 16 tests | No model hot path or GPU benchmark claimed |
| Model/data provenance | Synthetic blendshape/confidence values only; no MediaPipe model or dataset | Runtime model selection deferred to later work items |
| Licensing | No third-party runtime asset, model, or dependency added | No new license obligation |

The FaceControlV1 adapter provides a stable, deterministic interface for extracting face control data with pinned 52 MediaPipe blendshape categories, per-category confidence, transform parameters, validity state, and timestamps. All values are validated for bounds and finiteness, and outputs are byte-for-byte deterministic for identical inputs.

FlashVSR remains deferred. Runtime schemas, timing, authority, coordinates, deployment, and model selection belong to later ADR-backed work items.
## P1-004 resampler evidence

| Evidence | Revision or fixture | Result |
| --- | --- | --- |
| Resampler implementation | `src/nothuman/resampler.py` | Pinned 24000 -> 16000 path, anchored interpolation, exact `ceil(n*2/3)` output length |
| Contract tests | `tests/test_resampler.py` | 13 focused tests: normal, bounds, missing data, replay determinism, no duration drift |
| Metrics | CPU focused suite: `0.09s` for 13 tests | No model hot path or GPU benchmark claimed |
| Model/data provenance | Synthetic sine/DC/linear sample streams only; no model or dataset | Runtime assets deferred |
| Licensing | No third-party runtime asset, model, or dependency added | No new license obligation |

FlashVSR remains deferred. Runtime schemas, timing, authority, coordinates, deployment, and model selection belong to later ADR-backed work items.

## P0-006 artifact and license manifest evidence

| Evidence | Revision or fixture | Result |
| --- | --- | --- |
| Manifest implementation | `src/nothuman/asset_manifest_v1.py` | Deterministic asset manifest, SHA-256 checksums, SPDX permissive allowlist, canonical SBOM |
| Contract fixture | Synthetic `AssetRecord` fixtures in tests | No model, dataset, weight, or media artifact committed |
| Contract tests | `tests/test_asset_manifest_v1.py` | 11 focused tests: normal, bounds, missing data, duplicate rejection, replay determinism |
| Metrics | CPU focused suite recorded in PR evidence | No model hot path or GPU benchmark claimed |
| Model/data provenance | Records reference revisions only; no weights or media embedded | Synthetic provenance; runtime asset selection deferred |
| Licensing | SPDX permissive allowlist only (Apache-2.0, MIT, BSD-2/3-Clause, ISC, Unlicense, Zlib) | Non-permissive identifiers (e.g. CC-BY-NC-4.0) rejected at construction |

## P1-002 BodyControlV1 evidence

| Evidence | Revision or fixture | Result |
| --- | --- | --- |
| BodyControlV1 implementation | `src/nothuman/body_control_v1.py` | Pinned 33 MediaPipe Pose landmarks, visibility, canonical body frame |
| Contract fixture | `tests/fixtures/body_control_v1.json` | Canonical deterministic JSON with all 33 landmarks |
| Contract tests | `tests/test_body_control_v1.py` | 19 focused tests: normal, bounds, missing data, replay, fixture round-trip |
| Metrics | CPU focused suite: `0.08s` for 19 tests | No model hot path or GPU benchmark claimed |
| Model/data provenance | Synthetic landmark values only; no MediaPipe model or dataset | Runtime model selection deferred to later work items |
| Licensing | No third-party runtime asset, model, or dependency added | No new license obligation |

The BodyControlV1 adapter provides a stable, deterministic interface for extracting body control data with pinned 33 MediaPipe Pose landmarks, per-landmark visibility, canonical body frame coordinates, and timestamps. All values are validated for bounds and finiteness, and outputs are byte-for-byte deterministic for identical inputs.
## P1-003 KokoroV1 adapter evidence

| Evidence | Revision or fixture | Result |
| --- | --- | --- |
| KokoroV1 implementation | `src/nothuman/kokoro_v1.py` | Cancellable 24 kHz mono int16 PCM chunks with normalization, metrics, replay, provenance |
| Contract fixture | `tests/fixtures/kokoro_v1.json` | Canonical deterministic JSON |
| Contract tests | `tests/test_kokoro_v1.py` | 12 focused tests: normal, bounds, missing data, replay, cancellation, determinism |
| Metrics | CPU focused suite: `0.07s` for 12 tests | No model hot path or GPU benchmark claimed |
| Model/data provenance | Synthetic PCM samples only; `KokoroProvenance` pins model/data/revision/fixture | No model weights or generated media committed |
| Licensing | No third-party runtime asset, model, or dependency added | No new license obligation |

The KokoroV1 adapter wraps current Kokoro output as cancellable, timestamped 24 kHz mono int16 PCM chunks with deterministic normalization, metrics, replay, and model/data provenance. FlashVSR remains deferred. Runtime schemas, timing, authority, coordinates, deployment, and model selection belong to later ADR-backed work items.
## P1-007 TranscriptV1 adapter evidence

| Evidence | Revision or fixture | Result |
| --- | --- | --- |
| TranscriptV1 implementation | `src/nothuman/transcript_v1.py` | Cancellable ASR transcript segments (text, timing bounds, is_final, optional no_speech_prob) with normalization, metrics, replay, provenance |
| Contract fixture | `tests/fixtures/transcript_v1.json` | Canonical deterministic JSON |
| Contract tests | `tests/test_transcript_v1.py` | 12 focused tests: normal, bounds, missing data, empty-text-is-valid, replay, cancellation, determinism |
| Metrics | CPU focused suite: `0.10s` for 12 tests | No model hot path or GPU benchmark claimed |
| Model/data provenance | Synthetic transcript segments only; `TranscriptProvenance` pins model/data/revision/fixture | No model weights or audio artifact committed |
| Licensing | No third-party runtime asset, model, or dependency added | No new license obligation |

The TranscriptV1 adapter provides a stable, deterministic interface for ASR transcript segments -- named after what downstream consumers (starting with #94's listening-face fast reaction path) read, not the engine that produces it. Field shape matches a real streaming backend evaluated end-to-end outside this repo (CrispASR, `flight-deck` `sessions/2026-09-09-crispasr-hearing-stack-adversarial-test.md`, running live as this fleet's `musetalk-volta-crispasr-stt-1`), including `no_speech_prob` for distinguishing genuine silence from a dropped partial. Runtime model selection -- which STT engine this wraps in production, and the live HTTP/WS call itself -- is deferred to a later ADR-backed work item, same as every other adapter in this package.
## P0-005 authority resolver evidence

| Evidence | Revision or fixture | Result |
| --- | --- | --- |
| Authority implementation | `src/nothuman/authority.py` | Absolute/additive writers, priority, authority, masks, residual gating |
| Contract tests | `tests/test_authority.py` | 13 focused tests: normal, bounds, missing data, replay determinism |
| Metrics | CPU focused suite: `0.05s` for 13 tests (full suite 97 passed, `0.24s`) | No model hot path or GPU benchmark claimed |
| Model/data provenance | Synthetic authority values only; no model or dataset | Runtime assets deferred |
| Licensing | No third-party runtime asset or dependency added | No new license obligation |

The authority resolver provides a stable, deterministic interface: writers contribute per-key values in absolute or additive mode, are ordered by (priority, seq, name), per-key masks scale or disable contributions, and residual gating retains the previous value when a winning absolute value falls below the configured gate. All values are validated for finiteness and bounds, and outputs are deterministic for identical inputs.
