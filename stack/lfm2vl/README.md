# LFM2 tiling fix (pinned llama.cpp fork commit)

Built from `michael-dm/llama.cpp` (a fork of
`ggml-org/llama.cpp`) at branch `lfm2-configurable-tiling`, commit
`e9d636c` ("mtmd : read LFM2 tiling params from GGUF metadata") --
this is an **open, unmerged upstream PR**:
[ggml-org/llama.cpp#25524](https://github.com/ggml-org/llama.cpp/pull/25524),
opened 2026-07-10, one approval (tdakhran, 2026-08-07), two required. Last
updated 2026-08-17 (a rebase request from tdakhran, not yet actioned as of
this vendoring).

## Why this fix is required, not optional

Confirmed live 2026-09-10, testing `LiquidAI/LFM2.5-VL-450M` (and the
`-Extract` variant) as a faster alternative to MiniCPM-V-4.6 for the
`vision` service's image-based calls:

- **Mainline llama.cpp segfaults** running this model's vision path, on
  two different release builds days apart (`b10878` and `b10893`, the
  latter published the same day as this test) -- not a staleness issue,
  a genuine missing-architecture-support gap. Confirmed independently:
  LFM2/LFM2.5-VL isn't in llama.cpp's own supported-multimodal-models
  documentation at all.
- The official `ghcr.io/ggml-org/llama.cpp:server-cuda` Docker image
  (mainline, no crash) loads the model fine but processes prompts at
  **~0.2-0.5 tokens/second** -- unusable (a single request would take
  minutes). Root cause matches the PR's description: without correctly
  reading LFM2's tiling parameters from GGUF metadata, the vision encoder
  either reads garbage (crash) or computes a wildly inflated tile/token
  count for the image (pathological slowness), depending on what garbage
  it reads.
- **With this fix**: ~150-185ms per request (caption, count-style
  questions, and free-text description all tested), correct answers
  across every test tried. See `../../docs/VISION.md` for the fuller
  before/after comparison against MiniCPM-V.

## Build notes

- `LLAMA_BUILD_UI=OFF` + `LLAMA_USE_PREBUILT_UI=OFF` are required to build
  at all from this exact commit -- the default build tries to fetch a
  prebuilt web-UI bundle from an HF bucket keyed by commit hash, which
  doesn't have an entry for this fork's commit. It partially succeeds
  (most assets download) but is missing one required file
  (`loading.html`), which hard-fails the `llama-ui-embed` step. Disabling
  both the npm build and the HF-bucket fetch skips that path entirely --
  we only need the OpenAI-compatible API, not the browser UI. See
  `../Dockerfile`'s comment.
- `GGML_CUDA_MAX_DEVICES` note: this fleet has 17 GPUs total. Running the
  built binary with all GPUs visible (`CUDA_VISIBLE_DEVICES` unset, or
  `--gpus all` in plain `docker run`) hits a hard `GGML_ASSERT(info.device_count
  <= GGML_CUDA_MAX_DEVICES)` crash at startup. Always pin to one specific
  device (`--gpus device=N` in docker, or `CUDA_VISIBLE_DEVICES=N` when
  running the raw binary), matching every other GPU service in this repo's
  convention anyway.

## Refreshing this vendor copy

Once ggml-org/llama.cpp#25524 merges upstream (or LFM2/LFM2.5-VL support
lands some other way), re-vendor from a recent `ggml-org/llama.cpp`
release instead of this fork and drop this whole `README.md` caveat --
check `docs/multimodal.md` in that repo for whether LFM2-VL has been
added to the supported-models list as the signal it's safe to switch back
to mainline.
