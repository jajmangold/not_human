# Fast, small vision-language model (LiquidAI/LFM2.5-VL-450M, 449M params)
# for the `vision` service's image-based calls (ambient caption + on-demand
# targeted query) -- see musetalk-volta/docs/VISION.md and lfm2vl/README.md
# for the full story.
#
# Built from an UNMERGED llama.cpp PR (michael-dm/llama.cpp,
# lfm2-configurable-tiling, ggml-org/llama.cpp#25524, "mtmd: read LFM2
# tiling params from GGUF metadata"). Mainline llama.cpp segfaults on this
# model's vision path without it (confirmed on two different release
# builds, days apart) -- the tiling params it reads from GGUF metadata are
# required to correctly interpret this model's image patch layout;
# without them the vision encoder either reads garbage (segfault) or
# processes a wildly wrong tile count (measured ~0.2-0.5 tokens/sec on the
# official server-cuda image using mainline). With this fix: ~150-185ms
# per request, correct answers, confirmed live 2026-09-10.
#
# Multi-stage: build with the devel image (needs nvcc), ship the runtime
# image -- same pattern as crispasr-stt/Dockerfile in this repo.

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# Open, unmerged upstream PR (ggml-org/llama.cpp#25524): LFM2 tiling params
# from GGUF metadata. Mainline segfaults on this model's vision path.
ARG LLAMA_REPO=https://github.com/michael-dm/llama.cpp
ARG LLAMA_REV=e9d636c46df330e0372087d953b249c337898792
RUN git clone "$LLAMA_REPO" . && git checkout "$LLAMA_REV"

# sm_70: this fleet's actual silicon -- see crispasr-stt/Dockerfile's
# comment for the full explanation (every "V100"-labeled card here is
# really CMP-100-210/GV100, tensor cores firmware-disabled).
#
# LLAMA_BUILD_UI=OFF + LLAMA_USE_PREBUILT_UI=OFF: the default build tries
# to fetch a prebuilt web UI bundle from an HF bucket keyed by commit,
# which doesn't have an entry for this fork's exact commit -- it partially
# succeeds (most assets download) but is missing one required file
# (loading.html), which then hard-fails the whole build at the
# llama-ui-embed step. We don't want or need the browser UI at all here
# (only the OpenAI-compatible /v1/chat/completions API), so disabling both
# the npm build and the HF-bucket fetch skips that whole broken path
# instead of trying to fix it.
RUN echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig
RUN cmake -S . -B /build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_BUILD_UI=OFF \
      -DLLAMA_USE_PREBUILT_UI=OFF \
    && cmake --build /build --target llama-server -j16

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS runtime

# libgomp1: same gotcha as crispasr-stt -- built with -fopenmp, not shipped
# by the runtime base image by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/bin/llama-server ./llama-server
COPY --from=builder /build/bin/*.so* ./lib/
ENV LD_LIBRARY_PATH=/app/lib

EXPOSE 8080

# --host 0.0.0.0: reached through docker-compose bridge port-mapping, not
# --network host -- same reasoning as crispasr-stt's ENTRYPOINT comment.
# --n-gpu-layers 99: offload everything (this is a 449M model, comfortably
# fits well under this fleet's 16GB cards -- see docs/VISION.md for actual
# measured VRAM use).
ENTRYPOINT ["/app/llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080", \
     "--n-gpu-layers", "99", \
     "-m", "/models/LFM2.5-VL-450M-Q8_0.gguf", \
     "--mmproj", "/models/mmproj-LFM2.5-VL-450m-F32.gguf"]
