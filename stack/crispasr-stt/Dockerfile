# GPU-accelerated STT for the LiveKit interface (#45), replacing the
# CPU-only faster-whisper service. Backend: CrispASR (a whisper.cpp fork),
# built from a pinned upstream commit plus one local patch (patches/, five
# fixes: see docs/lab-notebook/01-hearing-stack.md). Verified on this
# fleet's actual hardware (CMP-100-210/GV100 "Volta", sm_70, tensor cores
# firmware-disabled) in the 2026-09-09 hearing-stack evaluation:
# large-v3-turbo, 0.066 clean WER (vs. 0.211 for base.en on the same GPU server (the CPU faster-whisper small.en was never benchmarked in that harness)
# setup) and 14.1x realtime batch throughput on that hardware.
#
# Multi-stage: build with the devel image (needs nvcc), ship the runtime
# image (much smaller, no compiler toolchain) -- same pattern as this
# fleet's other CUDA services (see repos/fni8-serve, issue #194).

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
ARG CRISPASR_REPO=https://github.com/CrispStrobe/CrispASR
ARG CRISPASR_REV=ce521ee178867ceaa5fdc11803616578c8936c19
RUN git clone "$CRISPASR_REPO" . && git checkout "$CRISPASR_REV" && git submodule update --init --recursive
COPY patches/ /patches/
RUN git apply --check /patches/crispasr-livekit-stt-fixes.patch && git apply /patches/crispasr-livekit-stt-fixes.patch

# sm_70: this fleet's actual silicon (see fni8 repo -- every "V100"-labeled
# GPU here is really CMP-100-210/GV100 with tensor cores firmware-disabled,
# so this targets plain FP32 CUDA cores, not tensor-core paths).
#
# Target is `crispasr-cli` (the big unified `crispasr` binary, OUTPUT_NAME
# override -- 54 ASR backends, 52 TTS engines, all compiled in), NOT the
# leaner `examples/server` target. The OpenAI-compatible routes this
# service exists for (/v1/audio/transcriptions, /backends, /v1/models) live
# in crispasr_server.cpp, which only this target builds -- verified by
# building the lean target first and getting a 404 on
# /v1/audio/transcriptions before finding this.
# The devel image ships libcuda.so (the link-time driver stub) only under
# lib64/stubs, deliberately off the default linker path -- the real driver
# is injected at container *run* time via nvidia-container-toolkit, so the
# stub must never be picked up by ldconfig at runtime. ggml-cuda's own
# CMakeLists.txt already anticipates exactly this (find_library(...
# PATH_SUFFIXES stubs), with a comment calling out "nvidia/cuda devel
# containers" by name) and links libggml-cuda.so against the stub fine --
# confirmed via the configure log ("ggml-cuda: linking CUDA driver
# /usr/local/cuda/lib64/stubs/libcuda.so"). But the FINAL crispasr-cli
# executable link still failed with undefined references to
# cuGetErrorString etc: the stub's SONAME is libcuda.so.1 (confirmed via
# objdump -p), while the file on disk is only named libcuda.so. A plain
# symlink (libcuda.so.1 -> libcuda.so) in the same directory wasn't
# enough either -- the linker warned "libcuda.so.1 ... not found" for
# THIS target's link step specifically, because that step's search path
# doesn't include lib64/stubs at all (only the ggml-cuda sub-target's own
# explicit find_library call did). Registering the directory with
# ldconfig makes it part of the system-wide linker search path for every
# subsequent link in this stage, not just the one CMake target that
# explicitly asked for it -- link-time only, harmless: the builder and
# runtime stages are separate, so this never reaches the shipped image.
RUN echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf && ldconfig
# -j16, not -j$(nproc): this fleet's hosts run dozens of other heavy
# containers concurrently, and -j$(nproc) (48 here) badly oversubscribed
# individual compile jobs during testing -- some sat at 0% CPU for
# minutes at a time under real host load while the rest of the build
# continued (not a true deadlock, just severe scheduling contention).
RUN cmake -S . -B /build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DGGML_NATIVE=ON \
    && cmake --build /build --target crispasr-cli -j16

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS runtime

# libgomp1: the binary is built with -fopenmp (confirmed via the configure
# log -- "Found OpenMP_CXX: -fopenmp") but the runtime image doesn't ship
# libgomp.so.1 by default. Caught by actually running the built image
# (`docker run musetalk-crispasr-stt:local`), which is exactly why that
# step matters beyond a successful `docker build` -- a clean build here
# still produced a binary that failed at container startup with "error
# while loading shared libraries: libgomp.so.1: cannot open shared object
# file". Confirmed via ldd that it's the only missing runtime dependency
# (libcuda.so.1 also shows "not found" under plain ldd, but that's
# expected and correct -- it's supplied by nvidia-container-toolkit only
# when the container actually runs with --gpus, not present in any static
# image).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /build/bin/crispasr ./crispasr
COPY --from=builder /build/src/*.so* ./lib/
COPY --from=builder /build/ggml/src/*.so* ./lib/
COPY --from=builder /build/ggml/src/ggml-cuda/*.so* ./lib/
ENV LD_LIBRARY_PATH=/app/lib

EXPOSE 8097

# --host 0.0.0.0: this service is reached through docker-compose bridge
# port-mapping (127.0.0.1:18097:8097, matching the previous whisper-stt
# service), not --network host, so it must bind all interfaces inside the
# container -- the default 127.0.0.1 would be unreachable from the host
# (verified directly -- an earlier smoke test against a --network host
# setup silently could not be reached at all through -p mapping).
ENTRYPOINT ["/app/crispasr"]
CMD ["--server", "--host", "0.0.0.0", "--port", "8097", \
     "-m", "/models/ggml-large-v3-turbo.bin"]
