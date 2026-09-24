#!/usr/bin/env bash
# Downloads the model crispasr-stt's Dockerfile CMD expects, into the
# directory docker-compose.yml mounts read-only at /models
# (CRISPASR_STT_MODEL_DIR, default ./models/crispasr-stt).
#
# large-v3-turbo won the 2026-09-09 quality investigation outright (no
# tradeoff found against base.en/small.en/medium.en/large-v3 -- see
# docs/lab-notebook/01-hearing-stack.md):
# 3.2x lower WER than the previous small.en/CPU setup, more noise-robust
# at every tested SNR, and still ~14x realtime on this fleet's GV100
# "Volta" silicon.
set -euo pipefail

DEST_DIR="${CRISPASR_STT_MODEL_DIR:-./models/crispasr-stt}"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
MODEL_PATH="${DEST_DIR}/ggml-large-v3-turbo.bin"

mkdir -p "${DEST_DIR}"
if [ -f "${MODEL_PATH}" ]; then
    echo "already present: ${MODEL_PATH}"
    exit 0
fi

echo "downloading large-v3-turbo (~1.6GB) to ${MODEL_PATH} ..."
curl -L -o "${MODEL_PATH}.tmp" "${MODEL_URL}"
mv "${MODEL_PATH}.tmp" "${MODEL_PATH}"
echo "done: ${MODEL_PATH}"
