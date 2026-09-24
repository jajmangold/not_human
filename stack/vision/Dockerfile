# CUDA runtime (not devel -- neither YOLO26n's nor MiniCPM-V's torch wheel
# needs anything compiled from source here) matching this fleet's actual
# silicon: sm_70 (Volta/CMP-100-210, tensor cores firmware-disabled -- see
# the 2026-09-09 nothuman vision-stack spike and the fni8 repo's
# AGENTS.md). cu126 base to match the torch/torchvision pin below, and this
# fleet's own production ComfyUI image (yanwk/comfyui-boot:cu126-megapak),
# which established that cu126 is the sm_70-safe index for this hardware.
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# Reinstall torch/torchvision from the cu126 wheel index as a separate,
# later step -- see requirements.txt's comment. Version matches what was
# actually verified working (real YOLO detections, real MiniCPM-V
# captioning/batching/classification, all smoke-tested live) in this
# session's vision-stack-test venv, not a guess.
RUN pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.9.1 torchvision==0.24.1
COPY yolo_runtime.py minicpmv_runtime.py lfm2vl_client.py app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8098"]
