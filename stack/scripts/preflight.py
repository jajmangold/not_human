#!/usr/bin/env python3
"""Check the host-side model and I/O mounts before starting Compose."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


REQUIRED = (
    "musetalkV15/unet.pth",
    "musetalkV15/musetalk.json",
    "whisper/config.json",
    "whisper/pytorch_model.bin",
    "sd-vae/config.json",
    "sd-vae/diffusion_pytorch_model.bin",
    "dwpose/dw-ll_ucoco_384.pth",
    "face-parse-bisent/79999_iter.pth",
    "face-parse-bisent/resnet18-5c106cde.pth",
)
S3FD_RELATIVE = "hub/checkpoints/s3fd-619a316812.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dotenv() -> None:
    """Load simple KEY=VALUE entries without adding a runtime dependency."""
    dotenv = Path(".env")
    if not dotenv.is_file():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    load_dotenv()
    model_root = Path(os.environ.get("MUSE_TALK_MODEL_CACHE", "./models/musetalk"))
    io_root = Path(os.environ.get("MUSE_TALK_IO_DIR", "./runtime"))
    torch_root = Path(os.environ.get("MUSE_TALK_TORCH_CACHE", "./runtime-data/torch-cache"))
    kokoro_root = Path(os.environ.get("KOKORO_MODEL_DIR", "./models/kokoro-82m-v1.0-onnx"))
    missing = [relative for relative in REQUIRED if not (model_root / relative).is_file()]
    if missing:
        print(f"missing model files under {model_root}:")
        for relative in missing:
            print(f"  {relative}")
        return 2
    (io_root / "muse_jobs").mkdir(parents=True, exist_ok=True)
    (io_root / "outputs").mkdir(parents=True, exist_ok=True)
    s3fd = torch_root / S3FD_RELATIVE
    if not s3fd.is_file():
        print(f"missing cached face detector (prevents an implicit network download): {s3fd}")
        return 3
    if os.environ.get("MUSE_KOKORO_ENABLED", "1") == "1":
        kokoro_files = (kokoro_root / "onnx" / os.environ.get("KOKORO_MODEL_FILE", "model_q8f16.onnx"), kokoro_root / "voices.npz")
        missing_kokoro = [str(path) for path in kokoro_files if not path.is_file()]
        if missing_kokoro:
            print("missing Kokoro assets:")
            for path in missing_kokoro:
                print(f"  {path}")
            return 5
    unet = model_root / "musetalkV15/unet.pth"
    print(f"model_root={model_root}")
    print(f"io_root={io_root.resolve()}")
    print(f"torch_cache={torch_root.resolve()}")
    print(f"kokoro_root={kokoro_root.resolve()}")
    print(f"unet_bytes={unet.stat().st_size}")
    print(f"unet_sha256={sha256(unet)}")
    taesd = model_root / "taesd/diffusion_pytorch_model.safetensors"
    if os.environ.get("MUSE_USE_TAESD", "1") == "1" and not taesd.is_file():
        print(f"TAESD requested but missing: {taesd}")
        return 4
    print("preflight=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
