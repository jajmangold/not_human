#!/usr/bin/env python3
"""Atomically enqueue one prepared-avatar MuseTalk audio chunk.

The resident emits ``00000000.png`` ... into ``--stream-dir`` as the chunk is
decoded. The directory is shared through ``/io``; a ``.done`` marker is created
by the resident after the last frame. Audio chunks should be short (typically
0.5--2 seconds) and use the same stable ``--avatar-id`` for a conversation.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--avatar-id", required=True)
    parser.add_argument("--video", required=True, help="container path, e.g. /io/input/portrait.png")
    parser.add_argument("--audio", required=True, help="short WAV chunk under /io")
    parser.add_argument("--stream-dir", required=True, help="frame output directory under /io")
    parser.add_argument("--bbox-shift", type=int, default=0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Optional microbatch size; smaller values reduce first-frame latency")
    parser.add_argument("--stream-format", choices=("png", "jpg", "jpeg"), default="png",
                        help="Frame encoding written to the stream directory")
    parser.add_argument("--chunk-id", default=None)
    args = parser.parse_args()

    for label, value in (("video", args.video), ("audio", args.audio), ("stream_dir", args.stream_dir)):
        if not value.startswith("/io/"):
            parser.error(f"{label} must be under /io (got {value!r})")
    io_root = Path(os.environ.get("MUSE_TALK_IO_DIR", "./runtime"))
    jobs = io_root / "muse_jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    chunk_id = args.chunk_id or f"chunk-{int(time.time())}-{secrets.token_hex(4)}"
    payload = {
        "avatar_id": args.avatar_id,
        "video": args.video,
        "audio": args.audio,
        "stream_dir": args.stream_dir,
        "bbox_shift": args.bbox_shift,
        "fps": args.fps,
        "stream_format": args.stream_format,
    }
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be positive")
        payload["batch_size"] = args.batch_size
    temporary = jobs / f".{chunk_id}.json.tmp"
    destination = jobs / f"{chunk_id}.json"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    print(json.dumps({"chunk_id": chunk_id, "job_file": str(destination), "payload": payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
