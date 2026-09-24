#!/usr/bin/env python3
"""Atomically enqueue one resident MuseTalk job.

Paths in the job JSON are container paths. Put source media below the configured
I/O mount and use `/io/...` paths when submitting a job.
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
    parser.add_argument("--video", required=True, help="container path, e.g. /io/input/portrait.png")
    parser.add_argument("--audio", required=True, help="container path, e.g. /io/input/voice.wav")
    parser.add_argument("--output", required=True, help="container path, e.g. /io/outputs/talk.mp4")
    parser.add_argument("--bbox-shift", type=int, default=0)
    parser.add_argument("--avatar-id", default=None,
                        help="Stable resident avatar id; omit to derive one from the portrait")
    parser.add_argument("--job-id", default=None)
    args = parser.parse_args()

    for label, value in (("video", args.video), ("audio", args.audio), ("output", args.output)):
        if not value.startswith("/io/"):
            parser.error(f"{label} must be under /io (got {value!r})")
    io_root = Path(os.environ.get("MUSE_TALK_IO_DIR", "./runtime"))
    jobs = io_root / "muse_jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    job_id = args.job_id or f"job-{int(time.time())}-{secrets.token_hex(4)}"
    payload = {"video": args.video, "audio": args.audio, "out": args.output,
               "bbox_shift": args.bbox_shift}
    if args.avatar_id:
        payload["avatar_id"] = args.avatar_id
    temporary = jobs / f".{job_id}.json.tmp"
    destination = jobs / f"{job_id}.json"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    print(json.dumps({"job_id": job_id, "job_file": str(destination), "payload": payload}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
