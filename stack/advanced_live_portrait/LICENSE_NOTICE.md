# License notice: unknown license, and a stated non-commercial dependency

This service builds the ComfyUI-AdvancedLivePortrait node (pinned commit, fetched at build time)
and its dependencies. Checked against upstream on 2026-09-24:

1. **The node's license is unknown.** GitHub reports no license file, and the repo was last pushed
   2024-08-21. Nothing from it is in this repository; we do not publish an image that contains it.
2. **It uses InsightFace's `buffalo_l` model pack** (`insightface.app.FaceAnalysis`). InsightFace
   states that models trained on its data are non-commercial research only and that buffalo_l
   requires contacting them for a license. The LivePortrait weights bundle includes an
   `insightface` directory.
3. **The Dockerfile installs `ultralytics==8.2.0` (AGPL-3.0).** See `../vision/AGPL_NOTICE.md`.

Possible ways forward: get a license from the node's author and from InsightFace, or replace the
face-analysis step with a permissive detector (the LivePortrait README lists a MediaPipe variant
as an alternative to InsightFace, in a different node).
