# License notice: research use only until resolved

This service builds the ComfyUI-AdvancedLivePortrait node (pinned commit, fetched at build time)
and its dependencies. Three separate problems, all checked against upstream on 2026-09-24:

1. **The node has no license.** GitHub reports none; the repo was last pushed 2024-08-21.
   No license means all rights reserved. Nothing from it is in this repository, but a built image
   would contain it, so **do not publish this image.**
2. **It uses InsightFace's `buffalo_l` model pack** (`insightface.app.FaceAnalysis`). InsightFace
   states that models trained on its data are non-commercial research only and that buffalo_l
   requires contacting them for a license. The LivePortrait weights bundle includes an
   `insightface` directory.
3. **The Dockerfile installs `ultralytics==8.2.0` (AGPL-3.0).** See `../vision/AGPL_NOTICE.md`.

Options: obtain a license from the node's author and from InsightFace, or replace the
face-analysis step with a permissive detector (the LivePortrait README lists a MediaPipe variant
as an alternative to InsightFace, in a different node).
