# License boundary: this service is not MIT

`yolo_runtime.py` imports `ultralytics` (pinned in `requirements.txt`), which is licensed
AGPL-3.0 upstream, and it loads YOLO26n weights distributed under the same terms. Anyone who
builds, distributes or network-serves this service therefore takes on AGPL-3.0 obligations for it.
The MIT license at the repository root does not change that.

Options if you need permissive terms:

- remove the object-detection layer (`/perceive` still has face, pose and caption sources);
- replace it with a permissively licensed detector behind the same `yolo_runtime` interface.

Neither has been done. The detector was kept because the latency work in
`docs/lab-notebook/03-perception-latency.md` (8.9x from a CUDA graph) is about this exact model,
and swapping it in GPU code that could not be run here would trade a licensing problem for an
untested one.
