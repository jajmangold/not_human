# 03 · Perception: what can watch the person, and how often

*A talking avatar that can't see is a podcast. What fits in a 40 ms frame?*

Code: [`stack/vision/`](../../stack/vision/) (service), [`stack/upper_body/`](../../stack/upper_body/),
[`stack/lfm2vl/`](../../stack/lfm2vl/) · Data: [`data/vision_latency.csv`](../../data/vision_latency.csv) ·
Design notes: [`stack/docs/VISION.md`](../../stack/docs/VISION.md)

There was no physical camera on the test host. "Real time" was simulated by replaying two
existing clips frame by frame and timing each layer against the source frame interval. This is a
*latency* study, not a live-capture one, and accuracy was checked only informally.

![perception latency](../../figures/perception_latency.png)

Single-frame latency on one GPU, no batching, no ROI cropping, no CUDA graphs. A deliberately
cold baseline:

| layer | mean | p95 | fps | fits 25 fps? |
|---|---:|---:|---:|:---:|
| MediaPipe Face Landmarker | 7.6–9.9 ms | 19–23 ms | 101–131 | yes, large headroom |
| MediaPipe Pose Landmarker (lite) | 40.3 ms | 61.3 ms | 24.8 | borderline (~0 headroom) |
| YOLO26n | 47.9 ms | 57.7 ms | 20.9 | no (as measured; see below) |
| RapidOCR (PP-OCRv6-small, CPU) | 252–348 ms | ~400 ms | 2.9–4.0 | no |
| MiniCPM-V-4.6 Q4_K_M (Vulkan) | ~1.0 s | 1.1–1.7 s | ~1.0 | 1 Hz only |
| SmolVLM2-256M | 1.4–2.0 s | 1.8–2.6 s | 0.5–0.7 | 1 Hz only, and wrong |

The MediaPipe tasks ran on the *CPU* delegate.

## The interesting result: it isn't compute

The YOLO26n miss looked like "the model is too slow." Profiling with `nsys` said otherwise:

- Actual GPU kernel time: **89.8 ms over 30 frames, ≈ 3.0 ms/frame.** About **94%** of the
  measured 47.9 ms was Python, launch dispatch and unbatched host-to-device copies.
- **530 kernel launches per frame** (convolution running as separate im2col + GEMM).

So the next step was a CUDA graph, and it worked:

| | eager | CUDA graph | speedup |
|---|---:|---:|---:|
| YOLO26n (backbone+head, fixed 640²) | 50.2 ms | **5.65 ms** | 8.9x |
| YOLO26n-pose (raw forward) | 57.4 ms | **7.0 ms** | 8.2x |

The graphed number lands almost exactly on the profiler's ~3 ms compute estimate, which confirms
the launch-count theory directly rather than inferring it. This is the same finding, independently,
as MuseTalk's UNet ([05-dispatch-bound.md](05-dispatch-bound.md)).

**Pose caveat.** Graphed YOLO-pose is ~5.7x faster than MediaPipe lite, and found all 3 people in
frame where MediaPipe's single-pose config found one. But it emits 17 COCO keypoints, and the
control contract in [`spec/`](../../spec/) is pinned to MediaPipe's 33. It is not a drop-in; the
remap is partial at best.

## Things that did not work

- **OCR on a GPU execution provider: no gain** (355–365 ms vs 250–350 ms on CPU), after real
  effort getting `CUDAExecutionProvider` to load. `cProfile` put ~280 ms in three sequential,
  unbatched `session.run()` calls. The cost is *call count*, not device. The known fix is to run
  detection every N frames and track/crop, not to accelerate the per-frame pattern. **Also
  unverified:** the test clips contained no text, so OCR *correctness* was never checked, only
  latency.
- **VLM request batching on llama-server did not batch:** concurrency 1 → 4 → 8 gave
  1.04 → 1.03 → 1.35 img/s while per-request latency went 0.96 → 3.83 → 5.27 s. Requests queued.
  Nothing suggests the vision encoder is batched across slots in the multimodal path. Only tested
  on Vulkan; CUDA not ruled out.
- **SDNQ int8 fused matmul fails for MiniCPM-V-4.6** (`PassManager::run failed` in Triton's
  compile) while running fine in production for another model on the same fleet. Three
  PyTorch/Triton/SDNQ combinations, including the exact production one, gave the identical
  failure, and forcing a batch-4 prefill ruled out a small-M shape problem. It looks like a
  model-specific dimension/alignment trigger. It was **not diagnosed**.
- **SmolVLM2-256M** hallucinated captions ("a rope swing in a boat" for a person practising by a
  harbour) at the same latency where MiniCPM-V-4.6 was accurate on all 16 sampled frames.

## Then the swap that mattered

In the running service the image-based calls moved from MiniCPM-V-4.6 to **LFM2.5-VL-450M**:
~150–185 ms per request against 0.94–3.1 s, same or better accuracy on the tests tried.
Mainline llama.cpp *segfaults* on this model's vision path (two release builds), and the official
CUDA server image loads it but crawls at ~0.2–0.5 tokens/s. The fix is an open, unmerged upstream
PR (ggml-org/llama.cpp#25524, LFM2 tiling parameters read from GGUF metadata), which is why the
Dockerfile builds from a pinned fork commit ([`stack/lfm2vl/README.md`](../../stack/lfm2vl/README.md)).

The small model was **unreliable at transcript classification** (answered ARTIFACT for
everything, including obviously real text), so MiniCPM-V stays for that one call only. A
3B sibling handles a slower ~20 s scene-narration loop; it answers short prompts at the same
latency class and describes detail the 450M one never attempts.

## Environment traps (each cost real time)

1. **An unpinned `pip install` silently replaced `torch` 2.5.1+cu121 with 2.14+cu130, which has
   dropped Volta entirely** (`no kernel image is available`). Reinstall the sm_70-capable torch
   *last*. Any dependency bump can do this to every card in the fleet.
2. **`CUDA_VISIBLE_DEVICES` is not `nvidia-smi`'s index** unless `CUDA_DEVICE_ORDER=PCI_BUS_ID`
   is also set. llama.cpp's Vulkan backend enumerates in a *third* order. Confirm idleness per
   tool.
3. **cuDNN 9.1 has no engine for some conv shapes on sm_70** (SmolVLM2's 14×14 patch embed).
   `torch.backends.cudnn.enabled = False` works but forces unfused im2col + GEMM.
4. **A thinking-capable model silently spent its whole token budget on reasoning** (empty
   `content`, `finish_reason: "length"`). The benchmark returned 200 OK with real timings while
   measuring nothing. Set `enable_thinking: false` explicitly.

## A self-correction

The first draft of the profiling section blamed the cuDNN workaround for "taking conv off the
tensor cores." Reading the hardware notes for this fleet showed there are no working tensor
cores to lose ([05-dispatch-bound.md](05-dispatch-bound.md)). The claim was corrected in place.
The launch-count part of the story survived; the tensor-core framing did not.
