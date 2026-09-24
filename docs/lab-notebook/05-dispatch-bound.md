# 05 · Dispatch-bound: the recurring shape of the bottleneck

*Every "the model is too slow" in this project turned out to be "the launches are too many."*

This page collects one pattern that showed up independently in three unrelated components, plus
the hardware facts that shape everything else. Numbers link back to the page they came from.

## The hardware

The GPUs are NVIDIA **CMP 100-210** cards (GV100 / Volta mining silicon, sm_70). Some are
flashed with a V100 VBIOS, so `nvidia-smi` calls them "Tesla V100-PCIE-12GB"; it is the same
silicon. The FP16/TF32 tensor cores are firmware-disabled: measured **6.9 TFLOP/s on the tensor
path, slower than the same card's 8.3 TFLOP/s FP32 CUDA-core path**. (Source: the fleet's kernel
project's own silicon audit, quoted in our notes; we did not re-measure it here.) The practical
rule is to never optimize a hot path onto tensor cores expecting a speedup. Plain FP32
`volta_sgemm_*` is close to the best matmul available. That is why the Dockerfiles target
`CMAKE_CUDA_ARCHITECTURES=70` and why the images pin an sm_70-capable torch.

> **We got this wrong once, in writing.** An early draft of the perception notes claimed a cuDNN
> workaround was "taking convolution off the tensor cores." There are no working tensor cores to
> take it off. Corrected in place; the launch-count part of the analysis survived.

You do not need this hardware to read the rest of this repo, but it explains why so much of it
is about avoiding work rather than doing it faster.

## Same shape, three times

### 1. MuseTalk UNet (lip sync)

`nsys` on a 101-frame run, one CMP 100-210:

- UNet: **84.3% of marked wall time (7.403 s)**, TAESD 14.2%, input projection 1.5%.
- The UNet launched **19,963 kernels** for 101 frames. Actual kernel time: **1.835 s.** The rest
  was dispatch and serialization.

Fix: a fixed-shape **CUDA graph** capturing UNet + TAESD. Output matches eager at
**72.9 dB mean PSNR** over 101 frames. Primed four-frame live batches then generate in 0.18 s
GPU / 0.35 s including blend and JPEG: **22.2 generated fps**, against an 8 fps mouth clock.

*Still not solved:* native 25 fps on the long benchmark (**15.4 fps steady**). Attention-only
kernel work can't close it because convolution dominates. A W8A8 DP4A Conv2d path is the
open item.

### 2. YOLO26n (object detection)

**94% of the measured latency was not GPU compute** (3.0 of 47.9 ms per frame; 530 launches per
frame). CUDA graph: **50.2 → 5.65 ms, 8.9x.** Details in [03](03-perception-latency.md).

### 3. OCR

GPU execution provider: **no improvement** (355–365 ms vs 250–350 ms CPU). Three sequential
unbatched calls each pay full dispatch. The fix is fewer calls, not a faster device.

### And the avatar bridge, one level up

The same disease at the systems level: a **memory leak** in the LiveKit avatar bridge (~80 MB to
3 GB+ within ~80 s) traced to an *unbounded queue* in front of a *blocking, bounded* consumer.
The publish path's internal queue holds ~100 ms, or about one frame at 12 fps. Any downstream
stall (a slow native call under GPU contention, a WebRTC reconnect) blocks it, which blocks the
consumer of the generator's output queue, and the generator's paced real-time loop never blocks
on an unbounded `put()`, so it absorbed the entire backlog as multi-MB RGB buffers. The fix bounds both queues
(output queue: 1 s of frames by default; segment queue: 4), so a downstream stall makes the
generator's own `put()` block (real backpressure) instead of growing without limit. Two
regression tests assert that the queues block when full and unblock when drained.

**Status, plainly:** this is a code-level fix from reading the call chain. It was **not
reproduced or confirmed in a live session**, and the avatar video path stayed switched off in the
deployment (`MUSE_AVATAR_ENABLED=0`). A second symptom seen alongside the leak (the agent
skipping user input during a scheduling pause) was not traced and may be unrelated.

## Things that also did not work

- **Qwen3-TTS via int8 DP4A** (a hand-written kernel port): talker cosine **0.9559** against a
  **0.999** gate on the real checkpoint (SQNR 10.5 dB vs a 30 dB gate). A selective-FP16 ablation
  topped out at 0.996. Against a deterministic oracle, **0% of complete frames matched** over the
  first 63 frames. **Rejected as a production backend**; the gates were not loosened. TTS runs
  elsewhere.
- **SDNQ int8 fused matmul on MiniCPM-V-4.6:** see [03](03-perception-latency.md).

## The sm_70 supply chain

Independent of speed, staying on this hardware is an ongoing maintenance cost:

- Stock PyPI `torch` (2.14+cu130) has dropped Volta entirely.
- The `cu126` builds are still sm_70-capable and are what the Dockerfiles use
  (`nvidia/cuda:12.6.3`).
- cuDNN 9.1 lacks an engine for some conv shapes on this arch.
- Upstream llama.cpp does not publish Linux+CUDA prebuilts, so image builds compile from source
  (~25 min for CrispASR).
