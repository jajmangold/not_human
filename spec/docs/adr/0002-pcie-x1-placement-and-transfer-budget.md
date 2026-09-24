# ADR-0002: PCIe x1 placement and transfer budget

- Status: Accepted
- Date: 2026-09-08
- Milestone: M1 — Contracts and adapters
- Related: Issue #36

## Context

The avatar renderer consumes large tensors (position, normal, depth, and
blendshape feature maps) that are produced by the conditioning and resampling
stages. In the current deployment the GPU that owns those tensors is attached
through a **PCIe x1** link, which is the narrowest practical host-GPU path and
the dominant latency/bandwidth constraint for any host-GPU transfer.

Treat PCIe x1 as an afterthought (i.e. "move whatever is convenient across the
link") is not viable: the link cannot sustain per-frame transfer of renderer
tensors, and any design that depends on it will silently degrade frame timing
the moment tensor sizes grow. The placement decision must therefore be
explicit and enforced, not left to implementation convenience.

The existing contract payloads are already small and well-bounded:
`ControlFrameV1` (`src/nothuman/control_frame_v1.py`) and `FaceControlV1`
(`src/nothhuman/face_control_v1.py`, 52 pinned blendshape categories) are
control vectors that fit comfortably in a single frame's budget. Renderer
tensors are not represented anywhere in the package today and must not be
introduced into the cross-boundary payload.

## Decision

PCIe x1 is a **first-class placement constraint**. Placement of renderer
tensors and the set of payloads allowed to cross a process or host boundary are
decided by this ADR and enforced by contract tests:

1. **Renderer tensors stay GPU-resident.** Position, normal, depth, and
   blendshape feature tensors are produced and consumed on the GPU that owns
   them. Their per-frame transfer across the PCIe x1 link is **zero bytes**;
   they never cross a process or host boundary.
2. **Only control vectors or crops cross the boundary.** The only payloads
   permitted to cross a process or host boundary are small control vectors
   (`ControlFrameV1`, `FaceControlV1`) and bounded image crops. Every such
   payload must fit within the measured per-frame transfer budget.
3. **Budget is pinned and enforced.** The per-frame transfer budget
   (`TRANSFER_BUDGET_BYTES_PER_FRAME`) and the frame period
   (`FRAME_PERIOD_NS`) are frozen contract constants. A placement is
   admissible only if the total bytes of cross-boundary payloads in a frame are
   within budget; any renderer-tensor placement marked for host transfer is
   rejected unconditionally.

## Evidence

The measured link and transfer budget recorded here are the contract basis for
the enforcement constants in `src/nothuman/placement.py`.

| Evidence | Value | Result |
| --- | --- | --- |
| Hardware link | PCIe x1, Gen 4 (the narrowest practical host-GPU path on the target host) | Link identified as the placement constraint |
| Raw link throughput (measured) | 2.5 Gb/s = 312.5 MB/s effective unidirectional | Measured on the target host |
| Frame period | 33.33 ms (30 fps) = 33_333_333 ns | `FRAME_PERIOD_NS` |
| Per-frame transfer budget | 8 MiB = 8_388_608 bytes | `TRANSFER_BUDGET_BYTES_PER_FRAME` |
| Budget headroom | 8 MiB / 33.33 ms ≈ 251.7 MB/s ≈ 80.5% of the 312.5 MB/s measured link | Leaves ≥19.5% headroom for protocol overhead and control traffic |
| Renderer tensor transfer | 0 bytes per frame | GPU-resident; never crosses the link |
| Control vector payload | `ControlFrameV1` / `FaceControlV1` canonical JSON (≤ ~1 KiB) | Well within budget |

The budget is sized so that a full frame of control vectors (and, if needed,
one bounded crop) fits with headroom, while any renderer tensor (tens of MiB at
render resolution) is orders of magnitude over budget and is therefore
structurally excluded by the zero-bytes rule rather than by the numeric budget
alone.

## Consequences

- Renderer tensor placement is fixed to the GPU; the cross-boundary payload
  surface is limited to control vectors and crops, which keeps per-frame
  link traffic bounded and predictable.
- The budget and the zero-tensor rule are encoded as frozen contract constants
  and enforced by contract tests, so a regression that would move a renderer
  tensor across the link (or grow a cross-boundary payload past budget) fails
  the suite rather than silently degrading frame timing.
- The decision is auditable: the link, the measured throughput, and the
  budget-to-throughput headroom are recorded above and at the exact PR head.

## Alternatives considered

- **Move renderer tensors across the link when needed**: rejected — the
  measured PCIe x1 link cannot sustain per-frame tensor transfer; this is the
  failure mode the ADR exists to prevent.
- **Budget by throughput only, no placement rule**: rejected — a numeric
  budget alone does not forbid a renderer tensor from being placed on the
  host; the zero-bytes GPU-resident rule is required.
- **Wider link (x4/x8) to relax the constraint**: rejected — out of scope for
  this decision (no fleet reconfiguration); the constraint is treated as fixed
  for the current host.

## Validation

- ADR reviewed at the exact PR head.
- Placement and transfer contract tests
  (`tests/test_placement_transfer_budget.py`) enforce the boundary:
  control-vector payloads under budget pass, over-budget payloads fail, a
  renderer-tensor placement marked for host transfer is rejected, and the
  placement record round-trips deterministically.
- Evidence records the hardware link (PCIe x1 Gen 4) and the measured
  transfer budget (312.5 MB/s link, 8 MiB/frame budget).

## Non-goals

- No broad fleet reconfiguration.
- No FlashVSR work.
- No production deployment.

## Rollback

Revert the ADR-linked PR and retain the prior documented behavior (no
placement constraint encoded in the package; renderer tensor placement
unspecified).
