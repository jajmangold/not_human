# ADR-0004: Accept IDOL/SMPL-X as a gated research `DenseBodyConditioning` capsule

- Status: Accepted
- Date: 2026-09-09
- Milestone: Phase 8 — Body and gesture renderer
- Related: ADR-0003, Issue #31 (P8-001), Issue #82, Issue #83, Issue #84

## Context

ADR-0003 established `DenseBodyConditioning` as a capsule abstraction behind
the body renderer (`rasterize({ position, normal, depth })`) and explicitly
excluded SMPL-X and AMASS as production dependencies, because both carry
non-commercial research licenses.

Since then, IDOL (yiyuzhuang/IDOL, CVPR 2025, MIT-licensed *code*) was
evaluated as a candidate dense body conditioning source: single-image
feed-forward reconstruction to an animatable SMPL-X-UV Gaussian avatar. It was
stood up end-to-end on this project's target hardware (CMP 100-210, matching
ADR-0001) and validated directly, not assumed:

- Confirmed working from a cold clone through a live, controllable HTTP demo
  driving the avatar via arbitrary SMPL-X parameters (no fixed reference
  motion required) — see Issue #82.
- Found and fixed a real rendering bug (`forward_render` unconditionally
  computed an unused KNN visibility mask plus a discarded second rasterization
  pass in eval mode): verified 1.95x speedup, taking every tested resolution
  from 2.2-3.6x short of a 30 FPS target to 1.3-1.7x short (1024px best at
  23.29 FPS). CUDA graph capture is a known, untried next lever.
- Confirmed true 4K-class rendering (2160x3024, 6.53MP) costs about the same
  as 768-1280px once the bug above is fixed (48ms/frame, 20.82 FPS,
  8.4GB peak VRAM) — pixel count is not the binding constraint in this range.
- Confirmed appearance fidelity is source-photo-quality-sensitive (a sharp,
  well-lit source photo produced a dramatic, independent improvement over a
  low-quality one) and confirmed a separate, known ceiling: the Gaussian
  point budget (~200k points) is fixed regardless of render resolution or
  source quality, capping fine texture/face/eye/mouth-interior detail.
- IDOL both requires the official SMPL-X model file at runtime and emits
  SMPL-X-space data (canonical template vertices, joint transforms) as part
  of its own output — the dependency is load-bearing, not incidental, and
  cannot be quietly avoided by only using IDOL's code.

This is real evidence IDOL is a technically viable `DenseBodyConditioning`
capsule implementation. It does not change the licensing fact ADR-0003 was
built around: SMPL-X remains a non-commercial research license, and nothing
in this evaluation obtained a commercial license for it.

## Decision

Accept IDOL as a **gated research capsule** behind the existing
`DenseBodyConditioning` interface, without reopening or weakening ADR-0003's
production stance:

1. **The interface does not change.** `DenseBodyConditioning.rasterize(...)`
   stays capsule-agnostic per ADR-0003's own design goal — an IDOL-backed
   capsule is swappable for a license-neutral one later without renderer
   changes.
2. **The IDOL/SMPL-X capsule is explicitly not production-approved.** The
   P0-006 asset manifest's license gate continues to reject SMPL-X artifacts
   by default, exactly as ADR-0003 specifies ("artifacts without a verified
   permissive license are rejected at load time"). Using this capsule
   requires an explicit development/research-only opt-in (e.g. a
   non-default environment flag checked at load time), never a production
   build default.
3. **Production shipping stays blocked** on one of: (a) resolving a
   commercial SMPL-X license (Meshcapade offers one; not pursued or costed
   as part of this ADR), or (b) a license-neutral capsule maturing to
   comparable quality (the mesh/primitive/atlas direction ADR-0003 already
   named as the default). This ADR does not choose between those — it only
   authorizes IDOL to keep being built out as a research capsule in the
   meantime, rather than being blocked entirely on an unresolved license
   question.
4. **Evidence and code land as research, not as the default path.** Any
   `nothuman`-native IDOL adapter lands outside the production dependency
   surface (research-flagged module, excluded from default builds/imports),
   carrying the same test rigor as production adapters but not the same
   shipping claim.

## Consequences

- Phase 8 (P8-001a/b/c, issues #82/#83/#84) can keep progressing on a
  technically-validated path instead of stalling on an unresolved licensing
  question — but "IDOL works" and "IDOL is shippable" remain separate claims,
  and only the first is true today.
- The commercial-SMPL-X-license decision is now explicit and visible (this
  ADR), not an implicit assumption buried in a PR.
- If a license-neutral capsule reaches comparable quality first, the IDOL
  capsule is dropped with no renderer-contract change required — the
  abstraction ADR-0003 chose is exactly what makes that cheap.
- Anyone extending the IDOL capsule inherits its real, measured limits: not
  yet real-time by a fixed margin (1.3-1.7x short, pending CUDA graph
  capture), fixed Gaussian detail ceiling, and no automated photo-to-betas
  shape fitting yet (#83).

## Alternatives considered

- **Reject IDOL entirely, wait for a license-neutral capsule to mature**:
  rejected — leaves Phase 8 with no validated dense-conditioning
  implementation at all while the license-neutral direction remains an
  open research problem, not a scheduled deliverable.
- **Resolve a commercial SMPL-X license now and ship IDOL as the default**:
  not rejected, but not decided here either — that's a real cost/business
  decision outside this ADR's scope. Left as a live option under
  Consequences, not foreclosed.
- **Treat IDOL as production-approved without a license resolution**:
  rejected outright — this would ship a non-commercial-licensed dependency
  in a production build, the exact outcome ADR-0003 exists to prevent.

## Validation

- ADR reviewed at the exact PR head.
- Real-time, resolution, and appearance-quality claims above are backed by
  measurements recorded on Issue #82 (2026-09-09 comments), not estimates.
- The research-only load-time gate is testable: default-configuration loads
  of the IDOL capsule must fail the P0-006 manifest check exactly as any
  other unlisted-license artifact would.

## Non-goals

- No change to `DenseBodyConditioning`'s renderer-facing contract.
- No commercial SMPL-X license obtained or costed by this ADR.
- No production deployment of the IDOL capsule.
- No claim that IDOL is the final or preferred long-term capsule — it is one
  validated option among the alternatives ADR-0003 already named.

## Rollback

Revert this ADR and remove the IDOL research capsule; ADR-0003's exclusion
returns to unqualified effect, and Phase 8 dense body conditioning proceeds
only on license-neutral candidates.
