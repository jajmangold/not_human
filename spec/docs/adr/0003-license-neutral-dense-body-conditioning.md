# ADR-0003: License-neutral dense body conditioning

- Status: Accepted
- Date: 2026-02-07
- Milestone: M1 — Contracts and adapters
- Related: Issue #37

## Context

CanonicalBodyV1 must feed the body renderer position, normal, and depth maps.
Sparse MediaPipe landmarks are insufficient: they are a small set of 2D/3D
points with no surface coverage, no per-pixel normals, and no occlusion-aware
depth. The renderer therefore needs a dense, per-pixel body conditioning
abstraction that can be rasterized into those three maps.

Any dependency used for this must be license-compatible with the project.
SMPL-X and AMASS are **excluded**: both are non-commercial research licenses
and are not permitted as production dependencies (see Non-goals).

## Decision

Introduce a license-neutral dense body conditioning capsule for
CanonicalBodyV1:

1. **Abstraction**: a `DenseBodyConditioning` capsule exposing
   `rasterize({ position, normal, depth })` that emits per-pixel maps for the
   body renderer. The capsule is the only place that knows how the dense
   representation is produced; the renderer consumes maps only.
2. **Representation**: a compact parametric mesh (low-poly body mesh plus a
   small set of pose parameters) is the default dense primitive. It is dense
   enough to rasterize all three maps and small enough to benchmark within a
   bounded budget. A capsule can wrap a mesh, a primitive stack, or a
   precomputed atlas without changing the renderer contract.
3. **Artifact provenance**: every dense artifact (mesh, pose prior, texture
   atlas) ships with a provenance record — source, commit/revision, and
   license identifier. Artifacts without a verified permissive license are
   rejected at load time.
4. **Phase 8 scope**: Phase 8 includes a *bounded* implementation and
   benchmark of dense body conditioning: rasterize a fixed reference pose to
   position/normal/depth maps at a fixed resolution, measure wall time and
   memory, and record the result. The benchmark is bounded (fixed input,
   fixed resolution, fixed iteration count) so it is reproducible in CI.

## Consequences

- The body renderer depends only on the three maps, not on the conditioning
  source, so the capsule can be swapped (e.g., from a generated mesh to a
  precomputed atlas) without renderer changes.
- Provenance records make license status auditable per artifact.
- The benchmark keeps dense conditioning cost visible without unbounded
  runtime.

## Alternatives considered

- **Keep sparse MediaPipe landmarks**: rejected — insufficient density for
  normal/depth maps.
- **SMPL-X / AMASS**: rejected — non-commercial research licenses violate the
  license-neutral constraint.
- **Full photogrammetry / neural implicit surfaces**: rejected — outside
  Phase 8 bounded budget and introduces heavy runtime dependencies.

## Validation

- ADR reviewed at the exact PR head.
- Body conditioning evidence identifies artifact provenance and license
  status (provenance record per artifact, verified at load).

## Non-goals

- No SMPL-X or AMASS production dependency.
- No FlashVSR work.
- No production deployment.

## Rollback

Revert the ADR-linked PR and retain the prior documented behavior (sparse
landmark conditioning only).
