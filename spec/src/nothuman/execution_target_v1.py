"""CMP 100-210 execution target contract (ADR-0001).

Defines the pinned execution target for the nothuman fleet: an HMMA-disabled
sm70 (Volta) dp4a / CUDA-core target. FP16 tensor paths are optional and only
admitted where the hardware proves support. This module is the single source of
truth for the target's identity, capability flags, and the evidence (resource
IDs + current limitations) that the ADR requires to be named.

The interface is stable and deterministic: the target is a frozen dataclass,
``assert_target`` validates a candidate against the pinned contract, and the
evidence helpers (``resource_ids`` / ``limitations``) return identical values
for identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

TARGET_ID = "CMP-100-210"
COMPUTE_CAPABILITY = "sm_70"
ARCHITECTURE = "volta"


@dataclass(frozen=True)
class ExecutionTarget:
    """A named execution target with pinned capability flags."""

    target_id: str
    compute_capability: str
    architecture: str
    dp4a_enabled: bool
    hmma_enabled: bool
    cuda_core_enabled: bool


@dataclass(frozen=True)
class GpuCaps:
    """Hardware-proven runtime capability report for a device."""

    hmma: bool = False
    fp16: bool = False


# The pinned CMP 100-210 target: HMMA disabled, sm70 dp4a / CUDA-core enabled.
CMP_100_210 = ExecutionTarget(
    target_id=TARGET_ID,
    compute_capability=COMPUTE_CAPABILITY,
    architecture=ARCHITECTURE,
    dp4a_enabled=True,
    hmma_enabled=False,
    cuda_core_enabled=True,
)


def assert_target(target: ExecutionTarget) -> None:
    """Validate that ``target`` matches the pinned CMP 100-210 contract.

    Raises ``ValueError`` if the target id, compute capability, architecture,
    or capability flags deviate from the pinned contract.
    """
    if target.target_id != TARGET_ID:
        raise ValueError(f"target id must be {TARGET_ID!r}, got {target.target_id!r}")
    if target.compute_capability != COMPUTE_CAPABILITY:
        raise ValueError(
            f"compute capability must be {COMPUTE_CAPABILITY!r}, got {target.compute_capability!r}"
        )
    if target.architecture != ARCHITECTURE:
        raise ValueError(
            f"architecture must be {ARCHITECTURE!r}, got {target.architecture!r}"
        )
    if not target.dp4a_enabled:
        raise ValueError("dp4a must be enabled for the CMP 100-210 target")
    if not target.cuda_core_enabled:
        raise ValueError("cuda-core must be enabled for the CMP 100-210 target")
    if target.hmma_enabled:
        raise ValueError("hmma must be disabled for the CMP 100-210 target")


def fp16_tensor_supported(target: ExecutionTarget, caps: GpuCaps) -> bool:
    """Return whether an FP16 tensor path is admitted for ``target``.

    FP16 tensor paths are optional only where the hardware proves support:
    the target must permit HMMA and the device must report HMMA capability.
    For the pinned HMMA-disabled CMP 100-210 target this is always False.
    """
    return bool(target.hmma_enabled and caps.hmma)


def resource_ids(target: ExecutionTarget) -> tuple[str, ...]:
    """Actual hardware/resource IDs that name the target for evidence.

    Names the compute capability, the integer dp4a resource, and the CUDA-core
    resource. HMMA is intentionally absent because it is disabled for this
    target.
    """
    ids = [target.compute_capability]
    if target.dp4a_enabled:
        ids.append("dp4a")
    if target.cuda_core_enabled:
        ids.append("cuda-core")
    if target.hmma_enabled:
        ids.append("hmma")
    return tuple(ids)


def limitations(target: ExecutionTarget) -> tuple[str, ...]:
    """Current limitations of the target, named for evidence."""
    lims = [
        f"HMMA tensor cores disabled on {COMPUTE_CAPABILITY}; no HMMA FP16/FP32 tensor path",
        "FP16 tensor paths are optional only where hardware proves HMMA support",
        f"compute capability pinned to {COMPUTE_CAPABILITY}; newer archs out of scope",
    ]
    return tuple(lims)
