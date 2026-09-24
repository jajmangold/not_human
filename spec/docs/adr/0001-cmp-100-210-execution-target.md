# ADR-0001: CMP 100-210 execution target

- Status: Accepted
- Date: 2026-09-08
- Milestone: M1 — Contracts and adapters
- Related: Issue #35

## Context

The fleet needs a single, pinned execution target so that model, adapter, and
renderer work can be validated against one deterministic hardware contract
instead of an unbounded fleet of GPUs. The contract must name actual
hardware/resource IDs and current limitations so that evidence is auditable and
reproducible.

The target hardware is the CMP 100-210 class: a Volta-generation (GV100) device at
compute capability `sm_70`. On this class the integer `dp4a` (dot-product-4)
instruction and the CUDA-core (scalar/vector) path are the baseline execution
resources. HMMA (half-precision matrix) tensor cores are **not** part of the
pinned contract: they are disabled by default, and any FP16 tensor path is
optional and admitted only where the hardware proves HMMA support at runtime.

## Decision

Pin the CMP 100-210 execution target as an **HMMA-disabled sm70 dp4a /
CUDA-core** target:

1. **Target identity**: `CMP-100-210`, compute capability `sm_70`,
   architecture `volta`.
2. **Baseline resources enabled**: integer `dp4a` and CUDA-core execution.
3. **HMMA disabled**: HMMA tensor cores are off by default. The target is not
   an HMMA target; enabling HMMA produces a different target and is rejected by
   the contract.
4. **FP16 tensor paths optional**: an FP16 tensor path is admitted only where
   the hardware proves support (the target permits HMMA *and* the device
   reports HMMA capability). For the pinned HMMA-disabled target this is
   always false.
5. **Single source of truth**: the target, its capability flags, and the
   evidence helpers live in `src/nothuman/execution_target_v1.py`.
   `assert_target` enforces the pinned contract; `resource_ids` and
   `limitations` name the actual hardware/resource IDs and current
   limitations for evidence.

## Consequences

- Model/adapter work is validated against one deterministic target, so
  benchmarks and contract tests are reproducible in CI.
- `assert_target` rejects any candidate that deviates from the pinned
  contract (wrong id, wrong compute capability, dp4a or CUDA-core disabled,
  or HMMA enabled).
- FP16 tensor paths remain opt-in and hardware-gated, so the baseline target
  never silently depends on an HMMA path that the pinned contract forbids.
- Evidence is auditable: `resource_ids` names `sm_70`, `dp4a`, and
  `cuda-core` (and omits `hmma`), and `limitations` names the HMMA/FP16 and
  compute-capability limits.

## Alternatives considered

- **Keep HMMA enabled**: rejected — the decision is explicitly an HMMA-disabled
  target; enabling HMMA changes the target identity and is out of scope.
- **Per-device auto-detect with no pinned contract**: rejected — an unbounded
  fleet of GPUs is exactly what this ADR removes; the contract must be pinned
  and enforced.
- **Require FP16 tensor support**: rejected — FP16 tensor paths are optional
  only where hardware proves support, not a baseline requirement.

## Validation

- ADR reviewed at the exact PR head.
- Hardware/runtime contract tests (`tests/test_execution_target_v1.py`) enforce
  the target: pinned sm70 dp4a/CUDA-core identity, HMMA-disabled contract,
  FP16 tensor optional-only-where-proven, and evidence naming actual
  hardware/resource IDs (`sm_70`, `dp4a`, `cuda-core`) and current
  limitations (HMMA disabled, FP16 optional, compute capability pinned).

## Non-goals

- No broad fleet reconfiguration.
- No FlashVSR work.
- No production deployment.

## Rollback

Revert the ADR-linked PR and retain the prior documented behavior (no pinned
execution-target contract; per-device behavior as previously documented).


## Correction (open-source cleanup)

The original text and the pinned `ARCHITECTURE` constant said `turing`. `sm_70` is Volta (GV100);
Turing is `sm_75`. The constant, its test and this ADR were corrected to `volta`. Nothing else
in the decision changes.
