---
category: workflow
conditions:
  kernel_language: triton
  operator_family:
  - general
  soc:
  - Ascend910B
  - Ascend910_93
  - Ascend950
domain_path: compilation/triton/migration-and-tooling
evidence_strength: reproduced_empirical
id: unit-triton-ascend-precision-debug-and-verifier-workflow
inherits_from:
- unit-triton-ascend-hardware-and-compiler-constraints-contract
- unit-triton-ascend-operator-domain-constraints-contract
provenance:
  evidence_refs:
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/tooling/precision_alignment_guide.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/tooling/precision_attention_lowprec_contract.md
  - https://gitcode.com/cann/cannbot-knowledge/blob/be5882ef29fa9943305085652d2025a038ba491c/knowledge/ops/triton/guides/tooling/precision_divide_scale_calibration.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/triton-precision-debug/SKILL.md
  - https://gitcode.com/cann/cannbot-skills/blob/f305e200aa4b1e1fed73cd21ff1f083b8df25a1e/ops/triton-op-verifier/SKILL.md
  source_locator: cann/cannbot-knowledge@be5882ef29fa9943305085652d2025a038ba491c:knowledge/ops/triton/guides/tooling/precision_alignment_guide.md
  source_revision: be5882ef29fa9943305085652d2025a038ba491c
  source_type: canonical_engineering_guide
  supersedes: []
revision: 1
schema: mindie-graded-unit/1
tier: L3
title: Triton-Ascend Five-Stage Precision Alignment Debugging and Op-Verifier Workflow
triggers:
  error_signatures:
  - 1. Triton Op-Verifier Three-Gate Acceptance Workflow
  - 'Gate 1: AST Anti-Cheating & Structural Static Check (`validate_triton_impl.py`)'
  - 'Gate 2: Multi-Shape / Multi-Dtype Precision Verification (`verify.py`)'
  - 'Gate 3: CANN Profiler Kernel-Only Latency Benchmark (`benchmark.py`)'
  keywords:
  - unit-triton-ascend-precision-debug-and-verifier-workflow
---
# Triton-Ascend Five-Stage Precision Alignment Debugging and Op-Verifier Workflow

## 1. Triton Op-Verifier Three-Gate Acceptance Workflow

Before claiming functional or performance completion on any Triton-Ascend operator, run the three verification gates in sequence:

### Gate 1: AST Anti-Cheating & Structural Static Check (`validate_triton_impl.py`)
1. **No Host/PyTorch Compute Fallback (Type-3 Violation)**: `ModelNew.forward()` (and optional `self._route()` dispatcher for multi-kernel operators) must only compute shapes, allocate output buffers, and launch `@triton.jit` kernels. Any `torch.*` or `torch_npu.*` compute call inside `forward()` is rejected.
2. **Forbidden Construct Scan**: Ensure zero GPU-only launch kwargs (`num_warps`, `num_stages`, `cache_modifier`), zero in-loop `return`/`continue`/`break`, and zero `triton.cdiv` calls inside `@triton.jit`.

### Gate 2: Multi-Shape / Multi-Dtype Precision Verification (`verify.py`)
1. Evaluate across all benchmark shapes and dtypes (`fp32`, `fp16`, `bf16`, `int8`) using the 4-metric NPU standard (`MERE` maximum relative error, mean relative error, small-value absolute error ratio, and `RMSE`).
2. Typical tolerances: `fp32` `rtol=1e-6` (`MERE <= 1.22e-4`), `fp16` `rtol=1e-3` (`MERE <= 9.766e-4`), `bf16` `rtol=4e-3` (`MERE <= 7.81e-3`).
3. **Diagnostic Fingerprint Triage**:
   - **All `fp16` cases fail while `bf16`/`fp32` pass**: Arithmetic rounding path mismatch (see Section 3.2 `aclnnMuls` scalar pre-rounding or missing `.to(native)` round-trip), **not** accumulator precision insufficiency.
   - **MERE $\approx 0.442695$ ($\log_2(e) - 1$) on `lse` output**: Reference uses natural `log` (`base-e`) while kernel multiplied by `LOG2E` (`base-2`).
   - **`matched_ratio ~ 0` when `H > 1` but passes when `H == 1`**: Head-interleaved KV cache flattening formula or stride order mismatch.
   - **Off-by-1 errors on `int8` quantized outputs**: Scalar vs vector division `1–2 ULP` drift amplified by `nearbyint`/`round`, or `tl.round` (round-half-to-even) vs PyTorch round-half-away-from-zero (`tl.where(q >= 0, q + 0.5, q - 0.5)`).

### Gate 3: CANN Profiler Kernel-Only Latency Benchmark (`benchmark.py`)
- Never rely on Python-level `torch.npu.Event` elapsed time for microsecond-scale kernels (`10–50 us`), because `3x` host wrapper dispatch overhead distorts speedups. Always extract `aicore_time` / `total_time` from `torch_npu.profiler` kernel-only traces.

---

## 2. Five-Stage Precision Alignment Debugging Workflow

When an operator fails Gate 2 precision verification, execute the **Five-Stage Isolation Protocol**:

### Stage 0: Pre-Flight Sanity Check
- Verify output tensor shape, stride contiguity, `NaN` bitmask equality, and `Inf` sign/location equality.
- Verify `tl.load(..., mask=mask, other=...)` neutral padding values: `other=0.0` is valid for `tl.sum` / `tl.dot`, but corrupts `tl.max` / `tl.min` (must use `-1.0e9` / `-3.0e38`, never `0.0`).
- For in-place or `tl.atomic_add` kernels with `@triton.autotune`, verify `reset_to_zero=["out_ptr"]` is set so autotune trials do not accumulate into dirty buffers.

### Stage 1: End-to-End Divergence Localization
- Run the smallest failing shape, compute `diff = (triton_out.float() - ref_out.float()).abs()`, and print the top-10 diverging indices `(idx, triton_val, ref_val)`.

### Stage 2: Atomic Operation Micro-Benchmark Bisection
- Decompose the composite operator into individual mathematical steps (e.g., for `DynamicQuant`: `abs` -> `max` -> `scale = max_val / 127.0` -> `clamp` -> `x / scale` -> `round` -> `saturate_cast`) and compare each step 1-to-1 against PyTorch on NPU using a minimal 10-line Triton kernel.

### Stage 3: Scalar vs. Vector Codegen Path Matrix Test
Triton-Ascend lowers scalar float operations (`PIPE_S` / scalar unidad) and vector float operations (`PIPE_V`) through different instruction sequences that can differ by `1–2 ULP` on division and reciprocal approximations:

| Test Case | `BLOCK_SIZE` | `grid` | Codegen Path | Diagnostic Outcome |
|---|---|---|---|---|
| **Test A** | `1` | `(M,)` | Scalar optimization (`BLOCK_SIZE=1` folded to scalar) | Fails (`1–2 ULP` drift vs `aclnnDiv`) |
| **Test B** | `2` | `(M // 2,)` | Vector SIMD (`PIPE_V`) | Passes (`0 diff` vs `aclnnDiv`) |
| **Test C** | `2048` | `(1,)` | Vector SIMD loop | Passes (`0 diff` vs `aclnnDiv`) |
| **Test D** | Scalar `for` loop | `(1,)` | Explicit scalar (`tl.load(ptr + i) / 127.0`) | Fails (`1–2 ULP` drift) |

- **Rule**: If Test A/D fail while Test B/C pass, the root cause is **Scalar vs. Vector codegen divergence**. Never compute scale factors (`max_abs / 127.0` or `int_scale / max_abs`) on scalar variables or with `BLOCK_SIZE=1`; store `max_abs` to memory and reload as a vector tile (`BLOCK_SIZE >= 2`, typically `2048`) to force `PIPE_V` vector division.

### Stage 4: Two-Level Fix Verification
1. **Level 1 (Single-Stage Replacement)**: Replace only the diverging stage with the fixed Triton vector kernel while keeping the remaining pipeline in PyTorch; verify `max_diff == 0`.
2. **Level 2 (Full-Pipeline Integration)**: Re-integrate into the full Triton operator (using `self._route()` if split into multiple kernels) and run the full `verify.py` suite.

---

## 3. High-Frequency Golden Rounding Chain Contracts

### 3.1 Replicating Explicit Golden Dtype Cast Chains (Attention & Composite Ops)
- **Contract**: When the PyTorch reference contains explicit intermediate casts—such as `scores = torch.einsum('qhd,khd->hqk', q, k).float()` followed by `p = torch.softmax(scores, dim=-1).to(v.dtype)`—doing higher-precision single-pass `fp32` online softmax in Triton will **fail** `bf16`/`fp16` verification because the rounding points differ by `1 ULP`.
- **Mandatory Replication**:
  1. Replicate the `einsum` round-trip: `scores = tl.dot(q, tl.trans(k)).to(native_dtype).to(tl.float32)`.
  2. When `p` is cast to `v.dtype` **after** global normalization, use a 2-pass KV loop: Pass 1 computes exact global `(m_g, l_g)`; Pass 2 recomputes `scores` with the identical round-trip, computes `p = (tl.exp(scores - m_g) / l_g).to(native_dtype)`, and accumulates `tl.dot(p, v, acc)`.

### 3.2 Host-Side `fp16` Scalar Pre-Rounding (`aclnnMuls` Contract)
- **Root Cause**: In PyTorch on NPU, `tensor_fp16 * python_float` dispatches to `aclnnMuls`, which **rounds the Python float scalar to `fp16` (round-to-nearest-even) before multiplying**. In contrast, passing a Python `float` to a Triton kernel keeps the scalar in `fp32`.
- **Fix**: Pre-round `scale` on the host when `dtype == torch.float16` before passing to the kernel:
  ```python
  import struct
  def scale_for_dtype(scale: float, dtype: torch.dtype) -> float:
      if dtype == torch.float16:
          return struct.unpack("e", struct.pack("e", scale))[0]
      return scale
  ```
