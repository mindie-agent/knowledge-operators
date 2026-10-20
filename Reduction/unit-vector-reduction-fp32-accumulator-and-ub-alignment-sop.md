---
schema: mindie-graded-unit/1
id: unit-vector-reduction-fp32-accumulator-and-ub-alignment-sop
revision: 1
tier: L2
category: sop
domain_path: Reduction
evidence_strength: software_contract
title: 'Ascend vector-reduction SOP: FP32 accumulation and 32 B UB tail alignment'
conditions:
  soc:
  - Ascend910B
  - Ascend910B1
  - Ascend910B2
  - Ascend910B3
  - Ascend910B4
  - Ascend910_93
  dtype:
  - bfloat16
  - float16
  op_family:
  - reduction
  reduction_axis:
  - last
inherits_from:
- unit-ascend-910b-a3-memory-alignment-and-tiling-axioms
- unit-ascend-operator-roofline-and-four-view-measurement
provenance:
  source_type: public_software_contract_review
  source_locator: https://github.com/Ascend/ascendc-samples
  source_url: https://github.com/Ascend/ascendc-samples
  evidence_refs:
  - https://github.com/Ascend/ascendc-samples
  - https://github.com/Ascend/pytorch
  - https://triton-lang.org/main/python-api/generated/triton.language.sum.html
  - https://triton-lang.org/main/python-api/generated/triton.language.max.html
  - https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/opdevg/Ascendcopdevg/atlas_ascendc_10_0001.html
  supersedes: []
triggers:
  error_signatures:
  - a bf16 or fp16 reduction returns NaN or Inf while the fp32 reference is finite
  - a last-axis reduction misreads the tail when the row byte length is not a multiple of 32
  - the padded UB tile is counted in the mean denominator or in the maximum
  - a timing number is published before finiteness and max-abs checks pass
  keywords:
  - reduction
  - vector reduction
  - fp32 accumulation
  - UB alignment
  - 32-byte alignment
  - DataCopyPad
  - tail mask
  - bfloat16
  - float16
  - sum
  - mean
  - amax
  - logsumexp
  - Ascend910B
  - Ascend910_93
---
# Ascend vector-reduction SOP: FP32 accumulation and 32 B UB tail alignment

## Use when

A last-axis reduction (`sum`, `mean`, `amax`, `amin`, `l2norm`, `logsumexp`, variance) on Ascend 910B/910_93 consumes `bfloat16` or `float16` and must return a half-precision output. The reduced extent, the row stride, or both need not be a whole number of 32 B UB DMA granules. A kernel whose reduction is already known to be wrong, or an op that is not a last-axis reduction, is out of scope.

Record the applicability facts before editing: SOC and device count, CANN/`torch_npu`/framework revision, input shape and stride, reduction axis, input dtype, output dtype, epsilon, tolerance, graph/eager mode, and whether the row base and row stride are 32 B aligned. A missing fact stays `unknown`; do not assume it.

## 1. Compute the row and tail geometry

For a contiguous last axis of length `N` and element size `e` bytes:

```text
row_bytes          = N * e
aligned_row_bytes  = ceil(row_bytes / 32) * 32
padding_elements   = aligned_row_bytes / e - N
```

For a 2-D `[R, N]` tensor the row stride must also be a multiple of 32 B. If only the last row is padded but `row_stride_bytes = N * e` is not 32 B aligned, every row after the first starts at a misaligned address; padding the allocation without padding the stride is not enough.

Reference values for the common dtypes:

| dtype | bytes/element | elements in 32 B | `N = 1000` row | `N = 4095` row |
| --- | ---: | ---: | --- | --- |
| bfloat16 | 2 | 16 | 2000 B, pad to 2016 B (1008 elements) | 8190 B, pad to 8192 B (4096 elements) |
| float16 | 2 | 16 | 2000 B, pad to 2016 B (1008 elements) | 8190 B, pad to 8192 B (4096 elements) |
| float32 | 4 | 8 | 4000 B, already 32 B aligned | 16380 B, pad to 16384 B (4096 elements) |

Run the geometry check on the actual NPU tensors, not on a CPU copy:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 - <<'PY'
import torch
import torch_npu

for name, t in {
    "x": torch.randn(4096, 1000, dtype=torch.bfloat16, device="npu"),
    "out": torch.empty(4096, dtype=torch.bfloat16, device="npu"),
}.items():
    row_bytes = t.shape[-1] * t.element_size() if t.dim() > 1 else t.numel() * t.element_size()
    print(name, t.dtype, tuple(t.shape), t.stride(),
          "base_mod32=", t.data_ptr() % 32,
          "row_bytes=", row_bytes,
          "row_bytes_mod32=", row_bytes % 32)
PY
```

Then choose one implementation:

- **Aligned stage:** copy or allocate with an aligned row stride and a zero/identity-filled tail, run the reduction over the padded tile, and mask the tail before the final combine.
- **`DataCopyPad` tail:** keep the natural row stride but use the platform's padded copy for the final partial granule and supply a mask. A padded copy alone does not define the tail value; the mask must.
- **Per-row scalar tail:** reduce the aligned prefix in the vector unit and add the remaining `N mod tile` elements with an explicit masked path. This is valid but usually loses vector throughput; measure it rather than assuming.

Never read an uninitialized tail. A deterministic fill is required even when the reduction's identity would make the value irrelevant, because NaN, Inf or a signaling value in the tail still contaminates the result.

## 2. Fix the numerical contract before tuning

- Load `bfloat16`/`float16` into the UB tile, convert to `float32` before the first accumulation, and keep the running accumulator and all partial sums in `float32`.
- Apply the reduction tree in `float32`. Do not cast a partial sum back to half between levels; that silently lowers the accumulator precision.
- Apply `epsilon` in `float32` only where the formula requires it (`l2norm`, variance, normalized reductions). Do not add epsilon to a plain `sum` or `mean`.
- Cast once at the store to the declared output dtype. Returning `float32` when the contract says `bfloat16`/`float16` is dtype drift, not extra accuracy.
- Use the correct identity for the padded lanes: `0` for `sum`/`mean`/`l2norm`, `-inf` for `amax`/`logsumexp`, `+inf` for `amin`. For `mean`, divide by the true `N`, never by `aligned_row_bytes / e`.
- For `logsumexp`, compute the row maximum in `float32`, subtract it, exponentiate and sum in `float32`, then add the maximum back before the final cast. Padding must not enter the maximum.

For Triton-Ascend, keep the tile extent used by `tl.arange` static, build a `col < N` mask for the tail, and use the identity as the `other=` value. `tl.sum` and `tl.max` over an axis are valid only when the masked lanes carry the correct identity. If the row stride is not 32 B aligned, materialize an aligned view or pass the natural strides to a kernel that explicitly handles them; a flat pointer plus `N` is not a layout contract.

## 3. Correctness gate before any timing

Freeze an `float32` reference and the inputs. For each shape, including an aligned shape, a `N mod 16` tail and a `N mod 16 == 1` tail, run:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 - <<'PY'
import torch
import torch_npu

def check(x, out, axis=-1, kind="sum"):
    torch.npu.synchronize()
    xf = x.float()
    if kind == "sum":
        ref = xf.sum(dim=axis)
    elif kind == "mean":
        ref = xf.mean(dim=axis)
    elif kind == "amax":
        ref = xf.amax(dim=axis)
    else:
        raise ValueError(kind)
    out_f = out.float()
    finite = bool(torch.isfinite(out_f).all().item())
    abs_err = (out_f - ref).abs()
    rel_err = abs_err / ref.abs().clamp_min(1e-6)
    print("finite=", finite,
          "dtype=", out.dtype,
          "max_abs=", abs_err.max().item(),
          "max_rel=", rel_err.max().item())
    assert finite, "NaN/Inf in reduction output"
    assert out.dtype == torch.bfloat16, "output dtype drift"
PY
```

For every case, require all of the following before recording a latency:

1. **Finiteness:** no NaN and no Inf, checked on the NPU output after `torch.npu.synchronize()`.
2. **Max-abs drift:** the maximum absolute error against the frozen `float32` reference is inside the pre-registered absolute tolerance. Report the value; do not report only a mean error.
3. **Relative drift:** the maximum relative error is inside tolerance where the reference is not near zero. Keep the near-zero mask and its count visible.
4. **Dtype:** the returned dtype is exactly the declared output dtype.
5. **Input preservation:** hash or compare the input before and after; the kernel must not mutate it.
6. **Repeatability:** run the same input at least three times and compare the output bytes. A changing result on the same device and kernel is a determinism failure, not noise.
7. **Tail-only control:** for `N` just below and just above a 32 B boundary, the error must not jump. A jump localizes the defect to the tail copy or mask.

If any check fails, repair the accumulator dtype, the tail identity, the mask, or the row stride before timing. A failed correctness gate is not a slow kernel; it is an invalid measurement.

## 4. Timing gate

Only after the correctness gate passes:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python3 - <<'PY'
import time
import torch
import torch_npu

x = torch.randn(4096, 1000, dtype=torch.bfloat16, device="npu")
# call your_reduce(x) here
for _ in range(2):
    your_reduce(x)
torch.npu.synchronize()

for block in range(3):
    torch.npu.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(12):
        your_reduce(x)
    torch.npu.synchronize()
    t1 = time.perf_counter_ns()
    print("block", block, "mean_us", (t1 - t0) / 12 / 1000)
PY
```

Keep one case identity per sample, exclude at least two warmups, synchronize immediately before and after each block, and report per-case block envelopes rather than one pooled IQR. Run the aligned and tail cases separately; do not average them into one speedup.

For attribution, profile the same invocation:

```bash
msprof --application="python3 bench_reduce.py --shape 4096x1000 --dtype bf16" \
  --output=./prof_reduce --ai-core=on --aic-metrics=PipeUtilization --l2=on
python3 - <<'PY'
import glob
import pandas as pd

path = sorted(glob.glob("prof_reduce/**/op_summary_*.csv", recursive=True))[-1]
df = pd.read_csv(path)
keep = [c for c in [
    "op_name", "op_type", "task_id",
    "aicore_time(us)", "aiv_time(us)",
    "aiv_vec_time(us)", "aiv_mte2_time(us)", "aiv_mte3_time(us)",
    "aic_mte2_time(us)", "aic_mac_time(us)",
] if c in df]
print(path)
print(df[keep].to_string(index=False))
PY
```

`aiv_mte2_time(us)` dominant with low `aiv_vec_time(us)` means the vector unit is waiting on GM-to-UB input; `aiv_mte3_time(us)` dominant means it is waiting on UB-to-GM writeback; `aiv_vec_time(us)` dominant means the vector ALU is the limit. Do not merge AIC and AIV MTE2 columns: they describe different memory paths. If the profiler does not export these columns, record `pipeline signal unavailable` and route to the L3 convergence workflow instead of guessing a bound.

## 5. Disposition

- **Tail or stride defect:** fix the row stride, use a masked padded copy, and re-run the tail-only control. Do not tune tile size around a wrong tail.
- **Precision or dtype defect:** restore the `float32` accumulator, remove intermediate half casts, correct the denominator/identity, and re-run all correctness checks.
- **NaN/Inf on a finite reference:** find the uninitialized tail, overflow, invalid `log`/`sqrt` domain, or missing epsilon; keep the failed artifact visible and rerun from a clean input.
- **All checks pass:** record the output hashes, tolerances, timing blocks and profiler artifact. Then use the L3 convergence workflow if the kernel is still slower than its matched baseline.
- **Shape outside this SOP:** collect a new alignment geometry and oracle; a pass at `N = 1000` is not a claim for `N = 4095`, another dtype, another SOC, or a different reduction kind.

## Claim boundaries

This is a correctness-and-measurement SOP, not a speedup claim. A finite output is not a proof of the intended reduction order; a matching mean can hide a wrong max; one shape does not establish a workload envelope. The inherited L1 alignment and measurement premises apply independently. A performance claim requires the matched baseline, synchronized blocks and per-case envelope defined by the inherited measurement contract.
