# GGFU Internal Audit — QuantForge

> **Scope:** Read-only audit. No code has been changed.
> **Date:** 2026-05-27
> **Model under test:** `facebook/opt-125m` (12 transformer layers, 125 M parameters)

---

## 1. Where GGFU Is Implemented

### Primary file

| Path | Role |
|------|------|
| `quantforge/quantization/ggfu.py` | All core logic: quantization, dequantization, layer class, model-level replacement |

### Supporting call sites

| File | How GGFU is referenced |
|------|----------------------|
| `quantforge/scripts/run_quantized.py` | Imports `apply_ggfu`; calls it with `group_size=32, bits=4, clip_percentile=99.9`; sets `is_simulated=True` and `quantization_bits=4` in the `extra` dict before memory measurement |
| `quantforge/scripts/run_all.py` | Registers `"ggfu"` as step 6 of 8 in the benchmark pipeline; passes `--method ggfu` to `run_quantized.main()` |
| `quantforge/server.py` | On FastAPI startup, if `QF_METHOD=ggfu`, calls `apply_ggfu(model, group_size=32)` (default `bits=4`, default `clip_percentile=99.9`) |
| `quantforge/optimization/fx_replace.py` | Registers `GGFULinear` in a `"ggfu"` → `(class, factory)` dispatch table for the FX graph replacement utility |
| `tests/test_quantization.py` | `test_ggfu()` — uses a mock model with one `nn.Linear(128, 64)` and asserts `cosine_similarity` is present in returned metrics |
| `quantforge/configs/opt_125m.yaml` | Documents the canonical hyperparameters: `group_size: 32`, `bits: 4`, `clip_percentile: 99.9` |

### Class and function inventory

| Symbol | Location | Purpose |
|--------|----------|---------|
| `quantize_group_wise(weight, group_size, bits, clip_percentile)` | `ggfu.py:36` | Pure function — converts an FP32 weight matrix into INT8 quantized groups + FP32 scales |
| `dequantize_group_wise(q_weight, scales, orig_in_features, group_size)` | `ggfu.py:89` | Pure function — reconstructs an FP32 weight tensor from INT8 + scales |
| `GGFULinear` | `ggfu.py:126` | `nn.Module` drop-in for `nn.Linear`; holds `q_weight` (INT8) and `scales` (FP32) as registered buffers |
| `GGFULinear.from_linear(linear, group_size, bits, clip_percentile)` | `ggfu.py:154` | Class method — quantizes an existing `nn.Linear` and returns a `GGFULinear` |
| `GGFULinear.forward(x)` | `ggfu.py:186` | Dequantizes weights on every call, then calls `F.linear` |
| `apply_ggfu(model, group_size, bits, clip_percentile, skip_names)` | `ggfu.py:206` | Recursively walks the model and replaces every eligible `nn.Linear` with `GGFULinear`; returns per-layer fidelity metrics |

---

## 2. What GGFU Stands For

The module docstring (line 2 of `ggfu.py`) is explicit:

> **GGFU: Grouped Gradient-Free Uniform Quantization.**

The same docstring adds a clarifying disclaimer:

> *"This is NOT the GGUF file format; it is a standalone algorithmic scheme that borrows the grouped + uniform approach for educational and benchmarking purposes."*

So the name is intentionally reminiscent of the GGUF file format used by llama.cpp, but it describes a custom research algorithm. The four components of the name map to:

- **Grouped** — weights are split into column groups of size 32.
- **Gradient-Free** — no training or calibration data is used during quantization (unlike GPTQ).
- **Uniform** — the quantization grid is uniform (evenly spaced levels), not learned.
- **Quantization** — the scheme produces integer-valued weight approximations.

---

## 3. Quantization Flow — Step by Step

### 3.1 Which layers are quantized

`apply_ggfu` performs a **recursive depth-first traversal** of the model. At each node it applies these two checks before replacing:

1. The layer's **name** must not contain any substring from `skip_names` (default: `("lm_head",)`).
2. The layer must be `isinstance(child, nn.Linear)` **and** `child.in_features >= 64`.

For OPT-125m this means all `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2` linear layers inside each of the 12 transformer blocks are replaced. `lm_head` is skipped by name, and any linear with fewer than 64 input features is skipped by the size guard.

### 3.2 Bit-width

The default call uses **`bits=4`** (4-bit). The arithmetic targets the range `[-7, 7]` (i.e. `2^(bits-1) - 1 = 7`). Weights are stored as **`torch.int8`** — there is no actual 4-bit packing in GGFU (see Section 5 for memory implications).

### 3.3 Group-wise quantization

Yes — GGFU uses **group-wise** quantization. The default `group_size` is **32 columns**. Each weight matrix of shape `(out_features, in_features)` is split along the `in_features` dimension into groups of 32, producing `ceil(in_features / 32)` groups per output channel.

If `in_features` is not divisible by 32, the weight is zero-padded before grouping, and the extra columns are discarded after quantization (line 83: `[:, :in_f]`).

### 3.4 Scale computation

Within each `(output_channel, group)` block:

1. The **absolute value** of every element in the group is computed.
2. The **99.9th percentile** of those absolute values is taken via `torch.quantile` — this is the clipping threshold `clip_val`.
3. `clip_val` is clamped to `1e-8` minimum to avoid division by zero.
4. The **scale** is set to `clip_val / 7`.

So the scale is directly derived from the clipping percentile, not from the full-range maximum. Each group gets its own independent FP32 scale.

### 3.5 Zero-points

**None.** GGFU uses pure symmetric quantization centred on zero.

### 3.6 Symmetric vs. asymmetric

**Symmetric.** The formula is:

```
q = round(clamp(w, -clip_val, +clip_val) / scale)
```

Zero always maps to zero. No asymmetric offset term exists.

### 3.7 Clipping

Yes — **outlier-aware clipping** is applied before quantization. The threshold is the 99.9th percentile of absolute values within each group of 32. Weights beyond this threshold are clipped to `[-clip_val, +clip_val]`. This is the key differentiator from plain max-based symmetric quantization.

### 3.8 Outlier handling

Outliers are handled by **hard clipping** to the calibrated percentile boundary before rounding. There is no channel-reordering, rotation, or SmoothQuant-style activation migration. Clipped outliers are saturated to the nearest boundary. No activation calibration data is used — the percentile is computed from the weight distribution alone.

### 3.9 Weight packing

**No packing is performed in GGFU.** The `q_weight` buffer is stored as `torch.int8` (1 byte per element). For 4-bit values in `[-8, 7]`, this wastes the upper 4 bits of every byte. No two-nibble packing is applied, unlike `quantization/int4.py` which uses `uint8` to pack two 4-bit values per byte.

### 3.10 Data types at rest

| Tensor | dtype | Shape |
|--------|-------|-------|
| `q_weight` | `torch.int8` | `(out_features, in_features)` |
| `scales` | `torch.float32` | `(out_features, ceil(in_features / group_size))` |
| `bias` | original dtype (FP16) | `(out_features,)` |

---

## 4. Forward Pass

### 4.1 Dequantization on every call

**Yes — weights are fully dequantized on every single forward call.** The `GGFULinear.forward` method (line 186) performs these steps every time:

1. Compute padding needed to align to a multiple of `group_size`.
2. Pad `q_weight` if needed (`F.pad`).
3. Reshape the padded INT8 weight to `(out_features, n_groups, group_size)`.
4. Cast INT8 groups to the input's dtype (FP16) and multiply by scales (also cast to FP16) — this is dequantization.
5. Reshape back to `(out_features, in_features)` and slice off padding.
6. Call `torch.nn.functional.linear(x, w_fp, self.bias)`.

### 4.2 Matrix multiplication

**Standard `torch.nn.functional.linear` is used.** No custom CUDA kernels, no INT4 GEMM, no fused dequantize+matmul. The matmul always runs in FP16.

### 4.3 Optimized CUDA kernel

**None.** GGFU is purely simulated quantization. The weight precision is reduced at rest, but computation always happens in floating point.

### 4.4 Temporary FP16/FP32 tensors

**Yes — a full FP16 copy of the weight matrix is allocated on every forward call.** The `w_fp` tensor of shape `(out_features, in_features)` in FP16 is created and immediately used for the GEMM, then released. For a layer like OPT-125m's `fc2` (`768→3072`), this is `768 × 3072 × 2 bytes ≈ 4.7 MB` per call. Across 96 layers and 50 decode steps, total transient allocation is approximately 22 GB of memory traffic.

---

## 5. Memory Accounting

### 5.1 `actual_storage_memory_mb`

Computed in `evaluation/memory.py` as:

```python
actual_storage_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
```

This iterates over `model.parameters()`. In `GGFULinear`, `q_weight` and `scales` are **registered buffers**, not parameters. `model.parameters()` does **not** yield buffers.

**Result:** `actual_storage_memory_mb` only counts `bias` (an `nn.Parameter`). The dominant storage tensors are invisible. The number is misleadingly small.

### 5.2 `effective_quantized_memory_mb`

Computed as:

```python
effective_quantized_mb = (total_elements * quantization_bits / 8) / (1024 ** 2)
```

where `total_elements = sum(p.numel() for p in model.parameters())` — again, parameters only. For GGFU, `quantization_bits=4` is passed, but since only bias elements are counted, the result is a tiny fraction of the actual weight storage.

**This metric does not reflect actual GGFU storage usage.**

### 5.3 Packed INT4 storage: is it counted?

**No.** GGFU does not pack weights at all (stored as INT8), and even if it did, buffers are excluded from `model.parameters()`. Nothing in `measure_memory` iterates over `model.buffers()`.

### 5.4 FP16 weights: are they retained?

**No.** The original `nn.Linear.weight` is replaced when `setattr(parent, name, ggfu_layer)` is called. The temporary FP32 copy used during quantization is a local variable that goes out of scope after `GGFULinear.from_linear` returns. A separate temporary CPU copy is made in `apply_ggfu` for fidelity metrics, then released.

---

## 6. Quality Risk — Why GGFU Has Higher Perplexity

The benchmark table shows GGFU perplexity at **49.52** (delta +4.23 vs. FP16 baseline). Here are the root causes:

### 6.1 Statistically unstable scale from small groups

A group size of 32 means the 99.9th percentile is estimated from **32 values**. `0.1% × 32 = 0.032` expected outliers per group — meaning in most groups, the clip threshold equals the group maximum and clipping does nothing. In the ~3% of groups that have an outlier, the scale is reduced to accommodate the rest of the group, which is the desired behaviour, but the transition is discontinuous and depends entirely on the distribution of that single group.

Larger group sizes (64, 128) produce more stable scale estimates.

### 6.2 Clipping behaviour is inconsistent

Due to the small group size, the 99.9th percentile clip either does nothing (no outlier in group) or aggressively clips a single value. This inconsistency means the scale quality varies widely across the weight matrix, adding noise to the reconstruction.

### 6.3 No calibration data

GGFU has no knowledge of which weight channels are activation-sensitive. Channels with small weights but high effective output contribution are quantized with the same granularity as insignificant channels. GPTQ (activation Hessian) and SmoothQuant (activation migration) avoid this.

### 6.4 Scales cast to FP16 during dequantization

The FP32 scales are cast to the input's dtype (FP16) during forward. FP16 has 10 bits of mantissa; the conversion introduces rounding in the scale itself, adding a second source of reconstruction error on top of the 4-bit quantization error.

### 6.5 Sensitive attention layers treated uniformly

`q_proj`, `k_proj`, `v_proj` — attention projection layers — are quantized with the same parameters as the large FFN layers. Attention weights are generally more sensitive to quantization noise because errors in attention scores compound across sequence positions and across layers. No per-layer sensitivity analysis is performed.

---

## 7. Speed Risk — Why GGFU Is Slower Than FP16

The benchmark shows GGFU at **1125 ms** latency and **44 tokens/s** — well below FP16 baseline.

### 7.1 Simulated quantization — no compute savings

GGFU stores weights as INT8 but all computation is FP16. The executed matmul kernel is a standard FP16 GEMM. No INT4 or INT8 GEMM, no Tensor Core low-bit path. Quantization provides **zero compute savings**.

### 7.2 Dequantization added to every forward call

For each of the ~96 linear layers, every forward call runs: an INT8→FP16 cast, an element-wise scale multiply, a reshape, and a slice. This extra work does not exist in FP16 baseline.

### 7.3 Full-size temporary FP16 tensor per call

A `(out_features, in_features)` FP16 tensor (`w_fp`) is allocated, filled, used for GEMM, and freed on every forward call. This creates cache pressure and memory bandwidth overhead proportional to model size × decode steps.

### 7.4 No kernel fusion

The dequantization (scale multiply) and the GEMM are two separate kernels. A fused dequantize+GEMM kernel would eliminate the intermediate tensor entirely, reducing memory bandwidth by roughly half for the weight data path. No such fusion is implemented.

### 7.5 Python-level overhead per layer

Shape calculations (padding, `view`, slicing) execute in Python on every forward call for every layer. While individually cheap, across 96 layers per decode step and many decode steps, this Python overhead accumulates.

---

## 8. Summary Comparison Table

| Property | GGFU | INT8 | INT4 (packed) | GPTQ |
|----------|------|------|--------------|------|
| Target bit-width | 4 | 8 | 4 | 8 |
| Storage dtype | `int8` (8 bits/weight) | `int8` (8 bits/weight) | `uint8` (4 bits/weight) | `int8` (8 bits/weight) |
| Group-wise | Yes (group=32) | No (per-channel) | Yes (group=128) | No (per-channel) |
| Clipping | Yes (99.9th pct) | No | Approx (99th pct) | No |
| Zero-points | No | No | No | No |
| Symmetric | Yes | Yes | Yes | Yes |
| Calibration data | None | None | None | Yes (activations) |
| Packing | **No** | No | **Yes** (nibble) | No |
| Dequantize on forward | Yes | Yes | Yes | Yes |
| Optimized kernel | None | None | None | None |
| `is_simulated` flag | `True` | `False` | `False` | `True` |
| `uses_packed_weights` flag | `False` | `False` | `True` | `False` |
| Memory accounting correct? | **No** (buffers excluded) | **No** | Partially | **No** |

---

## 9. Critical Findings

**Finding 1 — Memory metrics are broken for all quantized methods including GGFU.**
`measure_memory` iterates `model.parameters()` but quantized weights are stored in registered buffers (`q_weight`, `scales`). These are invisible to the metric. Both `actual_storage_memory_mb` and `effective_quantized_memory_mb` count only bias tensors and are wrong.

**Finding 2 — GGFU does not pack weights despite advertising 4-bit quantization.**
`q_weight` uses `torch.int8` (1 byte per element). The upper nibble of every byte is always zero. Real storage is 2× what a true packed INT4 would use. The `int4.py` module correctly packs two values per byte using `uint8`; GGFU does not do this.

**Finding 3 — Dequantization occurs on every forward call with no caching.**
There is no mechanism to cache the dequantized weight between decode steps. GGFU always runs slower than FP16 baseline.

**Finding 4 — The 99.9th percentile clip on groups of 32 is statistically fragile.**
With only 32 samples, the 99.9th percentile is effectively the group maximum in most cases. The clipping provides inconsistent benefit and can behave erratically depending on group-level outlier density.

**Finding 5 — No calibration data means no activation-aware sensitivity.**
Unlike GPTQ, GGFU cannot protect activation-sensitive weight channels from heavy quantization error. This is the principal algorithmic reason for higher perplexity versus GPTQ at the same 8-bit level.
