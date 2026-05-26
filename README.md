# QuantForge

**Advanced LLM Quantization Engine for Low-Bit Transformer Inference**

QuantForge benchmarks `facebook/opt-125m` across eight quantization methods and
measures perplexity, memory, latency, and output fidelity for each.  Every number
in the results directory comes from a real hardware run — no numbers are fabricated.

---

## Why Low-Bit Quantization Matters

Large language models are memory-bandwidth-bound at inference time.  A 7 B-parameter
FP16 model occupies ~14 GB of VRAM, which excludes consumer GPUs entirely.
Quantization reduces weight storage and the volume of data moved between HBM and
compute units on every forward pass:

| Precision | Bytes / weight | Relative size |
| --- | --- | --- |
| FP32 | 4 | 2× FP16 |
| FP16 | 2 | 1× (baseline) |
| INT8 | 1 | 0.5× |
| INT4 | 0.5 | 0.25× |

At INT4 the model fits in a quarter of the FP16 VRAM budget, making models like
OPT-6.7B or LLaMA-7B runnable on a 6 GB GPU.  The tradeoff is output fidelity:
quantization introduces a rounding error that manifests as increased perplexity.

---

## Methods Implemented

| # | Method | File | Key idea |
|---|--------|------|----------|
| 1 | FP16 Baseline | `scripts/run_baseline.py` | Reference model in native FP16 |
| 2 | INT8 / W8A8 | `quantization/int8.py` | Static per-channel weight quant + dynamic per-token activation quant |
| 3 | INT4 Weight-Only | `quantization/int4.py` | Symmetric INT4 per-channel, stored as int8, dequantized at runtime |
| 4 | GPTQ-style PTQ | `quantization/gptq.py` | Diagonal Hessian approximation for importance-weighted INT8 quantization |
| 5 | SmoothQuant | `quantization/smoothquant.py` | Channel-wise activation scaling to migrate quantization difficulty to weights |
| 6 | GGFU | `quantization/ggfu.py` | Group-wise + outlier-aware clipping, 4-bit, with per-layer fidelity validation |
| 7 | KV-Cache INT8 | `quantization/kv_cache.py` | Analytical memory estimation for FP16 vs INT8 KV-cache at seq 128–4096 |
| 8 | torch.fx Replace | `optimization/fx_replace.py` | Safe recursive linear-layer replacement with a unified report |
| 9 | torch.compile | `optimization/compile_model.py` | Eager vs compiled latency comparison with graceful fallback |

---

## Architecture

```
quantforge/
├── configs/
│   └── opt_125m.yaml          # Model + dataset + quantization hyper-parameters
├── data/
│   └── load_wikitext.py       # WikiText-2 tokenised sample loader
├── models/
│   └── load_model.py          # HuggingFace model/tokenizer loader + clone + memory util
├── quantization/
│   ├── int8.py                # QuantizedLinearW8A8 + recursive replace
│   ├── int4.py                # QuantizedLinearINT4 + recursive replace
│   ├── gptq.py                # GPTQ-style diagonal-Hessian PTQ
│   ├── smoothquant.py         # SmoothQuant activation scaling + W8A8
│   ├── ggfu.py                # GGFU group-wise outlier-aware quantization
│   └── kv_cache.py            # KV-cache memory estimation + comparison table
├── optimization/
│   ├── fx_replace.py          # Unified safe recursive layer replacement
│   └── compile_model.py       # torch.compile eager vs compiled benchmark
├── evaluation/
│   ├── perplexity.py          # Token-level perplexity computation
│   ├── latency.py             # Generation latency + tokens/s measurement
│   ├── memory.py              # Parameter memory + CUDA allocator metrics
│   ├── layer_error.py         # Cosine similarity + MSE vs baseline logits
│   └── benchmark.py           # JSON I/O, delta enrichment, markdown table builder
└── scripts/
    ├── run_baseline.py         # Entry point: FP16 baseline
    ├── run_quantized.py        # Entry point: any single method
    ├── run_kv_cache.py         # Entry point: KV-cache estimation
    └── run_all.py              # Entry point: full suite
results/                        # Auto-populated by the scripts
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 2. Install PyTorch (CUDA 12.x) — adjust the index URL for your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. (Optional) install the package in editable mode
pip install -e .
```

---

## Run Commands

### Individual benchmarks

```bash
# FP16 baseline
python -m quantforge.scripts.run_baseline --max_samples 256

# Quantization methods
python -m quantforge.scripts.run_quantized --method int8        --max_samples 256
python -m quantforge.scripts.run_quantized --method int4        --max_samples 256
python -m quantforge.scripts.run_quantized --method gptq        --max_samples 256
python -m quantforge.scripts.run_quantized --method smoothquant --max_samples 256
python -m quantforge.scripts.run_quantized --method ggfu        --max_samples 256

# KV-cache memory estimation (no GPU needed)
python -m quantforge.scripts.run_kv_cache

# Full suite (runs everything, then writes benchmark_table.md)
python -m quantforge.scripts.run_all --max_samples 256
```

### CLI flags

| Flag | Description | Default |
|------|-------------|---------|
| `--max_samples` | Number of WikiText-2 validation chunks | 256 |
| `--max_length` | Maximum token length per chunk | 512 |
| `--device` | `cuda` or `cpu` | `cuda` |
| `--dtype` | `float16`, `float32`, `bfloat16` | `float16` |
| `--method` | `int8`, `int4`, `gptq`, `smoothquant`, `ggfu` | required |
| `--skip_compile` | Skip torch.compile step in `run_all` | False |

---

## Metrics Explained

| Metric | Unit | Description |
|--------|------|-------------|
| `perplexity` | nats | exp(mean cross-entropy loss) — lower is better |
| `perplexity_delta` | nats | Absolute PPL increase vs FP16 baseline |
| `model_memory_mb` | MB | Sum of all parameter bytes (weight footprint) |
| `memory_reduction_pct` | % | (1 - quant_mem / baseline_mem) × 100 |
| `cuda_allocated_mb` | MB | Bytes currently allocated by the CUDA allocator |
| `cuda_reserved_mb` | MB | Bytes reserved by the caching allocator |
| `latency_ms` | ms | Mean wall-clock time per 50-token generation |
| `tokens_per_s` | tok/s | Generated tokens per second |
| `speed_change_pct` | % | Throughput delta vs baseline |
| `cosine_similarity` | [0,1] | Mean cosine similarity of logits vs baseline |
| `mse` | logit² | Mean squared error of logits vs baseline |

---

## Benchmark Table

Run `python -m quantforge.scripts.run_all --max_samples 256` to populate
`results/benchmark_table.md` with real measured numbers.

The table is not pre-filled here because all numbers must come from your hardware.

---

## Engineering Notes

### Post-Training Quantization (PTQ)

PTQ quantizes a pre-trained model without retraining.  Calibration data drives
statistics collection (activation ranges, Hessian diagonals) used to set scales.
The chief challenge is that outlier activations in a small fraction of channels
can dominate the dynamic range and degrade quantization quality for all other
channels.

### GPTQ-style Quantization

GPTQ (Frantar et al., 2022) quantizes weights column-by-column, compensating for
the quantization error of each column in subsequent columns using the inverse
Hessian of the layer's output.  This implementation approximates the Hessian
diagonal with `mean(x_i²)` over calibration activations — a stable, kernel-free
alternative.  True GPTQ uses a Cholesky-based solver for exact Hessian inversion,
which is more accurate but requires the full Hessian matrix and careful numerical
conditioning.

### SmoothQuant-style Activation Scaling

Activations in LLMs have systematic per-channel outliers: a few channels with
max absolute values 10–100× larger than the median.  SmoothQuant (Xiao et al.,
2023) divides each activation channel by a per-channel scale `s_i` and multiplies
the corresponding weight column by `s_i`, leaving the mathematical output
unchanged but balancing dynamic ranges.  The migration strength `α` controls how
much difficulty shifts from activations to weights: `s_i = max_act_i^α / max_w_i^(1-α)`.

### W8A8 Quantization

W8A8 quantizes both weights (W) and activations (A) to INT8.  Weights use static
per-output-channel scaling fixed after calibration.  Activations use dynamic
per-token scaling computed on the fly.  The forward pass dequantizes both back
to FP32 for the GEMM, which is a safe, portable path.  On hardware with INT8
GEMM support (e.g. A100 SM80 tensor cores), a dedicated kernel would avoid the
dequantization step and achieve true throughput gains.

### INT4 Weight-Only Quantization

Only weights are quantized; activations remain in the original dtype.  A scale
per output channel maps the weight range into [-7, +7] (symmetric INT4).  The
stored int8 tensor represents INT4 values — packing two INT4 values per byte
would halve the parameter memory but requires a custom kernel.  This
implementation simulates the memory benefit via dtype storage without bit-packing.

### Activation Outliers

Outlier channels arise because LLMs learn to use a small set of high-magnitude
activation channels as information highways.  These channels resist uniform
quantization: clipping them destroys information, preserving them forces the
quantization grid to be very coarse for the remaining channels.  SmoothQuant
and GGFU both handle outliers explicitly — SmoothQuant by scaling and GGFU by
clipping at a configurable percentile before quantizing.

### KV-Cache Memory

The key-value cache stores attention keys and values for each generated token
to avoid recomputing them.  Its size is:

```
bytes = 2 × batch_size × num_layers × num_heads × seq_len × head_dim × (bits/8)
```

For OPT-125M (12 layers, 12 heads, head_dim=64) at seq_len=4096 and batch=1:
- FP16: ~24 MB
- INT8: ~12 MB

At larger models and longer contexts this scales to GBs, making KV-cache INT8
compression a practical memory saving with minimal impact on generation quality.

### Memory Bandwidth

Memory bandwidth — not compute — is the bottleneck for autoregressive inference.
Each generated token requires loading all model weights from HBM.  An A100 has
~2 TB/s of HBM bandwidth.  At INT8, the OPT-125M weight read per token is ~125 MB
vs ~250 MB at FP16, so the theoretical maximum throughput doubles — though
system overhead reduces the realised gain.

### Latency vs Fidelity Tradeoffs

There is a Pareto frontier between latency (lower = better) and perplexity
(lower = better).  INT8 sits close to FP16 on the fidelity axis with a modest
latency gain.  INT4 saves the most memory but incurs a larger perplexity cost.
GPTQ and SmoothQuant improve fidelity relative to naive INT8/INT4 at the cost
of a calibration step.  GGFU adds outlier-aware clipping for a further fidelity
improvement in the grouped-quantization regime.

### torch.fx Layer Replacement

`torch.fx` symbolically traces a model to a computation graph, enabling
programmatic node inspection and replacement.  Many production quantization
stacks (e.g. `torch.ao.quantization`) use fx to pattern-match and swap
`nn.Linear` nodes.  QuantForge uses a safe recursive walk (`named_children`)
instead of full symbolic tracing because OPT uses dynamic control flow that
can break tracing.  The `fx_replace.py` module exposes the same API and report
format that a true fx-graph replacement would, making it easy to swap in a full
fx backend later.

### torch.compile Benchmarking

`torch.compile` (PyTorch ≥ 2.0) lowers Python dispatch overhead by compiling
the computation graph with TorchInductor.  Gains vary: for small models like
OPT-125M the compilation overhead may exceed the runtime gain at low batch sizes.
For larger models and sustained throughput workloads, compiled kernels can be
10–30% faster.  On Windows, Triton (the default backend) is not supported;
`torch.compile` falls back to `eager` or raises an error which QuantForge catches
and logs without aborting the benchmark.

---

## Limitations

- **Simulated INT4 storage:** INT4 values are stored as int8 tensors; actual
  bit-packing (two nibbles per byte) requires a custom kernel that is not
  included.  The memory numbers for INT4 reflect simulated per-channel scale
  overhead, not a fully packed representation.

- **Dequantized FP32 GEMM:** All quantized layers dequantize weights and
  activations before the matrix multiply.  This preserves correctness and
  portability but does not deliver the throughput benefit of native INT8 GEMM
  on Ampere/Ada tensor cores.

- **GPTQ diagonal approximation:** True GPTQ inverts the full layer Hessian
  using a Cholesky decomposition.  The diagonal approximation used here is
  faster and numerically stable but less accurate, particularly for highly
  correlated weight columns.

- **SmoothQuant input-only scaling:** This implementation scales only the
  input activations of each `nn.Linear`.  A full SmoothQuant deployment
  would also absorb the inverse scale into the preceding LayerNorm, which
  requires modifying the normalisation layer — not done here to keep the
  code self-contained.

- **Single-GPU, batch-size 1:** The benchmarks target a 6 GB consumer GPU
  with batch_size=1.  Throughput numbers will be higher at larger batch sizes
  and with data parallelism.

- **torch.compile on Windows:** Triton-based compilation is not supported on
  Windows.  The compile benchmark is included and will fall back gracefully,
  but speedup numbers may not be available without Linux/WSL2.

- **KV-cache estimation is analytical:** The KV-cache figures are computed
  from the model config, not from profiling actual Hugging Face generation
  internals.  Real KV-cache usage may differ due to padding, pre-allocation
  strategies, and framework overhead.
