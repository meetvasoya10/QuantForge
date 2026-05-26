"""
KV-cache INT8 memory estimation.

Provides:
  - ``quantize_tensor_int8``   – quantize any tensor to INT8.
  - ``dequantize_tensor_int8`` – reconstruct FP tensor from INT8.
  - ``estimate_kv_cache_memory`` – estimate KV-cache footprint at various
    sequence lengths and compare FP16 vs INT8.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tensor-level INT8 quantization / dequantization
# ---------------------------------------------------------------------------

def quantize_tensor_int8(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetrically quantize *tensor* to INT8 using a single global scale.

    Args:
        tensor: Float tensor of any shape.

    Returns:
        Tuple of:
            ``q_tensor`` – INT8 tensor, same shape as *tensor*.
            ``scale``    – Scalar FP32 scale factor.
    """
    max_abs = tensor.abs().max().clamp(min=1e-8)
    scale = max_abs / 127.0
    q_tensor = (tensor.float() / scale).round().clamp(-128, 127).to(torch.int8)
    return q_tensor, scale.to(torch.float32)


def dequantize_tensor_int8(
    q_tensor: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct a float tensor from INT8 quantized values.

    Args:
        q_tensor: INT8 tensor of any shape.
        scale:    Scalar FP32 scale produced by ``quantize_tensor_int8``.

    Returns:
        FP32 tensor with same shape as *q_tensor*.
    """
    return q_tensor.float() * scale


# ---------------------------------------------------------------------------
# KV-cache memory estimation
# ---------------------------------------------------------------------------

def estimate_kv_cache_memory(
    batch_size: int,
    num_layers: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    dtype_bits: int = 16,
) -> float:
    """
    Estimate KV-cache memory in megabytes for a single inference pass.

    The KV-cache stores key and value tensors for all layers:
      bytes = 2 (K+V) × batch × layers × heads × seq_len × head_dim × (bits/8)

    Args:
        batch_size:  Number of sequences processed in parallel.
        num_layers:  Number of transformer decoder layers.
        num_heads:   Number of attention heads.
        seq_len:     Sequence length (context window).
        head_dim:    Dimensionality per attention head.
        dtype_bits:  Bit-width of KV cache elements (16 for FP16, 8 for INT8).

    Returns:
        Memory estimate in MB.
    """
    bytes_per_element = dtype_bits // 8
    n_elements = 2 * batch_size * num_layers * num_heads * seq_len * head_dim
    total_bytes = n_elements * bytes_per_element
    return total_bytes / (1024 ** 2)


def compare_kv_cache(
    seq_lengths: List[int],
    batch_size: int = 1,
    num_layers: int = 12,       # OPT-125M
    num_heads: int = 12,        # OPT-125M
    head_dim: int = 64,         # hidden_size / num_heads = 768 / 12
) -> List[Dict[str, float]]:
    """
    Compare FP16 vs INT8 KV-cache memory across multiple sequence lengths.

    Args:
        seq_lengths: List of sequence lengths to evaluate.
        batch_size:  Batch size.
        num_layers:  Number of transformer layers.
        num_heads:   Number of attention heads.
        head_dim:    Per-head feature dimension.

    Returns:
        List of dicts with keys:
            ``seq_len``, ``fp16_mb``, ``int8_mb``, ``reduction_pct``.
    """
    results = []
    for sl in seq_lengths:
        fp16_mb = estimate_kv_cache_memory(batch_size, num_layers, num_heads, sl, head_dim, 16)
        int8_mb = estimate_kv_cache_memory(batch_size, num_layers, num_heads, sl, head_dim, 8)
        reduction = (1.0 - int8_mb / fp16_mb) * 100.0
        results.append({
            "seq_len": sl,
            "fp16_mb": round(fp16_mb, 4),
            "int8_mb": round(int8_mb, 4),
            "reduction_pct": round(reduction, 2),
        })
    return results


def build_kv_cache_table(comparison: List[Dict[str, float]]) -> str:
    """
    Render the KV-cache comparison as a markdown table.

    Args:
        comparison: Output of ``compare_kv_cache``.

    Returns:
        Markdown-formatted table string.
    """
    header = "| seq_len | fp16_mb | int8_mb | reduction_pct |\n"
    sep    = "| ---     | ---     | ---     | ---           |\n"
    rows = ""
    for row in comparison:
        rows += (
            f"| {row['seq_len']:>7} "
            f"| {row['fp16_mb']:>7.4f} "
            f"| {row['int8_mb']:>7.4f} "
            f"| {row['reduction_pct']:>13.2f}% |\n"
        )
    return header + sep + rows
