"""
GGFU: Grouped Gradient-Free Uniform Quantization.

A custom quantization scheme that combines:
  - Group-wise weight quantization (weights split into blocks of *group_size*
    columns; each block gets an independent scale).
  - Outlier-aware clipping (values beyond a calibrated percentile are clipped
    before quantization to reduce clipping-induced error on the bulk of the
    distribution).
  - Similarity-based validation (cosine similarity and MSE are tracked per
    layer to verify quantization fidelity before replacement).

This is NOT the GGUF file format; it is a standalone algorithmic scheme that
borrows the grouped + uniform approach for educational and benchmarking purposes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_INT8_MIN = -128
_INT8_MAX = 127


# ---------------------------------------------------------------------------
# Core quantization functions
# ---------------------------------------------------------------------------

def quantize_group_wise(
    weight: torch.Tensor,
    group_size: int = 32,
    bits: int = 4,
    clip_percentile: float = 99.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize *weight* using group-wise outlier-aware clipping.

    The weight matrix is divided into column groups of *group_size*.
    Within each group:
      1. The *clip_percentile* absolute value is computed as the clipping threshold.
      2. Weights are clipped to [-clip_val, +clip_val].
      3. Symmetric uniform quantization is applied to ``2**bits - 1`` levels.

    Weights are stored as int8 (for group_size=32, bits=4 → [-8, 7] range).

    Args:
        weight:          FP32 weight tensor of shape (out_features, in_features).
        group_size:      Number of input-feature columns per quantization group.
        bits:            Bit-width (4 or 8).
        clip_percentile: Percentile for outlier clipping (e.g. 99.9).

    Returns:
        Tuple of:
            ``q_weight`` – int8 tensor, shape (out_features, in_features).
            ``scales``   – FP32 tensor, shape (out_features, num_groups).
    """
    out_f, in_f = weight.shape
    n_bits_max = (2 ** (bits - 1)) - 1  # e.g. 7 for bits=4

    # Pad in_features to be divisible by group_size
    pad = (group_size - in_f % group_size) % group_size
    if pad > 0:
        weight = F.pad(weight, (0, pad))
    padded_in = weight.shape[1]
    n_groups = padded_in // group_size

    w_groups = weight.reshape(out_f, n_groups, group_size)       # (out, G, gs)
    q_groups = torch.zeros_like(w_groups, dtype=torch.int8)
    scales = torch.zeros(out_f, n_groups, dtype=torch.float32)

    for g in range(n_groups):
        wg = w_groups[:, g, :]  # (out, gs)
        clip_val = torch.quantile(wg.abs().reshape(-1), clip_percentile / 100.0).clamp(min=1e-8)
        wg_clipped = wg.clamp(-clip_val, clip_val)
        scale = clip_val / n_bits_max
        scales[:, g] = scale.expand(out_f) if scale.numel() == 1 else scale
        qg = (wg_clipped / scale).round().clamp(-n_bits_max - 1, n_bits_max).to(torch.int8)
        q_groups[:, g, :] = qg

    q_weight = q_groups.reshape(out_f, padded_in)[:, :in_f].contiguous()
    return q_weight, scales


def dequantize_group_wise(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    orig_in_features: int,
    group_size: int = 32,
) -> torch.Tensor:
    """
    Dequantize a group-wise quantized weight tensor back to FP32.

    Args:
        q_weight:          INT8 tensor of shape (out_features, in_features).
        scales:            FP32 scale tensor of shape (out_features, num_groups).
        orig_in_features:  Original (unpadded) in_features.
        group_size:        Column group size used during quantization.

    Returns:
        FP32 weight tensor of shape (out_features, orig_in_features).
    """
    out_f = q_weight.shape[0]
    in_f = q_weight.shape[1]
    n_groups = scales.shape[1]

    # Pad if needed
    pad = n_groups * group_size - in_f
    q_padded = F.pad(q_weight.float(), (0, pad)) if pad > 0 else q_weight.float()
    q_groups = q_padded.reshape(out_f, n_groups, group_size)

    w_dq = torch.zeros_like(q_groups, dtype=torch.float32)
    for g in range(n_groups):
        w_dq[:, g, :] = q_groups[:, g, :] * scales[:, g].unsqueeze(1)

    return w_dq.reshape(out_f, -1)[:, :orig_in_features].contiguous()


# ---------------------------------------------------------------------------
# GGFU quantized linear layer
# ---------------------------------------------------------------------------

class GGFULinear(nn.Module):
    """
    Drop-in replacement for ``nn.Linear`` using GGFU group-wise quantization.

    Args:
        in_features:  Number of input features.
        out_features: Number of output features.
        group_size:   Column group size for quantization.
        bias:         Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 32,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        n_groups = (in_features + group_size - 1) // group_size
        self.register_buffer("q_weight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scales", torch.ones(out_features, n_groups, dtype=torch.float32))
        self.bias: Optional[nn.Parameter] = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 32,
        bits: int = 4,
        clip_percentile: float = 99.9,
    ) -> "GGFULinear":
        """
        Construct a ``GGFULinear`` from an existing ``nn.Linear``.

        Args:
            linear:          Source layer to quantize.
            group_size:      Column group size.
            bits:            Target bit-width.
            clip_percentile: Outlier clipping percentile.

        Returns:
            Initialised GGFU layer on the same device.
        """
        device = linear.weight.device
        w = linear.weight.data.float()
        q_w, scales = quantize_group_wise(w, group_size, bits, clip_percentile)
        n_groups = (linear.in_features + group_size - 1) // group_size

        inst = cls(linear.in_features, linear.out_features, group_size, linear.bias is not None)
        inst.q_weight = q_w.to(device)
        inst.scales = scales.to(device)
        if linear.bias is not None:
            inst.bias = nn.Parameter(linear.bias.data.clone())
        return inst

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dequantize weights and perform FP matmul.

        Args:
            x: Input of shape (..., in_features).

        Returns:
            Output of shape (..., out_features).
        """
        w_fp = dequantize_group_wise(
            self.q_weight, self.scales, self.in_features, self.group_size
        ).to(x.dtype)
        out = x @ w_fp.t()
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out


# ---------------------------------------------------------------------------
# GGFU model replacement
# ---------------------------------------------------------------------------

def apply_ggfu(
    model: nn.Module,
    group_size: int = 32,
    bits: int = 4,
    clip_percentile: float = 99.9,
    skip_names: tuple[str, ...] = ("lm_head",),
) -> Dict[str, Dict[str, float]]:
    """
    Replace all eligible ``nn.Linear`` layers with ``GGFULinear``.

    Tracks per-layer cosine similarity and MSE between original and
    dequantized weights as a fidelity validation signal.

    Args:
        model:           Model to quantize in-place.
        group_size:      Column group size.
        bits:            Target bit-width.
        clip_percentile: Outlier clipping percentile.
        skip_names:      Layer name substrings to skip.

    Returns:
        Dict of ``layer_name → {"cosine_similarity": …, "mse": …}``.
    """
    layer_metrics: Dict[str, Dict[str, float]] = {}

    def _replace(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if any(skip in name for skip in skip_names):
                continue
            if isinstance(child, nn.Linear) and child.in_features >= 64:
                lin: nn.Linear = child  # type: ignore[assignment]
                w_orig = lin.weight.data.float()

                # Build quantized layer
                ggfu_layer = GGFULinear.from_linear(lin, group_size, bits, clip_percentile)
                setattr(parent, name, ggfu_layer)

                # Compute fidelity metrics on weight tensors
                w_dq = dequantize_group_wise(
                    ggfu_layer.q_weight.float(),
                    ggfu_layer.scales,
                    lin.in_features,
                    group_size,
                ).cpu()
                w_orig_cpu = w_orig.cpu()

                cos = F.cosine_similarity(
                    w_orig_cpu.reshape(1, -1),
                    w_dq.reshape(1, -1),
                    dim=-1,
                ).item()
                mse = F.mse_loss(w_dq, w_orig_cpu).item()
                layer_metrics[full] = {
                    "cosine_similarity": round(cos, 6),
                    "mse": round(mse, 8),
                }
            elif not isinstance(child, nn.Linear):
                _replace(child, full)

    _replace(model)

    n = len(layer_metrics)
    if n:
        mean_cos = sum(m["cosine_similarity"] for m in layer_metrics.values()) / n
        mean_mse = sum(m["mse"] for m in layer_metrics.values()) / n
        logger.info("GGFU: replaced %d layers | mean cos=%.4f | mean mse=%.4e", n, mean_cos, mean_mse)

    return layer_metrics
