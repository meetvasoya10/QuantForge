"""
INT8 / W8A8 quantization.

Implements:
  - Per-output-channel INT8 weight quantization (static).
  - Dynamic per-token activation quantization at runtime.
  - Safe PyTorch dequantization path (no custom CUDA kernels).
  - ``replace_linear_with_int8`` - recursive module replacement.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_INT8_MIN = -128
_INT8_MAX = 127


def quantize_weight_per_channel_int8(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize *weight* to INT8 using per-output-channel symmetric scaling.

    Each output channel is scaled independently so that the maximum absolute
    value maps to ±127.  Scales are stored as FP32.

    Args:
        weight: 2-D float tensor of shape (out_features, in_features).

    Returns:
        Tuple of:
            ``q_weight``  - INT8 tensor of shape (out_features, in_features).
            ``scale``     - FP32 scale tensor of shape (out_features, 1).
    """
    max_abs = weight.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    scale = max_abs / 127.0
    q_weight = (weight / scale).round().clamp(_INT8_MIN, _INT8_MAX).to(torch.int8)
    return q_weight, scale.to(torch.float32)


def quantize_activation_per_token_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dynamically quantize activation *x* to INT8 per-token (per-row).
    """
    max_abs = x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
    scale = max_abs / 127.0
    q_x = (x / scale).round_().clamp_(_INT8_MIN, _INT8_MAX).to(torch.int8)
    return q_x, scale


class QuantizedLinearW8A8(nn.Module):
    """
    Drop-in replacement for ``nn.Linear`` using W8A8 INT8 quantization.

    Weights are quantized once at construction time (static, per-channel).
    Activations are quantized dynamically at each forward pass (per-token).
    The matrix multiply is carried out in FP32 (dequantized path) to avoid
    requiring INT8 GEMM support from the hardware.

    Args:
        in_features:  Number of input features.
        out_features: Number of output features.
        bias:         Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Populated in ``from_linear``
        self.register_buffer("q_weight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("w_scale", torch.ones(out_features, 1, dtype=torch.float32))
        self.bias: Optional[nn.Parameter] = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinearW8A8":
        """
        Construct a ``QuantizedLinearW8A8`` from an existing ``nn.Linear``.

        Args:
            linear: Source layer whose weights are quantized.

        Returns:
            Initialised quantized layer on the same device as *linear*.
        """
        device = linear.weight.device
        inst = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        w = linear.weight.data.float()
        q_w, scale = quantize_weight_per_channel_int8(w)
        inst.q_weight = q_w.to(device)
        inst.w_scale = scale.to(device)
        if linear.bias is not None:
            inst.bias = nn.Parameter(linear.bias.data.clone())
        return inst

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic per-token activation quantization.
        """
        orig_dtype = x.dtype

        # Fast dynamic per-token activation scale
        x_scale = x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8).div_(127.0)
        
        # Simulate A8 using orig_dtype to avoid upcasting to float32
        x_q = (x / x_scale).round_().clamp_(_INT8_MIN, _INT8_MAX).to(orig_dtype) * x_scale

        # Dequantize W8 on the fly to orig_dtype
        w_fp = self.q_weight.to(orig_dtype) * self.w_scale.to(orig_dtype)

        # FP16/BF16 GEMM
        return torch.nn.functional.linear(x_q, w_fp, self.bias)


def replace_linear_with_int8(
    model: nn.Module,
    min_features: int = 64,
    skip_names: tuple[str, ...] = ("lm_head",),
) -> int:
    """
    Recursively replace ``nn.Linear`` layers with ``QuantizedLinearW8A8``.

    Layers whose name contains any element of *skip_names* or whose
    ``in_features`` is below *min_features* are left unchanged.

    Args:
        model:        Root module to traverse.
        min_features: Minimum in_features for replacement.
        skip_names:   Sub-strings that trigger skipping a layer by name.

    Returns:
        Number of layers replaced.
    """
    replaced = 0
    for name, module in list(model.named_children()):
        if any(skip in name for skip in skip_names):
            continue
        if isinstance(module, nn.Linear):
            if module.in_features < min_features:
                continue
            setattr(model, name, QuantizedLinearW8A8.from_linear(module))
            replaced += 1
        else:
            replaced += replace_linear_with_int8(module, min_features, skip_names)
    return replaced
