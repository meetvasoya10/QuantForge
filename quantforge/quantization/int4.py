"""
INT4 weight-only quantization.

Implements:
  - Per-output-channel symmetric INT4 quantization.
  - Weights stored as int8 tensors in the range [-8, 7] (simulated INT4).
  - Dequantization during the forward pass.
  - ``replace_linear_with_int4`` - recursive module replacement.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_INT4_MIN = -8
_INT4_MAX = 7


def pack_int4(q_weight: torch.Tensor) -> torch.Tensor:
    """
    Pack a 2D int8 tensor with values in [-8, 7] into a uint8 tensor,
    packing two 4-bit values per byte along the last dimension.
    """
    # Shift [-8, 7] to [0, 15]
    q_shifted = (q_weight + 8).to(torch.uint8)
    
    # Ensure divisible by 2
    out_features, in_features = q_shifted.shape
    assert in_features % 2 == 0
    
    q_pairs = q_shifted.view(out_features, in_features // 2, 2)
    
    # Pack: (left << 4) | right
    packed = (q_pairs[..., 0] << 4) | q_pairs[..., 1]
    return packed


def unpack_int4(packed: torch.Tensor, out_features: int, in_features: int) -> torch.Tensor:
    """
    Unpack a 2D uint8 tensor into a 2D int8 tensor with values in [-8, 7].
    """
    left = packed >> 4
    right = packed & 0x0F
    
    unpacked_shifted = torch.stack([left, right], dim=-1).view(out_features, in_features)
    return unpacked_shifted.to(torch.int8) - 8


def quantize_weight_groupwise_int4(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize *weight* to simulated INT4 using group-wise symmetric scaling.

    Args:
        weight: 2-D float tensor of shape (out_features, in_features).
        group_size: Number of elements per group.

    Returns:
        Tuple of:
            ``q_weight`` - int8 tensor with values in [-8, 7].
            ``scale``    - FP32 scale per group.
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, "in_features must be divisible by group_size"
    
    num_groups = in_features // group_size
    w_groups = weight.view(out_features, num_groups, group_size)
    
    # Optional clipping to reduce outlier impact
    # A simple search: 100% vs 99% max
    max_abs = w_groups.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8)
    # Simple percentile heuristic
    clip_val = torch.quantile(w_groups.abs().float(), 0.99, dim=-1, keepdim=True).to(weight.dtype)
    max_abs = torch.minimum(max_abs, clip_val).clamp_(min=1e-8)

    scale = max_abs / 7.0
    q_weight = (w_groups / scale).round_().clamp_(_INT4_MIN, _INT4_MAX).to(torch.int8)
    
    return q_weight.view(out_features, in_features), scale.view(out_features, num_groups)


class QuantizedLinearINT4(nn.Module):
    """
    Drop-in replacement for ``nn.Linear`` using INT4 weight-only quantization.

    Weights are quantized to 4-bit precision at construction time and stored
    compactly as int8.  At runtime they are dequantized back to the input
    dtype before the matrix multiply, so activation precision is unchanged.

    This simulates the memory footprint of INT4 (weights occupy ~50% of
    FP16 storage) while using standard PyTorch ops for correctness.

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

        self.group_size = 128
        assert in_features % self.group_size == 0, "in_features must be divisible by group_size"
        num_groups = in_features // self.group_size

        self.register_buffer("packed_weight", torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer("w_scale", torch.ones(out_features, num_groups, dtype=torch.float32))
        self.bias: Optional[nn.Parameter] = None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinearINT4":
        """
        Construct a ``QuantizedLinearINT4`` from an existing ``nn.Linear``.

        Args:
            linear: Source layer whose weights are quantized to INT4.

        Returns:
            Initialised quantized layer on the same device as *linear*.
        """
        device = linear.weight.device
        inst = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        w = linear.weight.data.float()
        q_w, scale = quantize_weight_groupwise_int4(w, group_size=inst.group_size)
        
        packed = pack_int4(q_w)
        inst.packed_weight = packed.to(device)
        inst.w_scale = scale.to(device)
        
        if linear.bias is not None:
            inst.bias = nn.Parameter(linear.bias.data.clone())
        return inst

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dequantize weights and perform standard FP matmul.

        Args:
            x: Input activation tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        orig_dtype = x.dtype
        
        # Unpack: uint8 → int8 → float32
        q_weight = unpack_int4(self.packed_weight, self.out_features, self.in_features)
        
        # w_scale has shape (out_features, num_groups)
        w_scale_exp = self.w_scale.repeat_interleave(self.group_size, dim=1)
        w_fp = q_weight.to(orig_dtype) * w_scale_exp.to(orig_dtype)
        
        out = torch.nn.functional.linear(x, w_fp, self.bias)
        return out


def replace_linear_with_int4(
    model: nn.Module,
    min_features: int = 64,
    skip_names: tuple[str, ...] = ("lm_head",),
) -> int:
    """
    Recursively replace ``nn.Linear`` layers with ``QuantizedLinearINT4``.

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
            setattr(model, name, QuantizedLinearINT4.from_linear(module))
            replaced += 1
        else:
            replaced += replace_linear_with_int4(module, min_features, skip_names)
    return replaced
