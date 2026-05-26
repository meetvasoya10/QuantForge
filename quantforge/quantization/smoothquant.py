"""
SmoothQuant-style activation scaling (Lin et al., 2023).

Key idea:
  Transformer activations contain per-channel outliers that make uniform
  quantization difficult.  SmoothQuant migrates the quantization difficulty
  from activations to weights by multiplying each activation channel by a
  per-channel scale s and compensating in the weight matrix:

      Y = (X / s) · (W * s^T)   ≡   X · W

  After scaling, both X/s and W*s have smaller dynamic ranges and are
  easier to quantize to INT8.

This implementation:
  1. Collects per-channel activation statistics on calibration data.
  2. Identifies outlier channels (those with max_abs >> median).
  3. Computes per-channel scales s_i = max_act_i^α / max_w_i^(1-α).
  4. Applies the scaling to the weight matrix in-place.
  5. Replaces each scaled linear layer with QuantizedLinearW8A8.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from quantforge.quantization.int8 import QuantizedLinearW8A8

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation statistics collector
# ---------------------------------------------------------------------------

class _ActStatCollector:
    """Track per-channel max absolute activation over calibration passes."""

    def __init__(self) -> None:
        self.max_abs: torch.Tensor | None = None
        self._handle = None

    def attach(self, module: nn.Module) -> None:
        self._handle = module.register_forward_hook(self._hook)

    def _hook(
        self,
        module: nn.Module,
        inp: tuple[torch.Tensor, ...],
        out: torch.Tensor,
    ) -> None:
        x = inp[0].detach().float()                          # (..., C_in)
        max_per_ch = x.reshape(-1, x.size(-1)).abs().max(0).values  # (C_in,)
        if self.max_abs is None:
            self.max_abs = max_per_ch
        else:
            self.max_abs = torch.max(self.max_abs, max_per_ch)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_activation_stats(
    model: nn.Module,
    calibration_samples: List[torch.Tensor],
    device: torch.device,
    skip_names: tuple[str, ...] = ("lm_head",),
) -> Dict[str, torch.Tensor]:
    """
    Run calibration forward passes and collect per-channel activation maxima.

    Args:
        model:               Model to profile (unmodified).
        calibration_samples: List of 1-D token-ID tensors.
        device:              Inference device.
        skip_names:          Layer name substrings to skip.

    Returns:
        Dict mapping full layer name → per-channel max-abs activation tensor.
    """
    collectors: Dict[str, _ActStatCollector] = {}

    def _attach(parent: nn.Module, prefix: str = "") -> None:
        for name, child in parent.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if any(skip in name for skip in skip_names):
                continue
            if isinstance(child, nn.Linear) and child.in_features >= 64:
                c = _ActStatCollector()
                c.attach(child)
                collectors[full] = c
            else:
                _attach(child, full)

    _attach(model)

    model.eval()
    with torch.no_grad():
        for ids in calibration_samples:
            try:
                model(input_ids=ids.unsqueeze(0).to(device))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
            except Exception as exc:
                logger.warning("SmoothQuant calibration error: %s", exc)

    for c in collectors.values():
        c.remove()

    return {k: v.max_abs for k, v in collectors.items() if v.max_abs is not None}


def apply_smoothquant(
    model: nn.Module,
    calibration_samples: List[torch.Tensor],
    device: torch.device,
    alpha: float = 0.5,
    skip_names: tuple[str, ...] = ("lm_head",),
) -> Dict[str, float]:
    """
    Apply SmoothQuant-style channel scaling then W8A8 quantization.

    For each target linear layer:
      1. Compute per-channel scale s_i = act_max_i^α / w_max_i^(1-α).
      2. Absorb s into weight columns: W_new[:, i] = W[:, i] * s_i.
      3. Replace the layer with QuantizedLinearW8A8(W_new).

    Args:
        model:               Model to quantize in-place.
        calibration_samples: Calibration token sequences.
        device:              Inference device.
        alpha:               Migration strength (0 = weight-heavy, 1 = act-heavy).
        skip_names:          Layer name substrings to skip.

    Returns:
        Dict of per-layer outlier ratios (fraction of channels that were outliers).
    """
    act_stats = collect_activation_stats(model, calibration_samples, device, skip_names)

    outlier_info: Dict[str, float] = {}

    def _replace(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if any(skip in name for skip in skip_names):
                continue
            if isinstance(child, nn.Linear) and full in act_stats and child.in_features >= 64:
                lin: nn.Linear = child  # type: ignore[assignment]
                act_max = act_stats[full].to(device).clamp(min=1e-8)  # (C_in,)
                w = lin.weight.data.float()                            # (C_out, C_in)
                w_max = w.abs().max(dim=0).values.clamp(min=1e-8)     # (C_in,)

                # Per-channel smoothing scale
                s = (act_max ** alpha) / (w_max ** (1.0 - alpha))
                s = s.clamp(min=1e-4)

                # Detect outlier channels (act_max > 3 * median)
                median_act = act_max.median()
                is_outlier = act_max > (3.0 * median_act)
                outlier_ratio = is_outlier.float().mean().item()
                outlier_info[full] = round(outlier_ratio, 4)

                # Absorb scale into weight columns
                w_smooth = w * s.unsqueeze(0)   # (C_out, C_in)

                # Build quantized layer with smoothed weight
                new_layer = QuantizedLinearW8A8.from_linear(
                    nn.Linear(lin.in_features, lin.out_features, bias=lin.bias is not None)
                )
                # Override with smoothed weight
                from quantforge.quantization.int8 import quantize_weight_per_channel_int8
                q_w, scale = quantize_weight_per_channel_int8(w_smooth.to(device))
                new_layer.q_weight = q_w
                new_layer.w_scale = scale
                if lin.bias is not None:
                    new_layer.bias = nn.Parameter(lin.bias.data.clone())
                setattr(parent, name, new_layer)
            elif not isinstance(child, nn.Linear):
                _replace(child, full)

    _replace(model)
    logger.info(
        "SmoothQuant: replaced layers. Mean outlier ratio: %.4f",
        sum(outlier_info.values()) / max(len(outlier_info), 1),
    )
    return outlier_info
