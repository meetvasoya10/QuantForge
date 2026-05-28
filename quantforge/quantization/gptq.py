"""
GPTQ-style post-training quantization (simplified).

Implements a layer-wise, error-aware quantization procedure inspired by
the GPTQ paper (Frantar et al., 2022).  Instead of the full Cholesky-based
OBD solver, we use a diagonal Hessian approximation computed from calibration
activations, which is stable and requires no custom CUDA kernels.

Pipeline:
  1. Register forward hooks on every target ``nn.Linear`` to collect
     calibration activations.
  2. Compute a per-column importance score (diagonal of X^T X, i.e. the
     squared activation norm per input feature).
  3. Scale weight columns by their importance, round to INT8, then rescale
     back - giving importance-weighted quantization.
  4. Replace the layer in-place with a ``QuantizedLinearW8A8`` whose weights
     reflect the GPTQ-adjusted quantization.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from quantforge.quantization.int8 import QuantizedLinearW8A8

logger = logging.getLogger(__name__)

class GPTQLinearW8A8(QuantizedLinearW8A8):
    """
    Extends QuantizedLinearW8A8 to support per-column importance scaling.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("col_scale_inv", torch.ones(1, in_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # Fast dynamic per-token activation scale
        x_scale = x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-8).div_(127.0)
        
        # Simulate A8
        x_q = (x / x_scale).round_().clamp_(-128, 127).to(orig_dtype) * x_scale

        # Dequantize W8 and apply inverse column scale
        w_fp = self.q_weight.to(orig_dtype) * self.w_scale.to(orig_dtype)
        w_fp = w_fp * self.col_scale_inv.to(orig_dtype)

        return torch.nn.functional.linear(x_q, w_fp, self.bias)


# ---------------------------------------------------------------------------
# Activation collection hook
# ---------------------------------------------------------------------------

class _ActivationCollector:
    """Collect input activations for one ``nn.Linear`` via a forward hook."""

    def __init__(self, max_samples: int = 64) -> None:
        self.inputs: List[torch.Tensor] = []
        self.max_samples = max_samples
        self._handle: Optional[torch.utils.hooks.RemovableHook] = None  # type: ignore[type-arg]

    def attach(self, module: nn.Module) -> None:
        self._handle = module.register_forward_hook(self._hook)

    def _hook(
        self,
        module: nn.Module,
        inp: tuple[torch.Tensor, ...],
        out: torch.Tensor,
    ) -> None:
        if len(self.inputs) < self.max_samples:
            x = inp[0].detach().float()
            self.inputs.append(x.reshape(-1, x.size(-1)))  # (tokens, C_in)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()


# ---------------------------------------------------------------------------
# GPTQ-style importance-weighted INT8 quantization
# ---------------------------------------------------------------------------

def _gptq_quantize_weight(
    weight: torch.Tensor,
    activation_inputs: List[torch.Tensor],
    damping: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize *weight* to INT8 with importance weighting from calibration data.

    The diagonal Hessian approximation H_ii ≈ mean(x_i^2) gives per-column
    importance scores.  Columns with high importance are quantized more
    carefully (higher effective scale precision) by temporarily scaling the
    weight matrix before uniform rounding.

    Args:
        weight:            FP32 weight tensor of shape (out, in).
        activation_inputs: List of calibration activation matrices (tokens, in).
        damping:           Ridge damping added to diagonal for stability.

    Returns:
        Tuple of:
            ``q_weight`` - INT8 tensor of shape (out, in).
            ``scale``    - FP32 per-channel scale of shape (out, 1).
    """
    if len(activation_inputs) == 0:
        # Fall back to plain per-channel INT8
        from quantforge.quantization.int8 import quantize_weight_per_channel_int8
        q_w, scale = quantize_weight_per_channel_int8(weight)
        return q_w, scale, torch.ones(1, weight.size(1), dtype=torch.float32)

    # Stack all calibration tokens → (N, C_in)
    X = torch.cat(activation_inputs, dim=0)  # (N, C_in)

    # Importance: mean(abs(X)) instead of H for stable AWQ-style scaling
    importance = X.abs().mean(dim=0).clamp_(min=1e-5) # (C_in,)
    importance = importance / importance.max()
    importance = importance.sqrt().unsqueeze(0)  # (1, C_in)
    
    # Scale columns by importance
    w_scaled = weight * importance            # (out, C_in)

    # Quantize scaled weight per-row
    max_abs = w_scaled.abs().amax(dim=1, keepdim=True).clamp_(min=1e-8)
    scale = max_abs / 127.0
    q_weight = (w_scaled / scale).round_().clamp_(-128, 127).to(torch.int8)

    col_scale_inv = 1.0 / importance
    return q_weight, scale.to(torch.float32), col_scale_inv.to(torch.float32)


def apply_gptq(
    model: nn.Module,
    calibration_samples: List[torch.Tensor],
    device: torch.device,
    calibration_count: int = 64,
    damping: float = 0.01,
    skip_names: tuple[str, ...] = ("lm_head",),
) -> Dict[str, float]:
    """
    Apply GPTQ-style post-training quantization to all eligible ``nn.Linear``
    layers in *model*.

    Args:
        model:               Model to quantize in-place.
        calibration_samples: List of 1-D token-ID tensors for calibration.
        device:              Inference device.
        calibration_count:   Maximum number of calibration samples to run.
        damping:             Diagonal damping for Hessian approximation.
        skip_names:          Layer names (substrings) to skip.

    Returns:
        Dict mapping ``layer_name → reconstruction_mse`` for each replaced layer.
    """
    model.eval()

    # ------------------------------------------------------------------
    # Step 1 - identify target layers and attach hooks
    # ------------------------------------------------------------------
    collectors: Dict[str, _ActivationCollector] = {}
    layer_refs: Dict[str, nn.Linear] = {}

    def _find_linears(parent: nn.Module, prefix: str = "") -> None:
        for name, child in parent.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if any(skip in name for skip in skip_names):
                continue
            if isinstance(child, nn.Linear) and child.in_features >= 64:
                c = _ActivationCollector(max_samples=calibration_count)
                c.attach(child)
                collectors[full_name] = c
                layer_refs[full_name] = child
            else:
                _find_linears(child, full_name)

    _find_linears(model)
    logger.info("GPTQ: attached hooks to %d linear layers.", len(collectors))

    # ------------------------------------------------------------------
    # Step 2 - run calibration forward passes to collect activations
    # ------------------------------------------------------------------
    with torch.no_grad():
        for i, ids in enumerate(calibration_samples[:calibration_count]):
            try:
                model(input_ids=ids.unsqueeze(0).to(device))
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM during GPTQ calibration pass %d - stopping early.", i)
                torch.cuda.empty_cache()
                break
            except Exception as exc:
                logger.warning("GPTQ calibration error at step %d: %s", i, exc)

    # Detach all hooks
    for c in collectors.values():
        c.remove()

    # ------------------------------------------------------------------
    # Step 3 - quantize each layer using importance-weighted INT8
    # ------------------------------------------------------------------
    layer_errors: Dict[str, float] = {}

    def _replace_recursive(parent: nn.Module, prefix: str = "") -> None:
        for name, child in list(parent.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if full_name in collectors:
                lin: nn.Linear = child  # type: ignore[assignment]
                w = lin.weight.data.float()
                acts = collectors[full_name].inputs

                q_w, scale, col_scale_inv = _gptq_quantize_weight(w, acts, damping)

                # Build a GPTQLinearW8A8 with importance weights
                new_layer = GPTQLinearW8A8(lin.in_features, lin.out_features, lin.bias is not None)
                new_layer.q_weight = q_w.to(device)
                new_layer.w_scale = scale.to(device)
                new_layer.col_scale_inv = col_scale_inv.to(device)
                if lin.bias is not None:
                    new_layer.bias = nn.Parameter(lin.bias.data.clone())

                setattr(parent, name, new_layer)

                # Track reconstruction error
                w_hat = q_w.float() * scale * col_scale_inv
                mse = ((w - w_hat) ** 2).mean().item()
                layer_errors[full_name] = round(mse, 8)
            else:
                _replace_recursive(child, full_name)

    _replace_recursive(model)
    logger.info("GPTQ: replaced %d layers. Mean recon MSE: %.6e",
                len(layer_errors),
                sum(layer_errors.values()) / max(len(layer_errors), 1))
    return layer_errors
