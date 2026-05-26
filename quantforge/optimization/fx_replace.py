"""
torch.fx-based (safe recursive) layer replacement for quantized inference.

Implements a reliable recursive walk that replaces ``nn.Linear`` sub-modules
with quantized alternatives without requiring torch.fx graph tracing (which
can fail on dynamic control flow in transformer models).

This module exposes a unified ``replace_layers_fx`` entry point that:
  1. Walks the model recursively.
  2. Replaces eligible ``nn.Linear`` layers with the chosen quantized class.
  3. Returns a report dict summarising the replacement.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type

import torch.nn as nn

logger = logging.getLogger(__name__)


def _recursive_replace(
    model: nn.Module,
    target_cls: Type[nn.Module],
    builder,
    skip_names: tuple[str, ...],
    min_features: int,
    prefix: str,
    report: Dict[str, Any],
) -> None:
    """
    Recursively traverse *model* and replace ``nn.Linear`` with *target_cls*.

    Args:
        model:        Module to traverse.
        target_cls:   Quantized replacement class.
        builder:      Callable ``(nn.Linear) → target_cls instance``.
        skip_names:   Child name substrings that trigger skipping.
        min_features: Minimum in_features for replacement.
        prefix:       Name prefix for reporting.
        report:       Mutable dict accumulating replaced layer names.
    """
    for name, child in list(model.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if any(skip in name for skip in skip_names):
            continue
        if isinstance(child, nn.Linear):
            if child.in_features < min_features:
                continue
            try:
                new_layer = builder(child)
                setattr(model, name, new_layer)
                report["replaced_layers"].append(full_name)
            except Exception as exc:
                logger.warning("FX-replace: could not replace %s - %s", full_name, exc)
                report["failed_layers"].append(full_name)
        else:
            _recursive_replace(child, target_cls, builder, skip_names, min_features, full_name, report)


def replace_layers_fx(
    model: nn.Module,
    method: str = "int8",
    skip_names: tuple[str, ...] = ("lm_head",),
    min_features: int = 64,
) -> Dict[str, Any]:
    """
    Replace ``nn.Linear`` layers in *model* with quantized alternatives.

    Supports the same method names as the broader QuantForge pipeline:
    ``"int8"``, ``"int4"``, ``"ggfu"``.

    Args:
        model:        Root module to modify in-place.
        method:       Quantization method name (determines replacement class).
        skip_names:   Layer name substrings to skip.
        min_features: Minimum in_features for replacement.

    Returns:
        Report dict with keys:
            ``method``            - method string.
            ``replaced_count``    - number of layers successfully replaced.
            ``failed_count``      - number of layers that failed replacement.
            ``replaced_layers``   - list of replaced layer full names.
            ``failed_layers``     - list of failed layer full names.
    """
    from quantforge.quantization.int8 import QuantizedLinearW8A8
    from quantforge.quantization.int4 import QuantizedLinearINT4
    from quantforge.quantization.ggfu import GGFULinear

    method_map = {
        "int8": (QuantizedLinearW8A8, QuantizedLinearW8A8.from_linear),
        "int4": (QuantizedLinearINT4, QuantizedLinearINT4.from_linear),
        "ggfu": (GGFULinear, lambda lin: GGFULinear.from_linear(lin)),
    }

    if method not in method_map:
        raise ValueError(f"Unknown FX-replace method '{method}'. Choose from {list(method_map)}.")

    target_cls, builder = method_map[method]

    report: Dict[str, Any] = {
        "method": method,
        "replaced_layers": [],
        "failed_layers": [],
    }

    _recursive_replace(model, target_cls, builder, skip_names, min_features, "", report)

    report["replaced_count"] = len(report["replaced_layers"])
    report["failed_count"] = len(report["failed_layers"])

    logger.info(
        "FX-replace (%s): %d replaced, %d failed.",
        method,
        report["replaced_count"],
        report["failed_count"],
    )
    return report
