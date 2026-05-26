"""
Memory measurement utilities: model parameters, CUDA allocated/reserved.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def measure_memory(model: PreTrainedModel, device: torch.device) -> Dict[str, float]:
    """
    Measure model parameter memory and CUDA runtime memory.

    Args:
        model:  Any nn.Module whose parameters are measured.
        device: The device the model lives on.

    Returns:
        Dict with keys:
            ``model_memory_mb``     – parameter bytes converted to MB.
            ``cuda_allocated_mb``   – CUDA memory currently allocated (MB).
            ``cuda_reserved_mb``    – CUDA memory reserved by the cache allocator (MB).
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_mb = param_bytes / (1024 ** 2)

    cuda_alloc_mb = 0.0
    cuda_reserved_mb = 0.0
    if device.type == "cuda":
        cuda_alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        cuda_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)

    return {
        "model_memory_mb": round(model_mb, 2),
        "cuda_allocated_mb": round(cuda_alloc_mb, 2),
        "cuda_reserved_mb": round(cuda_reserved_mb, 2),
    }
