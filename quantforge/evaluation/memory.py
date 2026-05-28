"""
Memory measurement utilities: model parameters, CUDA allocated/reserved.
"""

from __future__ import annotations

import logging
from typing import Dict

import torch
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def measure_memory(
    model: PreTrainedModel,
    device: torch.device,
    is_simulated: bool = False,
    uses_packed_weights: bool = False,
    quantization_bits: int = 16,
) -> Dict[str, Any]:
    """
    Measure model parameter memory and CUDA runtime memory.

    Args:
        model:  Any nn.Module whose parameters are measured.
        device: The device the model lives on.
        is_simulated: True if the quantization uses floating-point types for calculation.
        uses_packed_weights: True if sub-byte weights are packed (e.g., two INT4 in one INT8).
        quantization_bits: Theoretical bits per weight (e.g., 4 or 8 or 16).

    Returns:
        Dict with keys:
            fp16_model_memory_mb
            actual_storage_memory_mb
            effective_quantized_memory_mb
            cuda_allocated_mb
            cuda_reserved_mb
            is_simulated
            uses_packed_weights
    """
    total_elements = sum(p.numel() for p in model.parameters())
    
    fp16_model_mb = (total_elements * 2) / (1024 ** 2)
    actual_storage_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    effective_quantized_mb = (total_elements * quantization_bits / 8) / (1024 ** 2)

    cuda_alloc_mb = 0.0
    cuda_reserved_mb = 0.0
    cuda_peak_alloc_mb = 0.0
    cuda_peak_reserved_mb = 0.0
    if device.type == "cuda":
        cuda_alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        cuda_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
        cuda_peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        cuda_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    return {
        "fp16_model_memory_mb": round(fp16_model_mb, 2),
        "actual_storage_memory_mb": round(actual_storage_mb, 2),
        "effective_quantized_memory_mb": round(effective_quantized_mb, 2),
        "cuda_allocated_mb": round(cuda_alloc_mb, 2),
        "cuda_reserved_mb": round(cuda_reserved_mb, 2),
        "cuda_peak_allocated_mb": round(cuda_peak_alloc_mb, 2),
        "cuda_peak_reserved_mb": round(cuda_peak_reserved_mb, 2),
        "is_simulated": is_simulated,
        "uses_packed_weights": uses_packed_weights,
    }
