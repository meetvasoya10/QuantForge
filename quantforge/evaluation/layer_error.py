"""
Layer-wise reconstruction error: cosine similarity and MSE between
baseline and quantized logits / activations.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_logit_similarity(
    baseline_model: PreTrainedModel,
    quantized_model: PreTrainedModel,
    samples: List[torch.Tensor],
    device: torch.device,
    max_samples: int = 32,
) -> Dict[str, float]:
    """
    Compute cosine similarity and MSE between baseline and quantized logits.

    Runs both models on the same set of tokenized samples and compares
    output logits (pre-softmax) to quantify the fidelity drop introduced
    by quantization.

    Args:
        baseline_model:  FP16 reference model.
        quantized_model: Quantized model to evaluate.
        samples:         List of 1-D token-ID tensors.
        device:          Inference device.
        max_samples:     Cap on samples used (first *max_samples* only).

    Returns:
        Dict with keys:
            ``cosine_similarity`` – mean cosine similarity over all tokens.
            ``mse``               – mean squared error over all logit positions.
    """
    baseline_model.eval()
    quantized_model.eval()

    cos_total = 0.0
    mse_total = 0.0
    count = 0

    for ids in samples[:max_samples]:
        ids_dev = ids.to(device).unsqueeze(0)
        try:
            base_logits = baseline_model(input_ids=ids_dev).logits.float()  # (1, T, V)
            quant_logits = quantized_model(input_ids=ids_dev).logits.float()

            # Flatten to (T*1, V) for per-token metrics
            b = base_logits.view(-1, base_logits.size(-1))
            q = quant_logits.view(-1, quant_logits.size(-1))

            cos = F.cosine_similarity(b, q, dim=-1).mean().item()
            mse = F.mse_loss(q, b).item()

            cos_total += cos
            mse_total += mse
            count += 1
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM during logit similarity – skipping sample.")
            torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning("Error in logit similarity: %s", exc)

    if count == 0:
        return {"cosine_similarity": 0.0, "mse": float("inf")}

    return {
        "cosine_similarity": round(cos_total / count, 6),
        "mse": round(mse_total / count, 6),
    }


@torch.no_grad()
def compute_layer_reconstruction_error(
    baseline_out: torch.Tensor,
    quantized_out: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute per-layer reconstruction error between two activation tensors.

    Args:
        baseline_out:  Float tensor from the baseline layer.
        quantized_out: Float tensor from the quantized layer.

    Returns:
        Dict with keys:
            ``cosine_similarity`` – mean cosine similarity across the last dim.
            ``mse``               – mean squared error.
            ``relative_error``    – ||q - b|| / (||b|| + eps).
    """
    b = baseline_out.float().reshape(-1, baseline_out.size(-1))
    q = quantized_out.float().reshape(-1, quantized_out.size(-1))

    cos = F.cosine_similarity(b, q, dim=-1).mean().item()
    mse = F.mse_loss(q, b).item()
    rel_err = (q - b).norm() / (b.norm() + 1e-8)

    return {
        "cosine_similarity": round(cos, 6),
        "mse": round(mse, 6),
        "relative_error": round(rel_err.item(), 6),
    }
