"""
Perplexity evaluation on WikiText-2 validation samples.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_perplexity(
    model: PreTrainedModel,
    samples: List[torch.Tensor],
    device: torch.device,
    max_samples: Optional[int] = None,
) -> float:
    """
    Compute token-level perplexity over *samples*.

    Each sample in *samples* is a 1-D LongTensor of token IDs.
    The model's cross-entropy loss (averaged over tokens) is exponentiated
    to obtain perplexity.

    Args:
        model:       Causal-LM model in eval mode.
        samples:     List of 1-D token-ID tensors.
        device:      Device on which inference runs.
        max_samples: Optional cap on the number of samples used.

    Returns:
        Perplexity (float). Returns float('inf') if no valid samples exist.
    """
    model.eval()
    if max_samples is not None:
        samples = samples[:max_samples]

    total_loss = 0.0
    total_tokens = 0

    for ids in samples:
        ids = ids.to(device).unsqueeze(0)  # (1, seq_len)
        try:
            out = model(input_ids=ids, labels=ids)
            loss: torch.Tensor = out.loss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            n_tokens = (ids.numel() - 1)  # next-token prediction targets
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM during perplexity eval – skipping sample.")
            torch.cuda.empty_cache()
            continue
        except Exception as exc:
            logger.warning("Error during perplexity eval: %s", exc)
            continue

    if total_tokens == 0:
        logger.error("No valid tokens accumulated – returning inf perplexity.")
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)
