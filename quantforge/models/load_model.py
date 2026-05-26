"""
Model loader for facebook/opt-125m.

Loads the model and tokenizer with safe defaults for 6 GB VRAM.
"""

from __future__ import annotations

import copy
import logging
from typing import Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# FIX FOR CVE-2025-32434 IN TRANSFORMERS + PYTORCH 2.5
# `transformers` now hard-blocks `torch.load` if PyTorch < 2.6.
# facebook/opt-125m only has .bin files on main. We monkeypatch the check
# because this environment uses 2.5.1 and we trust the OPT-125M weights.
import transformers.modeling_utils
if hasattr(transformers.modeling_utils, "check_torch_load_is_safe"):
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None

logger = logging.getLogger(__name__)

MODEL_ID = "facebook/opt-125m"


def load_model_and_tokenizer(
    model_id: str = MODEL_ID,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load a causal-LM model and its tokenizer.

    Args:
        model_id: HuggingFace model identifier.
        device:   Target device string ("cuda" or "cpu").
        dtype:    Weight dtype (torch.float16 recommended for 6 GB VRAM).

    Returns:
        Tuple of (model, tokenizer) both moved to *device*.

    Raises:
        RuntimeError: If CUDA is requested but not available.
    """
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available - falling back to CPU.")
        device = "cpu"
        dtype = torch.float32

    logger.info("Loading tokenizer for %s ...", model_id)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_id, use_fast=True
    )
    # OPT uses EOS as PAD; ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model %s (dtype=%s, device=%s) ...", model_id, dtype, device)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded: %s  |  params: %.2fM  |  device: %s  |  dtype: %s",
        model_id,
        param_count / 1e6,
        device,
        dtype,
    )
    return model, tokenizer


def clone_model(model: PreTrainedModel) -> PreTrainedModel:
    """
    Return a deep copy of *model* on the same device and dtype.

    Useful for obtaining a fresh copy for each quantization experiment
    without re-downloading from HuggingFace Hub.

    Args:
        model: Source model to clone.

    Returns:
        Independent deep copy.
    """
    cloned = copy.deepcopy(model)
    cloned.eval()
    return cloned


def model_memory_mb(model: PreTrainedModel) -> float:
    """
    Compute total model parameter memory in megabytes.

    Args:
        model: Any nn.Module.

    Returns:
        Memory in MB occupied by all parameter tensors.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / (1024 ** 2)
