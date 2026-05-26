"""
Latency and throughput measurement for transformer inference.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_WARMUP_RUNS = 3
_BENCH_RUNS = 10


@torch.no_grad()
def measure_latency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 50,
    num_warmup: int = _WARMUP_RUNS,
    num_bench: int = _BENCH_RUNS,
) -> Dict[str, float]:
    """
    Measure average generation latency and tokens-per-second throughput.

    Runs *num_warmup* un-timed warm-up passes to prime caches, then
    averages *num_bench* timed generation passes.

    Args:
        model:          Causal-LM model in eval mode.
        tokenizer:      Matching tokenizer.
        device:         Inference device.
        prompt:         Text prompt to tokenize and feed as input.
        max_new_tokens: Number of new tokens to generate per run.
        num_warmup:     Number of warm-up runs (not timed).
        num_bench:      Number of benchmarked runs.

    Returns:
        Dict with keys:
            ``latency_ms``   – mean wall-clock generation time in ms.
            ``tokens_per_s`` – mean throughput (generated tokens / second).
    """
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]

    # Warm-up
    for _ in range(num_warmup):
        try:
            model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        except Exception:
            pass

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    total_time_s = 0.0
    successful = 0
    for _ in range(num_bench):
        try:
            t0 = time.perf_counter()
            model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_time_s += time.perf_counter() - t0
            successful += 1
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM during latency benchmark – skipping run.")
            torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning("Latency benchmark error: %s", exc)

    if successful == 0:
        return {"latency_ms": float("inf"), "tokens_per_s": 0.0}

    avg_s = total_time_s / successful
    return {
        "latency_ms": avg_s * 1000,
        "tokens_per_s": max_new_tokens / avg_s,
    }
