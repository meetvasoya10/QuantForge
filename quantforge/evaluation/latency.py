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


@torch.inference_mode()
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
            ``latency_ms``   - mean wall-clock generation time in ms.
            ``tokens_per_s`` - mean throughput (generated tokens / second).
    """
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")

    # Determine pad and eos token IDs safely
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    # Warm-up
    for _ in range(num_warmup):
        try:
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        except Exception:
            pass

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    latencies_s = []
    successful = 0
    for _ in range(num_bench):
        try:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_diff = time.perf_counter() - t0
            latencies_s.append(t_diff)
            successful += 1
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM during latency benchmark - skipping run.")
            torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning("Latency benchmark error: %s", exc)

    if successful == 0:
        return {"latency_ms": float("inf"), "tokens_per_s": 0.0, "latency_std_ms": 0.0, "latency_p50_ms": float("inf"), "latency_p95_ms": float("inf"), "latency_p99_ms": float("inf")}

    latencies_ms = [t * 1000 for t in latencies_s]
    latencies_ms.sort()
    import statistics
    
    avg_ms = statistics.mean(latencies_ms)
    std_ms = statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    
    def percentile(data, p):
        if not data:
            return 0.0
        k = (len(data) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        d0 = data[int(f)] * (c - k)
        d1 = data[int(c)] * (k - f)
        return d0 + d1

    import math
    p50_ms = percentile(latencies_ms, 0.50)
    p95_ms = percentile(latencies_ms, 0.95)
    p99_ms = percentile(latencies_ms, 0.99)

    avg_s = avg_ms / 1000.0
    return {
        "latency_ms": avg_ms,
        "latency_std_ms": std_ms,
        "latency_p50_ms": p50_ms,
        "latency_p95_ms": p95_ms,
        "latency_p99_ms": p99_ms,
        "tokens_per_s": max_new_tokens / avg_s,
    }
