"""
torch.compile inference benchmarking.

Wraps the model with ``torch.compile`` (if available) and compares
compiled vs non-compiled generation latency.  Errors are caught and
logged gracefully so the rest of the benchmark continues.
"""

from __future__ import annotations

import logging
import time
from typing import Dict

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

_WARMUP = 2
_BENCH = 5


def _generate(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    max_new_tokens: int,
) -> None:
    """Run one generation call."""
    model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )


@torch.no_grad()
def benchmark_compile(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 50,
) -> Dict[str, object]:
    """
    Benchmark compiled vs non-compiled generation latency.

    Attempts to compile the model with ``torch.compile``.  If compilation
    fails (unsupported backend, Windows Triton issues, etc.), the compiled
    benchmark is skipped and ``compile_supported`` is set to False.

    Args:
        model:          Causal-LM in eval mode.
        tokenizer:      Matching tokenizer.
        device:         Inference device.
        prompt:         Input text prompt.
        max_new_tokens: Tokens to generate per run.

    Returns:
        Dict with keys:
            ``compile_supported``         - bool.
            ``baseline_latency_ms``       - non-compiled mean latency.
            ``baseline_tokens_per_s``     - non-compiled throughput.
            ``compiled_latency_ms``       - compiled mean latency (or None).
            ``compiled_tokens_per_s``     - compiled throughput (or None).
            ``speedup_pct``               - % latency reduction (or None).
            ``error``                     - error message if compile failed.
    """
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]

    # ------------------------------------------------------------------
    # Baseline (eager) benchmark
    # ------------------------------------------------------------------
    for _ in range(_WARMUP):
        try:
            _generate(model, input_ids, tokenizer, max_new_tokens)
        except Exception:
            pass

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t_eager: list[float] = []
    for _ in range(_BENCH):
        try:
            t0 = time.perf_counter()
            _generate(model, input_ids, tokenizer, max_new_tokens)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_eager.append(time.perf_counter() - t0)
        except Exception as exc:
            logger.warning("Eager bench error: %s", exc)

    if not t_eager:
        eager_ms = float("inf")
        eager_tps = 0.0
    else:
        eager_ms = (sum(t_eager) / len(t_eager)) * 1000
        eager_tps = max_new_tokens / (sum(t_eager) / len(t_eager))

    result: Dict[str, object] = {
        "compile_supported": False,
        "baseline_latency_ms": round(eager_ms, 2),
        "baseline_tokens_per_s": round(eager_tps, 2),
        "compiled_latency_ms": None,
        "compiled_tokens_per_s": None,
        "speedup_pct": None,
        "error": None,
    }

    # ------------------------------------------------------------------
    # torch.compile benchmark
    # ------------------------------------------------------------------
    try:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile not available in this PyTorch version.")

        logger.info("Compiling model with torch.compile ...")
        compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

        # Warm up compiled model
        for _ in range(_WARMUP):
            try:
                _generate(compiled_model, input_ids, tokenizer, max_new_tokens)
            except Exception:
                pass

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t_compiled: list[float] = []
        for _ in range(_BENCH):
            t0 = time.perf_counter()
            _generate(compiled_model, input_ids, tokenizer, max_new_tokens)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_compiled.append(time.perf_counter() - t0)

        if t_compiled:
            comp_ms = (sum(t_compiled) / len(t_compiled)) * 1000
            comp_tps = max_new_tokens / (sum(t_compiled) / len(t_compiled))
            speedup = (eager_ms - comp_ms) / max(eager_ms, 1e-6) * 100

            result["compile_supported"] = True
            result["compiled_latency_ms"] = round(comp_ms, 2)
            result["compiled_tokens_per_s"] = round(comp_tps, 2)
            result["speedup_pct"] = round(speedup, 2)

    except Exception as exc:
        err_msg = str(exc)[:200]
        logger.warning("torch.compile benchmark failed: %s", err_msg)
        result["error"] = err_msg

    return result
