"""
run_quantized.py – Benchmark any single quantization method.

Usage:
    python -m quantforge.scripts.run_quantized --method int8   --max_samples 256
    python -m quantforge.scripts.run_quantized --method int4   --max_samples 256
    python -m quantforge.scripts.run_quantized --method gptq   --max_samples 256
    python -m quantforge.scripts.run_quantized --method smoothquant --max_samples 256
    python -m quantforge.scripts.run_quantized --method ggfu   --max_samples 256
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantforge.models.load_model import load_model_and_tokenizer, clone_model
from quantforge.data.load_wikitext import load_wikitext_samples
from quantforge.evaluation.perplexity import compute_perplexity
from quantforge.evaluation.latency import measure_latency
from quantforge.evaluation.memory import measure_memory
from quantforge.evaluation.layer_error import compute_logit_similarity
from quantforge.evaluation.benchmark import save_json, load_json, enrich_with_deltas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("int8", "int4", "gptq", "smoothquant", "ggfu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuantForge Quantization Benchmark")
    p.add_argument("--method", required=True, choices=SUPPORTED_METHODS,
                   help="Quantization method to benchmark")
    p.add_argument("--max_samples", type=int, default=256)
    p.add_argument("--max_length",  type=int, default=512)
    p.add_argument("--device",  default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype",   default="float16",
                   choices=["float16", "float32", "bfloat16"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-method quantization application
# ---------------------------------------------------------------------------

def _apply_method(
    method: str,
    model: torch.nn.Module,
    samples: list[torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Apply the chosen quantization method and return extra metadata.

    Args:
        method:  One of ``SUPPORTED_METHODS``.
        model:   Cloned model to quantize in-place.
        samples: Calibration/benchmark samples.
        device:  Inference device.

    Returns:
        Dict of extra metadata (layer counts, layer errors, etc.)
    """
    extra: Dict[str, Any] = {}

    if method == "int8":
        from quantforge.quantization.int8 import replace_linear_with_int8
        n = replace_linear_with_int8(model)
        extra["layers_replaced"] = n
        logger.info("INT8: replaced %d linear layers.", n)

    elif method == "int4":
        from quantforge.quantization.int4 import replace_linear_with_int4
        n = replace_linear_with_int4(model)
        extra["layers_replaced"] = n
        logger.info("INT4: replaced %d linear layers.", n)

    elif method == "gptq":
        from quantforge.quantization.gptq import apply_gptq
        layer_errors = apply_gptq(model, samples, device, calibration_count=64)
        extra["layer_reconstruction_errors"] = layer_errors
        extra["mean_reconstruction_mse"] = round(
            sum(layer_errors.values()) / max(len(layer_errors), 1), 8
        )

    elif method == "smoothquant":
        from quantforge.quantization.smoothquant import apply_smoothquant
        outlier_info = apply_smoothquant(model, samples[:64], device, alpha=0.5)
        extra["outlier_ratios"] = outlier_info
        extra["mean_outlier_ratio"] = round(
            sum(outlier_info.values()) / max(len(outlier_info), 1), 4
        )

    elif method == "ggfu":
        from quantforge.quantization.ggfu import apply_ggfu
        layer_metrics = apply_ggfu(model, group_size=32, bits=4, clip_percentile=99.9)
        extra["layer_metrics"] = layer_metrics
        if layer_metrics:
            extra["mean_weight_cosine"] = round(
                sum(m["cosine_similarity"] for m in layer_metrics.values()) / len(layer_metrics), 6
            )
            extra["mean_weight_mse"] = round(
                sum(m["mse"] for m in layer_metrics.values()) / len(layer_metrics), 8
            )

    return extra


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    method = args.method

    dtype_map = {
        "float16":  torch.float16,
        "float32":  torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype  = dtype_map[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    logger.info("=" * 60)
    logger.info("QuantForge  –  %s Benchmark", method.upper())
    logger.info("device=%s  dtype=%s  max_samples=%d  max_length=%d",
                device, dtype, args.max_samples, args.max_length)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load model + data (need baseline model for logit similarity)
    # ------------------------------------------------------------------
    baseline_model, tokenizer = load_model_and_tokenizer(device=str(device), dtype=dtype)
    samples = load_wikitext_samples(tokenizer, max_samples=args.max_samples,
                                    max_length=args.max_length)

    # ------------------------------------------------------------------
    # 2. Clone model for quantization
    # ------------------------------------------------------------------
    quant_model = clone_model(baseline_model)

    # ------------------------------------------------------------------
    # 3. Apply quantization
    # ------------------------------------------------------------------
    logger.info("Applying %s quantization …", method)
    extra = {}
    try:
        extra = _apply_method(method, quant_model, samples, device)
    except Exception as exc:
        logger.error("Quantization failed: %s", exc, exc_info=True)
        extra["quantization_error"] = str(exc)

    # ------------------------------------------------------------------
    # 4. Perplexity
    # ------------------------------------------------------------------
    logger.info("Computing perplexity …")
    try:
        ppl = compute_perplexity(quant_model, samples, device)
    except Exception as exc:
        logger.error("Perplexity eval failed: %s", exc)
        ppl = float("inf")
    logger.info("Perplexity: %.4f", ppl)

    # ------------------------------------------------------------------
    # 5. Memory
    # ------------------------------------------------------------------
    torch.cuda.empty_cache() if device.type == "cuda" else None
    mem = measure_memory(quant_model, device)
    logger.info("Model memory: %.2f MB", mem["model_memory_mb"])

    # ------------------------------------------------------------------
    # 6. Latency
    # ------------------------------------------------------------------
    logger.info("Measuring latency …")
    try:
        lat = measure_latency(quant_model, tokenizer, device)
    except Exception as exc:
        logger.error("Latency failed: %s", exc)
        lat = {"latency_ms": float("inf"), "tokens_per_s": 0.0}
    logger.info("Latency: %.1f ms  |  Tokens/s: %.1f", lat["latency_ms"], lat["tokens_per_s"])

    # ------------------------------------------------------------------
    # 7. Logit similarity vs baseline
    # ------------------------------------------------------------------
    logger.info("Computing logit similarity vs baseline …")
    try:
        sim = compute_logit_similarity(baseline_model, quant_model, samples, device,
                                       max_samples=16)
    except Exception as exc:
        logger.warning("Logit similarity failed: %s", exc)
        sim = {"cosine_similarity": 0.0, "mse": float("inf")}
    logger.info("Cosine similarity: %.4f  |  MSE: %.4e", sim["cosine_similarity"], sim["mse"])

    # ------------------------------------------------------------------
    # 8. Assemble result and enrich with deltas
    # ------------------------------------------------------------------
    result: Dict[str, Any] = {
        "method":            method,
        "device":            str(device),
        "dtype":             str(dtype),
        "max_samples":       args.max_samples,
        "max_length":        args.max_length,
        "perplexity":        round(ppl, 4),
        "model_memory_mb":   mem["model_memory_mb"],
        "cuda_allocated_mb": mem["cuda_allocated_mb"],
        "cuda_reserved_mb":  mem["cuda_reserved_mb"],
        "latency_ms":        round(lat["latency_ms"], 2),
        "tokens_per_s":      round(lat["tokens_per_s"], 2),
        "cosine_similarity": sim["cosine_similarity"],
        "mse":               sim["mse"],
        **extra,
    }

    # Load baseline for deltas
    baseline = load_json("baseline.json") or {}
    if baseline:
        result = enrich_with_deltas(result, baseline)

    path = save_json(result, f"{method}.json")
    logger.info("Results saved → %s", path)
    logger.info("%s benchmark complete.", method.upper())

    # Cleanup
    del baseline_model, quant_model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
