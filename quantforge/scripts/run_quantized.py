"""
run_quantized.py - Benchmark any single quantization method.

Usage:
    python -m quantforge.scripts.run_quantized --method int8        --max_samples 256
    python -m quantforge.scripts.run_quantized --method int4        --max_samples 256
    python -m quantforge.scripts.run_quantized --method gptq        --max_samples 256
    python -m quantforge.scripts.run_quantized --method smoothquant --max_samples 256
    python -m quantforge.scripts.run_quantized --method ggfu        --max_samples 256
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)   # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)   # type: ignore[union-attr]
except Exception:
    pass

print("QuantForge | run_quantized starting ...", flush=True)

try:
    import torch  # noqa: E402
except ModuleNotFoundError:
    print("", flush=True)
    print("ERROR: 'torch' not found. You are using the wrong Python.", flush=True)
    print("  .venv\\Scripts\\python.exe -m quantforge.scripts.run_quantized --method int8", flush=True)
    print("  OR:  .\\quantforge.bat -m quantforge.scripts.run_quantized --method int8", flush=True)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantforge.models.load_model import load_model_and_tokenizer, clone_model
from quantforge.data.load_wikitext import load_wikitext_samples
from quantforge.evaluation.perplexity import compute_perplexity
from quantforge.evaluation.latency import measure_latency
from quantforge.evaluation.memory import measure_memory
from quantforge.evaluation.layer_error import compute_logit_similarity
from quantforge.evaluation.benchmark import save_json, load_json, enrich_with_deltas

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
))
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger(__name__)

SUPPORTED_METHODS = ("int8", "int4", "gptq", "smoothquant", "ggfu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuantForge Quantization Benchmark")
    p.add_argument("--method", required=True, choices=SUPPORTED_METHODS)
    p.add_argument("--max_samples", type=int, default=256)
    p.add_argument("--max_length",  type=int, default=512)
    p.add_argument("--device",  default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype",   default="float16",
                   choices=["float16", "float32", "bfloat16"])
    return p.parse_args()


def _apply_method(
    method: str,
    model: torch.nn.Module,
    samples: list[torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """Apply quantization and return extra metadata dict."""
    extra: Dict[str, Any] = {}

    if method == "int8":
        from quantforge.quantization.int8 import replace_linear_with_int8
        n = replace_linear_with_int8(model)
        extra["layers_replaced"] = n
        logger.info("      INT8: replaced %d linear layers.", n)

    elif method == "int4":
        from quantforge.quantization.int4 import replace_linear_with_int4
        n = replace_linear_with_int4(model)
        extra["layers_replaced"] = n
        logger.info("      INT4: replaced %d linear layers.", n)

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


def main() -> None:
    args = parse_args()
    method = args.method

    dtype_map = {
        "float16":  torch.float16,
        "float32":  torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype  = dtype_map[args.dtype]
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )

    logger.info("=" * 60)
    logger.info("QuantForge  -  %s Benchmark", method.upper())
    logger.info("device=%s  dtype=%s  max_samples=%d  max_length=%d",
                device, dtype, args.max_samples, args.max_length)
    logger.info("=" * 60)

    # [1/7] Load model
    logger.info("[1/7] Loading model facebook/opt-125m ...")
    logger.info("      (first run downloads ~500 MB - progress bar appears below)")
    baseline_model, tokenizer = load_model_and_tokenizer(device=str(device), dtype=dtype)
    logger.info("      Model loaded OK.")

    # [2/7] Load data
    logger.info("[2/7] Loading WikiText-2 validation samples ...")
    samples = load_wikitext_samples(tokenizer, max_samples=args.max_samples,
                                    max_length=args.max_length)
    logger.info("      Loaded %d samples.", len(samples))

    # [3/7] Clone for quantization
    logger.info("[3/7] Cloning model for quantization ...")
    quant_model = clone_model(baseline_model)
    logger.info("      Clone ready.")

    # [4/7] Apply quantization
    logger.info("[4/7] Applying %s quantization ...", method.upper())
    extra: Dict[str, Any] = {}
    try:
        extra = _apply_method(method, quant_model, samples, device)
        logger.info("      Quantization applied.")
    except Exception as exc:
        logger.error("      Quantization failed: %s", exc, exc_info=True)
        extra["quantization_error"] = str(exc)

    # [5/7] Perplexity
    logger.info("[5/7] Computing perplexity over %d samples ...", len(samples))
    try:
        ppl = compute_perplexity(quant_model, samples, device)
    except Exception as exc:
        logger.error("      Perplexity eval failed: %s", exc)
        ppl = float("inf")
    logger.info("      Perplexity: %.4f", ppl)

    # Memory
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mem = measure_memory(quant_model, device)
    logger.info("      Model memory: %.2f MB", mem["model_memory_mb"])

    # [6/7] Latency
    logger.info("[6/7] Measuring generation latency ...")
    try:
        lat = measure_latency(quant_model, tokenizer, device)
    except Exception as exc:
        logger.error("      Latency failed: %s", exc)
        lat = {"latency_ms": float("inf"), "tokens_per_s": 0.0}
    logger.info("      Latency: %.2f ms  |  Tokens/s: %.2f",
                lat["latency_ms"], lat["tokens_per_s"])

    # [7/7] Logit similarity
    logger.info("[7/7] Computing logit similarity vs baseline (16 samples) ...")
    try:
        sim = compute_logit_similarity(baseline_model, quant_model, samples, device,
                                       max_samples=16)
    except Exception as exc:
        logger.warning("      Logit similarity failed: %s", exc)
        sim = {"cosine_similarity": 0.0, "mse": float("inf")}
    logger.info("      Cosine sim: %.4f  |  MSE: %.4e",
                sim["cosine_similarity"], sim["mse"])

    # Assemble + save
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

    baseline = load_json("baseline.json") or {}
    if baseline:
        result = enrich_with_deltas(result, baseline)

    path = save_json(result, f"{method}.json")

    del baseline_model, quant_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("")
    logger.info("Results saved -> %s", path)
    logger.info("=" * 60)
    logger.info("%s COMPLETE.", method.upper())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
