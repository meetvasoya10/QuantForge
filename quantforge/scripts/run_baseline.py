"""
run_baseline.py - FP16 baseline benchmark for facebook/opt-125m.

Usage:
    python -m quantforge.scripts.run_baseline [--max_samples N] [--max_length N]
                                              [--device cuda|cpu] [--dtype float16|float32]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Force unbuffered stdout so every log line shows immediately
# (must happen before any heavy imports)
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)

import torch

# ── project imports ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantforge.models.load_model import load_model_and_tokenizer
from quantforge.data.load_wikitext import load_wikitext_samples
from quantforge.evaluation.perplexity import compute_perplexity
from quantforge.evaluation.latency import measure_latency
from quantforge.evaluation.memory import measure_memory
from quantforge.evaluation.benchmark import save_json, RESULTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuantForge FP16 Baseline Benchmark")
    p.add_argument("--max_samples", type=int, default=256,
                   help="Number of WikiText-2 validation samples (default: 256)")
    p.add_argument("--max_length",  type=int, default=512,
                   help="Maximum token length per sample (default: 512)")
    p.add_argument("--device",  default="cuda", choices=["cuda", "cpu"],
                   help="Inference device (default: cuda)")
    p.add_argument("--dtype",   default="float16",
                   choices=["float16", "float32", "bfloat16"],
                   help="Model dtype (default: float16)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

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
    logger.info("QuantForge  -  FP16 Baseline")
    logger.info("device=%s  dtype=%s  max_samples=%d  max_length=%d",
                device, dtype, args.max_samples, args.max_length)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load model + tokenizer
    #   First run downloads ~500 MB from HuggingFace Hub.
    #   You will see a progress bar from the transformers library.
    # ------------------------------------------------------------------
    logger.info("[1/5] Loading model facebook/opt-125m ...")
    logger.info("      (first run downloads ~500 MB - progress bar will appear below)")
    model, tokenizer = load_model_and_tokenizer(device=str(device), dtype=dtype)
    logger.info("      Model loaded OK.")

    # ------------------------------------------------------------------
    # Step 2: Load WikiText-2 validation data
    # ------------------------------------------------------------------
    logger.info("[2/5] Loading WikiText-2 validation samples ...")
    samples = load_wikitext_samples(
        tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length,
    )
    logger.info("      Loaded %d samples.", len(samples))

    # ------------------------------------------------------------------
    # Step 3: Perplexity
    # ------------------------------------------------------------------
    logger.info("[3/5] Computing perplexity over %d samples ...", len(samples))
    ppl = compute_perplexity(model, samples, device)
    logger.info("      Perplexity: %.4f", ppl)

    # ------------------------------------------------------------------
    # Step 4: Memory
    # ------------------------------------------------------------------
    logger.info("[4/5] Measuring memory ...")
    mem = measure_memory(model, device)
    logger.info("      Model memory : %.2f MB", mem["model_memory_mb"])
    logger.info("      CUDA alloc   : %.2f MB", mem["cuda_allocated_mb"])
    logger.info("      CUDA reserved: %.2f MB", mem["cuda_reserved_mb"])

    # ------------------------------------------------------------------
    # Step 5: Latency (3 warmup + 10 timed runs)
    # ------------------------------------------------------------------
    logger.info("[5/5] Measuring generation latency (warmup + 10 runs) ...")
    lat = measure_latency(model, tokenizer, device)
    logger.info("      Latency : %.1f ms", lat["latency_ms"])
    logger.info("      Tokens/s: %.1f", lat["tokens_per_s"])

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    result = {
        "method":               "fp16_baseline",
        "device":               str(device),
        "dtype":                str(dtype),
        "max_samples":          args.max_samples,
        "max_length":           args.max_length,
        "perplexity":           round(ppl, 4),
        "perplexity_delta":     0.0,
        "model_memory_mb":      mem["model_memory_mb"],
        "memory_reduction_pct": 0.0,
        "cuda_allocated_mb":    mem["cuda_allocated_mb"],
        "cuda_reserved_mb":     mem["cuda_reserved_mb"],
        "latency_ms":           round(lat["latency_ms"], 2),
        "tokens_per_s":         round(lat["tokens_per_s"], 2),
        "speed_change_pct":     0.0,
        "cosine_similarity":    1.0,
        "mse":                  0.0,
    }

    path_json = save_json(result, "baseline.json")

    md = (
        "# QuantForge - FP16 Baseline\n\n"
        f"| Metric | Value |\n"
        f"| --- | --- |\n"
        f"| Perplexity | {result['perplexity']:.4f} |\n"
        f"| Model memory (MB) | {result['model_memory_mb']:.2f} |\n"
        f"| CUDA allocated (MB) | {result['cuda_allocated_mb']:.2f} |\n"
        f"| Latency (ms) | {result['latency_ms']:.2f} |\n"
        f"| Tokens/s | {result['tokens_per_s']:.2f} |\n"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "baseline.md").write_text(md, encoding="utf-8")

    logger.info("")
    logger.info("Results saved -> %s", path_json)
    logger.info("=" * 60)
    logger.info("Baseline COMPLETE.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
