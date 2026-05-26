"""
run_kv_cache.py - KV-cache INT8 memory estimation benchmark.

Usage:
    python -m quantforge.scripts.run_kv_cache [--batch_size N]
"""

# Auto-switch to venv Python if wrong interpreter is active
import quantforge.scripts._bootstrap  # noqa: F401

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantforge.quantization.kv_cache import compare_kv_cache, build_kv_cache_table
from quantforge.evaluation.benchmark import save_json, RESULTS_DIR

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
))
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuantForge KV-cache Memory Estimation")
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    seq_lengths = [128, 512, 1024, 2048, 4096]
    num_layers  = 12   # OPT-125M
    num_heads   = 12   # OPT-125M
    head_dim    = 64   # 768 / 12

    logger.info("=" * 60)
    logger.info("QuantForge  -  KV-Cache Memory Estimation")
    logger.info("Model: facebook/opt-125m | batch=%d", args.batch_size)
    logger.info("=" * 60)

    comparison = compare_kv_cache(
        seq_lengths=seq_lengths,
        batch_size=args.batch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
    )

    for row in comparison:
        logger.info(
            "seq_len=%5d | FP16=%.4f MB | INT8=%.4f MB | reduction=%.1f%%",
            row["seq_len"], row["fp16_mb"], row["int8_mb"], row["reduction_pct"],
        )

    table_md = build_kv_cache_table(comparison)

    payload = {
        "method":      "kv_cache_estimation",
        "batch_size":  args.batch_size,
        "num_layers":  num_layers,
        "num_heads":   num_heads,
        "head_dim":    head_dim,
        "results":     comparison,
    }
    save_json(payload, "kv_cache.json")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = RESULTS_DIR / "kv_cache_table.md"
    md_path.write_text(
        "# KV-Cache Memory Estimation (FP16 vs INT8)\n\n"
        f"Model: facebook/opt-125m | Batch size: {args.batch_size}\n\n"
        + table_md,
        encoding="utf-8",
    )
    logger.info("Markdown table saved -> %s", md_path)
    logger.info("KV-cache estimation complete.")


if __name__ == "__main__":
    main()
