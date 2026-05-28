"""
run_all.py - Run the complete QuantForge benchmark suite end-to-end.

Usage:
    python -m quantforge.scripts.run_all [--max_samples 256] [--max_length 512]
                                         [--device cuda] [--dtype float16]
                                         [--skip_compile]
"""

# Auto-switch to venv Python if wrong interpreter is active
import quantforge.scripts._bootstrap  # noqa: F401

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quantforge.evaluation.benchmark import save_benchmark_table, save_benchmark_csv

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(
    fmt="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
))
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QuantForge Full Benchmark Suite")
    p.add_argument("--max_samples", type=int, default=256)
    p.add_argument("--max_length",  type=int, default=512)
    p.add_argument("--device",  default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--dtype",   default="float16",
                   choices=["float16", "float32", "bfloat16"])
    p.add_argument("--skip_compile", action="store_true",
                   help="Skip torch.compile benchmark")
    return p.parse_args()


def _run_step(label: str, fn) -> bool:
    """Execute fn; log and continue on any exception. Returns True on success."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  STEP: %s", label)
    logger.info("=" * 60)
    try:
        fn()
        return True
    except Exception:
        logger.error("STEP '%s' failed:\n%s", label, traceback.format_exc())
        return False


def main() -> None:
    args = parse_args()

    common = [
        "--max_samples", str(args.max_samples),
        "--max_length",  str(args.max_length),
        "--device",      args.device,
        "--dtype",       args.dtype,
    ]

    def run_baseline():
        import quantforge.scripts.run_baseline as m
        old = sys.argv[:]
        sys.argv = ["run_baseline"] + common
        try:
            m.main()
        finally:
            sys.argv = old

    def make_quant_runner(method: str):
        def _run():
            import quantforge.scripts.run_quantized as m
            old = sys.argv[:]
            sys.argv = ["run_quantized", "--method", method] + common
            try:
                m.main()
            finally:
                sys.argv = old
        _run.__name__ = f"run_{method}"
        return _run

    def run_kv():
        import quantforge.scripts.run_kv_cache as m
        old = sys.argv[:]
        sys.argv = ["run_kv_cache"]
        try:
            m.main()
        finally:
            sys.argv = old

    def run_compile():
        from quantforge.models.load_model import load_model_and_tokenizer
        from quantforge.optimization.compile_model import benchmark_compile
        from quantforge.evaluation.benchmark import save_json

        dtype_map = {
            "float16":  torch.float16,
            "float32":  torch.float32,
            "bfloat16": torch.bfloat16,
        }
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        dtype  = dtype_map[args.dtype]

        model, tokenizer = load_model_and_tokenizer(device=str(device), dtype=dtype)
        result = benchmark_compile(model, tokenizer, device)
        result["method"] = "compile"
        save_json(result, "compile.json")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    steps = [
        ("FP16 Baseline",                   run_baseline),
        ("INT8 W8A8",                        make_quant_runner("int8")),
        ("INT4 Weight-Only",                 make_quant_runner("int4")),
        ("GPTQ-style PTQ",                   make_quant_runner("gptq")),
        ("SmoothQuant Activation Scaling",   make_quant_runner("smoothquant")),
        ("GGFU Group Quantization",          make_quant_runner("ggfu")),
        ("KV-Cache Estimation",              run_kv),
    ]

    try:
        import bitsandbytes
        steps.append(("BitsAndBytes 8-bit", make_quant_runner("bitsandbytes_8bit")))
        steps.append(("BitsAndBytes 4-bit", make_quant_runner("bitsandbytes_4bit")))
    except ImportError:
        logger.info("bitsandbytes not installed, skipping optimized backend benchmarks.")

    if not args.skip_compile:
        steps.append(("torch.compile Benchmark", run_compile))

    passed = 0
    for label, fn in steps:
        ok = _run_step(label, fn)
        if ok:
            passed += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("")
    logger.info("Building benchmark table ...")
    result_files = [
        "baseline.json", "int8.json", "int4.json",
        "gptq.json", "smoothquant.json", "ggfu.json",
        "bitsandbytes_8bit.json", "bitsandbytes_4bit.json"
    ]
    try:
        table_path = save_benchmark_table(result_files)
        csv_path = save_benchmark_csv(result_files)
        logger.info("Benchmark table -> %s", table_path)
        logger.info("Benchmark CSV   -> %s", csv_path)
    except Exception:
        logger.error("Failed to build benchmark table:\n%s", traceback.format_exc())

    logger.info("")
    logger.info("=" * 60)
    logger.info("QuantForge complete: %d/%d steps succeeded.", passed, len(steps))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
