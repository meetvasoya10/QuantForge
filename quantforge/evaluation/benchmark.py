"""
Master benchmark runner: assembles all per-method metrics into a
consolidated results dict and markdown table.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def save_json(data: Dict[str, Any], filename: str) -> Path:
    """
    Persist *data* as pretty-printed JSON inside the results directory.

    Args:
        data:     Dictionary to serialise.
        filename: Target filename (e.g. "baseline.json").

    Returns:
        Absolute path of the written file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved -> %s", path)
    return path


def load_json(filename: str) -> Optional[Dict[str, Any]]:
    """
    Load a previously saved results JSON file.

    Args:
        filename: Target filename relative to the results directory.

    Returns:
        Parsed dict or None if the file does not exist.
    """
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_benchmark_table(result_files: List[str]) -> str:
    """
    Load individual result JSONs and render a markdown comparison table.

    Args:
        result_files: List of JSON filenames (e.g. ["baseline.json", "int8.json"]).

    Returns:
        Markdown-formatted table string.
    """
    rows: List[Dict[str, Any]] = []
    for fname in result_files:
        data = load_json(fname)
        if data is None:
            logger.warning("Result file not found: %s - skipping.", fname)
            continue
        rows.append(data)

    if not rows:
        return "_No benchmark results found._\n"

    # Determine columns present across all rows
    columns = [
        "method",
        "perplexity",
        "perplexity_delta",
        "model_memory_mb",
        "memory_reduction_pct",
        "latency_ms",
        "tokens_per_s",
        "speed_change_pct",
        "cosine_similarity",
        "mse",
    ]
    present = [c for c in columns if any(c in r for r in rows)]

    header = "| " + " | ".join(present) + " |"
    sep = "| " + " | ".join(["---"] * len(present)) + " |"
    lines = [header, sep]

    for row in rows:
        cells = []
        for col in present:
            val = row.get(col, "--")
            if isinstance(val, float):
                cells.append(f"{val:.4f}" if abs(val) < 1e4 else f"{val:.2e}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def save_benchmark_table(result_files: List[str]) -> Path:
    """
    Build and write ``results/benchmark_table.md``.

    Args:
        result_files: List of JSON filenames.

    Returns:
        Path to the written markdown file.
    """
    table = build_benchmark_table(result_files)
    path = RESULTS_DIR / "benchmark_table.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# QuantForge Benchmark Results\n\n")
        fh.write(table)
    logger.info("Benchmark table saved -> %s", path)
    return path


def enrich_with_deltas(result: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute delta metrics relative to a baseline result dict.

    Adds the following keys to *result* (in-place and as return value):
        - ``perplexity_delta``      - absolute PPL increase vs baseline.
        - ``memory_reduction_pct``  - % reduction in model_memory_mb.
        - ``speed_change_pct``      - % change in tokens_per_s (positive = faster).

    Args:
        result:   Dict produced by one quantization method.
        baseline: Dict produced by the FP16 baseline run.

    Returns:
        Mutated *result* dict.
    """
    base_ppl = baseline.get("perplexity", 0.0)
    base_mem = baseline.get("model_memory_mb", 1.0)
    base_tps = baseline.get("tokens_per_s", 1.0)

    result["perplexity_delta"] = round(result.get("perplexity", 0.0) - base_ppl, 4)
    mem = result.get("model_memory_mb", base_mem)
    result["memory_reduction_pct"] = round((1.0 - mem / max(base_mem, 1e-6)) * 100, 2)
    tps = result.get("tokens_per_s", 0.0)
    result["speed_change_pct"] = round((tps / max(base_tps, 1e-6) - 1.0) * 100, 2)

    return result
