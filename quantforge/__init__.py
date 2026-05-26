"""
QuantForge: Advanced LLM Quantization Engine for Low-Bit Transformer Inference.

Benchmarks FP16 baseline against INT8/W8A8, INT4 weight-only, GPTQ-style PTQ,
SmoothQuant-style activation scaling, GGFU custom quantization, KV-cache INT8
memory estimation, torch.fx layer replacement, and torch.compile inference.
"""

__version__ = "1.0.0"
__author__ = "QuantForge"
