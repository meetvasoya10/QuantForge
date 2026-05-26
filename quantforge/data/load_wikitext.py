"""
Data loader for WikiText-2 validation split.
Returns tokenized batches ready for perplexity evaluation.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


def load_wikitext_samples(
    tokenizer: PreTrainedTokenizerBase,
    max_samples: int = 256,
    max_length: int = 512,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "validation",
) -> List[torch.Tensor]:
    """
    Load WikiText-2 validation samples and tokenize them.

    Each returned tensor is a 1-D LongTensor of token IDs with length <= max_length.
    Empty or very short sequences (< 2 tokens) are discarded.

    Args:
        tokenizer:      HuggingFace tokenizer matching the model.
        max_samples:    Maximum number of text chunks to return.
        max_length:     Maximum token length per chunk.
        dataset_name:   HuggingFace dataset identifier.
        dataset_config: Dataset configuration name.
        split:          Dataset split to use.

    Returns:
        List of 1-D LongTensors, one per usable text chunk.
    """
    logger.info("Loading %s / %s (%s split) ...", dataset_name, dataset_config, split)
    dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=False)

    samples: List[torch.Tensor] = []
    for row in dataset:
        text: str = row["text"].strip()  # type: ignore[index]
        if not text:
            continue

        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        ids: torch.Tensor = enc["input_ids"].squeeze(0)  # (seq_len,)
        if ids.numel() < 2:
            continue

        samples.append(ids)
        if len(samples) >= max_samples:
            break

    logger.info("Loaded %d tokenized samples (max_length=%d).", len(samples), max_length)
    return samples


def iterate_batches(
    samples: List[torch.Tensor],
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> Iterator[torch.Tensor]:
    """
    Yield batches of tokenized samples as padded LongTensors.

    Args:
        samples:    List of 1-D token-ID tensors.
        batch_size: Number of samples per batch.
        device:     Target device for the output tensor.

    Yields:
        2-D LongTensor of shape (batch_size, max_seq_len_in_batch).
        Sequences are right-padded with ``tokenizer.pad_token_id`` (default: 1).
    """
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        max_len = max(t.numel() for t in chunk)
        padded = torch.ones(len(chunk), max_len, dtype=torch.long)
        for j, t in enumerate(chunk):
            padded[j, : t.numel()] = t
        if device is not None:
            padded = padded.to(device)
        yield padded
