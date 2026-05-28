import logging
from typing import Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

def load_with_backend(
    model_name: str, 
    backend: str, 
    device: str = "cuda", 
    dtype: torch.dtype = torch.float16
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load model with a specified optimized backend if requested.
    Backends:
      - 'simulated' / 'fp16': Baseline load, simulated methods modify it.
      - 'bitsandbytes_8bit': Real INT8 using bitsandbytes (LLM.int8()).
      - 'bitsandbytes_4bit': Real INT4 using bitsandbytes.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    if backend == "bitsandbytes_8bit":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto" if device == "cuda" else None,
            )
            logger.info("Loaded with bitsandbytes 8-bit backend.")
        except ImportError:
            raise ImportError("bitsandbytes not installed. Please `pip install bitsandbytes accelerate`")
            
    elif backend == "bitsandbytes_4bit":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                device_map="auto" if device == "cuda" else None,
            )
            logger.info("Loaded with bitsandbytes 4-bit backend.")
        except ImportError:
            raise ImportError("bitsandbytes not installed. Please `pip install bitsandbytes accelerate`")
            
    else:
        # Default fp16 load for simulated or baseline
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        if device == "cuda":
            model.to(device)
            
    return model, tokenizer
