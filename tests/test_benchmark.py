import pytest
from quantforge.evaluation.benchmark import enrich_with_deltas

def test_enrich_with_deltas():
    baseline = {
        "perplexity": 20.0,
        "fp16_model_memory_mb": 1000.0,
        "actual_storage_memory_mb": 1000.0,
        "effective_quantized_memory_mb": 1000.0,
        "latency_ms": 50.0,
        "tokens_per_s": 20.0
    }
    
    result = {
        "perplexity": 21.0,
        "actual_storage_memory_mb": 500.0,
        "effective_quantized_memory_mb": 500.0,
        "latency_ms": 60.0,
        "tokens_per_s": 16.66,
        "cosine_similarity": 0.99
    }
    
    enriched = enrich_with_deltas(result, baseline)
    assert enriched["perplexity_delta"] == 1.0
    assert enriched["memory_reduction_pct"] == 50.0
    assert enriched["speed_change_pct"] == -16.7
    assert enriched["status"] == "success"

def test_enrich_failed_status():
    baseline = {
        "perplexity": 20.0,
        "fp16_model_memory_mb": 1000.0,
        "actual_storage_memory_mb": 1000.0,
        "effective_quantized_memory_mb": 1000.0,
        "latency_ms": 50.0,
        "tokens_per_s": 20.0
    }
    
    result = {
        "perplexity": 40.0, # Huge degradation
        "actual_storage_memory_mb": 500.0,
        "effective_quantized_memory_mb": 500.0,
        "latency_ms": 60.0,
        "tokens_per_s": 16.66,
        "cosine_similarity": 0.5 # Poor similarity
    }
    
    enriched = enrich_with_deltas(result, baseline)
    assert enriched["status"] == "failed"
