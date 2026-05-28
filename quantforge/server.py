import os
import sys
import time
import logging
from typing import Dict, Any, List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Bootstrap path
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantforge.models.load_model import load_model_and_tokenizer
from quantforge.evaluation.latency import measure_latency
from quantforge.evaluation.memory import measure_memory

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter("request_count", "Total requests", ["endpoint", "status"])
ERROR_COUNT = Counter("error_count", "Total errors", ["endpoint"])
LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])
GEN_TOKENS_SEC = Gauge("tokens_per_second", "Generation tokens per second")
GPU_MEM_ALLOCATED = Gauge("gpu_memory_allocated_mb", "GPU memory allocated in MB")

app = FastAPI(title="QuantForge Serving API")

# Global state
class ServerState:
    model = None
    tokenizer = None
    device = "cpu"
    dtype = "float16"
    method = "fp16_baseline"
    model_name = ""

state = ServerState()

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50

class BenchmarkRequest(BaseModel):
    max_samples: int = 64
    max_length: int = 512

@app.on_event("startup")
async def startup_event():
    import argparse
    
    # We parse manually from sys.argv because uvicorn passes its own args
    # But for a simple script, we'll use environment variables or defaults
    state.model_name = os.getenv("QF_MODEL", "facebook/opt-125m")
    state.method = os.getenv("QF_METHOD", "int8")
    state.device = os.getenv("QF_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading {state.model_name} with method {state.method} on {state.device}...")
    t0 = time.time()
    
    dtype_map = {"fp16_baseline": torch.float16, "int8": torch.float16, "int4": torch.float16, "gptq": torch.float16, "smoothquant": torch.float16, "ggfu": torch.float16}
    state.dtype = dtype_map.get(state.method, torch.float16)
    
    model, tokenizer = load_model_and_tokenizer(state.model_name, device=state.device, dtype=state.dtype)
    
    if state.method == "int8":
        from quantforge.quantization.int8 import replace_linear_with_int8
        replace_linear_with_int8(model)
    elif state.method == "int4":
        from quantforge.quantization.int4 import replace_linear_with_int4
        replace_linear_with_int4(model)
    elif state.method == "gptq":
        from quantforge.quantization.gptq import apply_gptq
        # Note: GPTQ requires calibration data. We'll use a dummy input for API
        dummy = [torch.randint(0, 1000, (10,)) for _ in range(2)]
        apply_gptq(model, dummy, torch.device(state.device), calibration_count=2)
    elif state.method == "smoothquant":
        from quantforge.quantization.smoothquant import apply_smoothquant
        dummy = [torch.randint(0, 1000, (10,)) for _ in range(2)]
        apply_smoothquant(model, dummy, torch.device(state.device))
    elif state.method == "ggfu":
        from quantforge.quantization.ggfu import apply_ggfu
        apply_ggfu(model, group_size=32)
        
    state.model = model
    state.tokenizer = tokenizer
    
    # Warmup
    try:
        inputs = tokenizer("Hello", return_tensors="pt").to(state.device)
        model.generate(**inputs, max_new_tokens=2, do_sample=False)
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
        
    logger.info(f"Model loaded and warmed up in {time.time() - t0:.2f}s")

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        status = "success" if response.status_code < 400 else "error"
        REQUEST_COUNT.labels(endpoint=request.url.path, status=status).inc()
        LATENCY.labels(endpoint=request.url.path).observe(time.time() - start_time)
        return response
    except Exception as e:
        REQUEST_COUNT.labels(endpoint=request.url.path, status="error").inc()
        ERROR_COUNT.labels(endpoint=request.url.path).inc()
        raise e

@app.get("/health")
def health():
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": state.model_name, "method": state.method}

@app.get("/metrics")
def metrics():
    if state.device == "cuda":
        GPU_MEM_ALLOCATED.set(torch.cuda.memory_allocated() / (1024**2))
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/generate")
def generate(req: GenerateRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    t0 = time.time()
    inputs = state.tokenizer(req.prompt, return_tensors="pt").to(state.device)
    
    with torch.inference_mode():
        outputs = state.model.generate(
            **inputs, 
            max_new_tokens=req.max_new_tokens,
            do_sample=False
        )
    
    gen_time = time.time() - t0
    
    input_len = inputs["input_ids"].shape[1]
    gen_tokens = outputs[0][input_len:]
    text = state.tokenizer.decode(gen_tokens, skip_special_tokens=True)
    
    tps = len(gen_tokens) / gen_time if gen_time > 0 else 0
    GEN_TOKENS_SEC.set(tps)
    
    return {
        "text": text,
        "latency_ms": gen_time * 1000,
        "tokens_generated": len(gen_tokens),
        "tokens_per_sec": tps
    }

@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    lat = measure_latency(state.model, state.tokenizer, torch.device(state.device))
    mem = measure_memory(state.model, torch.device(state.device))
    
    return {
        "latency": lat,
        "memory": mem
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="QuantForge API Server")
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--method", default="int8")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    os.environ["QF_MODEL"] = args.model
    os.environ["QF_METHOD"] = args.method
    os.environ["QF_DEVICE"] = args.device
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
