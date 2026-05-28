import pytest
from fastapi.testclient import TestClient
import torch

from quantforge.server import app, state

client = TestClient(app)

def test_health_not_loaded():
    # If state model is None, should return 503
    state.model = None
    response = client.get("/health")
    assert response.status_code == 503

def test_health_loaded():
    state.model = "dummy"
    state.model_name = "test-model"
    state.method = "fp16_baseline"
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "test-model", "method": "fp16_baseline"}

def test_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"request_count" in response.content

def test_generate_not_loaded():
    state.model = None
    response = client.post("/generate", json={"prompt": "Hello"})
    assert response.status_code == 503
