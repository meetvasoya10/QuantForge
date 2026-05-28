import torch
import torch.nn as nn
from quantforge.quantization.int8 import replace_linear_with_int8
from quantforge.quantization.int4 import replace_linear_with_int4
from quantforge.quantization.gptq import apply_gptq
from quantforge.quantization.smoothquant import apply_smoothquant
from quantforge.quantization.ggfu import apply_ggfu

def test_int8_replacement():
    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.Linear(32, 16) # Skip because < 64 features
    )
    replaced = replace_linear_with_int8(model, min_features=64)
    assert replaced == 1
    assert hasattr(model[0], "q_weight")

def test_int4_replacement():
    model = nn.Sequential(
        nn.Linear(128, 64)
    )
    replaced = replace_linear_with_int4(model, min_features=64)
    assert replaced == 1
    assert hasattr(model[0], "packed_weight")

def test_gptq():
    model = nn.Sequential(
        nn.Linear(128, 64)
    )
    samples = [torch.randint(0, 100, (10,)) for _ in range(2)]
    # Needs a mock model that takes input_ids
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 64)
        def forward(self, input_ids):
            x = torch.randn(input_ids.shape[0], input_ids.shape[1], 128)
            return self.linear(x)

    mock = MockModel()
    errors = apply_gptq(mock, samples, torch.device("cpu"), calibration_count=2)
    assert "linear" in errors

def test_smoothquant():
    samples = [torch.randint(0, 100, (10,)) for _ in range(2)]
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 64)
        def forward(self, input_ids):
            x = torch.randn(input_ids.shape[0], input_ids.shape[1], 128)
            return self.linear(x)

    mock = MockModel()
    outliers = apply_smoothquant(mock, samples, torch.device("cpu"))
    assert "linear" in outliers

def test_ggfu():
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 64)

    mock = MockModel()
    metrics = apply_ggfu(mock, group_size=32)
    assert "linear" in metrics
    assert "cosine_similarity" in metrics["linear"]

def test_int4_pack_unpack():
    from quantforge.quantization.int4 import pack_int4, unpack_int4
    
    # Create random int8 weights in [-8, 7]
    original = torch.randint(-8, 8, (16, 32), dtype=torch.int8)
    
    packed = pack_int4(original)
    assert packed.dtype == torch.uint8
    assert packed.shape == (16, 16)
    
    unpacked = unpack_int4(packed, 16, 32)
    assert unpacked.dtype == torch.int8
    assert unpacked.shape == (16, 32)
    
    assert torch.equal(original, unpacked)
