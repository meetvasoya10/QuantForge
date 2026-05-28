import os
import random
import yaml
import torch
import numpy as np
import datetime
from typing import Dict, Any

def set_seed(seed: int = 42) -> None:
    """Set deterministic seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_run_metadata() -> Dict[str, Any]:
    """Collect hardware and software metadata."""
    import transformers
    import sys
    
    # Try to get commit hash safely
    commit_hash = "unknown"
    try:
        import subprocess
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        pass
        
    gpu_name = "unknown"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu_name": gpu_name,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": sys.version.split()[0],
        "commit_hash": commit_hash
    }

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
