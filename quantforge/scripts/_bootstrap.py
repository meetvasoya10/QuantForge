"""
_bootstrap.py - Auto-relaunch with the venv Python if needed.

Import this at the very top of every QuantForge entry-point script
BEFORE any other project imports.  It will:
  1. Detect whether the current interpreter lives inside .venv/
  2. If not, re-exec the script using .venv/Scripts/python.exe
  3. If the venv python is missing, print a clear setup error

Usage (first two lines of any script):
    import quantforge.scripts._bootstrap  # noqa: F401
    # ... rest of imports ...
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# FIX WINDOWS SEGFAULT: datasets must be imported BEFORE torch to load DLLs correctly
import datasets  # noqa: F401

# Project root is 2 levels up from this file (quantforge/scripts/_bootstrap.py)
_ROOT = Path(__file__).resolve().parents[2]
_VENV_PYTHON = _ROOT / ".venv" / "Scripts" / "python.exe"


def _running_in_venv() -> bool:
    """Return True if sys.executable is inside .venv/."""
    try:
        Path(sys.executable).resolve().relative_to(_ROOT / ".venv")
        return True
    except ValueError:
        return False


def _ensure_venv() -> None:
    """Re-exec with venv Python if we're not already inside it."""
    # Proof of life
    print("QuantForge | Bootstrapping...", flush=True)

    if _running_in_venv():
        return  # Already correct Python — nothing to do

    if not _VENV_PYTHON.exists():
        print("", flush=True)
        print("ERROR: venv not found.", flush=True)
        print("Set it up first:", flush=True)
        print("  python -m venv .venv", flush=True)
        print("  .venv\\Scripts\\pip install torch --index-url https://download.pytorch.org/whl/cu121", flush=True)
        print("  .venv\\Scripts\\pip install -r requirements.txt", flush=True)
        sys.exit(1)

    # Re-launch: replace this process with the venv python + same args
    print(f"[QuantForge] Switching to venv Python: {_VENV_PYTHON}", flush=True)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    # Avoid passing -c without a script body if someone is testing with python -c
    args = sys.argv
    if args and args[0] == "-c":
        print("Cannot auto-relaunch a '-c' command. Please use the venv Python directly.")
        sys.exit(1)
        
    result = subprocess.run([str(_VENV_PYTHON)] + args)
    sys.exit(result.returncode)


# Run immediately on import
_ensure_venv()
