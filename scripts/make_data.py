"""Repository-local entry point for OULAD acquisition and exploration."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

if __name__ == "__main__":
    sys.path.insert(0, str(SOURCE_ROOT))
    runpy.run_module("study_planner.data.oulad", run_name="__main__")
