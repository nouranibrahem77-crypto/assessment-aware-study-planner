"""Repository-local entry point for FFF feature generation."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"

if __name__ == "__main__":
    sys.path.insert(0, str(SOURCE_ROOT))
    runpy.run_module(
        "study_planner.data.feature_engineering.pipeline", run_name="__main__"
    )
