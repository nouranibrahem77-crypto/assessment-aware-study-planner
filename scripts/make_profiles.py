"""Repository-local entry point for reproducible profile selection."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"

if __name__ == "__main__":
    sys.path.insert(0, str(SOURCE_ROOT))
    from study_planner.eval.profiles import main

    raise SystemExit(main())
