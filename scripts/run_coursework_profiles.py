"""Execute the documented 30-profile synthetic coursework scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

from study_planner.integration.coursework import (
    DEFAULT_RUNS_DIRECTORY,
    execute_coursework_profiles,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "resume in this dedicated run directory; omit to create a new run "
            f"under {DEFAULT_RUNS_DIRECTORY}"
        ),
    )
    args = parser.parse_args()
    run_directory = args.run_dir
    if run_directory is not None and not run_directory.is_absolute():
        run_directory = root / run_directory
    payload = execute_coursework_profiles(root, run_directory)
    print(payload["summary"])
    print(f"Run directory: {run_directory or DEFAULT_RUNS_DIRECTORY}")
    return 1 if payload["summary"]["execution_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
