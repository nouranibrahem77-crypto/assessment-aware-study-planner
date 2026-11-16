"""Execute the documented 30-profile synthetic coursework scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

from study_planner.integration.coursework import (
    DEFAULT_CSV_PATH,
    DEFAULT_RESULTS_PATH,
    execute_coursework_profiles,
)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / DEFAULT_RESULTS_PATH)
    parser.add_argument("--csv-output", type=Path, default=root / DEFAULT_CSV_PATH)
    args = parser.parse_args()
    payload = execute_coursework_profiles(root, args.output, args.csv_output)
    print(payload["summary"])
    print(f"JSON results: {args.output}")
    print(f"CSV results: {args.csv_output}")
    return 1 if payload["summary"]["execution_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
