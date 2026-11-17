"""Run the synthetic three-algorithm Search comparison without replacing demos."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_demo_experiments import _search_scenarios

from study_planner.eval.experiments import (
    run_search_comparison,
    write_experiment_bundle,
    write_experiment_plot,
)


def main() -> int:
    """Run Experiment 1 and write its run and summary tables plus plots."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=root / "reports/experiment_results/experiment1_astar",
    )
    args = parser.parse_args()
    output = args.output_directory
    if (output / "search_runs.json").exists():
        parser.error(f"refusing to overwrite existing Experiment 1 results: {output}")

    records = run_search_comparison(_search_scenarios())
    write_experiment_bundle(records, output, "search", "nodes_expanded")
    write_experiment_plot(
        records, output, "search_nodes_generated", "nodes_generated"
    )
    write_experiment_plot(records, output, "search_solution_cost", "solution_cost")
    print(f"Experiment 1 runs and summaries: {output}")
    print(f"Completed runs: {len(records)}")
    return 1 if any(record.error_type for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
