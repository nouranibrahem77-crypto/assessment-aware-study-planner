"""Tests for reproducible experiment execution and result persistence."""

import csv
import json

from study_planner.eval.experiment_runner import (
    DEFAULT_EXPERIMENT_SEEDS,
    ExperimentCase,
    ExperimentOutcome,
    ExperimentRunStatus,
    run_experiment,
    summarize_runs,
    write_experiment_results,
)


def test_run_experiment_uses_five_fixed_seeds_and_keeps_running_after_error() -> None:
    calls: list[int] = []
    case = ExperimentCase("UCS", "small-graph")

    def evaluate(_case: ExperimentCase, seed: int) -> ExperimentOutcome:
        calls.append(seed)
        if seed == 2025:
            raise RuntimeError("synthetic failure")
        return ExperimentOutcome(
            ExperimentRunStatus.FEASIBLE,
            {"nodes_expanded": seed},
        )

    records = run_experiment("search", (case,), evaluate)

    assert tuple(calls) == DEFAULT_EXPERIMENT_SEEDS
    assert len(records) == 5
    assert records[2].status is ExperimentRunStatus.ERROR
    assert records[2].error_type == "RuntimeError"
    assert records[3].seed == 31415
    assert records[3].status is ExperimentRunStatus.FEASIBLE


def test_summarize_runs_uses_sample_standard_deviation_and_counts_statuses() -> None:
    case = ExperimentCase("BT", "capacity-2")

    def evaluate(_case: ExperimentCase, seed: int) -> ExperimentOutcome:
        if seed == 271828:
            return ExperimentOutcome(
                ExperimentRunStatus.INCOMPLETE,
                {"nodes_visited": None},
                failure_reasons=("node_limit",),
            )
        return ExperimentOutcome(
            ExperimentRunStatus.FEASIBLE,
            {"nodes_visited": float(seed)},
        )

    records = run_experiment("csp", (case,), evaluate)
    summary = next(
        row for row in summarize_runs(records) if row.metric == "nodes_visited"
    )

    assert summary.valid_count == 4
    assert summary.mean == (42 + 123 + 2025 + 31415) / 4
    assert summary.standard_deviation is not None
    assert summary.feasible_count == 4
    assert summary.incomplete_count == 1
    assert summary.error_count == 0


def test_write_experiment_results_writes_csv_and_json(tmp_path) -> None:
    case = ExperimentCase(
        "Greedy",
        "profile-001",
        profile_id="001",
        configuration={"heuristic": "sessions_left"},
    )
    records = run_experiment(
        "search",
        (case,),
        lambda _case, seed: ExperimentOutcome(
            ExperimentRunStatus.FEASIBLE,
            {
                "solution_cost": 2,
                "nodes_expanded": 5,
                "expanded_order": ("start", "left", "right"),
            },
            provenance={"model": "baseline"},
        ),
        seeds=(42,),
    )

    runs_path, summary_path, json_path = write_experiment_results(
        records, tmp_path, "search"
    )

    with runs_path.open(newline="", encoding="utf-8") as stream:
        run_row = next(csv.DictReader(stream))
    with summary_path.open(newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert run_row["seed"] == "42"
    assert run_row["status"] == "feasible"
    assert json.loads(run_row["configuration_json"]) == {
        "heuristic": "sessions_left"
    }
    assert json.loads(run_row["expanded_order"]) == ["start", "left", "right"]
    assert len(summary_rows) == 2
    assert {row["metric"] for row in summary_rows} == {
        "nodes_expanded",
        "solution_cost",
    }
    assert payload["schema_version"] == 1
    assert payload["runs"][0]["provenance"] == {"model": "baseline"}
    assert payload["runs"][0]["metrics"]["expanded_order"] == [
        "start",
        "left",
        "right",
    ]
