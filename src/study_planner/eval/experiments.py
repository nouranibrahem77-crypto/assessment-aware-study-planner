"""Evaluation entry points for the three planner experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from study_planner.csp.ac3 import Ac3Options
from study_planner.csp.backtracking import BacktrackingOptions
from study_planner.csp.core import CspProblem
from study_planner.csp.forward_checking import ForwardCheckingOptions
from study_planner.csp.verification import WholePlanVerificationPolicy
from study_planner.eval.experiment_runner import (
    DEFAULT_EXPERIMENT_SEEDS,
    ExperimentCase,
    ExperimentOutcome,
    ExperimentRunRecord,
    ExperimentRunStatus,
    run_experiment,
    write_experiment_results,
)
from study_planner.integration.planner import FinalPlannerResult, PlannerStatus
from study_planner.integration.schedule_bridge import (
    CspAlgorithm,
    WholePlanEnumerationStatus,
    WholePlanScheduleEnumerator,
)
from study_planner.search.astar import a_star_search
from study_planner.search.core import SearchProblem, SearchResult
from study_planner.search.greedy import greedy_best_first_search
from study_planner.search.ucs import uniform_cost_search

SearchHeuristic = Callable[[Any], float]
PlannerScenarioRunner = Callable[[str, int], FinalPlannerResult]
_CspOptions = BacktrackingOptions | ForwardCheckingOptions | Ac3Options | None


@dataclass(frozen=True, slots=True)
class SearchScenario:
    """One shared finite problem with separate Greedy and A* heuristics."""

    scenario_id: str
    problem: SearchProblem[Any, Any]
    heuristic: SearchHeuristic
    configuration: Mapping[str, Any]
    astar_heuristic: SearchHeuristic | None = None


@dataclass(frozen=True, slots=True)
class CspScenario:
    """One fixed CSP instance shared by all propagation strategies."""

    scenario_id: str
    problem: CspProblem
    policy: WholePlanVerificationPolicy
    configuration: Mapping[str, Any]
    options: Mapping[CspAlgorithm, _CspOptions] | None = None


@dataclass(frozen=True, slots=True)
class PredictionScenario:
    """A profile and runner that constructs matched NN/baseline planner runs."""

    scenario_id: str
    profile_id: str
    configuration: Mapping[str, Any]
    run: PlannerScenarioRunner


def run_search_comparison(
    scenarios: Sequence[SearchScenario],
    *,
    seeds: Sequence[int] = DEFAULT_EXPERIMENT_SEEDS,
) -> tuple[ExperimentRunRecord, ...]:
    """Compare UCS, Greedy, and A* on identical scenarios and retain each run."""
    if not scenarios:
        raise ValueError("at least one search scenario is required")
    cases = tuple(
        ExperimentCase(
            method, scenario.scenario_id, configuration=scenario.configuration
        )
        for scenario in scenarios
        for method in ("ucs", "greedy_best_first", "a_star")
    )
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    if len(by_id) != len(scenarios):
        raise ValueError("search scenario IDs must be unique")
    if any(scenario.astar_heuristic is None for scenario in scenarios):
        raise ValueError("every search scenario must define a consistent A* heuristic")

    def evaluate(case: ExperimentCase, _seed: int) -> ExperimentOutcome:
        scenario = by_id[case.scenario_id]
        if case.method == "ucs":
            result = uniform_cost_search(scenario.problem)
        elif case.method == "greedy_best_first":
            result = greedy_best_first_search(scenario.problem, scenario.heuristic)
        else:
            assert scenario.astar_heuristic is not None
            result = a_star_search(scenario.problem, scenario.astar_heuristic)
        return _search_outcome(result)

    return run_experiment("search", cases, evaluate, seeds=seeds)


def run_csp_comparison(
    scenarios: Sequence[CspScenario],
    *,
    seeds: Sequence[int] = DEFAULT_EXPERIMENT_SEEDS,
) -> tuple[ExperimentRunRecord, ...]:
    """Compare BT, BT+FC, and BT+FC+AC-3 on identical CSP instances."""
    if not scenarios:
        raise ValueError("at least one CSP scenario is required")
    methods = {
        "bt": CspAlgorithm.BT,
        "bt_fc": CspAlgorithm.BT_FC,
        "bt_fc_ac3": CspAlgorithm.BT_FC_AC3,
    }
    cases = tuple(
        ExperimentCase(
            method, scenario.scenario_id, configuration=scenario.configuration
        )
        for scenario in scenarios
        for method in methods
    )
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    if len(by_id) != len(scenarios):
        raise ValueError("CSP scenario IDs must be unique")

    def evaluate(case: ExperimentCase, _seed: int) -> ExperimentOutcome:
        scenario = by_id[case.scenario_id]
        algorithm = methods[case.method]
        options = None if scenario.options is None else scenario.options.get(algorithm)
        enumerator = WholePlanScheduleEnumerator(
            scenario.problem,
            scenario.policy,
            algorithm=algorithm,
            options=options,
        )
        terminal = None
        while terminal is None:
            step = enumerator.next_schedule()
            if step.outcome is not None:
                terminal = step.outcome
        stats = terminal.statistics
        status = (
            ExperimentRunStatus.INCOMPLETE
            if terminal.status is WholePlanEnumerationStatus.INCOMPLETE
            else ExperimentRunStatus.FEASIBLE
            if terminal.feasible_schedules
            else ExperimentRunStatus.INFEASIBLE
        )
        return ExperimentOutcome(
            status,
            {
                "exhaustive": int(
                    terminal.status is not WholePlanEnumerationStatus.INCOMPLETE
                ),
                "feasible_schedules": terminal.feasible_schedules,
                "schedules_examined": terminal.schedules_examined,
                "rejected_schedules": terminal.rejected_schedules,
                "nodes_visited": stats.nodes_visited,
                "assignments_attempted": stats.assignments_attempted,
                "backtracks": stats.backtracks,
                "constraint_checks": stats.constraint_checks,
                "domain_prunings": stats.domain_prunings,
                "compatibility_checks": stats.compatibility_checks,
                "solver_time_seconds": stats.solver_time_seconds,
                "wall_time_seconds": stats.wall_time_seconds,
            },
            {
                "whole_plan_identity": terminal.whole_plan_identity,
                "problem_identity": terminal.problem_identity,
                "terminal_status": terminal.status.value,
                "rejection_reasons": terminal.rejection_reasons,
            },
            terminal.rejection_reasons,
        )

    return run_experiment("csp", cases, evaluate, seeds=seeds)


def run_prediction_comparison(
    scenarios: Sequence[PredictionScenario],
    *,
    seeds: Sequence[int] = DEFAULT_EXPERIMENT_SEEDS,
    experiment_id: str = "prediction",
) -> tuple[ExperimentRunRecord, ...]:
    """Compare matched NN and average-baseline planner outcomes per profile.

    Each scenario runner receives ``("nn" | "average", seed)`` and must build
    both runs from the same profile and planner configuration. Use this hook to
    bind approved application settings; the module intentionally invents none.
    """
    if not scenarios:
        raise ValueError("at least one prediction scenario is required")
    cases = tuple(
        ExperimentCase(
            method,
            scenario.scenario_id,
            scenario.profile_id,
            scenario.configuration,
        )
        for scenario in scenarios
        for method in ("nn", "average")
    )
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    if len(by_id) != len(scenarios):
        raise ValueError("prediction scenario IDs must be unique")

    def evaluate(case: ExperimentCase, seed: int) -> ExperimentOutcome:
        result = by_id[case.scenario_id].run(case.method, seed)
        status = {
            PlannerStatus.FEASIBLE: ExperimentRunStatus.FEASIBLE,
            PlannerStatus.INFEASIBLE: ExperimentRunStatus.INFEASIBLE,
            PlannerStatus.INCOMPLETE: ExperimentRunStatus.INCOMPLETE,
            PlannerStatus.ERROR: ExperimentRunStatus.ERROR,
        }[result.status]
        stats = result.search_statistics
        assignment_metrics: dict[str, float | int | None] = {}
        target_satisfying = 0
        for mark in result.predicted_marks:
            identity = _metric_identity(mark.assignment_identity)
            met_target = mark.predicted_mark >= mark.target_mark
            target_satisfying += int(met_target)
            assignment_metrics[f"predicted_mark_{identity}"] = mark.predicted_mark
            assignment_metrics[f"target_mark_{identity}"] = mark.target_mark
            assignment_metrics[f"target_met_{identity}"] = int(met_target)
        return ExperimentOutcome(
            status,
            {
                "solution_cost": result.solution_cost,
                "session_count": (
                    sum(sum(item.weekly_sessions) for item in result.allocations)
                    if result.allocations
                    else None
                ),
                "target_satisfying_assignments": target_satisfying
                if result.predicted_marks
                else None,
                "all_targets_met": int(
                    bool(result.predicted_marks)
                    and target_satisfying == len(result.predicted_marks)
                )
                if result.predicted_marks
                else None,
                "mean_predicted_mark": (
                    sum(mark.predicted_mark for mark in result.predicted_marks)
                    / len(result.predicted_marks)
                    if result.predicted_marks
                    else None
                ),
                "mean_target_mark": (
                    sum(mark.target_mark for mark in result.predicted_marks)
                    / len(result.predicted_marks)
                    if result.predicted_marks
                    else None
                ),
                "predicted_assignments": len(result.predicted_marks),
                "nodes_expanded": stats.search.nodes_expanded,
                "nodes_generated": stats.search.nodes_generated,
                "candidate_proposals": stats.candidate_proposals,
                "schedules_examined": stats.schedules_examined,
                "csp_nodes_visited": stats.csp.nodes_visited,
                "csp_constraint_checks": stats.csp.constraint_checks,
                "elapsed_seconds": stats.orchestration_time_seconds,
                **assignment_metrics,
            },
            {
                "model_identity": result.nn_model_identity,
                "model_mode": result.nn_model_mode.value,
                "artifact_digest": result.nn_fitted_artifact_digest,
                "feature_schema_identity": result.nn_feature_schema_identity,
                "prediction_family": result.prediction_family,
                "planner_outcome": result.outcome.value,
                "search_complete": result.search_complete,
                "failure_reasons": result.failure_reasons,
            },
            result.failure_reasons,
        )

    return run_experiment(experiment_id, cases, evaluate, seeds=seeds)


def write_experiment_plot(
    records: Sequence[ExperimentRunRecord],
    output_dir: Path,
    experiment_id: str,
    metric: str,
    *,
    group_by_scenario: bool = True,
) -> Path:
    """Write a dependency-free SVG mean comparison from run records."""
    if not records:
        raise ValueError("cannot plot empty experiment results")
    groups: dict[tuple[str, str], list[float]] = {}
    for row in records:
        value = row.metrics.get(metric)
        if value is not None:
            group = row.scenario_id if group_by_scenario else "all scenarios"
            groups.setdefault((group, row.method), []).append(float(value))
    if not groups:
        raise ValueError(f"metric {metric!r} has no defined values")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{experiment_id}_comparison.svg"
    width, height = 720, 420
    margin = 64
    maximum = max(max(values) for values in groups.values()) or 1.0
    bar_width = min(90, (width - 2 * margin) // (len(groups) * 2))
    gap = (width - 2 * margin - bar_width * len(groups)) / (len(groups) + 1)
    colors = ("#3568a8", "#e18b36", "#48966b")
    elements = [
        _svg(
            "svg",
            {
                "xmlns": "http://www.w3.org/2000/svg",
                "width": width,
                "height": height,
                "viewBox": f"0 0 {width} {height}",
            },
            close=False,
        ),
        _svg("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        _svg(
            "text",
            {
                "x": width / 2,
                "y": 30,
                "text-anchor": "middle",
                "font-family": "sans-serif",
                "font-size": 18,
            },
            f"{experiment_id}: mean {metric}",
        ),
        _svg(
            "line",
            {
                "x1": margin,
                "y1": height - margin,
                "x2": width - margin,
                "y2": height - margin,
                "stroke": "#333",
            },
        ),
    ]
    plot_height = height - 2 * margin
    for index, ((scenario_id, method), values) in enumerate(sorted(groups.items())):
        mean = sum(values) / len(values)
        bar_height = mean / maximum * plot_height
        x = margin + gap * (index + 1) + bar_width * index
        y = height - margin - bar_height
        elements.extend(
            (
                _svg(
                    "rect",
                    {
                        "x": f"{x:.1f}",
                        "y": f"{y:.1f}",
                        "width": bar_width,
                        "height": f"{bar_height:.1f}",
                        "fill": colors[index % len(colors)],
                    },
                ),
                _svg(
                    "text",
                    {
                        "x": f"{x + bar_width / 2:.1f}",
                        "y": height - margin + 22,
                        "text-anchor": "middle",
                        "font-family": "sans-serif",
                        "font-size": 12,
                    },
                    f"{scenario_id} / {method}",
                ),
                _svg(
                    "text",
                    {
                        "x": f"{x + bar_width / 2:.1f}",
                        "y": f"{max(y - 8, 48):.1f}",
                        "text-anchor": "middle",
                        "font-family": "sans-serif",
                        "font-size": 12,
                    },
                    f"{mean:.3g}",
                ),
            )
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")
    return path


def write_experiment_bundle(
    records: Sequence[ExperimentRunRecord],
    output_dir: Path,
    experiment_id: str,
    plot_metric: str,
    *,
    group_by_scenario: bool = True,
) -> tuple[Path, Path, Path, Path]:
    """Write standard CSV/JSON result tables and an SVG comparison plot."""
    run_path, summary_path, json_path = write_experiment_results(
        records, output_dir, experiment_id
    )
    plot_path = write_experiment_plot(
        records,
        output_dir,
        experiment_id,
        plot_metric,
        group_by_scenario=group_by_scenario,
    )
    return run_path, summary_path, json_path, plot_path


def _search_outcome(result: SearchResult[Any, Any]) -> ExperimentOutcome:
    status = (
        ExperimentRunStatus.FEASIBLE
        if result.found
        else ExperimentRunStatus.INFEASIBLE
    )
    return ExperimentOutcome(
        status,
        {
            "nodes_expanded": result.statistics.nodes_expanded,
            "nodes_generated": result.statistics.nodes_generated,
            "solution_cost": result.solution_cost,
            "elapsed_seconds": result.statistics.execution_time_seconds,
            "path_length": len(result.action_path),
            "expanded_order": result.statistics.expanded_order,
        },
    )


def _svg(
    tag: str,
    attributes: Mapping[str, Any],
    text: str | None = None,
    *,
    close: bool = True,
) -> str:
    """Serialize a small SVG element, escaping labels and attribute values."""
    attrs = " ".join(
        f'{name}="{_xml(str(value))}"' for name, value in attributes.items()
    )
    opening = f"<{tag} {attrs}"
    if not close:
        return f"{opening}>"
    if text is None:
        return f"{opening}/>"
    return f"{opening}>{_xml(text)}</{tag}>"


def _xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _metric_identity(value: str) -> str:
    """Make assignment identifiers safe as stable CSV metric suffixes."""
    normalized = "".join(
        character if character.isalnum() else "_" for character in value
    )
    return normalized.strip("_") or "assignment"


__all__ = [
    "CspScenario",
    "PredictionScenario",
    "SearchScenario",
    "run_csp_comparison",
    "run_prediction_comparison",
    "run_search_comparison",
    "write_experiment_bundle",
    "write_experiment_plot",
]
