"""Focused coverage for explicitly TEST-configured demo scenarios."""

from __future__ import annotations

import pytest
from run_demo import run_scenario

from study_planner.integration import PlannerOutcome, PlannerStatus
from study_planner.search.nn_adapter import (
    EvaluationStatus,
    NNModelMode,
    SessionNNAdapter,
)


def test_normal_demo_returns_feasible_three_assignment_plan(capsys) -> None:
    result = run_scenario("NORMAL")

    output = capsys.readouterr().out
    assert result.status is PlannerStatus.FEASIBLE
    assert len(result.allocations) == len(result.weekly_timetable) == 3
    assert len(result.predicted_marks) == 3
    assert "=== NORMAL (TEST CONFIGURATION) ===" in output
    assert "Timetable:" in output
    assert "Search:" in output and "CSP:" in output
    assert len(result.search_state_path) == len(result.search_action_path) + 1
    assert "Search solution path (actual SearchResult path):" in output
    assert "Search expansion trace: not available" in output


def test_edge_demo_returns_valid_zero_session_plan(capsys) -> None:
    result = run_scenario("EDGE")

    output = capsys.readouterr().out
    assert result.status is PlannerStatus.FEASIBLE
    assert all(not any(item.weekly_sessions) for item in result.allocations)
    assert "=== EDGE (TEST CONFIGURATION) ===" in output


def test_impossible_demo_reports_best_feasible_plan_after_exhaustion(capsys) -> None:
    result = run_scenario("IMPOSSIBLE")

    output = capsys.readouterr().out
    assert result.status is PlannerStatus.INFEASIBLE
    assert result.outcome.value == "target_unreachable"
    assert result.search_complete
    assert len(result.allocations) == 3
    assert result.weekly_timetable is not None
    assert len(result.predicted_marks) == 3
    assert all(
        mark.predicted_mark < mark.target_mark for mark in result.predicted_marks
    )
    assert result.search_statistics.schedules_examined > 0
    assert (
        result.search_statistics.final_predictions
        == 3 * result.search_statistics.schedules_examined
    )
    assert "Status: infeasible" in output
    assert "Outcome: target_unreachable" in output
    assert "Target marks:" in output
    assert "Best verified achievable marks" in output
    assert "Target shortfall:" in output
    assert "Search complete: True" in output
    assert "Timetable:" in output
    assert "CSP: schedules=" in output
    assert result.search_statistics.verification_rejections == 0
    assert (
        result.search_statistics.terminal_csp_outcomes[-1].status.value
        == "exhausted_with_feasible"
    )


@pytest.mark.parametrize("model", ["strict", "proxy"])
def test_nn_accepts_target_but_real_csp_proves_slot_conflict_infeasible(
    capsys, monkeypatch, model: str
) -> None:
    provisional_evaluations = []
    real_evaluate_provisional = SessionNNAdapter.evaluate_provisional

    def record_provisional(adapter, allocation, context):
        evaluation = real_evaluate_provisional(adapter, allocation, context)
        provisional_evaluations.append((context.assignment_identity, evaluation))
        return evaluation

    monkeypatch.setattr(
        SessionNNAdapter, "evaluate_provisional", record_provisional
    )
    result = run_scenario("NN_ACCEPTS_CSP_INFEASIBLE", model)

    output = capsys.readouterr().out
    expected_mode = (
        NNModelMode.MODEL_A_STRICT
        if model == "strict"
        else NNModelMode.MODEL_B_PREVIOUS_MARKS
    )
    assert result.nn_model_mode is expected_mode
    assert result.status is PlannerStatus.INFEASIBLE
    assert result.outcome is PlannerOutcome.NO_FEASIBLE_SCHEDULE
    assert result.search_complete
    assert result.search_statistics.provisional_predictions == 3
    assert result.search_statistics.schedules_examined == 0
    assert result.search_statistics.verification_rejections == 0
    outcomes = result.search_statistics.terminal_csp_outcomes
    assert len(outcomes) == 1
    assert all(item.status.value == "exhausted_no_feasible" for item in outcomes)
    assert result.weekly_timetable is None
    assert result.whole_plan_verification is None
    assert result.predicted_marks == ()
    assert len(provisional_evaluations) == 6
    targets = {
        target.assignment_identity: target.target_mark for target in result.targets
    }
    assert all(target > 0 for target in targets.values())
    assert all(
        evaluation.status is EvaluationStatus.PREDICTED
        and evaluation.prediction is not None
        and evaluation.prediction >= targets[assignment_identity]
        for assignment_identity, evaluation in provisional_evaluations
    )
    assert result.search_statistics.csp.constraint_checks > 0
    assert result.search_statistics.csp.domain_prunings > 0
    assert result.search_statistics.csp.backtracks > 0
    assert "NN provisional target: ACCEPTED" in output
    assert "targets=(70.0, 70.0, 70.0)" in output
    assert "CSP infeasibility setup" in output
    assert "CSP proposal allocations:" in output
    assert "weekly_slot_unique" in output
    assert "reuse=True" in output
    assert "CSP result: exhausted with no feasible schedule" in output
    assert "verification rejections=0" in output
    assert "Search then exhausted" in output


def test_verification_fault_injection_remains_distinct_and_continues(capsys) -> None:
    result = run_scenario("VERIFICATION_REJECTED")

    output = capsys.readouterr().out
    assert result.status is PlannerStatus.FEASIBLE
    assert result.search_statistics.provisional_predictions >= 3
    assert result.search_statistics.verification_rejections == 1
    assert result.search_statistics.schedules_examined >= 2
    assert result.predicted_marks
    assert "assignment_contains_out_of_domain_values" in result.failure_reasons
    assert "NN provisional target: ACCEPTED" in output
    assert "TEST-injected first verification rejection" in output
    assert "VERIFICATION_REJECTED:" in output
    assert "CSP rejection" not in output
    assert "assignment_contains_out_of_domain_values" in output
    assert output.index("NN provisional target: ACCEPTED") < output.index(
        "TEST-injected first verification rejection"
    )


def test_old_csp_rejection_scenario_name_is_no_longer_exposed():
    with pytest.raises(ValueError, match="unknown scenario"):
        run_scenario("NN_SUFFICIENT_CSP_REJECTED")
