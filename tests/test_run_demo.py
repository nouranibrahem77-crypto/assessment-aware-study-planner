"""Focused coverage for the four explicitly TEST-configured demo scenarios."""

from __future__ import annotations

from run_demo import run_scenario

from study_planner.integration import PlannerStatus


def test_normal_demo_returns_feasible_three_assignment_plan(capsys) -> None:
    result = run_scenario("NORMAL")

    output = capsys.readouterr().out
    assert result.status is PlannerStatus.FEASIBLE
    assert len(result.allocations) == len(result.weekly_timetable) == 3
    assert len(result.predicted_marks) == 3
    assert "=== NORMAL (TEST CONFIGURATION) ===" in output
    assert "Timetable:" in output
    assert "Search:" in output and "CSP:" in output


def test_edge_demo_returns_valid_zero_session_plan(capsys) -> None:
    result = run_scenario("EDGE")

    output = capsys.readouterr().out
    assert result.status is PlannerStatus.FEASIBLE
    assert all(not any(item.weekly_sessions) for item in result.allocations)
    assert "=== EDGE (TEST CONFIGURATION) ===" in output


def test_impossible_demo_reports_infeasibility_without_predictions(capsys) -> None:
    result = run_scenario("IMPOSSIBLE")

    output = capsys.readouterr().out
    assert result.status is PlannerStatus.INFEASIBLE
    assert result.predicted_marks == ()
    assert "Status: infeasible" in output
    assert "none (NN was not run" in output


def test_nn_sufficient_csp_rejection_is_recorded_then_planner_continues(
    capsys,
) -> None:
    result = run_scenario("NN_SUFFICIENT_CSP_REJECTED")

    output = capsys.readouterr().out
    assert result.status is PlannerStatus.FEASIBLE
    assert result.search_statistics.provisional_predictions >= 3
    assert result.search_statistics.verification_rejections == 1
    assert result.search_statistics.schedules_examined >= 2
    assert result.predicted_marks
    assert "assignment_contains_out_of_domain_values" in result.failure_reasons
    assert "NN provisional target: ACCEPTED" in output
    assert "TEST-injected first verification rejection" in output
    assert "assignment_contains_out_of_domain_values" in output
    assert output.index("NN provisional target: ACCEPTED") < output.index(
        "TEST-injected first verification rejection"
    )
