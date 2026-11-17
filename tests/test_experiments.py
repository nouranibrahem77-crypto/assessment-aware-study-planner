"""Tests for the three seeded experiment entry points."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from scripts import run_demo_experiments

from study_planner.csp.ac3 import ac3_propagate
from study_planner.csp.constraints import (
    SameAssignmentOrderingConstraint,
    build_required_constraints,
)
from study_planner.csp.core import (
    AssignmentCandidateIdentity,
    AssignmentEntry,
    AssignmentPlan,
    CspAssignment,
    CspEvaluationStatus,
    CspProblem,
    WholePlanManifest,
    evaluate_assignment,
)
from study_planner.csp.verification import WholePlanVerificationPolicy
from study_planner.eval.experiment_runner import (
    DEFAULT_EXPERIMENT_SEEDS,
    ExperimentRunRecord,
    ExperimentRunStatus,
)
from study_planner.eval.experiments import (
    CspScenario,
    PredictionScenario,
    SearchScenario,
    run_csp_comparison,
    run_prediction_comparison,
    run_search_comparison,
    write_experiment_bundle,
)
from study_planner.integration.planner import PlannerStatus


class _NoOrdering:
    policy_identity = "no-session-order-v1"

    def required_pairs(self, sessions):
        del sessions
        return ()

    def is_ordered(self, earlier, later):
        del earlier, later
        return True


@dataclass(frozen=True)
class _SearchProblem:
    initial_state: str = "start"

    def is_goal(self, state: str) -> bool:
        return state == "goal"

    def actions(self, state: str) -> tuple[str, ...]:
        return {"start": ("detour", "goal"), "detour": ("goal",)}.get(state, ())

    def transition(self, state: str, action: str) -> str:
        del state
        return action

    def action_cost(self, state: str, action: str, next_state: str) -> float:
        del state, next_state
        return 5.0 if action == "goal" else 1.0


def test_search_comparison_runs_both_methods_for_every_seed() -> None:
    records = run_search_comparison(
        (
            SearchScenario(
                "tiny",
                _SearchProblem(),
                lambda state: 0.0 if state == "goal" else 1.0,
                {"fixture": "tiny"},
                lambda state: 0.0 if state == "goal" else 1.0,
            ),
        )
    )
    assert len(records) == 3 * len(DEFAULT_EXPERIMENT_SEEDS)
    assert {record.method for record in records} == {
        "ucs",
        "greedy_best_first",
        "a_star",
    }
    assert {record.seed for record in records} == set(DEFAULT_EXPERIMENT_SEEDS)
    assert all(record.metrics["solution_cost"] is not None for record in records)


def test_prediction_comparison_retains_paired_profile_runs() -> None:
    calls: list[tuple[str, int]] = []

    def planner_run(method: str, seed: int) -> SimpleNamespace:
        calls.append((method, seed))
        return SimpleNamespace(
            status=PlannerStatus.FEASIBLE,
            outcome=SimpleNamespace(value="target_achieved"),
            solution_cost=3.0 if method == "nn" else 4.0,
            search_complete=True,
            allocations=(SimpleNamespace(weekly_sessions=(1, 0, 0, 0)),),
            predicted_marks=(
                SimpleNamespace(
                    assignment_identity="a1",
                    predicted_mark=80.0,
                    target_mark=70.0,
                ),
                SimpleNamespace(
                    assignment_identity="a2",
                    predicted_mark=65.0,
                    target_mark=70.0,
                ),
                SimpleNamespace(
                    assignment_identity="a3",
                    predicted_mark=75.0,
                    target_mark=70.0,
                ),
            ),
            search_statistics=SimpleNamespace(
                search=SimpleNamespace(nodes_expanded=seed % 5, nodes_generated=4),
                candidate_proposals=2,
                schedules_examined=3,
                csp=SimpleNamespace(nodes_visited=5, constraint_checks=6),
                orchestration_time_seconds=0.1,
            ),
            nn_model_identity=method,
            nn_model_mode=SimpleNamespace(value="strict"),
            nn_fitted_artifact_digest=f"digest-{method}",
            nn_feature_schema_identity="strict-v1",
            prediction_family="neural_network" if method == "nn" else "average",
            failure_reasons=(),
        )

    records = run_prediction_comparison(
        (PredictionScenario("profile-1", "student-1", {"case": 1}, planner_run),)
    )
    assert len(records) == 2 * len(DEFAULT_EXPERIMENT_SEEDS)
    assert {record.profile_id for record in records} == {"student-1"}
    assert {record.method for record in records} == {"nn", "average"}
    assert set(calls) == {
        (method, seed)
        for method in ("nn", "average")
        for seed in DEFAULT_EXPERIMENT_SEEDS
    }
    assert len(calls) == 2 * len(DEFAULT_EXPERIMENT_SEEDS)
    assert {record.experiment_id for record in records} == {"prediction"}
    assert all(record.metrics["all_targets_met"] == 0 for record in records)
    assert all(
        record.metrics["target_satisfying_assignments"] == 2 for record in records
    )
    assert all(record.metrics["predicted_mark_a1"] == 80.0 for record in records)
    assert all(
        record.metrics["mean_predicted_mark"] == 220.0 / 3 for record in records
    )


def test_csp_comparison_runs_all_strategies_for_each_seed() -> None:
    assignments = tuple(
        AssignmentPlan(
            assignment_id,
            AssignmentCandidateIdentity(
                "student-1",
                "course-1",
                assignment_id,
                (0, 0, 0, 0),
                f"features-{assignment_id}",
                "mapping-v1",
            ),
            f"features-{assignment_id}",
        )
        for assignment_id in ("a1", "a2", "a3")
    )
    manifest = WholePlanManifest(
        "student-1", "course-1", assignments, "mapping-v1"
    )
    constraints = build_required_constraints(
        manifest=manifest,
        variables=(),
        eligible_slots=(),
        ordering_policy=_NoOrdering(),
        weekly_capacity=3,
    )
    problem = CspProblem(manifest, (), (), constraints, 3, _NoOrdering())
    policy = WholePlanVerificationPolicy(
        "policy-v1",
        "deadline-v1",
        problem.ordering_policy.configuration_identity,
        "slots-v1",
        "capacity-v1",
        "slot-v1",
        3,
    )
    records = run_csp_comparison(
        (CspScenario("empty-plan", problem, policy, {"capacity": 3}),)
    )
    assert len(records) == 3 * len(DEFAULT_EXPERIMENT_SEEDS)
    assert {record.method for record in records} == {"bt", "bt_fc", "bt_fc_ac3"}
    assert all(record.metrics.get("exhaustive") == 1 for record in records), records


def test_experiment2_scenarios_have_exact_physical_slots_and_feasible_plans() -> None:
    scenarios = run_demo_experiments._csp_scenarios()
    assert len(scenarios) == 3

    for slots_per_week, scenario in zip((2, 3, 4), scenarios, strict=True):
        problem = scenario.problem
        physical_slots = {
            slot for domain in problem.domains for slot in domain.values
        }
        assert len(problem.variables) == 4
        assert {
            week: sum(slot.week_index == week for slot in physical_slots)
            for week in (0, 1)
        } == {0: slots_per_week, 1: slots_per_week}
        assert problem.weekly_capacity == slots_per_week
        assert scenario.configuration["physical_slots_per_week"] == slots_per_week
        assert any(
            isinstance(constraint, SameAssignmentOrderingConstraint)
            for constraint in problem.constraints
        )

        domains = {domain.variable: domain.values for domain in problem.domains}
        ordered = sorted(
            variable
            for variable in problem.variables
            if variable.assignment_id == "a_34899"
        )
        others = sorted(set(problem.variables) - set(ordered))
        def slot_at(variable, week, index, domain_map=domains):
            return next(
                slot
                for slot in domain_map[variable]
                if slot.week_index == week
                and slot.slot_id.endswith(f"slot-{index + 1}")
            )
        if slots_per_week == 2:
            placements = (
                (ordered[0], 0, 0),
                (ordered[1], 1, 0),
                (others[0], 0, 1),
                (others[1], 1, 1),
            )
        else:
            placements = (
                (ordered[0], 0, 0),
                (ordered[1], 1, 0),
                (others[0], 0, 1),
                (others[1], 0, 2),
            )
        feasible_assignment = CspAssignment(
            tuple(
                AssignmentEntry(variable, slot_at(variable, week, index))
                for variable, week, index in placements
            )
        )
        assert (
            evaluate_assignment(problem, feasible_assignment).status
            is CspEvaluationStatus.FEASIBLE
        )

        arc_result = ac3_propagate(problem)
        assert arc_result.consistent
        assert arc_result.domain_prunings > 0
        propagated = {domain.variable: domain.values for domain in arc_result.domains}
        assert {slot.week_index for slot in propagated[ordered[0]]} == {0}
        assert {slot.week_index for slot in propagated[ordered[1]]} == {1}


def test_csp_only_command_updates_only_experiment2_report_rows(
    tmp_path, monkeypatch
) -> None:
    report_path = tmp_path / "experiment_report.md"
    report_path.write_text(
        "| search | ucs | 5 | 5 | 0 | 0 | 0 |\n"
        "| csp | bt | 15 | 10 | 5 | 0 | 0 |\n"
        "| csp | bt_fc | 15 | 10 | 5 | 0 | 0 |\n"
        "| csp | bt_fc_ac3 | 15 | 10 | 5 | 0 | 0 |\n"
        "| prediction | nn | 150 | 150 | 0 | 0 | 0 |\n",
        encoding="utf-8",
    )
    csp_records = tuple(
        ExperimentRunRecord(
            "csp",
            method,
            "slots-per-week-2",
            None,
            seed,
            ExperimentRunStatus.FEASIBLE,
            {},
            {},
            {},
        )
        for method in ("bt", "bt_fc", "bt_fc_ac3")
        for seed in DEFAULT_EXPERIMENT_SEEDS
    )
    monkeypatch.setattr(
        run_demo_experiments, "run_csp_comparison", lambda _scenarios: csp_records
    )
    monkeypatch.setattr(
        run_demo_experiments,
        "write_experiment_bundle",
        lambda *_args, **_kwargs: (),
    )

    assert run_demo_experiments.main(
        ["--csp-only", "--output-directory", str(tmp_path)]
    ) == 0
    lines = report_path.read_text(encoding="utf-8").splitlines()
    assert "| search | ucs | 5 | 5 | 0 | 0 | 0 |" in lines
    assert "| prediction | nn | 150 | 150 | 0 | 0 | 0 |" in lines
    assert all(
        f"| csp | {method} | 5 | 5 | 0 | 0 | 0 |" in lines
        for method in ("bt", "bt_fc", "bt_fc_ac3")
    )


def test_experiment_bundle_writes_tables_and_plot(tmp_path) -> None:
    records = run_search_comparison(
        (
            SearchScenario(
                "tiny",
                _SearchProblem(),
                lambda _state: 0.0,
                {},
                lambda _state: 0.0,
            ),
        ),
        seeds=(42,),
    )
    paths = write_experiment_bundle(
        records, tmp_path, "search", "nodes_expanded"
    )
    assert [path.suffix for path in paths] == [".csv", ".csv", ".json", ".svg"]
    assert all(path.is_file() for path in paths)

