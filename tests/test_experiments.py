"""Tests for the three seeded experiment entry points."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from study_planner.csp.constraints import build_required_constraints
from study_planner.csp.core import (
    AssignmentCandidateIdentity,
    AssignmentPlan,
    CspProblem,
    WholePlanManifest,
)
from study_planner.csp.verification import WholePlanVerificationPolicy
from study_planner.eval.experiment_runner import DEFAULT_EXPERIMENT_SEEDS
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
            ),
        )
    )
    assert len(records) == 2 * len(DEFAULT_EXPERIMENT_SEEDS)
    assert {record.method for record in records} == {"ucs", "greedy_best_first"}
    assert {record.seed for record in records} == set(DEFAULT_EXPERIMENT_SEEDS)
    assert all(record.metrics["solution_cost"] is not None for record in records)


def test_prediction_comparison_retains_paired_profile_runs() -> None:
    def planner_run(method: str, seed: int) -> SimpleNamespace:
        return SimpleNamespace(
            status=PlannerStatus.FEASIBLE,
            solution_cost=3.0 if method == "nn" else 4.0,
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


def test_experiment_bundle_writes_tables_and_plot(tmp_path) -> None:
    records = run_search_comparison(
        (
            SearchScenario(
                "tiny",
                _SearchProblem(),
                lambda _state: 0.0,
                {},
            ),
        ),
        seeds=(42,),
    )
    paths = write_experiment_bundle(
        records, tmp_path, "search", "nodes_expanded"
    )
    assert [path.suffix for path in paths] == [".csv", ".csv", ".json", ".svg"]
    assert all(path.is_file() for path in paths)

