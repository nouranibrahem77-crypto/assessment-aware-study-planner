"""Focused checks for independent, matched Experiment 3 planner runs."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from scripts import run_demo_experiments as demo

from study_planner.eval.experiment_runner import DEFAULT_EXPERIMENT_SEEDS
from study_planner.eval.experiments import run_prediction_comparison
from study_planner.integration import application_inputs
from study_planner.integration.planner import PlannerOutcome, PlannerStatus


def test_each_predictor_and_seed_executes_integrated_planner(monkeypatch) -> None:
    feature_rows = pd.DataFrame({"id_student": ["student-1"]})
    common = {
        "targets": (60.0, 60.0, 60.0),
        "slots": ("slot-1", "slot-2", "slot-3"),
        "capacity": 2,
        "mapping": ("low", "content_navigation", 22),
    }
    adapters = {
        "nn": SimpleNamespace(prediction_family="neural_network", identity="nn"),
        "average": SimpleNamespace(
            prediction_family="simple_average_mark", identity="average"
        ),
    }
    application_calls = []
    planner_calls = []

    monkeypatch.setattr(
        demo, "_synthetic_config", lambda adapter: {**common, "model": adapter.identity}
    )
    monkeypatch.setattr(
        demo,
        "build_application_inputs",
        lambda config, *, student_identity, feature_rows, adapter: (
            application_calls.append(
                (config, student_identity, feature_rows, adapter.identity)
            )
            or config
        ),
    )
    monkeypatch.setattr(
        application_inputs,
        "build_application_planning_problem",
        lambda inputs, *, adapter, ordering_policy: SimpleNamespace(
            allocation_space=inputs,
            candidate_factory=ordering_policy,
        ),
    )

    def run_planner(allocation_space, candidate_factory, adapter, *, csp_algorithm):
        planner_calls.append(
            (allocation_space, candidate_factory, adapter.identity, csp_algorithm)
        )
        return SimpleNamespace(
            status=PlannerStatus.FEASIBLE,
            outcome=PlannerOutcome.TARGET_ACHIEVED,
            solution_cost=0.0,
            allocations=(),
            predicted_marks=(),
            search_complete=True,
            search_statistics=SimpleNamespace(
                search=SimpleNamespace(nodes_expanded=0, nodes_generated=0),
                candidate_proposals=1,
                schedules_examined=1,
                csp=SimpleNamespace(nodes_visited=1, constraint_checks=1),
                orchestration_time_seconds=0.01,
            ),
            nn_model_identity=adapter.identity,
            nn_model_mode=SimpleNamespace(value="strict"),
            nn_fitted_artifact_digest=adapter.identity,
            nn_feature_schema_identity="schema",
            prediction_family=adapter.prediction_family,
            failure_reasons=(),
        )

    monkeypatch.setattr(demo, "integrated_uniform_cost_search", run_planner)
    planner_results = {}
    scenarios = demo._prediction_scenarios(
        feature_rows, ("student-1",), adapters, planner_results
    )

    records = run_prediction_comparison(scenarios, seeds=DEFAULT_EXPERIMENT_SEEDS)

    assert len(records) == 2 * len(DEFAULT_EXPERIMENT_SEEDS)
    assert len(application_calls) == len(planner_calls) == len(records)
    assert {call[1] for call in application_calls} == {"student-1"}
    assert all(call[2] is feature_rows for call in application_calls)
    assert {call[3] for call in application_calls} == {"nn", "average"}
    assert all(
        {key: value for key, value in config.items() if key != "model"} == common
        for config, _, _, _ in application_calls
    )
    assert all(
        record.provenance["planner_outcome"] == "target_achieved"
        for record in records
    )
