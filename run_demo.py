"""Run explicitly TEST-configured Search–NN–CSP integration scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from study_planner.csp import (
    AssignmentEntry,
    CspAssignment,
    SessionPair,
    SessionVariable,
    WeeklySlot,
)
from study_planner.integration import (
    AllocationBounds,
    ApplicationAssignmentRule,
    ApplicationInputConfig,
    ApplicationPlanningBinding,
    ApplicationSlot,
    CspAlgorithm,
    FinalPlannerResult,
    ModelArtifactIdentity,
    OuladPresentationSplit,
    OutOfSupportAction,
    SupportPolicy,
    TargetRule,
    ValidatedApplicationInputs,
    WeeklySlotCatalog,
    WeeklyTimeWindow,
    build_application_planning_problem,
    integrated_uniform_cost_search,
    load_application_inputs,
    load_session_nn_adapter,
)
from study_planner.search.nn_adapter import (
    CandidateSessionAllocation,
    NNModelMode,
    SessionNNAdapter,
)
from study_planner.search.session_mapping import (
    ClickScenario,
    SessionClickMappingConfig,
)

ROOT = Path(__file__).resolve().parent
ASSIGNMENTS = ("TEST-a1", "TEST-a2", "TEST-a3")
ASSESSMENTS = ("34899", "34900", "34901")
DEADLINES = (24, 52, 94)
Scenario = Literal["NORMAL", "EDGE", "IMPOSSIBLE", "NN_SUFFICIENT_CSP_REJECTED"]
SCENARIOS: tuple[Scenario, ...] = (
    "NORMAL",
    "EDGE",
    "IMPOSSIBLE",
    "NN_SUFFICIENT_CSP_REJECTED",
)


class _TestOrdering:
    """Explicit test-only relation ordering same-assignment sessions by week."""

    policy_identity = "TEST-week-order-v1"

    def required_pairs(
        self, sessions: tuple[SessionVariable, ...]
    ) -> tuple[SessionPair, ...]:
        grouped: dict[str, list[SessionVariable]] = {}
        for session in sessions:
            grouped.setdefault(session.assignment_id, []).append(session)
        return tuple(
            SessionPair(first, second)
            for group in grouped.values()
            for first, second in zip(group, group[1:], strict=False)
        )

    def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
        return earlier.week_index < later.week_index


def _test_config(
    adapter: SessionNNAdapter, scenario: Scenario
) -> ApplicationInputConfig:
    zero_session = scenario == "EDGE"
    impossible = scenario == "IMPOSSIBLE"
    bounds = AllocationBounds(0, 0) if zero_session else AllocationBounds(1, 1)
    eligible = () if impossible else ("TEST-slot-1", "TEST-slot-2", "TEST-slot-3")
    rules = tuple(
        ApplicationAssignmentRule(
            identity,
            assessment_id,
            0.0,
            deadline,
            eligible,
            bounds,
        )
        for identity, assessment_id, deadline in zip(
            ASSIGNMENTS, ASSESSMENTS, DEADLINES, strict=True
        )
    )
    relative_windows = tuple(
        (identity, index + 1) for index, identity in enumerate(ASSIGNMENTS)
    )
    slots = tuple(
        ApplicationSlot(
            WeeklySlot(f"TEST-slot-{index}", index - 1),
            WeeklyTimeWindow(1, 9 * 60, 10 * 60, "TEST/UTC"),
            relative_windows,
        )
        for index in range(1, 4)
    )
    targets = tuple((identity, 0.0) for identity in ASSIGNMENTS)
    return ApplicationInputConfig(
        OuladPresentationSplit("FFF", ("2013B", "2013J"), "2014J"),
        rules,
        TargetRule("TEST-zero-targets", "v1", targets),
        WeeklySlotCatalog(
            "TEST-slot-catalog",
            "v1",
            2,
            "TEST-deadline-policy",
            "v1",
            slots,
        ),
        SupportPolicy(
            "TEST-marginal-support",
            "v1",
            "strict-training-range",
            OutOfSupportAction.REPORT_ONLY,
        ),
        SessionClickMappingConfig(ClickScenario.LOW),
        ModelArtifactIdentity.from_adapter(adapter),
    )


def _load_binding(
    scenario: Scenario, model: Literal["strict", "proxy"]
) -> tuple[SessionNNAdapter, ValidatedApplicationInputs, ApplicationPlanningBinding]:
    if model == "strict":
        mode = NNModelMode.MODEL_A_STRICT
        model_path = ROOT / "data/processed/nn_experiment_60ep"
        feature_path = ROOT / "data/processed/strict"
    else:
        mode = NNModelMode.MODEL_B_PREVIOUS_MARKS
        model_path = ROOT / "data/processed/nn_experiment_previous_marks"
        feature_path = ROOT / "data/processed/proxy"
    mapping = SessionClickMappingConfig(ClickScenario.LOW)
    adapter = load_session_nn_adapter(model_path, mode, mapping)
    config = _test_config(adapter, scenario)
    inputs = load_application_inputs(
        feature_path, config, student_identity="40878", adapter=adapter
    )
    binding = build_application_planning_problem(
        inputs, adapter=adapter, ordering_policy=_TestOrdering()
    )
    return adapter, inputs, binding


def _print_result(scenario: Scenario, result: FinalPlannerResult) -> None:
    print(f"\n=== {scenario} (TEST CONFIGURATION) ===")
    print(f"Status: {result.status.value}")
    print("Assignments:")
    for assignment in ASSIGNMENTS:
        allocation = next(
            (
                item
                for item in result.allocations
                if item.assignment_identity == assignment
            ),
            None,
        )
        print(
            f"  {assignment}: allocation="
            f"{allocation.weekly_sessions if allocation else 'unproposed'}"
        )
    if result.weekly_timetable is None:
        print("Timetable: unavailable (no verified feasible schedule)")
    else:
        print("Timetable:")
        for assignment_schedule in result.weekly_timetable:
            sessions = tuple(
                (session.slot_id, session.week_index)
                for session in assignment_schedule.sessions
            )
            print(f"  {assignment_schedule.assignment_identity}: {sessions}")
    print("Predicted marks:")
    if result.predicted_marks:
        for mark in result.predicted_marks:
            print(
                f"  {mark.assignment_identity}: {mark.predicted_mark:.2f} / "
                f"target {mark.target_mark:.2f}"
            )
    else:
        print("  none (NN was not run for an infeasible allocation)")
    stats = result.search_statistics
    print(
        "Search: "
        f"generated={stats.search.nodes_generated}, "
        f"expanded={stats.search.nodes_expanded}, "
        f"candidate_proposals={stats.candidate_proposals}, "
        f"provisional_predictions={stats.provisional_predictions}, "
        f"final_predictions={stats.final_predictions}"
    )
    csp = stats.csp
    print(
        "CSP: "
        f"schedules={stats.schedules_examined}, "
        f"rejections={stats.verification_rejections}, "
        f"nodes={csp.nodes_visited}, attempts={csp.assignments_attempted}, "
        f"backtracks={csp.backtracks}, checks={csp.constraint_checks}, "
        f"prunings={csp.domain_prunings}, seconds={csp.wall_time_seconds:.6f}"
    )
    if stats.terminal_csp_outcomes:
        print(f"CSP terminal: {stats.terminal_csp_outcomes[-1].status.value}")
    if result.failure_reasons:
        print(f"Reasons: {', '.join(result.failure_reasons)}")


def run_scenario(
    scenario: Scenario, model: Literal["strict", "proxy"] = "strict"
) -> FinalPlannerResult:
    """Run one scenario using local artifacts and TEST-only policy values."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    adapter, inputs, binding = _load_binding(scenario, model)
    if scenario == "NN_SUFFICIENT_CSP_REJECTED":
        provisional_marks = tuple(
            adapter.evaluate_provisional(
                CandidateSessionAllocation((1, 0, 0, 0)), context
            ).prediction
            for context in inputs.feature_contexts
        )
        if any(mark is None or mark < 0.0 for mark in provisional_marks):
            raise RuntimeError("TEST scenario did not meet its explicit zero targets")
        print(
            "NN provisional target: ACCEPTED "
            f"(predictions={tuple(round(mark, 2) for mark in provisional_marks)}, "
            "targets=(0, 0, 0))"
        )
        # Keep this demonstration deterministic: reject the first schedule at
        # the existing verification boundary, then delegate subsequent ones to
        # the real whole-plan verifier. This is an explicit TEST fault injection.
        import study_planner.integration.schedule_bridge as bridge_module

        real_verify = bridge_module.verify_whole_plan
        rejected_once = False

        def reject_first_schedule(problem, assignment, policy, **kwargs):
            nonlocal rejected_once
            if not rejected_once:
                rejected_once = True
                invalid = CspAssignment(
                    (
                        AssignmentEntry(
                            problem.variables[0], WeeklySlot("TEST-outside", 99)
                        ),
                    )
                )
                return real_verify(problem, invalid, policy, **kwargs)
            return real_verify(problem, assignment, policy, **kwargs)

        bridge_module.verify_whole_plan = reject_first_schedule
        try:
            result = integrated_uniform_cost_search(
                binding.allocation_space,
                binding.candidate_factory,
                adapter,
                csp_algorithm=CspAlgorithm.BT_FC,
            )
        finally:
            bridge_module.verify_whole_plan = real_verify
        if not rejected_once:
            raise RuntimeError("TEST rejection scenario did not reach verification")
        print(
            "NN_SUFFICIENT_CSP_REJECTED: TEST-injected first verification "
            "rejection; planner continued to the next schedule."
        )
    else:
        result = integrated_uniform_cost_search(
            binding.allocation_space,
            binding.candidate_factory,
            adapter,
            csp_algorithm=CspAlgorithm.BT_FC,
        )
    _print_result(scenario, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=SCENARIOS, nargs="?", default="NORMAL")
    parser.add_argument("--model", choices=("strict", "proxy"), default="strict")
    args = parser.parse_args()
    run_scenario(args.scenario, args.model)


if __name__ == "__main__":
    main()
