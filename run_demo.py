"""Run explicitly TEST-configured Search–NN–CSP integration scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from study_planner.csp import (
    AssignmentEntry,
    CspAssignment,
    UniqueSlotConstraint,
    WeeklySlot,
)
from study_planner.integration import (
    AllocationBounds,
    ApplicationAssignmentRule,
    ApplicationInputConfig,
    ApplicationPlanningBinding,
    CatalogChronologicalOrderingPolicy,
    CspAlgorithm,
    FinalPlannerResult,
    ModelArtifactIdentity,
    OuladPresentationSplit,
    OutOfSupportAction,
    SupportPolicy,
    SyntheticWeeklySlotPattern,
    TargetRule,
    ValidatedApplicationInputs,
    build_application_planning_problem,
    build_synthetic_weekly_slot_catalog,
    integrated_uniform_cost_search,
    load_application_inputs,
    load_session_nn_adapter,
)
from study_planner.search.nn_adapter import (
    CandidateSessionAllocation,
    EvaluationStatus,
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
Scenario = Literal[
    "NORMAL",
    "EDGE",
    "IMPOSSIBLE",
    "NN_ACCEPTS_CSP_INFEASIBLE",
    "VERIFICATION_REJECTED",
]
CSP_INFEASIBLE_TEST_TARGET = 70.0
SCENARIOS: tuple[Scenario, ...] = (
    "NORMAL",
    "EDGE",
    "IMPOSSIBLE",
    "NN_ACCEPTS_CSP_INFEASIBLE",
    "VERIFICATION_REJECTED",
)


def _test_config(
    adapter: SessionNNAdapter, scenario: Scenario
) -> ApplicationInputConfig:
    zero_session = scenario == "EDGE"
    impossible = scenario == "IMPOSSIBLE"
    csp_infeasible = scenario == "NN_ACCEPTS_CSP_INFEASIBLE"
    verification_rejected = scenario == "VERIFICATION_REJECTED"
    target_mark = (
        100.0
        if impossible
        else CSP_INFEASIBLE_TEST_TARGET
        if csp_infeasible
        else 0.0
    )
    default_bounds = (
        AllocationBounds(0, 0) if zero_session else AllocationBounds(1, 1)
    )
    catalog, eligible_by_assignment = build_synthetic_weekly_slot_catalog(
        identity="TEST-P1-SYNTH",
        version="test-v1",
        weekly_capacity=3,
        deadline_policy_identity="TEST-course-day-start-exclusive",
        deadline_policy_version="test-v1",
        assignment_deadlines=tuple(
            zip(ASSIGNMENTS, DEADLINES, strict=True)
        ),
        week_indices=(3, 7, 13),
        pattern=SyntheticWeeklySlotPattern(
            (1, 3) if verification_rejected else (3,),
            9 * 60,
            10 * 60,
            "TEST/UTC",
            1,
        ),
    )
    eligible = dict(eligible_by_assignment)
    rules = tuple(
        ApplicationAssignmentRule(
            identity,
            assessment_id,
            target_mark,
            deadline,
            eligible[identity],
            (
                AllocationBounds(2, 2)
                if csp_infeasible and identity == "TEST-a1"
                else AllocationBounds(0, 0)
                if csp_infeasible
                else default_bounds
            ),
        )
        for identity, assessment_id, deadline in zip(
            ASSIGNMENTS, ASSESSMENTS, DEADLINES, strict=True
        )
    )
    targets = tuple((identity, target_mark) for identity in ASSIGNMENTS)
    return ApplicationInputConfig(
        OuladPresentationSplit("FFF", ("2013B", "2013J"), "2014J"),
        rules,
        TargetRule(f"TEST-targets-{scenario.lower()}", "v1", targets),
        catalog,
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
        inputs,
        adapter=adapter,
        ordering_policy=CatalogChronologicalOrderingPolicy(inputs.config.slot_catalog),
    )
    return adapter, inputs, binding


def _print_result(scenario: Scenario, result: FinalPlannerResult) -> None:
    print(f"\n=== {scenario} (TEST CONFIGURATION) ===")
    print(f"Status: {result.status.value}")
    print(f"Outcome: {result.outcome.value}")
    if result.targets:
        print(
            "Target marks: "
            + ", ".join(
                f"{target.assignment_identity}={target.target_mark:.2f}"
                for target in result.targets
            )
        )
    if result.outcome.value == "target_achieved":
        print("Result: target achieved by the verified schedule.")
    elif result.outcome.value == "target_unreachable":
        if result.search_complete:
            print(
                "Result: target unreachable at configured capacity "
                "(exhaustive search)."
            )
        else:
            print(
                "Result: target not achieved; search completeness is not "
                "established."
            )
    elif result.outcome.value == "incomplete":
        print("Result: search incomplete; target reachability is unknown.")
    elif result.outcome.value == "no_feasible_schedule":
        print("Result: no verified feasible schedule was found.")
    if result.predicted_marks:
        label = (
            "Best verified marks so far"
            if result.outcome.value == "incomplete"
            else "Best verified achievable marks"
            if result.outcome.value == "target_unreachable"
            else "Final predicted marks"
        )
        print(f"{label} (comparison: {result.best_plan_comparison_rule}).")
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
        if result.outcome.value in {"target_unreachable", "incomplete"}:
            print("Target shortfall:")
            for mark in result.predicted_marks:
                shortfall = max(0.0, mark.target_mark - mark.predicted_mark)
                print(f"  {mark.assignment_identity}: {shortfall:.2f}")
    else:
        print("  none (no final predictions from a verified feasible schedule)")
    stats = result.search_statistics
    print(f"Search complete: {result.search_complete}")
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
    if result.search_state_path:
        print("Search solution path (actual SearchResult path):")
        for state in result.search_state_path:
            allocations = getattr(state, "allocations", state)
            print(f"  state: {allocations}")
        print(
            "  actions: "
            + repr(
                tuple(
                    (
                        getattr(action, "assignment_identity", None),
                        getattr(action, "relative_window", None),
                    )
                    for action in result.search_action_path
                )
            )
        )
    else:
        print(
            "Search solution path: none (SearchResult returns no path after "
            "exhaustion)."
        )
    print("Search expansion trace: not available; Search counters above are aggregate.")
    if result.failure_reasons:
        print(f"Reasons: {', '.join(result.failure_reasons)}")


def run_scenario(
    scenario: Scenario, model: Literal["strict", "proxy"] = "strict"
) -> FinalPlannerResult:
    """Run one scenario using local artifacts and TEST-only policy values."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    adapter, inputs, binding = _load_binding(scenario, model)
    if scenario == "NN_ACCEPTS_CSP_INFEASIBLE":
        state = binding.allocation_space.initial_state
        for _ in range(2):
            action = next(
                action
                for action in binding.allocation_space.actions(state)
                if action.assignment_identity == "TEST-a1"
            )
            state = binding.allocation_space.transition(state, action)
        candidate = binding.candidate_factory(state)
        allocations = {
            plan.assignment_identity:
                plan.candidate_identity.canonical_allocation_identity
            for plan in candidate.problem.manifest.canonical_assignments
        }
        provisional = tuple(
            adapter.evaluate_provisional(
                CandidateSessionAllocation(
                    allocations[context.assignment_identity]
                ),
                context,
            )
            for context in inputs.feature_contexts
        )
        targets = {
            target.assignment_identity: target.target_mark
            for target in candidate.targets
        }
        if any(
            result.status is not EvaluationStatus.PREDICTED
            or result.prediction is None
            or result.prediction < targets[context.assignment_identity]
            for context, result in zip(
                inputs.feature_contexts, provisional, strict=True
            )
        ):
            raise RuntimeError("TEST NN provisional target was not accepted")
        requested_allocations = tuple(
            (
                plan.assignment_identity,
                plan.candidate_identity.canonical_allocation_identity,
            )
            for plan in candidate.problem.manifest.canonical_assignments
        )
        print(f"CSP proposal allocations: {requested_allocations}")
        provisional_targets = tuple(
            targets[item.assignment_identity] for item in inputs.feature_contexts
        )
        print(
            "NN provisional target: ACCEPTED "
            f"(predictions={tuple(round(item.prediction, 2) for item in provisional)}, "
            f"targets={provisional_targets})"
        )
        a1_domains = tuple(
            domain
            for domain in candidate.problem.domains
            if domain.variable.assignment_id == "TEST-a1"
        )
        unique_constraints = tuple(
            constraint
            for constraint in candidate.problem.constraints
            if isinstance(constraint, UniqueSlotConstraint)
            and all(item.assignment_id == "TEST-a1" for item in constraint.scope)
        )
        if len(a1_domains) != 2 or not unique_constraints:
            raise RuntimeError("TEST CSP conflict was not constructed as expected")
        distinct_slots = {
            slot.slot_id for slot in a1_domains[0].values
        }
        if (
            len(distinct_slots) != 1
            or len(a1_domains[1].values) != 1
            or a1_domains[0].values[0] != a1_domains[1].values[0]
        ):
            raise RuntimeError("TEST CSP conflict must have exactly one eligible slot")
        only_slot = a1_domains[0].values[0]
        conflict = unique_constraints[0]
        rejects_reuse = not conflict.is_compatible(
            a1_domains[0].variable,
            only_slot,
            a1_domains[1].variable,
            only_slot,
        )
        print(
            "CSP infeasibility setup: TEST-a1 requests 2 sessions, but its "
            f"eligible domain has {len(distinct_slots)} distinct slot; "
            f"{conflict.constraint_id} rejects reuse={rejects_reuse}."
        )
        result = integrated_uniform_cost_search(
            binding.allocation_space,
            binding.candidate_factory,
            adapter,
            csp_algorithm=CspAlgorithm.BT_FC,
        )
        if (
            result.outcome.value != "no_feasible_schedule"
            or result.status.value != "infeasible"
            or len(result.search_statistics.terminal_csp_outcomes) != 1
            or any(
                outcome.status.value != "exhausted_no_feasible"
                for outcome in result.search_statistics.terminal_csp_outcomes
            )
            or result.search_statistics.verification_rejections
            or result.search_statistics.schedules_examined
            or result.weekly_timetable is not None
            or result.whole_plan_verification is not None
            or result.predicted_marks
            or not result.search_complete
        ):
            raise RuntimeError(
                "real CSP did not report one exhaustive no-feasible outcome"
            )
        print(
            "CSP result: exhausted with no feasible schedule; verification "
            "rejections=0. Search then exhausted the remaining TEST allocation graph."
        )
    elif scenario == "VERIFICATION_REJECTED":
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
        # Explicitly named TEST fault injection at the verification boundary;
        # this is not a CSP constraint rejection.
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
            "VERIFICATION_REJECTED: TEST-injected first verification "
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
