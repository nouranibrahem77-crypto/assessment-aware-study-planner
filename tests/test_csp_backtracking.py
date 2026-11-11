"""Deterministic regression tests for baseline CSP Backtracking Search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import pytest

from study_planner.csp import (
    AssignmentEntry,
    AssignmentPlan,
    BacktrackingEnumerationStatus,
    BacktrackingOptions,
    BacktrackingTerminationReason,
    BacktrackingValueOrder,
    BacktrackingVariableOrder,
    CspAssignment,
    CspEvaluationStatus,
    CspInputError,
    CspSolverStatus,
    SessionPair,
    SessionVariable,
    WeeklySlot,
    WholePlanManifest,
    backtracking_enumerator,
    backtracking_search,
    build_required_constraints,
    evaluate_assignment,
    expected_session_variables,
)
from study_planner.csp.constraints import (
    DeadlineConstraint,
    SameAssignmentOrderingConstraint,
    UniqueSlotConstraint,
    WeeklyCapacityConstraint,
)
from study_planner.csp.core import (
    AssignmentCandidateIdentity,
    CspProblem,
    VariableDomain,
)


@dataclass(frozen=True)
class _IncreasingWeeksPolicy:
    policy_identity: str = "backtracking-test-increasing-weeks-v1"

    def required_pairs(
        self, sessions: tuple[SessionVariable, ...]
    ) -> tuple[SessionPair, ...]:
        by_assignment: dict[str, list[SessionVariable]] = {}
        for session in sessions:
            by_assignment.setdefault(session.assignment_id, []).append(session)
        return tuple(
            SessionPair(first, second)
            for assignment_sessions in by_assignment.values()
            for first, second in zip(
                sorted(assignment_sessions),
                sorted(assignment_sessions)[1:],
                strict=False,
            )
        )

    def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
        return earlier.week_index < later.week_index


def _make_manifest() -> WholePlanManifest:
    assignments = tuple(
        AssignmentPlan(
            assignment_id,
            AssignmentCandidateIdentity(
                "student-1",
                "course-1",
                assignment_id,
                allocation,
                f"features-{assignment_id}",
                "central",
            ),
            f"features-{assignment_id}",
        )
        for assignment_id, allocation in (
            ("a1", (1, 1, 0, 0)),
            ("a2", (0, 0, 0, 0)),
            ("a3", (1, 0, 0, 0)),
        )
    )
    return WholePlanManifest("student-1", "course-1", assignments, "central")


@pytest.fixture
def problem_factory() -> Callable[..., CspProblem]:
    """Build small deterministic CSP fixtures for BT behavior tests."""

    def build(
        *,
        manifest: WholePlanManifest | None = None,
        domains_override: tuple[VariableDomain, ...] | None = None,
        capacity: int = 2,
    ) -> CspProblem:
        selected_manifest = manifest or _make_manifest()
        ordering = _IncreasingWeeksPolicy()
        variables = expected_session_variables(selected_manifest)
        catalog = (
            WeeklySlot("w0-a", 0),
            WeeklySlot("w0-b", 0),
            WeeklySlot("w1-a", 1),
            WeeklySlot("w2-a", 2),
        )
        domains = domains_override or tuple(
            VariableDomain(variable, catalog) for variable in variables
        )
        eligible = tuple((variable, frozenset(catalog)) for variable in variables)
        constraints = build_required_constraints(
            manifest=selected_manifest,
            variables=variables,
            eligible_slots=eligible,
            ordering_policy=ordering,
            weekly_capacity=capacity,
        )
        return CspProblem(
            manifest=selected_manifest,
            variables=variables,
            domains=domains,
            constraints=constraints,
            weekly_capacity=capacity,
            ordering_policy=ordering,
        )

    return build


def test_known_feasible_problem_returns_complete_identity_bound_solution(
    problem_factory,
) -> None:
    problem = problem_factory()
    original_identity = problem.problem_identity
    original_domains = problem.domains
    result = backtracking_search(problem)

    assert result.status is CspSolverStatus.FEASIBLE
    assert result.termination_reason is BacktrackingTerminationReason.SOLUTION_FOUND
    assert result.outcome.exhaustive is False
    assert result.evaluation is not None
    assert result.evaluation.whole_plan_identity == problem.whole_plan_identity
    assert result.evaluation.problem_identity == problem.problem_identity
    assert len(result.evaluation.assignment.entries) == len(problem.variables)
    assert result.statistics.nodes_visited == (
        result.statistics.assignments_attempted + 1
    )
    assert result.statistics.maximum_depth == len(problem.variables)
    assert problem.problem_identity == original_identity
    assert problem.domains == original_domains


def test_unsatisfiable_problem_is_infeasible_only_after_exhaustion(
    problem_factory,
) -> None:
    problem = problem_factory()
    only_slot = WeeklySlot("w0-a", 0)
    narrowed = tuple(
        VariableDomain(variable, (only_slot,)) for variable in problem.variables
    )
    impossible = replace(problem, domains=narrowed)

    result = backtracking_search(impossible)

    assert result.status is CspSolverStatus.INFEASIBLE
    assert result.outcome.exhaustive is True
    assert result.termination_reason is BacktrackingTerminationReason.EXHAUSTED
    assert result.evaluation is None
    assert result.statistics.backtracks > 0


def test_partial_constraint_rejection_backtracks_to_another_value(
    problem_factory,
) -> None:
    problem = problem_factory()
    choices = {
        SessionVariable("a1", 0): (
            WeeklySlot("w0-a", 0),
            WeeklySlot("w1-a", 1),
        ),
        SessionVariable("a1", 1): (WeeklySlot("w2-a", 2),),
        SessionVariable("a3", 0): (WeeklySlot("w0-a", 0),),
    }
    narrowed = tuple(
        VariableDomain(variable, choices[variable]) for variable in problem.variables
    )
    constrained = replace(problem, domains=narrowed)

    result = backtracking_search(
        constrained,
        BacktrackingOptions(value_order=BacktrackingValueOrder.CANONICAL),
    )

    assert result.status is CspSolverStatus.FEASIBLE
    assert result.statistics.backtracks >= 1
    assert result.evaluation is not None
    placement = result.evaluation.assignment.placements
    assert placement[SessionVariable("a1", 0)] == WeeklySlot("w1-a", 1)


def test_first_solution_is_deterministic_when_multiple_schedules_exist(
    problem_factory,
) -> None:
    problem = problem_factory()
    first = backtracking_search(problem)
    second = backtracking_search(problem)
    alternate = CspAssignment(
        (
            AssignmentEntry(SessionVariable("a1", 0), WeeklySlot("w0-a", 0)),
            AssignmentEntry(SessionVariable("a1", 1), WeeklySlot("w1-a", 1)),
            AssignmentEntry(SessionVariable("a3", 0), WeeklySlot("w2-a", 2)),
        )
    )

    assert (
        evaluate_assignment(problem, alternate).status is CspEvaluationStatus.FEASIBLE
    )
    assert first.evaluation is not None and second.evaluation is not None
    assert first.evaluation.assignment == second.evaluation.assignment
    assert first.evaluation.assignment != alternate
    assert first.statistics.nodes_visited == second.statistics.nodes_visited
    assert first.statistics.assignments_attempted == (
        second.statistics.assignments_attempted
    )


def test_empty_domain_is_proven_infeasible_without_mutating_problem(
    problem_factory,
) -> None:
    problem = problem_factory()
    original_domains = problem.domains
    domains = list(problem.domains)
    domains[0] = VariableDomain(domains[0].variable, ())
    impossible = replace(problem, domains=tuple(domains))

    result = backtracking_search(impossible)

    assert result.status is CspSolverStatus.INFEASIBLE
    assert result.outcome.exhaustive is True
    assert result.statistics.nodes_visited == 1
    assert result.statistics.constraint_checks == 0
    assert problem.domains == original_domains


def test_zero_variable_problem_is_feasible_at_root(problem_factory) -> None:
    assignment_plans = tuple(
        AssignmentPlan(
            assignment_id,
            AssignmentCandidateIdentity(
                "student-1",
                "course-1",
                assignment_id,
                (0, 0, 0, 0),
                f"features-{assignment_id}",
                "central",
            ),
            f"features-{assignment_id}",
        )
        for assignment_id in ("a1", "a2", "a3")
    )
    manifest = WholePlanManifest("student-1", "course-1", assignment_plans, "central")
    problem = problem_factory(manifest=manifest)

    result = backtracking_search(problem)

    assert problem.variables == ()
    assert result.status is CspSolverStatus.FEASIBLE
    assert result.evaluation is not None
    assert result.evaluation.assignment == CspAssignment()
    assert result.statistics.nodes_visited == 1
    assert result.statistics.assignments_attempted == 0
    assert result.statistics.maximum_depth == 0


def test_node_limit_returns_incomplete_not_infeasible(problem_factory) -> None:
    problem = problem_factory()
    result = backtracking_search(problem, BacktrackingOptions(max_nodes=1))

    assert result.status is CspSolverStatus.INCOMPLETE
    assert result.outcome.exhaustive is False
    assert result.outcome.reason == "node_limit"
    assert result.termination_reason is BacktrackingTerminationReason.NODE_LIMIT
    assert result.statistics.nodes_visited == 1
    assert result.statistics.assignments_attempted == 1


def test_zero_node_limit_stops_before_visiting_root(problem_factory) -> None:
    result = backtracking_search(problem_factory(), BacktrackingOptions(max_nodes=0))

    assert result.status is CspSolverStatus.INCOMPLETE
    assert result.statistics.nodes_visited == 0
    assert result.statistics.assignments_attempted == 0


def test_node_limit_exact_boundary_does_not_claim_exhaustion(problem_factory) -> None:
    problem = problem_factory()
    # One node reaches a depth-one partial assignment, leaving the remaining
    # branch space unexplored. Reaching the limit at that boundary is incomplete.
    result = backtracking_search(problem, BacktrackingOptions(max_nodes=2))

    assert result.status is CspSolverStatus.INCOMPLETE
    assert result.statistics.nodes_visited == 2
    assert result.statistics.assignments_attempted == 2
    assert result.outcome.exhaustive is False


def test_lazy_enumerator_returns_distinct_solutions_then_exhausts(
    problem_factory,
) -> None:
    problem = problem_factory()
    enumerator = backtracking_enumerator(
        problem,
        BacktrackingOptions(value_order=BacktrackingValueOrder.CANONICAL),
    )

    first = enumerator.next_solution()
    second = enumerator.next_solution()
    seen = [first, second]
    while True:
        next_result = enumerator.next_solution()
        if next_result.status is not BacktrackingEnumerationStatus.SOLUTION_FOUND:
            terminal = next_result
            break
        seen.append(next_result)
    terminal_again = enumerator.next_solution()

    assert first.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    assert second.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    assert first.evaluation is not None and second.evaluation is not None
    assert first.schedule_identity != second.schedule_identity
    assert first.evaluation.assignment != second.evaluation.assignment
    assert len(seen) > 2
    assert terminal.status is BacktrackingEnumerationStatus.EXHAUSTED
    assert terminal.evaluation is None
    assert terminal.termination_reason is BacktrackingTerminationReason.EXHAUSTED
    assert terminal_again == terminal
    assert first.evaluation.problem_identity == enumerator.problem_identity
    assert first.evaluation.whole_plan_identity == enumerator.whole_plan_identity


def test_lazy_enumeration_is_deterministic_and_has_no_duplicate_schedules(
    problem_factory,
) -> None:
    problem = problem_factory()

    def all_ids() -> tuple[str, ...]:
        enumerator = backtracking_enumerator(problem)
        identities: list[str] = []
        while True:
            result = enumerator.next_solution()
            if result.status is not BacktrackingEnumerationStatus.SOLUTION_FOUND:
                assert result.status is BacktrackingEnumerationStatus.EXHAUSTED
                break
            assert result.schedule_identity is not None
            identities.append(result.schedule_identity)
        return tuple(identities)

    first_run = all_ids()
    second_run = all_ids()
    assert first_run
    assert first_run == second_run
    assert len(first_run) == len(set(first_run))


def test_lazy_enumerator_node_limit_returns_incomplete_after_solutions(
    problem_factory,
) -> None:
    problem = problem_factory()
    baseline = backtracking_enumerator(problem)
    first_baseline = baseline.next_solution()
    second_baseline = baseline.next_solution()
    assert first_baseline.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    assert second_baseline.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    exact_limit = second_baseline.statistics.nodes_visited
    enumerator = backtracking_enumerator(
        problem, BacktrackingOptions(max_nodes=exact_limit)
    )
    first = enumerator.next_solution()
    second = enumerator.next_solution()
    next_result = enumerator.next_solution()

    assert first.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    assert second.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    assert next_result.status is BacktrackingEnumerationStatus.INCOMPLETE
    assert next_result.termination_reason is BacktrackingTerminationReason.NODE_LIMIT
    assert next_result.statistics.nodes_visited == exact_limit
    assert enumerator.next_solution() is next_result
    restarted = backtracking_enumerator(
        problem, BacktrackingOptions(max_nodes=exact_limit)
    )
    assert restarted.next_solution().schedule_identity == first.schedule_identity


def test_solver_time_excludes_pauses_between_enumeration_requests(
    problem_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = problem_factory()
    evaluation = backtracking_search(problem).evaluation
    assert evaluation is not None
    clock = [0.0]
    monkeypatch.setattr(
        "study_planner.csp.backtracking.perf_counter", lambda: clock[0]
    )
    enumerator = backtracking_enumerator(problem)

    def timed_solutions():
        clock[0] += 2.0
        yield evaluation
        clock[0] += 3.0

    enumerator._solutions = timed_solutions()
    first = enumerator.next_solution()
    assert first.statistics.solver_time_seconds == 2.0
    clock[0] += 100.0  # simulated caller or NN time between requests
    exhausted = enumerator.next_solution()

    assert exhausted.status is BacktrackingEnumerationStatus.EXHAUSTED
    assert exhausted.statistics.solver_time_seconds == 5.0
    assert exhausted.statistics.wall_time_seconds == 105.0


def test_predicate_count_matches_rejected_and_incomplete_paths(
    problem_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    constraint_types = (
        DeadlineConstraint,
        UniqueSlotConstraint,
        SameAssignmentOrderingConstraint,
        WeeklyCapacityConstraint,
    )
    originals = {
        constraint_type: constraint_type.is_satisfied
        for constraint_type in constraint_types
    }
    for constraint_type, original in originals.items():

        def counted(self, assignment, *, _original=original):
            nonlocal calls
            calls += 1
            return _original(self, assignment)

        monkeypatch.setattr(constraint_type, "is_satisfied", counted)

    problem = problem_factory()
    only_slot = WeeklySlot("w0-a", 0)
    impossible = replace(
        problem,
        domains=tuple(
            VariableDomain(variable, (only_slot,)) for variable in problem.variables
        ),
    )
    rejected_enumerator = backtracking_enumerator(impossible)
    while True:
        rejected = rejected_enumerator.next_solution()
        if rejected.status is not BacktrackingEnumerationStatus.SOLUTION_FOUND:
            break
    assert rejected.status is BacktrackingEnumerationStatus.EXHAUSTED
    assert rejected.statistics.constraint_checks == calls
    assert calls > 0

    calls = 0
    incomplete_enumerator = backtracking_enumerator(
        problem, BacktrackingOptions(max_nodes=1)
    )
    incomplete = incomplete_enumerator.next_solution()
    assert incomplete.status is BacktrackingEnumerationStatus.INCOMPLETE
    assert incomplete.statistics.constraint_checks == calls
    assert calls > 0


def test_predicate_count_matches_every_actual_constraint_invocation(
    problem_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = problem_factory()
    calls = 0
    constraint_types = (
        DeadlineConstraint,
        UniqueSlotConstraint,
        SameAssignmentOrderingConstraint,
        WeeklyCapacityConstraint,
    )
    originals = {
        constraint_type: constraint_type.is_satisfied
        for constraint_type in constraint_types
    }

    for constraint_type, original in originals.items():

        def counted(self, assignment, *, _original=original):
            nonlocal calls
            calls += 1
            return _original(self, assignment)

        monkeypatch.setattr(constraint_type, "is_satisfied", counted)

    enumerator = backtracking_enumerator(problem)
    result = enumerator.next_solution()
    assert result.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    assert result.statistics.constraint_checks == calls

    # The convenience wrapper performs the solver-boundary revalidation too;
    # its reported value includes those calls rather than silently omitting them.
    calls = 0
    wrapped = backtracking_search(problem)
    assert wrapped.status is CspSolverStatus.FEASIBLE
    assert wrapped.statistics.constraint_checks == calls


def test_canonical_variable_order_is_supported_and_deterministic(
    problem_factory,
) -> None:
    problem = problem_factory()
    options = BacktrackingOptions(variable_order=BacktrackingVariableOrder.CANONICAL)
    first = backtracking_search(problem, options)
    second = backtracking_search(problem, options)

    assert first.evaluation is not None and second.evaluation is not None
    assert first.evaluation.assignment == second.evaluation.assignment


def test_dynamic_mrv_uses_currently_legal_values(problem_factory) -> None:
    assignments = tuple(
        AssignmentPlan(
            assignment_id,
            AssignmentCandidateIdentity(
                "student-1",
                "course-1",
                assignment_id,
                allocation,
                f"features-{assignment_id}",
                "central",
            ),
            f"features-{assignment_id}",
        )
        for assignment_id, allocation in (
            ("a1", (1, 1, 1, 0)),
            ("a2", (0, 0, 0, 0)),
            ("a3", (1, 0, 0, 0)),
        )
    )
    manifest = WholePlanManifest("student-1", "course-1", assignments, "central")
    slots = (
        WeeklySlot("w0-a", 0),
        WeeklySlot("w0-b", 0),
        WeeklySlot("w1-a", 1),
        WeeklySlot("w2-a", 2),
    )
    variables = expected_session_variables(manifest)
    domains = tuple(
        VariableDomain(
            variable,
            (
                (slots[0], slots[2], slots[3])
                if variable == SessionVariable("a1", 2)
                else (slots[1], slots[3])
                if variable == SessionVariable("a3", 0)
                else slots
            ),
        )
        for variable in variables
    )
    ordering = _IncreasingWeeksPolicy()
    constraints = build_required_constraints(
        manifest=manifest,
        variables=variables,
        eligible_slots=tuple((variable, frozenset(slots)) for variable in variables),
        ordering_policy=ordering,
        weekly_capacity=2,
    )
    problem = CspProblem(
        manifest,
        variables,
        domains,
        constraints,
        2,
        ordering,
    )
    enumerator = backtracking_enumerator(
        problem,
        BacktrackingOptions(value_order=BacktrackingValueOrder.CANONICAL),
    )
    enumerator._assigned.extend(
        (
            (SessionVariable("a1", 0), slots[0]),
            (SessionVariable("a1", 1), slots[2]),
        )
    )

    selected = enumerator._select_variable(
        (SessionVariable("a1", 2), SessionVariable("a3", 0))
    )

    assert selected == SessionVariable("a1", 2)
    assert len(problem.domain_for(selected)) == 3
    assert len(problem.domain_for(SessionVariable("a3", 0))) == 2
    assert enumerator.statistics.constraint_checks > 0


def test_enumerator_continues_after_a_rejected_branch(problem_factory) -> None:
    problem = problem_factory()
    choices = {
        SessionVariable("a1", 0): (
            WeeklySlot("w0-a", 0),
            WeeklySlot("w1-a", 1),
        ),
        SessionVariable("a1", 1): (WeeklySlot("w2-a", 2),),
        SessionVariable("a3", 0): (WeeklySlot("w0-a", 0),),
    }
    constrained = replace(
        problem,
        domains=tuple(
            VariableDomain(variable, choices[variable])
            for variable in problem.variables
        ),
    )
    enumerator = backtracking_enumerator(
        constrained,
        BacktrackingOptions(value_order=BacktrackingValueOrder.CANONICAL),
    )

    result = enumerator.next_solution()

    assert result.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    assert result.evaluation is not None
    assert result.evaluation.assignment.placements[SessionVariable("a1", 0)] == (
        WeeklySlot("w1-a", 1)
    )
    assert result.statistics.backtracks >= 1


@pytest.mark.parametrize("max_nodes", [-1, True, 1.5])
def test_invalid_node_limits_are_rejected(max_nodes: object) -> None:
    with pytest.raises(CspInputError, match="max_nodes"):
        BacktrackingOptions(max_nodes=max_nodes)  # type: ignore[arg-type]


def test_invalid_variable_order_is_rejected() -> None:
    with pytest.raises(CspInputError, match="variable_order"):
        BacktrackingOptions(variable_order="random")  # type: ignore[arg-type]


def test_statistics_have_consistent_nonnegative_values(problem_factory) -> None:
    result = backtracking_search(problem_factory())
    stats = result.statistics

    assert stats.nodes_visited > 0
    assert stats.assignments_attempted >= stats.backtracks
    assert stats.constraint_checks >= stats.nodes_visited * len(
        problem_factory().constraints
    )
    assert stats.maximum_depth == len(result.evaluation.assignment.entries)  # type: ignore[union-attr]
    assert stats.solver_time_seconds >= 0
    assert stats.wall_time_seconds >= stats.solver_time_seconds
