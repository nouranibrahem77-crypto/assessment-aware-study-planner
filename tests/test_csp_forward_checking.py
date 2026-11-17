"""Forward Checking behavior and contract tests."""

from __future__ import annotations

from dataclasses import dataclass, field

from study_planner.csp import (
    AssignmentCandidateIdentity,
    AssignmentEntry,
    AssignmentPlan,
    BacktrackingEnumerationStatus,
    CspEvaluationStatus,
    CspSolverStatus,
    ForwardCheckingOptions,
    ForwardCheckingValueOrder,
    ForwardCheckingVariableOrder,
    SessionPair,
    SessionVariable,
    WeeklySlot,
    WholePlanManifest,
    backtracking_enumerator,
    forward_checking_enumerator,
    forward_checking_search,
)
from study_planner.csp.constraints import build_required_constraints
from study_planner.csp.core import (
    CspAssignment,
    CspProblem,
    VariableDomain,
    evaluate_assignment,
)


@dataclass(frozen=True)
class _Policy:
    policy_identity: str = "fc-test-policy"

    def required_pairs(self, sessions: tuple[SessionVariable, ...]) -> tuple:
        return tuple(
            SessionPair(a, b)
            for a, b in zip(sessions, sessions[1:], strict=False)
            if a.assignment_id == b.assignment_id
        )

    def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
        return earlier.week_index < later.week_index


def _problem(domain: tuple[WeeklySlot, ...] | None = None) -> CspProblem:
    assignments = tuple(
        AssignmentPlan(
            name,
            AssignmentCandidateIdentity(
                "s",
                "c",
                name,
                (1, 0, 0, 0) if name == "a" else (0, 0, 0, 0),
                f"f-{name}",
                "scenario",
            ),
            f"f-{name}",
        )
        for name in ("a", "b", "c")
    )
    manifest = WholePlanManifest("s", "c", assignments, "scenario")
    variables = (SessionVariable("a", 0),)
    slots = domain if domain is not None else (WeeklySlot("x", 0), WeeklySlot("y", 1))
    domains = (VariableDomain(variables[0], slots),)
    constraints = build_required_constraints(
        manifest=manifest,
        variables=variables,
        eligible_slots=((variables[0], frozenset(slots)),),
        ordering_policy=_Policy(),
        weekly_capacity=1,
    )
    return CspProblem(manifest, variables, domains, constraints, 1, _Policy())


def _two_variable_problem() -> CspProblem:
    assignments = tuple(
        AssignmentPlan(
            name,
            AssignmentCandidateIdentity(
                "s",
                "c",
                name,
                (2, 0, 0, 0) if name == "a" else (0, 0, 0, 0),
                f"f-{name}",
                "scenario",
            ),
            f"f-{name}",
        )
        for name in ("a", "b", "c")
    )
    manifest = WholePlanManifest("s", "c", assignments, "scenario")
    variables = (SessionVariable("a", 0), SessionVariable("a", 1))
    slots = (WeeklySlot("x", 0), WeeklySlot("y", 1), WeeklySlot("z", 2))
    domains = tuple(VariableDomain(variable, slots) for variable in variables)
    constraints = build_required_constraints(
        manifest=manifest,
        variables=variables,
        eligible_slots=tuple((variable, frozenset(slots)) for variable in variables),
        ordering_policy=_Policy(),
        weekly_capacity=1,
    )
    return CspProblem(manifest, variables, domains, constraints, 1, _Policy())


def _three_variable_problem(
    *, extra_constraint: object | None = None,
    slots: tuple[WeeklySlot, ...] | None = None,
) -> CspProblem:
    assignments = tuple(
        AssignmentPlan(
            name,
            AssignmentCandidateIdentity(
                "s",
                "c",
                name,
                (1, 0, 0, 0),
                f"f-{name}",
                "scenario",
            ),
            f"f-{name}",
        )
        for name in ("a", "b", "c")
    )
    manifest = WholePlanManifest("s", "c", assignments, "scenario")
    variables = tuple(SessionVariable(name, 0) for name in ("a", "b", "c"))
    catalog = slots or tuple(
        WeeklySlot(f"w{week}-{letter}", week)
        for week in (0, 1)
        for letter in "abc"
    )
    domains = tuple(VariableDomain(variable, catalog) for variable in variables)
    constraints = build_required_constraints(
        manifest=manifest,
        variables=variables,
        eligible_slots=tuple((variable, frozenset(catalog)) for variable in variables),
        ordering_policy=_Policy(),
        weekly_capacity=3,
    )
    if extra_constraint is not None:
        constraints = (*constraints, extra_constraint)
    return CspProblem(manifest, variables, domains, constraints, 3, _Policy())


def _zero_variable_problem() -> CspProblem:
    assignments = tuple(
        AssignmentPlan(
            name,
            AssignmentCandidateIdentity(
                "s", "c", name, (0, 0, 0, 0), f"f-{name}", "scenario"
            ),
            f"f-{name}",
        )
        for name in ("a", "b", "c")
    )
    manifest = WholePlanManifest("s", "c", assignments, "scenario")
    constraints = build_required_constraints(
        manifest=manifest,
        variables=(),
        eligible_slots=(),
        ordering_policy=_Policy(),
        weekly_capacity=3,
    )
    return CspProblem(manifest, (), (), constraints, 3, _Policy())


@dataclass
class _NaryConstraint:
    """A three-variable sum constraint that records predicate invocations."""

    scope: tuple[SessionVariable, ...]
    calls: list[tuple[int, ...]] = field(default_factory=list)

    @property
    def constraint_id(self) -> str:
        return "test_three_variable_sum"

    @property
    def configuration_identity(self) -> str:
        return "sum-of-weeks-at-most-one-v1"

    def is_satisfied(self, assignment: CspAssignment) -> bool:
        assigned = tuple(
            assignment.placements[variable].week_index
            for variable in self.scope
            if variable in assignment.placements
        )
        self.calls.append(assigned)
        return sum(assigned) <= 1


def test_fc_enumerates_distinct_solutions_then_exhausts_without_mutating_problem() -> (
    None
):
    problem = _problem()
    original = problem.domains
    enum = forward_checking_enumerator(
        problem, ForwardCheckingOptions(value_order=ForwardCheckingValueOrder.CANONICAL)
    )
    first, second, end = (enum.next_solution() for _ in range(3))
    assert first.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    assert second.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    assert first.schedule_identity != second.schedule_identity
    assert end.status is BacktrackingEnumerationStatus.EXHAUSTED
    assert problem.domains == original
    assert enum.statistics.nodes_visited == 3
    assert enum.statistics.constraint_checks > 0


def test_fc_zero_node_limit_returns_incomplete() -> None:
    result = forward_checking_search(_problem(), ForwardCheckingOptions(max_nodes=0))
    assert result.status is CspSolverStatus.INCOMPLETE
    assert result.outcome.reason == "node_limit"
    assert forward_checking_search(_problem()).status is CspSolverStatus.FEASIBLE


def test_fc_zero_variable_problem_is_one_empty_solution_then_exhaustion() -> None:
    problem = _zero_variable_problem()
    enumerator = forward_checking_enumerator(problem)

    solution = enumerator.next_solution()
    assert solution.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    assert solution.evaluation is not None
    assert solution.evaluation.assignment.entries == ()
    assert solution.evaluation.problem_identity == problem.problem_identity
    assert solution.evaluation.whole_plan_identity == problem.whole_plan_identity
    assert solution.schedule_identity is not None
    assert solution.statistics.nodes_visited == 1
    assert solution.statistics.assignments_attempted == 0
    assert solution.statistics.maximum_depth == 0

    exhausted = enumerator.next_solution()
    assert exhausted.status is BacktrackingEnumerationStatus.EXHAUSTED
    assert exhausted.statistics.nodes_visited == 1
    assert exhausted.statistics.assignments_attempted == 0


def test_fc_empty_domain_is_exhaustively_infeasible() -> None:
    result = forward_checking_search(_problem(()))
    assert result.status is CspSolverStatus.INFEASIBLE
    assert result.outcome.exhaustive
    assert result.statistics.domain_prunings == 0


def test_fc_binary_pruning_restoration_and_solution_set_match_backtracking() -> None:
    problem = _two_variable_problem()
    fc = forward_checking_enumerator(
        problem, ForwardCheckingOptions(value_order=ForwardCheckingValueOrder.CANONICAL)
    )
    bt = backtracking_enumerator(problem)
    fc_solutions = set()
    bt_solutions = set()
    while True:
        item = fc.next_solution()
        if item.status is not BacktrackingEnumerationStatus.SOLUTION_FOUND:
            assert item.status is BacktrackingEnumerationStatus.EXHAUSTED
            break
        assert item.evaluation is not None
        fc_solutions.add(frozenset(item.evaluation.assignment.placements.items()))
    while True:
        item = bt.next_solution()
        if item.status is not BacktrackingEnumerationStatus.SOLUTION_FOUND:
            assert item.status is BacktrackingEnumerationStatus.EXHAUSTED
            break
        assert item.evaluation is not None
        bt_solutions.add(frozenset(item.evaluation.assignment.placements.items()))
    assert fc_solutions == bt_solutions
    assert fc.statistics.domain_prunings > 0
    assert problem.domains == tuple(
        VariableDomain(
            variable, (WeeklySlot("x", 0), WeeklySlot("y", 1), WeeklySlot("z", 2))
        )
        for variable in problem.variables
    )


def test_fc_scope_index_preserves_order_and_complete_solution_set() -> None:
    variables = tuple(SessionVariable(name, 0) for name in ("a", "b", "c"))
    extra = _NaryConstraint(variables)
    problem = _three_variable_problem(extra_constraint=extra)
    enumerator = forward_checking_enumerator(problem)

    for variable in problem.variables:
        assert enumerator._constraints_by_variable[variable] == tuple(
            constraint
            for constraint in problem.constraints
            if variable in constraint.scope
        )

    solutions = set()
    while True:
        result = enumerator.next_solution()
        if result.status is BacktrackingEnumerationStatus.EXHAUSTED:
            break
        assert result.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
        assert result.evaluation is not None
        solutions.add(frozenset(result.evaluation.assignment.placements.items()))

    baseline = backtracking_enumerator(problem)
    baseline_solutions = set()
    while True:
        result = baseline.next_solution()
        if result.status is BacktrackingEnumerationStatus.EXHAUSTED:
            break
        assert result.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
        assert result.evaluation is not None
        baseline_solutions.add(
            frozenset(result.evaluation.assignment.placements.items())
        )

    assert len(solutions) == 60
    assert solutions == baseline_solutions
    assert enumerator.next_solution().status is BacktrackingEnumerationStatus.EXHAUSTED


def test_fc_domain_wipeout_restores_values_before_later_branch() -> None:
    assignments = tuple(
        AssignmentPlan(
            name,
            AssignmentCandidateIdentity(
                "s",
                "c",
                name,
                (1, 0, 0, 0) if name in {"a", "b"} else (0, 0, 0, 0),
                f"f-{name}",
                "scenario",
            ),
            f"f-{name}",
        )
        for name in ("a", "b", "c")
    )
    manifest = WholePlanManifest("s", "c", assignments, "scenario")
    variables = (SessionVariable("a", 0), SessionVariable("b", 0))
    shared = WeeklySlot("a-shared", 0)
    free = WeeklySlot("z-free", 1)
    domains = (
        VariableDomain(variables[0], (shared, free)),
        VariableDomain(variables[1], (shared,)),
    )
    constraints = build_required_constraints(
        manifest=manifest,
        variables=variables,
        eligible_slots=tuple(
            (variable, frozenset(domain.values))
            for variable, domain in zip(variables, domains, strict=True)
        ),
        ordering_policy=_Policy(),
        weekly_capacity=1,
    )
    problem = CspProblem(manifest, variables, domains, constraints, 1, _Policy())
    enumerator = forward_checking_enumerator(
        problem,
        ForwardCheckingOptions(
            variable_order=ForwardCheckingVariableOrder.CANONICAL,
            value_order=ForwardCheckingValueOrder.CANONICAL,
        ),
    )

    first = enumerator.next_solution()
    assert first.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    assert first.evaluation is not None
    assert first.evaluation.assignment.placements == {
        variables[0]: free,
        variables[1]: shared,
    }
    assert first.statistics.nodes_visited == 3
    assert first.statistics.assignments_attempted == 3
    assert first.statistics.backtracks == 1
    assert first.statistics.domain_prunings == 1
    assert enumerator.next_solution().status is BacktrackingEnumerationStatus.EXHAUSTED
    assert problem.domains == domains


def test_fc_defers_nary_constraint_until_one_scope_variable_remains() -> None:
    variables = tuple(SessionVariable(name, 0) for name in ("a", "b", "c"))
    constraint = _NaryConstraint(variables)
    problem = _three_variable_problem(extra_constraint=constraint)
    enumerator = forward_checking_enumerator(
        problem,
        ForwardCheckingOptions(
            variable_order=ForwardCheckingVariableOrder.CANONICAL,
            value_order=ForwardCheckingValueOrder.CANONICAL,
        ),
    )
    fc_solutions = set()
    while True:
        result = enumerator.next_solution()
        if result.status is BacktrackingEnumerationStatus.EXHAUSTED:
            break
        assert result.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
        assert result.evaluation is not None
        placements = result.evaluation.assignment.placements
        weeks = tuple(placements[variable].week_index for variable in variables)
        assert sum(weeks) <= 1
        fc_solutions.add(frozenset(placements.items()))

    assert len(fc_solutions) == 60
    assert enumerator.statistics.nodes_visited == 91
    # Every entered depth-one and depth-two state is validated once. The exact
    # counts show there were no extra n-ary candidate probes while two scoped
    # variables remained unassigned.
    calls_by_depth = {
        depth: sum(len(call) == depth for call in constraint.calls)
        for depth in range(4)
    }
    assert calls_by_depth == {0: 1, 1: 6, 2: 24, 3: 180}

    # There are 60 valid complete states, each checked once by state validation.
    # The other 120 three-value calls are FC support probes for the last
    # unassigned variable. All 60 invalid tuples occur among those probes and
    # were rejected before entering a depth-three node.
    complete_state_checks = len(fc_solutions)
    support_probes = calls_by_depth[3] - complete_state_checks
    invalid_support_probes = sum(
        len(call) == 3 and sum(call) > 1 for call in constraint.calls
    )
    assert support_probes == 120
    assert invalid_support_probes == 60

    # Independently confirm that full validation rejects the same kind of
    # invalid complete tuple if it reaches the validator boundary.
    invalid_assignment = CspAssignment(
        tuple(
            AssignmentEntry(variable, WeeklySlot(f"w1-{letter}", 1))
            for variable, letter in zip(variables[:2], "ab", strict=True)
        )
        + (AssignmentEntry(variables[2], WeeklySlot("w0-c", 0)),)
    )
    invalid_evaluation = evaluate_assignment(problem, invalid_assignment)
    assert invalid_evaluation.status is CspEvaluationStatus.BRANCH_REJECTED
    assert constraint.constraint_id in invalid_evaluation.failed_checks

    # Compare the final feasible schedule set with baseline Backtracking.
    bt_constraint = _NaryConstraint(variables)
    bt = backtracking_enumerator(
        _three_variable_problem(extra_constraint=bt_constraint)
    )
    bt_solutions = set()
    while True:
        result = bt.next_solution()
        if result.status is BacktrackingEnumerationStatus.EXHAUSTED:
            break
        assert result.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
        assert result.evaluation is not None
        bt_solutions.add(frozenset(result.evaluation.assignment.placements.items()))
    assert bt_solutions == fc_solutions


def test_fc_node_limit_after_solution_is_terminal_and_exact_boundary_exhausts() -> None:
    problem = _problem()
    limited = forward_checking_enumerator(
        problem,
        ForwardCheckingOptions(
            max_nodes=2,
            value_order=ForwardCheckingValueOrder.CANONICAL,
        ),
    )
    first = limited.next_solution()
    assert first.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    incomplete = limited.next_solution()
    assert incomplete.status is BacktrackingEnumerationStatus.INCOMPLETE
    assert incomplete.statistics.nodes_visited == 2
    assert incomplete.statistics.assignments_attempted == 2
    assert limited.next_solution() == incomplete

    exact = forward_checking_enumerator(
        problem,
        ForwardCheckingOptions(
            max_nodes=3,
            value_order=ForwardCheckingValueOrder.CANONICAL,
        ),
    )
    assert exact.next_solution().status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    assert exact.next_solution().status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    exhausted = exact.next_solution()
    assert exhausted.status is BacktrackingEnumerationStatus.EXHAUSTED
    assert exhausted.statistics.nodes_visited == 3
    assert exhausted.statistics.assignments_attempted == 2
