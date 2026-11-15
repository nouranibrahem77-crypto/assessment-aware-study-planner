"""Regression tests for binary AC-3 propagation and integrated search."""

from __future__ import annotations

from dataclasses import dataclass

from study_planner.csp import (
    Ac3Options,
    BacktrackingEnumerationStatus,
    CspAssignment,
    SessionVariable,
    WeeklySlot,
    ac3_enumerator,
    ac3_propagate,
    ac3_search,
    backtracking_enumerator,
    forward_checking_enumerator,
)
from study_planner.csp.constraints import (
    DeadlineConstraint,
    UniqueSlotConstraint,
    WeeklyCapacityConstraint,
    build_required_constraints,
)
from study_planner.csp.core import (
    AssignmentCandidateIdentity,
    AssignmentEntry,
    AssignmentPlan,
    BinaryConstraint,
    CspEvaluationStatus,
    CspProblem,
    CspSolverStatus,
    VariableDomain,
    WholePlanManifest,
    evaluate_assignment,
)


@dataclass(frozen=True)
class _Policy:
    policy_identity: str = "ac3-test-policy"

    def required_pairs(self, sessions: tuple[SessionVariable, ...]) -> tuple:
        return ()

    def is_ordered(self, earlier: WeeklySlot, later: WeeklySlot) -> bool:
        return earlier.week_index < later.week_index


class _TableConstraint(BinaryConstraint):
    def __init__(
        self,
        constraint_id: str,
        scope: tuple[SessionVariable, SessionVariable],
        allowed: frozenset[tuple[WeeklySlot, WeeklySlot]],
    ) -> None:
        self._constraint_id = constraint_id
        self._scope = scope
        self.allowed = allowed
        self.calls: list[
            tuple[SessionVariable, WeeklySlot, SessionVariable, WeeklySlot]
        ] = []

    @property
    def constraint_id(self) -> str:
        return self._constraint_id

    @property
    def scope(self) -> tuple[SessionVariable, SessionVariable]:
        return self._scope

    @property
    def configuration_identity(self) -> str:
        pairs = sorted((left.slot_id, right.slot_id) for left, right in self.allowed)
        return f"{self.constraint_id}:{pairs!r}"

    def is_compatible(
        self,
        first_variable: SessionVariable,
        first_value: WeeklySlot,
        second_variable: SessionVariable,
        second_value: WeeklySlot,
    ) -> bool:
        self.calls.append((first_variable, first_value, second_variable, second_value))
        if (first_variable, second_variable) == self.scope:
            pair = (first_value, second_value)
        elif (second_variable, first_variable) == self.scope:
            pair = (second_value, first_value)
        else:
            raise ValueError("variables do not match the table scope")
        return pair in self.allowed

    def is_satisfied(self, assignment: CspAssignment) -> bool:
        placements = assignment.placements
        if any(variable not in placements for variable in self.scope):
            return True
        return self.is_compatible(
            self.scope[0],
            placements[self.scope[0]],
            self.scope[1],
            placements[self.scope[1]],
        )


@dataclass
class _NaryConstraint:
    scope: tuple[SessionVariable, ...]
    calls: int = 0

    @property
    def constraint_id(self) -> str:
        return "test_nary_no_arc_api"

    @property
    def configuration_identity(self) -> str:
        return "sum-of-weeks-at-most-one-v1"

    def is_satisfied(self, assignment: CspAssignment) -> bool:
        self.calls += 1
        values = tuple(
            assignment.placements[v].week_index
            for v in self.scope
            if v in assignment.placements
        )
        return sum(values) <= 1


def _problem(
    domains: dict[str, tuple[WeeklySlot, ...]],
    extras: tuple[object, ...] = (),
    *,
    capacity: int = 3,
) -> CspProblem:
    active = set(domains)
    manifest = WholePlanManifest(
        "s",
        "c",
        tuple(
            AssignmentPlan(
                name,
                AssignmentCandidateIdentity(
                    "s",
                    "c",
                    name,
                    (1, 0, 0, 0) if name in active else (0, 0, 0, 0),
                    f"f-{name}",
                    "scenario",
                ),
                f"f-{name}",
            )
            for name in ("a", "b", "c")
        ),
        "scenario",
    )
    variables = tuple(SessionVariable(name, 0) for name in sorted(active))
    variable_domains = tuple(
        VariableDomain(variable, domains[variable.assignment_id])
        for variable in variables
    )
    constraints = build_required_constraints(
        manifest=manifest,
        variables=variables,
        eligible_slots=tuple(
            (domain.variable, frozenset(domain.values)) for domain in variable_domains
        ),
        ordering_policy=_Policy(),
        weekly_capacity=capacity,
    )
    return CspProblem(
        manifest,
        variables,
        variable_domains,
        (*constraints, *extras),
        capacity,
        _Policy(),
    )


def _solutions(enumerator) -> set[frozenset[tuple[SessionVariable, WeeklySlot]]]:
    found = set()
    while True:
        result = enumerator.next_solution()
        if result.status is BacktrackingEnumerationStatus.EXHAUSTED:
            return found
        assert result.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
        assert result.evaluation is not None
        found.add(frozenset(result.evaluation.assignment.placements.items()))


def test_ac3_revises_each_arc_direction_and_counts_compatibility_calls() -> None:
    a, b = SessionVariable("a", 0), SessionVariable("b", 0)
    a0, a1 = WeeklySlot("a0", 0), WeeklySlot("a1", 1)
    b0, b1 = WeeklySlot("b0", 2), WeeklySlot("b1", 3)
    relation = _TableConstraint("table_ab", (a, b), frozenset({(a1, b0)}))
    problem = _problem({"a": (a0, a1), "b": (b0, b1)}, (relation,))
    original_domains = problem.domains

    result = ac3_propagate(problem)

    assert result.consistent
    assert result.domains == (VariableDomain(a, (a1,)), VariableDomain(b, (b0,)))
    assert result.domain_prunings == 2
    assert len(relation.calls) == 7
    assert result.compatibility_checks == 11
    assert {call[0] for call in relation.calls} == {a, b}
    assert problem.domains == original_domains


def test_ac3_reenqueues_neighboring_arcs_after_domain_revision() -> None:
    a, b, c = (SessionVariable(name, 0) for name in ("a", "b", "c"))
    a0 = WeeklySlot("a0", 0)
    b0, b1 = WeeklySlot("b0", 1), WeeklySlot("b1", 2)
    c0, c1 = WeeklySlot("c0", 3), WeeklySlot("c1", 4)
    bc = _TableConstraint("a_bc", (b, c), frozenset({(b0, c0), (b1, c1)}))
    ab = _TableConstraint("z_ab", (a, b), frozenset({(a0, b0)}))
    problem = _problem({"a": (a0,), "b": (b0, b1), "c": (c0, c1)}, (bc, ab))

    result = ac3_propagate(problem)

    assert result.consistent
    assert result.domains == (
        VariableDomain(a, (a0,)),
        VariableDomain(b, (b0,)),
        VariableDomain(c, (c0,)),
    )
    initial_arcs = 2 * len(problem.binary_constraints)
    assert result.arcs_enqueued > initial_arcs
    assert result.arc_revisions > initial_arcs


def test_ac3_detects_empty_neighbor_and_propagation_wipeout() -> None:
    a, b = SessionVariable("a", 0), SessionVariable("b", 0)
    a0, b0 = WeeklySlot("a0", 0), WeeklySlot("b0", 1)
    no_support = _TableConstraint("no_support", (a, b), frozenset())
    problem = _problem({"a": (a0,), "b": (b0,)}, (no_support,))
    result = ac3_propagate(problem)
    assert not result.consistent
    assert any(not domain.values for domain in result.domains)
    assert result.domain_prunings >= 1

    empty_neighbor = ac3_propagate(
        problem,
        (VariableDomain(a, (a0,)), VariableDomain(b, ())),
    )
    assert not empty_neighbor.consistent


def test_ac3_search_enumerates_distinct_solutions_and_matches_bt_and_fc() -> None:
    a, b = SessionVariable("a", 0), SessionVariable("b", 0)
    a0, a1 = WeeklySlot("a0", 0), WeeklySlot("a1", 1)
    b0, b1 = WeeklySlot("b0", 2), WeeklySlot("b1", 3)
    relation = _TableConstraint(
        "table_ab", (a, b), frozenset({(a0, b1), (a1, b0), (a1, b1)})
    )
    problem = _problem({"a": (a0, a1), "b": (b0, b1)}, (relation,))
    original_domains = problem.domains

    bt_solutions = _solutions(backtracking_enumerator(problem))
    fc_solutions = _solutions(forward_checking_enumerator(problem))
    ac3 = ac3_enumerator(problem)
    ac3_solutions = _solutions(ac3)

    assert len(ac3_solutions) == 3
    assert ac3_solutions == fc_solutions == bt_solutions
    assert ac3.statistics.compatibility_checks > 0
    assert ac3.statistics.constraint_checks >= ac3.statistics.compatibility_checks
    assert problem.domains == original_domains


def test_ac3_exhaustive_infeasibility_and_zero_variable_solution() -> None:
    a, b = SessionVariable("a", 0), SessionVariable("b", 0)
    a0, b0 = WeeklySlot("a0", 0), WeeklySlot("b0", 1)
    relation = _TableConstraint("no_support", (a, b), frozenset())
    impossible = _problem({"a": (a0,), "b": (b0,)}, (relation,))
    result = ac3_search(impossible)
    assert result.status is CspSolverStatus.INFEASIBLE
    assert result.outcome.exhaustive
    assert result.statistics.domain_prunings == 1

    zero = _problem({})
    enumerator = ac3_enumerator(zero)
    first = enumerator.next_solution()
    assert first.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    assert first.evaluation is not None
    assert first.evaluation.assignment.entries == ()
    assert first.evaluation.problem_identity == zero.problem_identity
    assert first.evaluation.whole_plan_identity == zero.whole_plan_identity
    assert first.statistics.nodes_visited == 1
    assert enumerator.next_solution().status is BacktrackingEnumerationStatus.EXHAUSTED


def test_ac3_node_limit_enumeration_and_terminal_incomplete_semantics() -> None:
    a, b = SessionVariable("a", 0), SessionVariable("b", 0)
    a_slots = (WeeklySlot("a0", 0), WeeklySlot("a1", 1))
    b_slots = (WeeklySlot("b0", 2), WeeklySlot("b1", 3))
    allowed = frozenset((first, second) for first in a_slots for second in b_slots)
    problem = _problem(
        {"a": a_slots, "b": b_slots},
        (_TableConstraint("all_pairs", (a, b), allowed),),
    )
    limited = ac3_enumerator(problem, Ac3Options(max_nodes=3))
    assert (
        limited.next_solution().status is BacktrackingEnumerationStatus.SOLUTION_FOUND
    )
    incomplete = limited.next_solution()
    assert incomplete.status is BacktrackingEnumerationStatus.INCOMPLETE
    assert limited.next_solution() == incomplete

    zero_limit = ac3_search(problem, Ac3Options(max_nodes=0))
    assert zero_limit.status is CspSolverStatus.INCOMPLETE
    assert zero_limit.statistics.nodes_visited == 0


def test_ac3_exact_node_limit_exhausts_without_an_extra_node() -> None:
    a, b = SessionVariable("a", 0), SessionVariable("b", 0)
    a_slots = (WeeklySlot("a0", 0), WeeklySlot("a1", 1))
    b_slots = (WeeklySlot("b0", 2), WeeklySlot("b1", 3))
    relation = _TableConstraint(
        "all_pairs",
        (a, b),
        frozenset((first, second) for first in a_slots for second in b_slots),
    )
    problem = _problem(
        {"a": a_slots, "b": b_slots},
        (relation,),
    )
    exact = ac3_enumerator(problem, Ac3Options(max_nodes=7))
    schedules = []
    while True:
        result = exact.next_solution()
        if result.status is BacktrackingEnumerationStatus.EXHAUSTED:
            break
        assert result.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
        assert result.evaluation is not None
        schedules.append(frozenset(result.evaluation.assignment.placements.items()))

    assert len(schedules) == 4
    assert len(set(schedules)) == 4
    assert exact.statistics.nodes_visited == 7
    assert exact.next_solution().statistics.nodes_visited == 7

    smaller = ac3_enumerator(problem, Ac3Options(max_nodes=6))
    earlier_schedules = []
    while True:
        result = smaller.next_solution()
        if result.status is BacktrackingEnumerationStatus.INCOMPLETE:
            break
        assert result.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
        assert result.evaluation is not None
        earlier_schedules.append(
            frozenset(result.evaluation.assignment.placements.items())
        )
    assert earlier_schedules
    assert smaller.statistics.nodes_visited == 6


def test_ac3_is_deterministic_and_nary_constraints_are_not_arc_revised() -> None:
    variables = tuple(SessionVariable(name, 0) for name in ("a", "b", "c"))
    slots = tuple(
        WeeklySlot(f"w{week}-{name}", week) for week in (0, 1) for name in "abc"
    )
    nary = _NaryConstraint(variables)
    problem = _problem({name: slots for name in "abc"}, (nary,))

    first = ac3_propagate(problem)
    first_calls = nary.calls
    second = ac3_propagate(problem)

    assert first.consistent and second.consistent
    assert first.domains == problem.domains
    assert first.domains == second.domains
    assert first.arc_revisions == second.arc_revisions
    assert first.compatibility_checks == second.compatibility_checks
    assert nary.calls == first_calls
    assert nary.calls == 0

    def ordered_solutions(selected_problem: CspProblem):
        enumerator = ac3_enumerator(selected_problem)
        ordered = []
        while True:
            result = enumerator.next_solution()
            if result.status is BacktrackingEnumerationStatus.EXHAUSTED:
                return ordered
            assert result.status is BacktrackingEnumerationStatus.SOLUTION_FOUND
            assert result.evaluation is not None
            ordered.append(frozenset(result.evaluation.assignment.placements.items()))

    ordered = ordered_solutions(problem)
    second_constraint = _NaryConstraint(variables)
    repeated = ordered_solutions(
        _problem({name: slots for name in "abc"}, (second_constraint,))
    )
    assert ordered == repeated
    solutions = set(ordered)
    assert solutions
    assert nary.calls > 0
    for placements in solutions:
        assert sum(slot.week_index for _, slot in placements) <= 1


def test_ac3_predicate_and_pruning_statistics_are_exact_for_binary_arc() -> None:
    a, b = SessionVariable("a", 0), SessionVariable("b", 0)
    a0, a1 = WeeklySlot("a0", 0), WeeklySlot("a1", 1)
    b0 = WeeklySlot("b0", 2)
    relation = _TableConstraint("table_ab", (a, b), frozenset({(a1, b0)}))
    problem = _problem({"a": (a0, a1), "b": (b0,)}, (relation,))

    result = ac3_propagate(problem)

    assert result.domain_prunings == 1
    assert result.compatibility_checks >= len(relation.calls)
    assert result.compatibility_checks > 0
    assert result.arcs_enqueued >= 2 * len(problem.binary_constraints)


def test_ac3_search_counts_satisfaction_and_nested_compatibility_calls_exactly(
    monkeypatch,
) -> None:
    a, b = SessionVariable("a", 0), SessionVariable("b", 0)
    a0, a1 = WeeklySlot("a0", 0), WeeklySlot("a1", 1)
    b0, b1 = WeeklySlot("b0", 2), WeeklySlot("b1", 3)
    relation = _TableConstraint(
        "table_ab", (a, b), frozenset({(a0, b1), (a1, b0), (a1, b1)})
    )
    problem = _problem({"a": (a0, a1), "b": (b0, b1)}, (relation,))
    observed = {"satisfied": 0, "compatible": 0}

    def instrument(cls, method_name: str, counter: str) -> None:
        original = getattr(cls, method_name)

        def counted(self, *args, **kwargs):
            observed[counter] += 1
            return original(self, *args, **kwargs)

        monkeypatch.setattr(cls, method_name, counted)

    for constraint_type in (
        DeadlineConstraint,
        UniqueSlotConstraint,
        WeeklyCapacityConstraint,
        _TableConstraint,
    ):
        instrument(constraint_type, "is_satisfied", "satisfied")
    instrument(UniqueSlotConstraint, "is_compatible", "compatible")
    instrument(_TableConstraint, "is_compatible", "compatible")

    result = ac3_search(problem)

    assert result.status is CspSolverStatus.FEASIBLE
    assert observed == {"satisfied": 28, "compatible": 26}
    assert result.statistics.constraint_checks == 54
    assert result.statistics.compatibility_checks == 26
    assert result.statistics.compatibility_checks == observed["compatible"]
    assert result.statistics.constraint_checks == (
        observed["satisfied"] + observed["compatible"]
    )


def test_ac3_rejects_invalid_complete_assignment_at_validation_boundary() -> None:
    variables = tuple(SessionVariable(name, 0) for name in ("a", "b", "c"))
    slots = tuple(
        WeeklySlot(f"w{week}-{name}", week) for week in (0, 1) for name in "abc"
    )
    nary = _NaryConstraint(variables)
    problem = _problem({name: slots for name in "abc"}, (nary,))
    assignment = CspAssignment(
        tuple(
            AssignmentEntry(variable, WeeklySlot(slot_id, week))
            for variable, slot_id, week in zip(
                variables, ("w1-a", "w1-b", "w0-c"), (1, 1, 0), strict=True
            )
        )
    )

    evaluation = evaluate_assignment(problem, assignment)

    assert evaluation.status is CspEvaluationStatus.BRANCH_REJECTED
    assert nary.constraint_id in evaluation.failed_checks
